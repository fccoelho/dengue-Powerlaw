#!/usr/bin/env python3
"""
run_classifier.py -- out-of-fold rarity classifier at a chosen aggregation level.

This is the *producer* of the artefacts that Figures 12 and 13 plot. It exists
because those two figures used to recompute their own experiment inside
``Manuscript/figures/generate_figures.py``: a different feature list, a single
train/test split instead of rolling origins, and -- at municipality level -- a
Q_75 pooled across every municipality in the country. That last choice made the
task answerable from population size alone, so the ROC panel was reporting the
ability to tell a large municipality from a small one.

What it writes
--------------
``<out>.parquet``            out-of-fold predictions, one row per
                             (origin, rung, unit, year): y_true, y_prob. This
                             is what Figure 12 plots. Rung ``POP`` is the
                             population-only reference, not a ladder rung.
``<out>_metrics.csv``        Brier, AUC, log loss, prevalence and n per
                             origin x rung, so the figure can be checked
                             against a table.
``<out>_importance.csv``     permutation importance on the held-out fold, per
                             origin x feature, plus a joint row
                             (``SCALING_BLOCK``) for the block shuffled as a
                             unit. What Figure 13 plots.
``<out>.meta.json``          panel, ladder, threshold scope, seeds, versions.

Guarantees
----------
* The threshold Q_p is estimated inside each fold from training years only.
* With ``--threshold-scope per_unit`` (the default at ``--level city``) a unit
  needs ``--min-unit-years`` seasons of its own history to receive a threshold;
  units below it are excluded from the fold and counted, never handed the
  pooled value.
* A run that produces no evaluable fold writes nothing and exits non-zero,
  because an empty artefact is indistinguishable from a run with no skill.

Usage
-----
    python run_classifier.py --level city --origins 2021,2022,2023,2024,2025
    python run_classifier.py --level state --percentile 0.90 --out clf_state
    python run_classifier.py --panel panel.parquet --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from run_ablation import (
    SEED,
    assert_ladder_is_nested,
    assert_no_contemporaneous_leak,
    attach_climate,
    build_ladder,
    classifier_fold,
    load_panel,
    resolve_threshold_scope,
    validate_panel,
)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Out-of-fold rarity classifier for the dengue power-law "
                    "manuscript (feeds Figures 12 and 13).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--level", choices=["state", "city"], default="city")
    ap.add_argument("--region", default="BR")
    ap.add_argument("--origins", default="2021,2022,2023,2024,2025")
    ap.add_argument("--percentile", type=float, default=0.75,
                    help="rarity threshold: Y = 1[S_t > Q_p]")
    ap.add_argument("--threshold-scope", choices=["pooled", "per_unit"],
                    default=None,
                    help="default: per_unit at --level city, pooled at "
                         "--level state")
    ap.add_argument("--min-unit-years", type=int, default=5)
    ap.add_argument("--max-iter", type=int, default=300)
    ap.add_argument("--classifier-model", default=None,
                    choices=["logistic", "hist_gradient_boosting"],
                    help="classifier passed to run_ablation.make_classifier; "
                         "default is run_ablation.DEFAULT_CLASSIFIER (logistic, "
                         "which extrapolates in log-odds)")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--panel", default=None, help="reuse a cached panel parquet")
    ap.add_argument("--save-panel", default=None)
    ap.add_argument("--attach-climate", action="store_true",
                    help="build the climate block from data/[0-9]*.parquet")
    ap.add_argument("--climate-parquet", default=None)
    ap.add_argument("--importance-rungs", default="M3",
                    help="comma-separated rungs whose held-out permutation "
                         "importance is written")
    ap.add_argument("--n-repeats", type=int, default=5,
                    help="permutation repeats per feature")
    ap.add_argument("--importance-max-rows", type=int, default=4000,
                    help="cap on held-out rows used for permutation "
                         "importance; subsampled, never augmented")
    ap.add_argument("--out", default="classifier_results")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    return ap.parse_args(argv)


def _auc(y, p):
    from run_ablation import _auc as _a

    return _a(y, p)


def _log_loss(y, p, eps=1e-12):
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    y = np.asarray(y, dtype=float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _permutation_importance(model, X, y, feature_names, n_repeats, seed,
                            max_rows=None):
    """Held-out permutation importance on the Brier score of the fold."""
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import make_scorer, brier_score_loss

    if len(np.unique(y)) < 2:
        return None, "only one class in the held-out fold"
    X = X.copy()
    if max_rows is not None and len(X) > max_rows:
        rng = np.random.default_rng(seed)
        keep = rng.choice(len(X), size=max_rows, replace=False)
        X, y = X.iloc[keep], np.asarray(y)[keep]
    scorer = make_scorer(brier_score_loss, greater_is_better=False)
    res = permutation_importance(
        model, X, y, scoring=scorer, n_repeats=n_repeats,
        random_state=seed, n_jobs=1,
    )
    return pd.DataFrame({
        "feature": feature_names,
        # greater_is_better=False, so the sign is flipped back to read
        # "how much worse does the Brier score get when this is shuffled"
        "importance": -res.importances_mean,
        "sd": res.importances_std,
    }).sort_values("importance", ascending=False), ""


#: The power-law block. Kept as an explicit list rather than discovered by a
#: name pattern: a block that silently matches nothing turns "the scaling
#: descriptors did not help" into a statement about an empty feature set.
SCALING_BLOCK = ("prev_alpha_size", "prev_xmin_size", "prev_alpha_trend")


def _permutation_importance_joint(model, X, y, cols, n_repeats, seed,
                                  max_rows=None):
    """Brier degradation when ``cols`` are shuffled together, as one block.

    Permuting one column of a collinear group while its partners stay in place
    lets the model read the information off the partners, so per-column
    importance collapses towards zero even when the group carries signal. The
    scaling descriptors are rank-correlated at about 0.65 within units, which is
    exactly this case, so the group is also scored jointly.
    """
    from sklearn.metrics import brier_score_loss

    cols = [c for c in cols if c in X.columns]
    if not cols:
        return None, "no block columns present"
    if len(np.unique(y)) < 2:
        return None, "only one class in the held-out fold"

    X = X.copy()
    y = np.asarray(y)
    if max_rows is not None and len(X) > max_rows:
        rng = np.random.default_rng(seed)
        keep = rng.choice(len(X), size=max_rows, replace=False)
        X, y = X.iloc[keep], y[keep]

    base = brier_score_loss(y, model.predict_proba(X)[:, 1])
    rng = np.random.default_rng(seed + 1)
    deltas = np.empty(n_repeats)
    block = X[cols].to_numpy()
    for i in range(n_repeats):
        Xp = X.copy()
        Xp[cols] = block[rng.permutation(len(X))]
        deltas[i] = brier_score_loss(y, model.predict_proba(Xp)[:, 1]) - base
    return pd.DataFrame({"feature": ["SCALING_BLOCK"],
                         "importance": [float(deltas.mean())],
                         "sd": [float(deltas.std(ddof=0))]}), ""


def main(argv=None):
    args = parse_args(argv)
    verbose = not args.quiet
    warnings.filterwarnings("ignore", category=FutureWarning)
    t0 = time.time()

    scope = resolve_threshold_scope(args)
    origins = [int(o) for o in args.origins.split(",") if o.strip()]

    df = load_panel(args.level, args.region, args.panel, verbose=verbose)
    df, panel_notes = validate_panel(df, args.level, verbose=verbose)
    unit_col = "state" if args.level == "state" else "geocode"

    if args.attach_climate or args.climate_parquet:
        df, climate_notes = attach_climate(
            df, args.level, args.region, parquet=args.climate_parquet,
            verbose=verbose)
        panel_notes += climate_notes

    if args.save_panel:
        df.to_parquet(args.save_panel, index=False)
        if verbose:
            print(f"[panel] cached to {args.save_panel}")

    ladder, dropped, climate_info = build_ladder(df)
    assert_ladder_is_nested(ladder)
    assert_no_contemporaneous_leak(ladder, df)

    if verbose:
        print("\n[rungs]")
        for rung in (r for r in ("M1", "M2", "M3") if r in ladder):
            print(f"  {rung:3s} {len(ladder[rung]):2d} features  {ladder[rung]}")
        print(f"[classifier] Q{int(args.percentile*100)}, scope={scope}, "
              f"min_unit_years={args.min_unit_years}")

    if args.dry_run:
        print(f"\n[dry-run] {len(origins)} origin(s) x {len(ladder)} rung(s) "
              f"+ population-only reference; panel {len(df)} rows, "
              f"{df[unit_col].nunique()} units")
        print("[dry-run] nothing was fitted and nothing was written")
        return 0

    importance_rungs = [r.strip() for r in args.importance_rungs.split(",") if r.strip()]

    oof_rows, metric_rows, imp_rows = [], [], []
    for origin in origins:
        fold = classifier_fold(
            df, ladder, unit_col, origin, percentile=args.percentile,
            threshold_scope=scope, min_unit_years=args.min_unit_years,
            seed=args.seed, max_iter=args.max_iter,
            model=args.classifier_model, verbose=verbose,
        )
        if fold is None:
            if verbose:
                print(f"[classifier {origin}] no evaluable fold")
            continue

        y_val, val_ok = fold["y_val"], fold["val_ok"]
        units = fold["val_ok"][unit_col].to_numpy()
        years = fold["val_ok"]["year"].to_numpy()
        probs = dict(fold["rungs"])
        base = float(fold["y_train"].mean())
        probs["M0"] = np.full(len(y_val), base)
        if fold["pop_prob"] is not None:
            probs["POP"] = fold["pop_prob"]

        for rung, p in probs.items():
            oof_rows.append(pd.DataFrame({
                "origin": int(origin), "rung": rung, "level": args.level,
                "unit": units, "year": years,
                "y_true": y_val, "y_prob": np.asarray(p, dtype=float),
                "threshold_scope": scope, "percentile": args.percentile,
            }))
            metric_rows.append({
                "origin": int(origin), "rung": rung, "level": args.level,
                "threshold_scope": scope, "percentile": args.percentile,
                "brier": float(np.mean((np.asarray(p) - y_val) ** 2)),
                "auc": _auc(y_val, p),
                "log_loss": _log_loss(y_val, p),
                "prevalence": float(np.mean(y_val)),
                "n_val": int(len(y_val)),
                "n_units": int(val_ok[unit_col].nunique()),
                "n_train": int(len(fold["train_ok"])),
                "n_units_below_history_floor": int(fold["n_short_units"]),
                "n_val_excluded_short_history": int(fold["n_unlabelled_val"]),
                "threshold": fold["thr_desc"],
            })

        for rung in importance_rungs:
            model = fold["models"].get(rung)
            if model is None:
                continue
            feats = ladder[rung]
            imp, why = _permutation_importance(
                model, val_ok[feats], y_val, feats, args.n_repeats, args.seed,
                max_rows=args.importance_max_rows)
            if imp is None:
                if verbose:
                    print(f"[importance {origin}/{rung}] skipped: {why}")
                continue
            imp.insert(0, "origin", int(origin))
            imp.insert(1, "rung", rung)
            imp_rows.append(imp)

            blk, bwhy = _permutation_importance_joint(
                model, val_ok[feats], y_val,
                [c for c in feats if c in SCALING_BLOCK],
                args.n_repeats, args.seed, max_rows=args.importance_max_rows)
            if blk is None:
                if verbose:
                    print(f"[importance {origin}/{rung}] block skipped: {bwhy}")
            else:
                blk.insert(0, "origin", int(origin))
                blk.insert(1, "rung", rung)
                imp_rows.append(blk)

    if not oof_rows:
        raise SystemExit(
            "No fold could be evaluated, so nothing was written. Check the "
            f"origins ({origins}), the threshold scope ({scope}) and "
            f"--min-unit-years ({args.min_unit_years})."
        )

    out = Path(args.out)
    oof = pd.concat(oof_rows, ignore_index=True)
    oof.to_parquet(out.with_suffix(".parquet"), index=False)

    metrics = pd.DataFrame(metric_rows).sort_values(["origin", "rung"])
    metrics.to_csv(Path(f"{out}_metrics.csv"), index=False)

    if imp_rows:
        importance = pd.concat(imp_rows, ignore_index=True)
        importance["is_scaling"] = importance["feature"].isin(
            set(SCALING_BLOCK) | {"SCALING_BLOCK"})
        importance.to_csv(Path(f"{out}_importance.csv"), index=False)

    # A population-only reference that matches the full model is the signature
    # of a task that is really about unit size. Say so in the metadata rather
    # than leaving it for a reader to notice.
    pop = metrics[metrics["rung"] == "POP"][["origin", "auc"]]
    m3 = metrics[metrics["rung"] == "M3"][["origin", "auc"]]
    # aligned by origin, not by row position: both frames are filtered from
    # the same table, and a positional subtraction would compare different
    # forecast origins whenever one of them is missing.
    pair = m3.merge(pop, on="origin", suffixes=("_m3", "_pop"))
    pop_gap = (float((pair["auc_m3"] - pair["auc_pop"]).mean())
               if len(pair) else None)

    from run_ablation import DEFAULT_CLASSIFIER

    meta = {
        "classifier_model": args.classifier_model or DEFAULT_CLASSIFIER,
        "level": args.level, "region": args.region,
        "panel_rows": int(len(df)), "panel_units": int(df[unit_col].nunique()),
        "origins": origins, "percentile": args.percentile,
        "threshold_scope": scope, "min_unit_years": args.min_unit_years,
        "ladder": ladder, "unavailable_columns": dropped,
        "climate": climate_info, "panel_notes": panel_notes,
        "max_iter": args.max_iter, "seed": args.seed,
        "importance_rungs": importance_rungs, "n_repeats": args.n_repeats,
        "auc_gap_full_minus_population_only": pop_gap,
        "files": {
            "oof": str(out.with_suffix(".parquet")),
            "metrics": str(Path(f"{out}_metrics.csv")),
            "importance": (str(Path(f"{out}_importance.csv"))
                           if imp_rows else None),
        },
        "versions": {"python": sys.version.split()[0],
                     "pandas": pd.__version__, "numpy": np.__version__},
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    Path(f"{out}.meta.json").write_text(json.dumps(meta, indent=2, default=str))

    if verbose:
        print("\n[metrics]")
        piv = metrics.pivot_table(index="origin", columns="rung",
                                  values=["brier", "auc"])
        print(piv.round(4).to_string())
        if pop_gap is not None:
            print(f"\n[diagnostic] mean AUC gap (M3 - population only) = {pop_gap:+.3f}")
            if pop_gap < 0.05:
                print("             The full model barely beats population. Read "
                      "the rarity task as a size comparison, or switch to "
                      "--threshold-scope per_unit.")
        print(f"\n[out] {out.with_suffix('.parquet')}  ({len(oof)} OOF rows)")
        print(f"[out] {out}_metrics.csv")
        if imp_rows:
            print(f"[out] {out}_importance.csv")
        print(f"[out] {out}.meta.json")
        print(f"[time] {meta['elapsed_seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
