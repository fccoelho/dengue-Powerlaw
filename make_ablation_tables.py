"""Derive Tables 3 and 4, and the ablation facts, from the forecasting artefacts.

Why this exists
---------------
``make_tables.py`` derives everything the manuscript says about the *fits*, and
``--check`` fails when ``main.tex`` disagrees with the database. Nothing did the
same for the *predictive* results, and that gap produced a concrete defect: after
the estimator change and the data refresh, Figure 11 was regenerated from
``ablation_results.csv`` while the Table 3 cells and the figure caption were
still the hand-typed values of the superseded run. The figure and its own caption
then contradicted each other on the same page -- the panel said the scaling block
*helped* the 95th percentile by 4.9% in median, the caption said it *hurt* by
18.3%.

So the rule is the same here as there: every ablation number in the manuscript is
emitted from an artefact, never typed.

    python make_ablation_tables.py            # write generated/{tab_ablation,tab_exceed}.tex
    python make_ablation_tables.py --check    # audit only, exit 1 on staleness
    python make_ablation_tables.py --print    # dump the facts
    python make_ablation_tables.py --sources  # which artefacts exist, and why not

"Table 3 cannot be rendered: no size ladder" is therefore not a rendering bug --
it means ``ablation_results.csv`` is missing or holds something else, usually
because a run with a different ``--level``/``--tasks`` wrote over it. Run
``--sources`` first; it names the file and the reason.

Three verdicts, because a missing artefact is neither fresh nor stale:

* ``STALE``      -- ``main.tex`` carries a number an artefact contradicts.
* ``MISSING``    -- ``main.tex`` omits a number an artefact supplies.
* ``UNVERIFIED`` -- a source artefact is absent, so the number cannot be checked
                    either way. This is the state the state-level exceedance
                    results are in: they were produced by a run whose output was
                    overwritten by the size-only refresh, so the manuscript
                    quotes them, the repository cannot regenerate them, and no
                    one can tell whether they still hold after the new
                    extraction. It fails the check by default: a number nobody
                    can reproduce should not ship quietly.

Sign conventions, fixed once here because they are the easiest thing to get
backwards: a ``contrast`` row carries ``value = mean(L_reference - L_rung)`` and
``rel_improvement = value / mean(L_reference)``, so **positive means the later
rung has the lower loss** -- the later rung is better. ``improves`` is the
stricter flag (same direction *and* a bootstrap interval excluding zero);
``ci_excludes_zero`` alone also fires on intervals that exclude zero downwards.
Table 4's ``Origins CI excl. 0`` column reports ``improves``, not
``ci_excludes_zero``, and is labelled accordingly.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

GEN_DIR = os.path.join("Manuscript", "generated")

#: The primary loss for the decision rule: pinball at the median. Section 2.10
#: fixes this, so it is not a free parameter here.
PRIMARY_TAU = 0.5

#: The quantile the scaling block was expected to earn its keep at: a tail
#: exponent concerns extreme seasons, so any gain should concentrate here.
EXTREME_TAU = 0.95

#: Adjacent rungs of the ladder, as contrasts.
ADJACENT = ("M1-M0", "M2-M1", "M3-M2")

#: Every artefact the tables depend on. ``columns`` is the minimum a file must
#: have to be used at all: a file that exists but carries the wrong shape is
#: worse than a missing one, because it reads as present.
SOURCES = {
    "state_size": {
        "path": "ablation_results.csv",
        "meta": "ablation_results.meta.json",
        "columns": ("record_type", "task", "loss", "tau", "contrast", "rung"),
        # Content, not just shape. A file with the right columns but the wrong
        # ``task`` is the signature of one run overwriting another, and it used
        # to pass every check here while Table 3 quietly lost its source.
        "values": {"task": ("size",), "level": ("state",)},
        "what": "state-level size ablation",
        # The committed copy of this file is usually the run the text quotes, so
        # restoring it is the fix nine times out of ten -- and it is the only fix
        # that keeps the frozen hashes. Refitting is the fallback, and it needs
        # --force precisely because the clobbered file now belongs to another
        # experiment (run_ablation.guard_output refuses it otherwise).
        "restore": "git checkout -- ablation_results.csv ablation_results.meta.json",
        "how": ("python run_ablation.py --level state --attach-climate --force  "
                "(refit; new hashes, so re-freeze and recompile)"),
        # Keys written by the current run_ablation.py. Their absence means the
        # file was produced before the estimator change, so its numbers describe
        # an experiment the manuscript no longer reports.
        "design": ("extrapolating_regressor", "classifier_model"),
    },
    "state_exceedance": {
        "path": "ablation_state_exceedance.csv",
        "meta": "ablation_state_exceedance.meta.json",
        "columns": ("record_type", "task", "loss", "contrast", "rung"),
        "values": {"task": ("exceedance",), "level": ("state",)},
        "what": "state-level exceedance classifier",
        "design": ("extrapolating_classifier", "classifier_model"),
        # Not produced by any script in this repository as committed: the
        # size-only refresh overwrote ablation_results.csv without the
        # exceedance task. Left as a declared source so its absence is loud.
        "how": ("python run_ablation.py --level state --tasks exceedance "
                "--output ablation_state_exceedance.csv"),
    },
    "city_exceedance": {
        "path": "ablation_city.csv",
        "meta": "ablation_city.meta.json",
        "columns": ("record_type", "task", "loss", "contrast", "rung"),
        "values": {"task": ("exceedance",), "level": ("city",)},
        "what": "municipal-level exceedance ablation",
        "design": ("extrapolating_classifier", "classifier_model"),
        "how": ("python run_ablation.py --level city --tasks exceedance "
                "--output ablation_city.csv"),
    },
    "city_classifier": {
        "path": "classifier_results_metrics.csv",
        "meta": "classifier_results.meta.json",
        "columns": ("origin", "rung", "brier", "auc"),
        "values": {"level": ("city",)},
        "what": "municipal-level out-of-fold classifier metrics",
        "design": ("classifier_model",),
    },
    "classifier_oof": {
        "path": "classifier_results.parquet",
        "meta": "classifier_results.meta.json",
        "columns": ("origin", "rung", "unit", "y_true", "y_prob"),
        "what": "municipal-level out-of-fold predictions",
        "design": ("classifier_model",),
    },
    "classifier_importance": {
        "path": "classifier_results_importance.csv",
        "meta": "classifier_results.meta.json",
        "columns": ("origin", "rung", "feature", "importance"),
        "values": {"rung": ("M3",)},
        "what": "held-out permutation importance of the municipal classifier",
        "design": ("classifier_model",),
    },
}


def facts_get(objs=()):
    """Keys present in the classifier facts dict (a tiny readability shim)."""
    return dict(objs or {})


def _pct(x: float, d: int = 1) -> str:
    """Signed percentage for LaTeX prose, e.g. ``+4.9\\%``."""
    return f"{100 * float(x):+.{d}f}\\%"


def _num(x: float, d: int = 2) -> str:
    return f"{float(x):.{d}f}"


def _cases(x: float) -> str:
    """Case count with a LaTeX thousands separator, e.g. ``44{,}471``."""
    return f"{float(x):,.0f}".replace(",", "{,}")


def _sci(x: float) -> str:
    """Compact power-of-ten literal for LaTeX, e.g. ``$2.99\\times10^{5}$``."""
    x = float(x)
    if not np.isfinite(x) or x == 0:
        return "$0$"
    e = int(np.floor(np.log10(abs(x))))
    return f"${x / 10 ** e:.2f}\\times10^{{{e}}}$"


def _load(name: str, root: str = ".") -> tuple[pd.DataFrame | None, dict, str | None]:
    """Load one artefact. Returns ``(df, meta, problem)``; df is None if absent."""
    spec = SOURCES[name]
    path = os.path.join(root, spec["path"])
    if not os.path.exists(path):
        return None, {}, f"{spec['path']} absent ({spec['what']})"
    try:
        # The out-of-fold predictions are a parquet; everything else is csv.
        df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    except Exception as e:  # noqa: BLE001
        return None, {}, f"{spec['path']} unreadable: {e}"
    miss = [c for c in spec["columns"] if c not in df.columns]
    if miss:
        return None, {}, (f"{spec['path']} lacks the columns {miss}: it is not a "
                          f"{spec['what']} artefact")
    for col, allowed in spec.get("values", {}).items():
        if col not in df.columns:
            continue
        got = sorted(str(v) for v in df[col].dropna().unique())
        absent = [v for v in allowed if v not in got]
        if absent:
            return None, {}, (
                f"{spec['path']} holds {col}={got or ['nothing']} but a "
                f"{spec['what']} artefact needs {col} in {list(allowed)}: "
                f"another run has written over it")
    meta = {}
    mpath = os.path.join(root, spec["meta"])
    if os.path.exists(mpath):
        try:
            meta = json.loads(open(mpath, encoding="utf-8").read())
        except Exception:  # noqa: BLE001
            meta = {}
    return df, meta, None


def _signed_list(per_origin: dict, d: int = 3) -> str:
    """LaTeX list of signed per-origin values: ``$-0.064$, $+0.074$ and $+0.126$``."""
    vals = [per_origin[k] for k in sorted(per_origin)]
    body = ", ".join(f"${v:+.{d}f}$" for v in vals[:-1])
    return f"{body} and ${vals[-1]:+.{d}f}$"


def _contrast_summary(df: pd.DataFrame, task: str, tau: float | None = None) -> dict:
    """Per-contrast medians and counts over origins, for one task.

    ``tau`` selects one quantile of the grid. The size ladder writes a contrast
    row at each of the 19 quantiles, so a summary that forgets to filter reads
    95 rows and reports a median over the grid instead of over origins -- which
    would look plausible and be wrong. Exceedance rows carry no ``tau`` at all,
    so ``tau=None`` selects them.
    """
    sub = df[(df["record_type"] == "contrast") & (df["task"] == task)]
    if tau is None:
        sub = sub[sub["tau"].isna()]
    else:
        sub = sub[np.isclose(sub["tau"].astype(float), float(tau))]
    out = {}
    for c in ADJACENT:
        d = sub[sub["contrast"] == c]
        if d.empty:
            continue
        out[c] = {
            "n_origins": int(len(d)),
            "median_diff": float(d["value"].median()),
            "median_rel": (float(d["rel_improvement"].median())
                           if "rel_improvement" in d else None),
            "n_diff_positive": int((d["value"] > 0).sum()),
            "n_same_direction_ci": int(d["improves"].sum()),
            "n_ci_excludes_zero": int(d["ci_excludes_zero"].sum()),
            # Per-origin values, so the sentence that lists them can be
            # audited instead of retyped. Sorted by origin, not by row order.
            "per_origin": {int(o): float(v) for o, v in
                           d.sort_values("origin").set_index("origin")["value"].items()},
        }
        if "rel_improvement" in d:
            out[c]["n_rel_positive"] = int((d["rel_improvement"] > 0).sum())
    dec = df[(df["record_type"] == "decision") & (df["task"] == task)]
    for _, r in dec.iterrows():
        if r.get("contrast") in out:
            out[r["contrast"]]["verdict"] = r.get("verdict")
            out[r["contrast"]]["decision_origins"] = (
                int(r["value"]) if pd.notna(r.get("value")) else None)
    return out


def compute_facts(root: str = ".") -> dict:
    """Every quantity Tables 3 and 4 state, plus the warnings about them."""
    facts: dict = {
        "primary_tau": PRIMARY_TAU,
        "extreme_tau": EXTREME_TAU,
        "sources": {},
        "size": {},
        "exceed": {"state": None, "city": None},
        "classifier": {},
        "warnings": [],
    }

    for name, spec in SOURCES.items():
        _df, _meta, problem = _load(name, root)
        fresh = problem is None
        if fresh:
            missing = [k for k in spec.get("design", ()) if k not in _meta]
            if missing:
                fresh = False
                problem = (f"{spec['path']} predates the current design (meta has no "
                           f"{missing}): its numbers describe a superseded experiment")
        facts["sources"][name] = {"path": spec["path"], "what": spec["what"],
                                  "present": fresh, "problem": problem,
                                  "restore": spec.get("restore"),
                                  "how": spec.get("how")}
        if problem:
            if spec.get("restore"):
                hint = f" -- restore with: {spec['restore']}"
            elif spec.get("how"):
                hint = f" -- regenerate with: {spec['how']}"
            else:
                hint = ""
            facts["warnings"].append(problem + hint)

    # ---- Table 3: the size ladder at state level -------------------------
    df, meta, problem = _load("state_size", root)
    if df is not None and facts["sources"]["state_size"]["present"]:
        prim = df[(df["task"] == "size") & (df["loss"] == "pinball")
                  & np.isclose(df["tau"].astype(float), PRIMARY_TAU)]
        skill = prim[prim["record_type"] == "skill"]
        piv = skill.pivot_table(index="origin", columns="rung", values="value")
        facts["size"]["skill"] = {
            r: {"median": float(piv[r].median()), "min": float(piv[r].min()),
                "max": float(piv[r].max()),
                "n_positive": int((piv[r] > 0).sum()),
                "n_origins": int(piv[r].notna().sum())}
            for r in ("M1", "M2", "M3") if r in piv
        }
        facts["size"]["contrasts"] = _contrast_summary(df, "size", PRIMARY_TAU)

        # Absolute error of the same forecasts. Table 3 compares the rungs as a
        # ratio against persistence, which is the right scale for ranking them
        # and the wrong one for asking how large the miss is. These are the
        # case-scale errors, plus MASE, whose denominator is the mean absolute
        # one-step error the same unit showed in *training*: a value above one
        # therefore reads as "worse than carrying the previous count forward
        # unchanged", which is the only form comparable across units whose
        # counts differ by orders of magnitude.
        lr = df[(df["record_type"] == "loss") & (df["task"] == "size")
                & np.isclose(df["tau"].astype(float), PRIMARY_TAU)]
        absolute: dict = {}
        for loss in ("mae", "rmse", "mase"):
            sub = lr[lr["loss"] == loss]
            if sub.empty:
                continue
            piv = sub.pivot_table(index="origin", columns="rung", values="value")
            piv = piv.reindex(columns=[c for c in ("M0", "M1", "M2", "M3")
                                       if c in piv.columns])
            entry = {
                "by_origin": {str(int(o)): {r: float(v) for r, v in row.items()
                                            if pd.notna(v)}
                              for o, row in piv.iterrows()},
                "median": {r: float(piv[r].median()) for r in piv.columns},
                "n_origins": int(piv.notna().any(axis=1).sum()),
            }
            if loss == "mase":
                entry["n_above_one"] = {r: int((piv[r] > 1).sum())
                                        for r in piv.columns}
            absolute[loss] = entry
        if "mae" in absolute:
            # The record origin is whichever carries the largest mean error
            # across rungs. Deriving it rather than writing "2024" keeps the
            # comparison attached to the data if the series is extended.
            mae_bo = absolute["mae"]["by_origin"]
            mean_by_origin = {o: float(np.mean(list(v.values())))
                              for o, v in mae_bo.items()}
            record = max(mean_by_origin, key=mean_by_origin.get)
            earlier = [o for o in mae_bo if int(o) < int(record)]
            absolute["mae"]["record"] = {
                "origin": record,
                "min": min(mae_bo[record].values()),
                "max": max(mae_bo[record].values()),
            }
            if earlier:
                absolute["mae"]["before_record"] = {
                    "origins": sorted(earlier),
                    "min": min(min(mae_bo[o].values()) for o in earlier),
                    "max": max(max(mae_bo[o].values()) for o in earlier),
                }
        if absolute:
            facts["size"]["absolute"] = absolute

        # The quantile grid, for the extreme-tail claim.
        c = df[(df["record_type"] == "contrast") & (df["task"] == "size")
               & (df["contrast"] == "M3-M2") & (df["loss"] == "pinball")]
        if not c.empty:
            # ``ci_excludes_zero`` is direction-agnostic: an interval can exclude
            # zero while pointing against the block. Count both directions
            # separately, or the "three origins agree" criterion silently counts
            # agreement that the block is worse as agreement that it is better.
            c = c.copy()
            c["_excl"] = (c["ci_excludes_zero"].astype("boolean")
                          .fillna(False).astype(bool))
            c["_imp"] = c["improves"].astype("boolean").fillna(False).astype(bool)
            c["_against"] = c["_excl"] & ~c["_imp"]
            g = c.groupby("tau").agg(
                median_rel=("rel_improvement", "median"),
                n_favouring_ci=("_imp", "sum"),
                n_ci=("_excl", "sum"),
                n_against_ci=("_against", "sum"),
            )
            ext = c[np.isclose(c["tau"].astype(float), EXTREME_TAU)]
            facts["size"]["scaling_vs_tau"] = {
                "n_taus": int(len(g)),
                "grid_median_min": float(g["median_rel"].min()),
                "grid_median_max": float(g["median_rel"].max()),
                "taus_with_three_favouring": [
                    float(t) for t in g.index[g["n_favouring_ci"] >= 3]],
                "taus_with_three_against": [
                    float(t) for t in g.index[g["n_against_ci"] >= 3]],
                "at_extreme": {
                    "median_rel": float(ext["rel_improvement"].median()),
                    "n_positive": int((ext["rel_improvement"] > 0).sum()),
                    "n_negative": int((ext["rel_improvement"] < 0).sum()),
                    "n_same_direction_ci": int(ext["improves"].sum()),
                    "n_ci_excludes_zero": int(ext["ci_excludes_zero"].sum()),
                    "per_origin": {int(o): float(v) for o, v in
                                   ext.set_index("origin")["rel_improvement"].items()},
                },
            }
        facts["size"]["model"] = meta.get("model")
        facts["size"]["estimator_extrapolates"] = bool(
            meta.get("extrapolating_regressor", False))
        facts["size"]["decision_rule"] = meta.get("decision_rule")

    # ---- Table 4, municipal block ---------------------------------------
    cdf, cmeta, _ = _load("city_exceedance", root)
    if cdf is not None and facts["sources"]["city_exceedance"]["present"]:
        facts["exceed"]["city"] = {
            "contrasts": _contrast_summary(cdf, "exceedance"),
            "threshold_scope": cmeta.get("classifier_threshold_scope"),
            "n_units": cmeta.get("panel_units"),
            "auc_median": {r: float(v) for r, v in
                           cdf[cdf["record_type"] == "loss"]
                           .groupby("rung")["auc"].median().items()},
        }
        # AUC lives on the per-unit ``loss`` rows of the exceedance artefact.
        # The scaling block's gain and its rank among the rungs are quoted in
        # the manuscript from *this* artefact, not from the classifier one,
        # because this is the run that carries the bootstrap intervals and the
        # verdicts; the classifier artefact exists for figures 12 and 13.
        loss = cdf[cdf["record_type"] == "loss"]
        if "auc" in loss and not loss.empty:
            piv_auc = loss.pivot_table(index="origin", columns="rung", values="auc")
            if {"M1", "M2", "M3"}.issubset(piv_auc.columns):
                gain = piv_auc["M3"] - piv_auc["M2"]
                facts["exceed"]["city"]["auc_gain_scaling_median"] = float(gain.median())
                facts["exceed"]["city"]["n_origins_scaling_best"] = int(
                    (piv_auc[["M1", "M2", "M3"]].idxmax(axis=1) == "M3").sum())
        pops = cdf[cdf["record_type"] == "diagnostic"]
        if not pops.empty:
            facts["exceed"]["city"]["population_only_auc_median"] = float(
                pops["auc"].median())
            facts["exceed"]["city"]["population_only_auc_range"] = [
                float(pops["auc"].min()), float(pops["auc"].max())]

    # ---- Table 4, state block (may be absent) ---------------------------
    sdf, _smeta, _ = _load("state_exceedance", root)
    if sdf is not None and facts["sources"]["state_exceedance"]["present"]:
        facts["exceed"]["state"] = {"contrasts": _contrast_summary(sdf, "exceedance")}

    # ---- The classifier artefact behind figures 12 and 13 ---------------
    kdf, kmeta, _ = _load("city_classifier", root)
    if kdf is not None and facts["sources"]["city_classifier"]["present"]:
        piv_auc = kdf.pivot_table(index="origin", columns="rung", values="auc")
        piv_b = kdf.pivot_table(index="origin", columns="rung", values="brier")
        facts["classifier"] = {
            "auc_median": {r: float(piv_auc[r].median()) for r in piv_auc},
            "brier_median": {r: float(piv_b[r].median()) for r in piv_b},
            "prevalence_min": float(kdf["prevalence"].min()),
            "prevalence_max": float(kdf["prevalence"].max()),
            "threshold_scope": kmeta.get("threshold_scope"),
            "n_repeats": kmeta.get("n_repeats"),
            "brier_by_origin": {
                str(int(o)): {r: float(v) for r, v in row.items() if pd.notna(v)}
                for o, row in piv_b.iterrows()},
        }
        if {"M1", "M2", "M3"}.issubset(piv_auc.columns):
            gain = piv_auc["M3"] - piv_auc["M2"]
            facts["classifier"]["auc_gain_scaling_median"] = float(gain.median())
            facts["classifier"]["n_origins_scaling_best"] = int(
                (piv_auc[["M1", "M2", "M3"]].idxmax(axis=1) == "M3").sum())
        if "train_prevalence" in kdf.columns:
            facts["classifier"]["train_prevalence"] = {
                int(o): float(v) for o, v in
                kdf.drop_duplicates("origin").set_index("origin")["train_prevalence"].items()}

        # Two artefacts of the same municipal classifier disagree slightly on
        # AUC. They are different runs (one with bootstrap intervals, one with
        # five repeated out-of-fold fits), so a small gap is expected; a large
        # one would mean they are not the same experiment, which is what this
        # bounds.
        city = facts["exceed"]["city"]
        if city and "M3" in city["auc_median"]:
            gap = abs(city["auc_median"]["M3"] - facts["classifier"]["auc_median"].get("M3", np.nan))
            facts["classifier"]["auc_gap_vs_ablation_city"] = float(gap)
            if np.isfinite(gap) and gap > 0.05:
                facts["warnings"].append(
                    f"the two municipal artefacts disagree on median AUC(M3) by "
                    f"{gap:.3f}; they may not describe the same experiment")
            elif np.isfinite(gap) and gap > 0.005:
                facts["warnings"].append(
                    f"the two municipal artefacts give median AUC(M3) as "
                    f"{city['auc_median']['M3']:.3f} (ablation_city) and "
                    f"{facts['classifier']['auc_median']['M3']:.3f} "
                    f"(classifier_results); the manuscript quotes the first and "
                    f"figure 12 draws the second")

    # ---- Predicted-versus-observed agreement, the map's own numbers -------
    odf, _ometa, _ = _load("classifier_oof", root)
    if odf is not None and facts["sources"]["classifier_oof"]["present"]:
        # The map draws one origin. It is the origin in which the largest share
        # of municipalities exceeded their own upper quartile -- the record
        # season -- rather than a fixed year, so that extending the series moves
        # the figure instead of freezing it on 2024.
        rate = odf.groupby("origin")["y_true"].mean()
        origin = int(rate.idxmax())
        rungs = set(odf["rung"].dropna())
        rung = "M3" if "M3" in rungs else sorted(rungs)[-1]
        s = odf[(odf["rung"] == rung) & (odf["origin"] == origin)].copy()
        if len(s) > 5 and s["y_prob"].nunique() > 5:
            s["q"] = pd.qcut(s["y_prob"], 5, labels=False, duplicates="drop")
            g = (s.groupby("q").agg(observed=("y_true", "mean"),
                                    predicted=("y_prob", "mean"))
                 .sort_index())
            gap = g["predicted"] - g["observed"]
            facts["map_agreement"] = {
                "origin": str(origin), "rung": rung, "n_units": int(len(s)),
                "observed_rate": float(s["y_true"].mean()),
                "predicted_mean": float(s["y_prob"].mean()),
                "first": float(g["observed"].iloc[0]),
                "fourth": float(g["observed"].iloc[len(g) - 2]),
                "last": float(g["observed"].iloc[-1]),
                "gap_min": float(gap.min()), "gap_max": float(gap.max()),
            }

    # ---- Permutation importance behind figure 10 -------------------------
    idf, _imeta, _ = _load("classifier_importance", root)
    if idf is not None and facts["sources"]["classifier_importance"]["present"]:
        agg = (idf.groupby("feature")["importance"]
                  .agg(median="median",
                       n_positive=lambda s: int((s > 0).sum()),
                       n_origins="size"))
        indiv = agg.drop(index="SCALING_BLOCK", errors="ignore")
        block_rows = idf[idf["feature"] == "SCALING_BLOCK"]["importance"]
        facts["importance"] = {
            "n_features": int(len(indiv)),
            "n_negative": int((indiv["median"] < 0).sum()),
            "worst": [
                {"feature": f, "median": float(r["median"]),
                 "n_positive": int(r["n_positive"])}
                for f, r in indiv.sort_values("median").head(3).iterrows()
            ],
            "block": ({
                "median": float(block_rows.median()),
                "min": float(block_rows.min()),
                "max": float(block_rows.max()),
                "n_positive": int((block_rows > 0).sum()),
                "n_origins": int(len(block_rows)),
            } if not block_rows.empty else None),
        }

    # ---- Fallback: municipal block from the fresh classifier artefact ----
    # The exceedance ablation carries the bootstrap intervals and therefore the
    # verdicts, so it is preferred. When it is stale, the out-of-fold classifier
    # artefact still gives the differentials and their signs -- just not the
    # intervals. Reporting the signs and marking the rule ``not tested`` is
    # honest; reporting the stale differentials would not be, because they were
    # produced by a tree classifier that the manuscript no longer uses (in 2024
    # it emits the training base rate, which is why every rung scores a Brier of
    # 0.40 in that artefact but 0.25 in the logistic one).
    if facts["exceed"]["city"] is None and facts["classifier"]:
        piv = kdf.pivot_table(index="origin", columns="rung", values="brier")
        contrasts = {}
        for later, ref in (("M1", "M0"), ("M2", "M1"), ("M3", "M2")):
            if later in piv and ref in piv:
                d = piv[ref] - piv[later]
                contrasts[f"{later}-{ref}"] = {
                    "n_origins": int(d.notna().sum()),
                    "median_diff": float(d.median()),
                    "n_diff_positive": int((d > 0).sum()),
                    "n_same_direction_ci": None,
                    "n_ci_excludes_zero": None,
                    "verdict": None,
                }
        facts["exceed"]["city"] = {
            "source": "classifier_results_metrics.csv",
            "no_intervals": True,
            "contrasts": contrasts,
            "auc_median": {r: v for r, v in facts["classifier"]["auc_median"].items()
                           if r not in ("POP",)},
            "population_only_auc_median": facts["classifier"]["auc_median"].get("POP"),
            "n_units": kmeta.get("panel_units"),
            "threshold_scope": kmeta.get("threshold_scope"),
        }
        for k in ("auc_gain_scaling_median", "n_origins_scaling_best"):
            if k in facts["classifier"]:
                facts["exceed"]["city"][k] = facts["classifier"][k]
        facts["warnings"].append(
            "the municipal block is drawn from the classifier artefact because "
            "ablation_city.csv is stale; the CI column and the pre-registered "
            "verdicts need the re-run")

    return facts


# ---------------------------------------------------------------- rendering

def render_ablation_table(f: dict) -> str:
    """LaTeX float for Table 3, built from the size ladder."""
    if not f.get("size", {}).get("skill"):
        prob = (f.get("sources", {}).get("state_size") or {}).get("problem")
        raise ValueError(
            "no size ladder in the artefacts; cannot render Table 3\n"
            f"    ablation_results.csv: {prob or 'absent'}\n"
            "    if the committed copy is the run the text quotes, restore it:\n"
            "        git checkout -- ablation_results.csv "
            "ablation_results.meta.json\n"
            "    otherwise refit it (new hashes: re-freeze, then recompile):\n"
            "        python run_ablation.py --level state --attach-climate --force")
    s, c = f["size"]["skill"], f["size"]["contrasts"]
    cells = []
    for r in ("M1", "M2", "M3"):
        d = s[r]
        cells.append(f"${_num(d['median'])}$ [${_num(d['min'])}$, ${_num(d['max'])}$]")
    d32 = c.get("M3-M2", {})
    diff = (f"${_pct(d32['median_rel'])}$ "
            f"({d32['n_rel_positive']}/{d32['n_origins']})") if d32 else "--"
    return (
        "% GENERATED by make_ablation_tables.py from ablation_results.csv -- do not edit.\n"
        "% Regenerate with: python make_ablation_tables.py\n"
        "\\begin{table}[htbp]\n\\centering\n\\small\n"
        "\\setlength{\\tabcolsep}{4pt}\n"
        "\\caption{Ablation ladder over the five forecast origins 2021--2025, state\n"
        "level (26 units, 23--25 evaluated per origin), for the single target. Cells\n"
        "give the median across origins of the skill score\n"
        "$1-L(\\text{rung})/L(\\text{M0})$ against calibrated persistence, with the\n"
        "range over origins in brackets; the primary loss is the pinball loss at\n"
        "$\\tau=0.5$. The last column is the M3$-$M2 differential at that loss:\n"
        "median relative change, positive meaning that the scaling block improves\n"
        "the forecast, and in parentheses the number of origins in which M3 does so\n"
        "by sign alone. The classifier is reported separately in\n"
        "Table~\\ref{tab:exceed}.}\n"
        "\\label{tab:ablation_results}\n"
        "\\begin{tabular}{@{}lrrrr@{}}\n\\toprule\n"
        "Target & M1 & M2 & M3 & M3$-$M2 \\\\\n\\midrule\n"
        f"Size $S_t$ & {cells[0]} & {cells[1]} & {cells[2]} & {diff}\\\\\n"
        "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )


def render_exceed_table(f: dict) -> str:
    """LaTeX float for Table 4, state and municipal blocks.

    A block whose artefact is absent is rendered as a single conspicuous row
    rather than omitted. Dropping it would silently delete the paper's only
    ``supported`` verdict; keeping the old numbers would be worse. A placeholder
    cannot be overlooked in the compiled PDF and disappears the moment the run
    happens.
    """
    def rows(block: dict, label: str) -> list[str]:
        names = ("M1$-$M0 (lagged metrics)", "M2$-$M1 (climate)",
                 "M3$-$M2 (scaling)")
        out = []
        for i, (c, name) in enumerate(zip(ADJACENT, names)):
            d = block["contrasts"][c]
            head = label if i == 0 else ""
            if d.get("verdict") == "supported":
                verdict = "\\textbf{supported}"
            elif d.get("verdict") == "not_supported":
                verdict = "not supported"
            else:
                verdict = "\\emph{not tested}"
            ci = ("--" if d.get("n_same_direction_ci") is None
                  else f"{d['n_same_direction_ci']}/{d['n_origins']}")
            out.append(
                f"{head} & {name} & ${d['median_diff']:+.3f}$ & "
                f"{d['n_diff_positive']}/{d['n_origins']} & {ci} & {verdict}\\\\")
        return out

    lines: list[str] = []
    state = f["exceed"].get("state")
    if state:
        lines += rows(state, "State (26 units)")
    else:
        lines.append(
            "\\multicolumn{6}{@{}l}{\\emph{State level: no committed artefact; run "
            "\\texttt{run\\_ablation.py --level state --tasks exceedance}}}\\\\")
    lines.append("\\addlinespace")

    city = f["exceed"].get("city")
    if city:
        n_units = city.get("n_units")
        label = (f"Municipal ({int(n_units):,} units)".replace(",", "{,}")
                 if n_units else "Municipal")
        lines += rows(city, label)
    else:
        lines.append("\\multicolumn{6}{@{}l}{\\emph{Municipal level: no artefact}}\\\\")

    pop = (city or {}).get("population_only_auc_median")
    aucs = [v for v in ((city or {}).get("auc_median") or {}).values() if v > 0.5]
    auc_txt = f"${min(aucs):.2f}$--${max(aucs):.2f}$" if aucs else "the fitted rungs"

    caption = (
        "\\caption{The large-season classifier at both aggregation levels, five\n"
        "forecast origins 2021--2025. The target is whether a unit exceeds the 75th\n"
        "percentile of its own history; the loss is the Brier score. Positive values\n"
        "mean the later rung has the lower Brier score, and ``median'' is over the\n"
        "five origins. ``Origins CI excl.\\ 0'' counts the origins with a\n"
        "same-direction improvement whose 95\\% unit-bootstrap interval excludes\n"
        "zero, which is what the decision rule of Section~\\ref{sec:ablation}\n"
        "requires in at least 3 of 5. The municipal rows use a within-unit\n"
        "threshold, without which the label is a population indicator; there the\n"
        "population-only reference reaches a median AUC of "
        f"${pop:.2f}$ against {auc_txt} for the fitted rungs.}}"
        if pop else
        "\\caption{The large-season classifier at both aggregation levels.}"
    )
    return (
        "% GENERATED by make_ablation_tables.py -- do not edit.\n"
        "% Regenerate with: python make_ablation_tables.py\n"
        "\\begin{table}[htbp]\n\\centering\n\\small\n"
        "\\setlength{\\tabcolsep}{5pt}\n" + caption + "\n"
        "\\label{tab:exceed}\n"
        "\\begin{tabular}{@{}llrrrr@{}}\n\\toprule\n"
        "Level & Contrast & $\\Delta$Brier, median & Origins $\\Delta>0$ & "
        "Origins CI excl.\\ 0 & Verdict\\\\\n\\midrule\n"
        + "\n".join(lines) +
        "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )



def render_regression_table(f: dict) -> str:
    """LaTeX float for the absolute error of the size forecast.

    Table 3 ranks the rungs by a ratio against persistence and says nothing
    about magnitude: a skill of zero is compatible with a miss of a thousand
    cases or of three hundred thousand. This is the same forecast measured on
    the case scale, with MASE beside it so that the reader can see where the
    ladder falls relative to doing nothing.
    """
    ab = (f.get("size") or {}).get("absolute") or {}
    if "mae" not in ab or "mase" not in ab:
        prob = (f.get("sources", {}).get("state_size") or {}).get("problem")
        raise ValueError(
            "ablation_results.csv carries no absolute-error rows (mae/rmse/mase "
            "at tau=0.5); cannot render the absolute table\n"
            f"    ablation_results.csv: {prob or 'absent'}\n"
            "    regenerate with: python run_ablation.py --level state "
            "--attach-climate --force")

    origins = sorted(ab["mae"]["by_origin"], key=int)
    rungs = [r for r in ("M0", "M1", "M2", "M3") if r in ab["mae"]["median"]]
    span = len(rungs) + 1
    rule = r"\\"                      # LaTeX row terminator
    head = "Origin & " + " & ".join(rungs) + " " + rule

    def block_title(label: str) -> str:
        return (r"\multicolumn{%d}{@{}l}{\emph{%s}}%s"
                % (span, label, rule))

    def rows(loss: str, fmt) -> list[str]:
        out = []
        for o in origins:
            cells = [fmt(ab[loss]["by_origin"].get(o, {}).get(r)) for r in rungs]
            out.append(o + " & " + " & ".join(cells) + " " + rule)
        out.append(r"\addlinespace")
        out.append("median & "
                   + " & ".join(fmt(ab[loss]["median"].get(r)) for r in rungs)
                   + " " + rule)
        return out

    body = [
        "% GENERATED by make_ablation_tables.py from ablation_results.csv "
        "-- do not edit.\n% Regenerate with: python make_ablation_tables.py",
        r"\begin{table}[htbp]", r"\centering", r"\small",
        r"\setlength{\tabcolsep}{5pt}",
    ]
    body.append(
        "\\caption{Absolute out-of-sample error of the size forecast, state "
        "level,\nfive forecast origins, at $\\tau=0.5$. Upper block: median "
        "absolute error\nin cases. Lower block: MASE, the same error divided by "
        "the mean absolute\none-step error the unit showed in training, so that "
        "a value above one\nmeans the forecast is worse than carrying the "
        "previous season's count\nforward unchanged. The last two rows of the "
        "lower block give the median\nMASE and the number of origins in which it "
        "exceeds one. The ratios these\nerrors produce against persistence are "
        "in Table~\\ref{tab:ablation_results}.}"
    )
    body.append(r"\label{tab:regression}")
    body.append(r"\begin{tabular}{@{}l" + "r" * len(rungs) + r"@{}}")
    body.append(r"\toprule")
    body.append(block_title("MAE, cases"))
    body.append(r"\addlinespace")
    body.append(head)
    body.append(r"\midrule")
    body += rows("mae", lambda v: _cases(v) if v is not None else "--")
    body.append(r"\midrule")
    body.append(block_title("MASE, scaled by the naive within-unit error"))
    body.append(r"\addlinespace")
    body.append(head)
    body.append(r"\midrule")
    body += rows("mase", lambda v: f"{v:.2f}" if v is not None else "--")
    up = ab["mase"].get("n_above_one") or {}
    if up:
        body.append(r"\addlinespace")
        body.append("origins with MASE $>1$ & "
                    + " & ".join(f"{up[r]} of {len(origins)}" for r in rungs
                                 if r in up) + " " + rule)
    body.append(r"\bottomrule")
    body.append(r"\end{tabular}")
    body.append(r"\end{table}")
    return "\n".join(body) + "\n"


# ------------------------------------------------------------------- claims


def claims(f: dict) -> list[dict]:
    """Prose statements the artefacts must satisfy.

    ``want`` must appear in ``main.tex``; each ``stale`` entry is a value that
    was actually published in an earlier draft, so a regression to it is caught
    by name rather than by re-deriving it. ``needs`` names the artefact a claim
    depends on: when that artefact is missing the claim is reported as
    UNVERIFIED instead of being silently skipped.
    """
    s = f.get("size", {}).get("skill") or {}
    c = f.get("size", {}).get("contrasts") or {}
    tau = f.get("size", {}).get("scaling_vs_tau") or {}
    ext = tau.get("at_extreme") or {}
    clf = f.get("classifier") or {}
    city = f.get("exceed", {}).get("city") or {}
    out: list[dict] = []

    if s:
        m1 = s["M1"]
        out += [
            {"name": "M1 median skill",
             "want": f"median skill ${_num(m1['median'])}$",
             "stale": ["median skill $0.10$"], "needs": "state_size"},
            {"name": "M1 skill range",
             "want": f"${_num(m1['min'])}$ to ${_num(m1['max'])}$",
             "stale": ["$-0.59$ to $0.71$"], "needs": "state_size"},
            {"name": "M2 median skill",
             "want": f"M2 has median ${_num(s['M2']['median'])}$",
             "stale": ["M2 has median $-0.02$"], "needs": "state_size"},
            {"name": "M3 median skill",
             "want": f"and M3 ${_num(s['M3']['median'])}$",
             "stale": ["M3 $-0.14$"], "needs": "state_size"},
        ]
    if "M3-M2" in c:
        d = c["M3-M2"]
        out.append(
            {"name": "M3-M2 differential at primary tau",
             "want": f"of ${_pct(d['median_rel'])}$",
             "stale": ["$-1.5\\%$"], "needs": "state_size"})
    if ext:
        out += [
            {"name": "extreme-tau median differential",
             "want": f"$\\tau={f['extreme_tau']:.2f}$ the median differential is "
                     f"${_pct(ext['median_rel'])}$",
             "stale": ["median $-18.3\\%$", "median of $18.3\\%$"],
             "needs": "state_size"},
            {"name": "extreme-tau sign count",
             "want": f"positive in {ext['n_positive']} of "
                     f"{ext['n_positive'] + ext['n_negative']} origins",
             "stale": ["negative in \\emph{all five} origins",
                       "in every origin tested"],
             "needs": "state_size"},
            {"name": "extreme-tau interval count",
             "want": f"2 of {ext['n_positive'] + ext['n_negative']}",
             "stale": [], "needs": "state_size"},
            {"name": "no tau favours the block",
             "want": ("no quantile has three of the five origins agreeing in favour"
                      if not tau.get("taus_with_three_favouring")
                      else f"{len(tau['taus_with_three_favouring'])} of "
                           f"{tau.get('n_taus')} quantiles"),
             "stale": ["0 of 19 quantiles clearing the criterion"],
             "needs": "state_size"},
            {"name": "taus where three origins agree against the block",
             "want": f"{len(tau.get('taus_with_three_against', []))} of "
                     f"the {tau.get('n_taus')} levels",
             "stale": [], "needs": "state_size"},
        ]
    if city:
        auc = city.get("auc_median") or {}
        d32 = city["contrasts"]["M3-M2"]
        # Which artefact backs the municipal claims: the ablation when it is
        # fresh (it has the intervals), otherwise the classifier.
        src = ("city_exceedance" if not city.get("no_intervals")
               else "city_classifier")
        if {"M1", "M2", "M3"}.issubset(auc):
            out.append(
                {"name": "municipal AUC ladder",
                 "want": f"{auc['M1']:.2f}$, ${auc['M2']:.2f}$ and ${auc['M3']:.2f}$",
                 "stale": ["$0.59$, $0.63$ and $0.64$"], "needs": src})
        out.append(
            {"name": "municipal scaling Brier",
             "want": f"median of ${d32['median_diff']:+.3f}$",
             "stale": [], "needs": src})
        out.append(
            {"name": "municipal origins where the scaling block helps",
             "want": f"in {d32['n_diff_positive']} of {d32['n_origins']} origins",
             "stale": ["improves the Brier score in 4 of\n5 origins"],
             "needs": src})
        if "auc_gain_scaling_median" in city:
            out.append(
                {"name": "municipal AUC gain of the scaling block",
                 "want": f"median ${city['auc_gain_scaling_median']:+.3f}$ over M2",
                 "stale": ["median $+0.008$ over M2"], "needs": src})
        if "n_origins_scaling_best" in city:
            out.append(
                {"name": "municipal origins where M3 ranks best",
                 "want": f"best of the three in {city['n_origins_scaling_best']} of 5",
                 "stale": ["best of the three in 4 of 5"], "needs": src})

    # State-level exceedance. These used to be two hand-typed literals, one of
    # which asserted a `supported` verdict that the refreshed run no longer
    # produces: the lagged block now clears the rule at municipal level and
    # fails at state level, on 23--25 units per origin. Both directions are
    # derived here so the sentence cannot outlive the artefact again.
    st = f.get("exceed", {}).get("state") or {}
    sm1 = (st.get("contrasts") or {}).get("M1-M0") or {}
    sm32 = (st.get("contrasts") or {}).get("M3-M2") or {}
    if sm1:
        out += [
            {"name": "state lagged block, sign count",
             "want": f"positive in {sm1['n_diff_positive']} of "
                     f"{sm1['n_origins']} origins",
             "stale": [],
             "needs": "state_exceedance"},
            {"name": "state lagged block, decision count",
             "want": f"exclude zero in only {sm1['n_same_direction_ci']} of the "
                     f"{sm1['n_origins']} origins",
             "stale": ["excluding zero in 3 of the 5 origins"],
             "needs": "state_exceedance"},
            {"name": "state lagged block, per-origin differentials",
             "want": _signed_list(sm1["per_origin"]),
             "stale": ["$-0.047$, $+0.074$, $+0.134$, $+0.175$ and $+0.159$"],
             "needs": "state_exceedance"},
            {"name": "state lagged block, verdict",
             "want": ("fails the rule at state level"
                      if sm1.get("verdict") != "supported"
                      else "clears the rule at state level"),
             "stale": [], "needs": "state_exceedance"},
        ]
    if sm32:
        out.append(
            {"name": "state scaling Brier at the municipal comparison",
             "want": f"median of ${sm32['median_diff']:+.3f}$",
             "stale": [], "needs": "state_exceedance"})

    # The municipal lagged block is now the paper's only `supported` verdict.
    cm1 = ((f.get("exceed", {}).get("city") or {}).get("contrasts") or {}).get("M1-M0") or {}
    if cm1:
        out.append(
            {"name": "municipal lagged block, verdict",
             "want": ("the only contrast in this study to satisfy the "
                      "pre-specified rule"
                      if cm1.get("verdict") == "supported"
                      else "no rung can be credited at this level"),
             "stale": [("no rung can be credited at this level"
                        if cm1.get("verdict") == "supported"
                        else "the only contrast in this paper to satisfy")],
             "needs": "city_exceedance"})
        out.append(
            {"name": "municipal lagged block, decision count",
             "want": f"excluding zero in {cm1['n_same_direction_ci']} of "
                     f"{cm1['n_origins']} origins",
             "stale": [], "needs": "city_exceedance"})
    # The record-year paragraph: its numbers come from the classifier artefact,
    # and its base rate needs a column that only a re-run of run_classifier.py
    # writes.
    if clf:
        b24 = (clf.get("brier_by_origin") or {}).get("2024") if clf else None
        if b24:
            out.append(
                {"name": "2024 Brier by rung",
                 "want": f"the fitted rungs about ${b24['M3']:.2f}$",
                 "stale": ["every rung scores a Brier score of about $0.40$"],
                 "needs": "city_classifier"})
        out.append(
            {"name": "2024 training base rate",
             "want": "training base rate of 26.4\\%", "stale": [],
             "needs": "train_prevalence"})

    # Absolute error of the size forecast. Table 3 is a ratio and these are the
    # levels; both are read from the same loss rows, so a drift in either shows
    # up here rather than in a reader's arithmetic.
    ab = (f.get("size") or {}).get("absolute") or {}
    mae, rmse, mase = ab.get("mae"), ab.get("rmse"), ab.get("mase")
    if mae and mase:
        m, ms = mae["median"], mase["median"]
        up = mase.get("n_above_one") or {}
        out += [
            {"name": "median MAE of the size forecast",
             "want": f"${_cases(m['M1'])}$ cases under M1 and "
                     f"${_cases(m['M3'])}$ under M3",
             "stale": [], "needs": "state_size"},
            {"name": "median MAE of calibrated persistence",
             "want": f"against ${_cases(m['M0'])}$ for calibrated persistence",
             "stale": [], "needs": "state_size"},
            {"name": "median MASE of the best rung",
             "want": f"The best rung on that measure is M2, at ${ms['M2']:.2f}$",
             "stale": [], "needs": "state_size"},
            {"name": "origins where MASE exceeds one",
             # Every rung clears the naive benchmark in the same number of
             # origins; if a refit ever breaks that, the sentence has to change
             # rather than silently describe four rungs with one number.
             "want": (f"exceeds one in {min(up.values())} of the "
                      f"{mase['n_origins']} origins for every rung"
                      if len(set(up.values())) == 1
                      else f"exceeds one in {min(up.values())} to "
                           f"{max(up.values())} of the {mase['n_origins']} origins"),
             "stale": [], "needs": "state_size"},
        ]
        rec, aft = mae.get("record") or {}, mae.get("before_record") or {}
        if rec:
            out.append(
                {"name": "MAE in the record season",
                 "want": f"runs from {_sci(rec['min'])} to {_sci(rec['max'])} cases",
                 "stale": [], "needs": "state_size"})
        if aft:
            out.append(
                {"name": "MAE in the origins before the record",
                 "want": (f"against {_sci(aft['min'])} to {_sci(aft['max'])} in "
                          f"the {len(aft['origins'])} origins before it"),
                 "stale": [], "needs": "state_size"})
    if rmse:
        r = rmse["median"]
        out.append(
            {"name": "median RMSE ladder",
             "want": (f"${_cases(r['M1'])}$, ${_cases(r['M3'])}$ and "
                      f"${_cases(r['M0'])}$"),
             "stale": [], "needs": "state_size"})

    # What the classifier is using. Permutation importance answers a different
    # question from the ablation -- how much a predictor is *relied on*, not how
    # much it *adds* -- and the two numbers are quoted together, so both are
    # derived here to keep the contrast attached to the artefacts.
    imp = f.get("importance") or {}
    if imp.get("block") and imp.get("worst"):
        b, w = imp["block"], imp["worst"]
        out += [
            {"name": "predictors with negative permutation importance",
             "want": (f"{imp['n_negative']} of the {imp['n_features']} individual "
                      f"predictors have a negative median permutation"),
             "stale": [], "needs": "classifier_importance"},
            {"name": "the most detrimental predictors",
             "want": (f"${w[0]['median']:.3f}$, ${w[1]['median']:.3f}$ and "
                      f"${w[2]['median']:.3f}$"),
             "stale": [], "needs": "classifier_importance"},
            {"name": "joint scaling block importance",
             "want": (f"by a median of ${b['median']:+.3f}$ (range "
                      f"${b['min']:.3f}$ to ${b['max']:.3f}$)"),
             "stale": [], "needs": "classifier_importance"},
        ]

    # The map's own numbers: predicted quintile against the realised outcome.
    ma = f.get("map_agreement") or {}
    if ma:
        out += [
            {"name": "realised exceedance by predicted quintile",
             "want": (f"rises from ${ma['first']:.2f}$ in the lowest quintile to "
                      f"${ma['fourth']:.2f}$ in the fourth and settles at "
                      f"${ma['last']:.2f}$ in the fifth"),
             "stale": [], "needs": "classifier_oof"},
            {"name": "level gap of the map prediction",
             "want": (f"between ${abs(ma['gap_max']):.2f}$ and "
                      f"${abs(ma['gap_min']):.2f}$ below the observed share"),
             "stale": [], "needs": "classifier_oof"},
            {"name": "overall means of the map",
             "want": (f"${ma['predicted_mean']:.2f}$ against an observed "
                      f"${ma['observed_rate']:.2f}$"),
             "stale": [], "needs": "classifier_oof"},
        ]

    for o in out:
        if o["needs"] == "train_prevalence":
            o["available"] = "train_prevalence" in (
                facts_get(objs=(f.get("classifier") or {})))
        else:
            o["available"] = f["sources"].get(o["needs"], {}).get("present", False)
    return [o for o in out if o["want"]]


def _squash(text: str) -> str:
    """Collapse all whitespace runs to single spaces.

    Prose wraps, so a claim that is present in the manuscript straddles a
    newline about as often as not. Searching the raw ``.tex`` reported those as
    MISSING and asked for a rebuild that was not needed -- and a check that
    cries wolf over its own line breaks is a check that gets ignored, which is
    how a real stale value slips through. Both comparisons below, the stale
    literals as well as the wanted one, run on squashed text.
    """
    return " ".join(text.split())


def audit_prose(tex_path: str, f: dict) -> dict:
    """Sort every claim into ok / stale / missing / unverified."""
    res = {"stale": [], "missing": [], "unverified": [], "ok": []}
    if not os.path.exists(tex_path):
        res["stale"].append({"name": "main.tex", "found": ["absent"],
                             "want": tex_path})
        return res
    tex = _squash(open(tex_path, encoding="utf-8", errors="replace").read())
    for c in claims(f):
        if not c["available"]:
            res["unverified"].append({"name": c["name"], "want": c["want"],
                                      "source": c["needs"]})
            continue
        stale = [s for s in c["stale"] if _squash(s) in tex]
        if stale:
            res["stale"].append({"name": c["name"], "found": stale,
                                 "want": c["want"]})
        elif _squash(c["want"]) not in tex:
            res["missing"].append({"name": c["name"], "want": c["want"]})
        else:
            res["ok"].append(c["name"])
    # Stale literals are checked even when their source is missing: a retired
    # value is a retired value whether or not the replacement is available.
    seen = {x["name"] for x in res["stale"]}
    for c in claims(f):
        if c["available"] or c["name"] in seen or not c["stale"]:
            continue
        hit = [s for s in c["stale"] if _squash(s) in tex]
        if hit:
            res["stale"].append({"name": c["name"], "found": hit,
                                 "want": "value of the superseded run"})
    return res


#: Sources whose absence blocks a table outright, as opposed to one that only
#: withdraws a column. ``state_size`` is Table 3; ``state_exceedance`` is the
#: paper's only ``supported`` verdict, and it is currently reproducible by no
#: committed script -- so it must fail loudly rather than read as an omission.
HARD_SOURCES = ("state_size", "state_exceedance")


def format_sources(f: dict) -> str:
    """One line per source artefact, with the reason when it cannot be used."""
    rows = []
    for name, s in f.get("sources", {}).items():
        hard = " (required)" if name in HARD_SOURCES else ""
        rows.append(f"{'ok     ' if s.get('present') else 'MISSING'}  "
                    f"{s['path']:32s}  {s['what']}{hard}")
        if not s.get("present"):
            rows.append(f"         -> {s.get('problem')}")
            if s.get("restore"):
                rows.append(f"         -> restore: {s['restore']}")
            if s.get("how"):
                rows.append(f"         -> rebuild: {s['how']}")
    return "\n".join(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=".")
    ap.add_argument("--tex", default=os.path.join("Manuscript", "main.tex"))
    ap.add_argument("--check", action="store_true", help="audit, do not write")
    ap.add_argument("--print", dest="dump", action="store_true")
    ap.add_argument("--sources", action="store_true",
                    help="report the state of the ablation artefacts and exit; "
                         "exit 1 when a source a table needs is unusable")
    ap.add_argument("--allow-unverified", action="store_true",
                    help="treat unverifiable claims as warnings, not failures")
    a = ap.parse_args(argv)

    f = compute_facts(a.root)

    if a.dump:
        print(json.dumps(f, indent=2, sort_keys=True, default=float))
        return 0

    if a.sources:
        print(format_sources(f))
        for w in f["warnings"]:
            print(f"note: {w}")
        bad = [n for n in HARD_SOURCES
               if not f["sources"].get(n, {}).get("present")]
        if bad:
            print(f"unusable required source(s): {', '.join(bad)}", file=sys.stderr)
        return 1 if bad else 0

    audit = audit_prose(os.path.join(a.root, a.tex), f)

    if a.check:
        for s in audit["stale"]:
            print(f"STALE      {s['name']}: found {s['found']} -> want {s['want']!r}",
                  file=sys.stderr)
        for m in audit["missing"]:
            print(f"MISSING    {m['name']}: {m['want']!r} not present", file=sys.stderr)
        for u in audit["unverified"]:
            print(f"UNVERIFIED {u['name']}: needs {u['source']}", file=sys.stderr)
        for w in f["warnings"]:
            print(f"NOTE       {w}", file=sys.stderr)
        bad = bool(audit["stale"] or audit["missing"]
                   or (audit["unverified"] and not a.allow_unverified))
        print(f"ablation tables: {len(audit['ok'])} claims verified, "
              f"{len(audit['stale'])} stale, {len(audit['missing'])} missing, "
              f"{len(audit['unverified'])} unverifiable")
        return 1 if bad else 0

    gen = os.path.join(a.root, GEN_DIR)
    os.makedirs(gen, exist_ok=True)
    # Render both before writing either. Writing them in a loop meant a failure
    # on the second left a fresh Table 3 beside a stale Table 4, which is worse
    # than writing neither: the manuscript would be internally inconsistent and
    # nothing would say so.
    rendered = {}
    for name, fn in (("tab_ablation.tex", render_ablation_table),
                     ("tab_exceed.tex", render_exceed_table),
                     ("tab_regression.tex", render_regression_table)):
        try:
            rendered[name] = fn(f)
        except ValueError as e:
            print(f"cannot write {name}: {e}", file=sys.stderr)
            print("no table was written, so Manuscript/generated/ is unchanged",
                  file=sys.stderr)
            return 2
    for name, tex in rendered.items():
        open(os.path.join(gen, name), "w", encoding="utf-8").write(tex)
        print(f"wrote {GEN_DIR}/{name}")
    with open(os.path.join(gen, "ablation_facts.json"), "w", encoding="utf-8") as fh:
        json.dump(f, fh, indent=2, sort_keys=True, default=float)
        fh.write("\n")
    print(f"wrote {GEN_DIR}/ablation_facts.json")
    for w in f["warnings"]:
        print(f"  note: {w}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
