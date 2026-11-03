#!/usr/bin/env python3
"""
run_ablation.py -- rolling-origin ablation ladder for the dengue power-law paper.

Writes ``ablation_results.csv``: the M0..M3 ladder of Manuscript
Section 2.10 ("Evaluation protocol and ablation ladder", Table 4),
evaluated by rolling origin over 2021-2025 at one aggregation level.

The ladder
----------
Four nested rungs built on the lagged panel returned by
``predictive_model.prepare_lagged_data``:

    M0  persistence        S_hat = S_{t-1}; D_hat = D_{t-1}; climatological
                           quantile for the peak week; training base rate for
                           the classifier
    M1  lagged metrics     M0 predictors + P_{t-1}, R0_{t-1}, log10(pop)
    M2  + climate          M1 + quarterly mean T and H of year t-1
                           (eight ``prev_avg_{temp,umid}_Q1..Q4`` columns, listed
                           in ``CLIMATE_BLOCK`` rather than discovered; the run
                           aborts if the panel supplies none of them)
    M3  + scaling          M2 + alpha_{t-1}, x_min,t-1, alpha-dot_{t-1}

An optional fifth rung ``M3dur`` (off by default, ``--include-dur-scaling``)
adds the *duration* scaling descriptors. It is a diagnostic, never the
quantity the manuscript's decision rule refers to.

The only contrast the paper needs is M2 -> M3. Everything else is context
for reading it. The script therefore (a) asserts that M2 and M3 differ
*only* in scaling columns, and (b) applies the decision rule stated in
Section 2.10 to that contrast and to the neighbours, and reports the
verdict mechanically.

Decision rule (fixed in advance, Section 2.10)
---------------------------------------------
The scaling block adds predictive value only if M3 beats M2 on the primary
loss with a 95% bootstrap interval for the paired difference that excludes
zero, in the same direction, for at least three of the five origins.
The script does not reinterpret that rule after seeing the numbers.

Losses
------
Primary: pinball loss at tau=0.5 for the regression targets (size, duration,
peak week) and the Brier score for the classifier -- both proper scoring
rules. Secondary: MAE, RMSE (case scale), and MASE scaled by the in-sample
naive error. The pinball loss is also reported on the full tau grid, which
is what Figure 11 needs.

Size is fitted on log1p and back-transformed with expm1. No Jensen bias
correction is applied to the quantile models: expm1 is monotone, so it maps
the log-space tau-quantile to the case-space tau-quantile, and a mean-matching
factor would destroy exactly that property.

Interval estimation
-------------------
Paired loss differentials are computed within unit-year and their
uncertainty is bootstrapped by resampling *units* (2,000 replicates by
default), preserving the within-unit correlation across years. A
Diebold-Mariano statistic with a Newey-West correction is attached to each
``decision`` row together with ``dm_n``, the number of origins it was
computed from: with five origins ``dm_n=5`` and the statistic is low-powered
by construction, so it is description, not evidence. The bootstrap interval
is the primary evidence.

The decision rule is evaluated on the *primary* loss only -- pinball at
tau=0.5 for the regressions, the Brier score for the classifier. Emitting a
verdict at all nineteen tau levels would multiply the decision without
adding evidence.

What lands in the CSV
---------------------
A tidy long table, one row per
``(record_type, task, loss, tau, rung, origin)``:

  ``loss``      the loss of each rung for each origin
  ``skill``     SS = 1 - L(rung)/L(M0), on the rows where both are defined
  ``contrast``  the paired differential against the previous rung, with a
                bootstrap interval; positive means the later rung is better
  ``decision``  the pre-registered verdict, one row per contrast and task

A rung that cannot be evaluated is written with an empty ``value`` and the
reason in ``note``, never as a zero.

Runtime
-------
At state level the panel is a few hundred rows, so a full run (5 origins,
19 tau, 3 targets, 3 rungs) is 855 model fits and typically a few minutes.
At ``--level city`` the panel is tens of thousands of rows and the same run
takes hours; use ``--dry-run`` first, ``--tau-grid none`` to get the
decision quickly, and ``--max-iter`` only for a pilot (the manuscript
specifies 300, and the value actually used is recorded in the metadata).

Usage
-----
    python run_ablation.py                       # state level, full tau grid
    python run_ablation.py --level city           # municipality cross-section
    python run_ablation.py --skip-tau-grid        # tau=0.5 only (fast)
    python run_ablation.py --panel panel.parquet  # reuse a cached panel
    python run_ablation.py --include-classifier   # add the rarity task

The output CSV is long/tidy: one row per
``(record_type, level, task, target, origin, rung, loss, tau)``.
``record_type`` is one of ``loss``, ``skill``, ``contrast``, ``decision``.
Rows are never written with a fabricated value: if a fold cannot be
evaluated the row is emitted with ``value`` empty and ``note`` explaining
why, so a missing number is never mistaken for a zero.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42

# --------------------------------------------------------------------------
# Rung definitions
# --------------------------------------------------------------------------
# Column names as produced by predictive_model.prepare_lagged_data().
# Every predictor is a lag (``prev_*``) or a static column; the unshifted
# climate aggregates (avg_temp_Q1..Q4) describe the target year and must
# never appear here.
LAGGED_SIZE = ["prev_total_cases"]
LAGGED_DUR = ["prev_ep_dur"]
LAGGED_PEAK = ["prev_peak_week"]
EPI = ["prev_R0"]
DEMO = ["prev_log_pop"]
SCALING_SIZE = ["prev_alpha_size", "prev_xmin_size", "prev_alpha_trend"]
SCALING_DUR = ["prev_alpha_dur", "prev_xmin_dur"]

#: The climate block that *defines* M2, named instead of discovered.
#:
#: Discovering it by scanning the panel for ``prev_avg_*`` made the rung
#: silently identical to M1 whenever the panel carried no climate at all, and
#: a rung that adds nothing yields a contrast of exactly 0.0 with a [0, 0]
#: interval -- the table then says "climate adds nothing" when the truth is
#: "climate was never entered". Listing the block makes that failure mode
#: impossible: what is missing is reported, and a rung that adds no column is
#: refused rather than scored.
CLIMATE_BLOCK = [
    f"prev_avg_{var}_Q{q}" for var in ("temp", "umid") for q in (1, 2, 3, 4)
]
CLIMATE_RE = re.compile(r"^prev_avg_(?:temp|umid)_Q[1-4]$")

#: Loss reported as primary for each task (Section 2.10).
PRIMARY_LOSS = {
    "size": ("pinball", 0.5),
    "duration": ("pinball", 0.5),
    "peak_week": ("pinball", 0.5),
    "exceedance": ("brier", None),
}

TARGETS = {"size": "total_cases", "duration": "ep_dur", "peak_week": "peak_week"}


# --------------------------------------------------------------------------
# Panel loading and validation
# --------------------------------------------------------------------------
def load_panel(level="state", region="BR", panel_path=None, verbose=True):
    """Return the lagged panel, from cache or from prepare_lagged_data()."""
    if panel_path:
        path = Path(panel_path)
        if not path.exists():
            raise FileNotFoundError(f"--panel {path} does not exist")
        df = pd.read_parquet(path)
        source = f"cache:{path}"
    else:
        try:
            import predictive_model  # imported lazily: it pulls duckdb, sqlite
        except ImportError as exc:  # missing project dependency
            raise SystemExit(
                f"cannot import predictive_model ({exc}). Install the project "
                "dependencies (e.g. `pip install -e .`) or pass --panel with a "
                "previously saved panel."
            ) from None

        df = predictive_model.prepare_lagged_data(region, target_year=None, level=level)
        source = f"prepare_lagged_data({region!r}, level={level!r})"

    if df is None or df.empty:
        raise RuntimeError(
            "The panel is empty, so no ablation can be run. "
            f"Source: {source}. "
            "prepare_lagged_data() needs (i) data/episcanner_*.parquet for the "
            "region and years and (ii) the episcanner/ powerlaw fits in "
            "powerlaw_results.db. Build the episcanner extracts with "
            "data_utils.ensure_episcanner_files(region, years) or point "
            "--panel at a previously saved panel."
        )

    df = df.copy()
    if verbose:
        print(f"[panel] {source}: {len(df)} rows x {df.shape[1]} cols")
    return df


def validate_panel(df, level, verbose=True):
    """Coerce types, drop non-calendar years and unknown units.

    Returns (df, warnings) where warnings is a list of human-readable strings
    that also end up in the run metadata.
    """
    notes = []
    unit_col = "state" if level == "state" else "geocode"

    if unit_col not in df.columns:
        raise RuntimeError(f"panel has no {unit_col!r} column; cannot identify units")
    if "year" not in df.columns:
        raise RuntimeError("panel has no 'year' column")

    if not pd.api.types.is_numeric_dtype(df["year"]):
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        notes.append("year column was non-numeric and was coerced to int")
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)

    # year == 0 holds the *pooled* fits written by the fitter; it is not a
    # calendar year and must not enter a rolling-origin evaluation.
    n_pooled = int((df["year"] == 0).sum())
    if n_pooled:
        df = df[df["year"] > 0]
        notes.append(f"dropped {n_pooled} row(s) with year=0 (pooled fits, not calendar years)")

    if level == "state" and (df["state"].astype(str) == "Unknown").any():
        n_unknown = int((df["state"].astype(str) == "Unknown").sum())
        df = df[df["state"].astype(str) != "Unknown"]
        notes.append(f"dropped {n_unknown} row(s) whose geocode mapped to 'Unknown' state")

    # One row per unit-year, or the paired bootstrap is meaningless.
    dup = df.duplicated(subset=[unit_col, "year"]).sum()
    if dup:
        raise RuntimeError(
            f"panel has {dup} duplicated ({unit_col}, year) rows; "
            "the paired bootstrap needs exactly one row per unit-year"
        )

    for target in TARGETS.values():
        if target not in df.columns:
            raise RuntimeError(f"panel is missing target column {target!r}")

    df = df.sort_values([unit_col, "year"]).reset_index(drop=True)
    if verbose:
        print(
            f"[panel] units={df[unit_col].nunique()} "
            f"years={df['year'].min()}-{df['year'].max()} "
            f"({df['year'].nunique()} distinct)"
        )
    return df, notes


# --------------------------------------------------------------------------
# The ladder
# --------------------------------------------------------------------------
def attach_climate(df, level, region, parquet=None, block=None, verbose=True):
    """Merge the quarterly climate block onto ``df`` as year-1 predictors.

    This is what ``prepare_lagged_data`` does, minus the silent skip: when
    ``get_quarterly_climate`` fails it returns an empty frame, the merge is
    never attempted, and the panel simply has no climate columns. Rebuilding
    the block here makes the failure visible instead of turning it into an
    ablation rung that is quietly empty.

    The merge is on ``(unit, year + 1)``: the climate of season t-1 becomes a
    predictor of season t, which is exactly what the ``prev_avg_*`` name
    asserts and what the one-year ``shift`` in ``prepare_lagged_data`` does.
    """
    unit_col = "state" if level == "state" else "geocode"
    block = list(CLIMATE_BLOCK if block is None else block)
    raw_cols = [c[len("prev_"):] if c.startswith("prev_") else c for c in block]

    available, missing = climate_availability(df, block)
    if not missing:
        if verbose:
            print(f"[climate] panel already carries all {len(block)} climate columns")
        return df, []

    if parquet:
        clim = pd.read_parquet(parquet)
        source = f"parquet:{parquet}"
    else:
        try:
            import predictive_model
        except ImportError as exc:
            raise SystemExit(f"cannot import predictive_model ({exc})") from None
        try:
            clim = predictive_model.get_quarterly_climate(region, level=level, strict=True)
        except SystemExit:
            raise
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(
                f"cannot build the climate block: {type(exc).__name__}: {exc}\n"
                "get_quarterly_climate() reads data/[0-9]*.parquet (Infodengue). "
                "Fetch those extracts (data_utils.ensure_episcanner_files / "
                "update_data.py) or pass --climate-parquet with a pre-aggregated "
                "file."
            ) from None
        source = "get_quarterly_climate(strict=True)"

    if unit_col not in clim.columns:
        raise SystemExit(
            f"climate source {source} has no {unit_col!r} column "
            f"(it has {list(clim.columns)})"
        )
    absent = [c for c in raw_cols if c not in clim.columns]
    if absent:
        raise SystemExit(
            f"climate source {source} is missing {absent}; expected the quarterly "
            f"aggregates {raw_cols} plus {unit_col!r} and 'year'"
        )

    clim = clim[[unit_col, "year"] + raw_cols].copy()
    clim["year"] = pd.to_numeric(clim["year"], errors="coerce")
    clim = clim.dropna(subset=["year"])
    clim["year"] = clim["year"].astype(int) + 1  # lag: climate of t-1 predicts t
    clim = clim.rename(columns={c: f"prev_{c}" for c in raw_cols})

    before = set(df.columns)
    df = df.merge(clim, on=[unit_col, "year"], how="left")
    added = [c for c in block if c in df.columns and c not in before]

    notes = []
    if not added:
        raise SystemExit(
            f"merging climate from {source} added no column; the panel keys "
            f"({unit_col!r}, 'year') did not match the climate frame"
        )
    coverage = float(df[added].notna().any(axis=1).mean())
    notes.append(
        f"climate block attached from {source}: {len(added)}/{len(block)} columns, "
        f"non-null on {coverage:.1%} of panel rows"
    )
    if verbose:
        print(f"[climate] {source}: +{len(added)} columns, coverage {coverage:.1%}")
        if missing:
            print(f"[climate] still unavailable: {missing}")
    if coverage < 0.5:
        notes.append(
            f"climate coverage is only {coverage:.1%}; the M2 rung will be fitted "
            "on mostly-missing climate and its contrast is hard to interpret"
        )
    return df, notes


def climate_availability(df, block=None):
    """Return ``(available, missing)`` for the declared climate block.

    ``available`` keeps the declared order and requires the column to hold at
    least one observation: a column that is present but entirely null is not a
    predictor and must not be counted as one.
    """
    block = list(CLIMATE_BLOCK if block is None else block)
    available, missing = [], []
    for col in block:
        ok = col in df.columns and not df[col].isnull().all()
        (available if ok else missing).append(col)
    return available, missing


def _diagnose_missing_climate(df, missing):
    """Actionable message for a panel that cannot supply the climate block."""
    climate_like = sorted(
        c for c in df.columns
        if "avg_" in c or "temp" in c.lower() or "umid" in c.lower()
    )
    lines = [
        f"The M2 rung cannot be built: the panel supplies none of the "
        f"{len(missing)} climate columns that define it.",
        "",
        f"  expected : {missing}",
        ("  present  : " + str(climate_like)) if climate_like
        else "  present  : none -- the panel has no climate-like column at all",
        "",
        "prepare_lagged_data() drops climate silently: it merges only",
        "``if not df_climate.empty``, so when get_quarterly_climate() fails it",
        "returns an empty frame, the merge is skipped, and avg_temp_Q* /",
        "avg_umid_Q* never reach the panel. Their prev_* lags then do not",
        "exist either, and nothing downstream notices.",
        "",
        "Options, in order of preference:",
        "  1. python run_ablation.py --attach-climate",
        "     rebuild the block from data/[0-9]*.parquet (Infodengue) here and",
        "     lag it to year-1",
        "  2. python run_ablation.py --climate-parquet CLIMATE.parquet",
        "     a pre-aggregated file with columns <unit>, year, avg_temp_Q1..Q4,",
        "     avg_umid_Q1..Q4",
        "  3. rebuild the panel with the Infodengue extracts in place and rerun",
        "     with --panel",
        "",
        "Refusing to run: M2 identical to M1 writes M2-M1 = 0.0 with a [0, 0]",
        "interval, which reads as 'climate adds nothing' when the truth is",
        "'climate was never entered'.",
    ]
    return "\n".join(lines)


def build_ladder(df, include_dur_scaling=False, climate_block=None,
                 require_climate=True):
    """Return ``(ladder, dropped, climate_info)``.

    A column is dropped from a rung when it is absent from the panel or is
    entirely null, because a predictor with no observations is not a
    predictor. Every drop is reported -- including the climate columns that
    were *expected* and did not arrive -- so the table can be read knowing
    which predictors were actually available.

    ``require_climate`` aborts when the panel supplies none of the declared
    climate block. That is not defensiveness: an M2 with no climate column is
    M1, and the M2-M1 contrast it produces is a zero by construction rather
    than a finding.
    """
    climate_cols, climate_missing = climate_availability(df, climate_block)
    if require_climate and not climate_cols:
        raise SystemExit(_diagnose_missing_climate(df, climate_missing))

    m1 = LAGGED_SIZE + LAGGED_DUR + LAGGED_PEAK + EPI + DEMO
    ladder = {
        "M1": m1,
        "M2": m1 + climate_cols,
        "M3": m1 + climate_cols + SCALING_SIZE,
    }
    if include_dur_scaling:
        ladder["M3dur"] = m1 + climate_cols + SCALING_SIZE + SCALING_DUR

    available, dropped = {}, {}
    for rung, cols in ladder.items():
        keep, gone = [], []
        for c in cols:
            if c not in df.columns or df[c].isnull().all():
                gone.append(c)
            else:
                keep.append(c)
        available[rung] = keep
        if gone:
            dropped[rung] = gone

    if climate_missing:
        # Expected for M2, absent from the panel -- and therefore absent from
        # ladder["M2"], which only holds what survived. Without this line they
        # disappear from the metadata too, which is exactly how the original
        # run came to report "unavailable_columns: {}" while M2 was empty.
        dropped.setdefault("M2", []).extend(climate_missing)

    # A block that vanishes leaves two consecutive rungs with the very same
    # features, and their contrast would be a fabricated 0.0. Catch it here,
    # where the message can still say which column went missing.
    order = [r for r in ("M1", "M2", "M3", "M3dur") if r in available]
    for prev, nxt in zip(order, order[1:]):
        if set(available[nxt]) == set(available[prev]):
            raise SystemExit(
                f"rung {nxt} ended up with exactly the same features as {prev} "
                f"({sorted(set(available[nxt]))}), so the {nxt}-{prev} contrast "
                "would be identically zero. Columns listed for "
                f"{nxt} but absent from the panel: {dropped.get(nxt, [])}"
            )

    climate_info = {
        "block": list(CLIMATE_BLOCK if climate_block is None else climate_block),
        "available": climate_cols,
        "missing": climate_missing,
    }
    return available, dropped, climate_info


def assert_ladder_is_nested(ladder, climate_block=None):
    """Structural guarantee that consecutive rungs differ only in their block.

    This is what licenses the sentence "a contrast between consecutive rungs
    is attributable to the block that was added". If a future edit adds a
    column to the wrong rung, the run dies here rather than producing a table
    that quietly supports the wrong claim.

    Nesting alone is not enough: M2 must be a *strict* superset of M1, or the
    M2-M1 contrast is zero by construction. The emptiness check is here as
    well as in ``build_ladder`` so that a hand-built ladder handed to this
    function cannot slip through either.
    """
    for rung in ("M1", "M2", "M3", "M3dur"):
        cols = ladder.get(rung, [])
        dupes = sorted({c for c in cols if cols.count(c) > 1})
        if dupes:
            raise AssertionError(
                f"rung {rung} lists {dupes} more than once; a column may belong "
                "to exactly one rung"
            )

    expected_climate = set(
        CLIMATE_BLOCK if climate_block is None else climate_block
    )
    order = [r for r in ("M1", "M2", "M3", "M3dur") if r in ladder]
    for prev, nxt in zip(order, order[1:]):
        added = set(ladder[nxt]) - set(ladder[prev])
        removed = set(ladder[prev]) - set(ladder[nxt])
        if removed:
            raise AssertionError(
                f"rung {nxt} drops columns relative to {prev}: {sorted(removed)}. "
                "Rungs must be nested."
            )
        if not added:
            raise AssertionError(
                f"rung {nxt} adds no column to {prev}: the {nxt}-{prev} contrast "
                "is identically zero and is not a result"
            )
        if nxt == "M2":
            bad = sorted(added - expected_climate)
            if bad:
                raise AssertionError(
                    f"M2 must add only the declared climate block; it also added "
                    f"{bad}"
                )
        if nxt in ("M3", "M3dur") and prev == "M2":
            bad = [c for c in added if c not in SCALING_SIZE + SCALING_DUR]
            if bad:
                raise AssertionError(
                    f"M3 must add only scaling descriptors; it also added {sorted(bad)}"
                )
        if nxt == "M3dur":
            bad = [c for c in added if c not in SCALING_DUR]
            if bad:
                raise AssertionError(
                    f"M3dur must add only duration scaling descriptors; it also added {sorted(bad)}"
                )
    return True


def assert_no_contemporaneous_leak(ladder, df):
    """Fail if a rung contains an unshifted (target-year) column.

    ``binary_classifier._get_feature_columns`` encodes the sanctioned feature
    set after the B4/B5 fix; when that module is importable we check against
    it directly. Otherwise we fall back to a prefix test.
    """
    try:
        from binary_classifier import _get_feature_columns

        sanctioned = set(_get_feature_columns(df))
    except Exception:  # xgboost/lightgbm absent, or module changed
        sanctioned = {c for c in df.columns if c.startswith("prev_")} | {"log_pop"}

    offenders = {}
    for rung, cols in ladder.items():
        bad = [c for c in cols if c not in sanctioned]
        if bad:
            offenders[rung] = bad
    if offenders:
        raise AssertionError(
            "rung(s) contain columns outside the sanctioned lagged feature set "
            f"(possible target-year leakage): {offenders}"
        )
    return True


# --------------------------------------------------------------------------
# Losses
# --------------------------------------------------------------------------
def pinball_loss(y_true, y_pred, tau):
    """Mean pinball (quantile) loss. Lower is better; tau=0.5 is half the MAE."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    diff = y_true - y_pred
    return float(np.mean(np.maximum(tau * diff, (tau - 1.0) * diff)))


def brier(y_true, p):
    """Brier score for binary probabilities."""
    y_true = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
    return float(np.mean((p - y_true) ** 2))


def mase_denominators(train_df, unit_col, target):
    """Per-unit in-sample naive MAE, used to scale absolute errors.

    MASE divides the out-of-sample absolute error by the mean absolute
    one-step (previous-season) error observed on the *training* years only.
    A value above 1 therefore means "worse than the naive forecast on the
    same unit", which is the only reading that is comparable across units
    differing by orders of magnitude.
    """
    denom = {}
    for unit, grp in train_df.groupby(unit_col):
        y = grp.sort_values("year")[target].to_numpy(dtype=float)
        if y.size < 2:
            continue
        naive = np.abs(np.diff(y))
        naive = naive[np.isfinite(naive)]
        if naive.size and naive.mean() > 0:
            denom[unit] = float(naive.mean())
    return denom


# --------------------------------------------------------------------------
# Baselines and estimators
# --------------------------------------------------------------------------
def persistence_predictions(train_df, val_df, unit_col, target, tau, kind):
    """The M0 forecast.

    For size and duration, persistence is scaled by the empirical tau-quantile
    of the training ratio (y_t / y_{t-1}) so that the rung also produces a
    quantile forecast at every tau on the grid, not only at the median. For
    peak week, which has no meaningful "previous value" forecast, the
    climatological training tau-quantile is used (Table 4: "historical median
    for P_t", generalised to the tau-quantile).
    """
    if kind == "peak_week":
        q = float(np.nanquantile(train_df[target].to_numpy(dtype=float), tau))
        return np.full(len(val_df), q, dtype=float)

    prev_col = f"prev_{target}"
    prev = val_df[prev_col].to_numpy(dtype=float)
    if kind == "size":
        # ratio computed on the case scale, on training rows only
        t = train_df.dropna(subset=[prev_col, target])
        prev_t = t[prev_col].to_numpy(dtype=float)
        y_t = t[target].to_numpy(dtype=float)
        ok = (prev_t > 0) & (y_t > 0)
        ratio = np.where(ok, y_t[ok] / prev_t[ok], np.nan) if ok.any() else np.array([])
    else:
        t = train_df.dropna(subset=[prev_col, target])
        prev_t = t[prev_col].to_numpy(dtype=float)
        y_t = t[target].to_numpy(dtype=float)
        ok = np.abs(prev_t) > 0
        ratio = np.where(ok, y_t[ok] / prev_t[ok], np.nan) if ok.any() else np.array([])

    ratio = ratio[np.isfinite(ratio)]
    if ratio.size < 5:
        # Not enough training ratios to calibrate the spread: fall back to a
        # pure persistence point forecast at every tau. Recorded, not hidden.
        return prev.copy()
    scale = float(np.quantile(ratio, tau))
    return np.clip(prev * scale, 0.0, None)


def fit_quantile_model(X_train, y_train, tau, model="hist_gradient_boosting", seed=SEED,
                       max_iter=300):
    """Fit the quantile regressor used for rungs M1..M3."""
    if model == "lightgbm":
        import lightgbm as lgb

        est = lgb.LGBMRegressor(
            objective="quantile",
            alpha=tau,
            n_estimators=max_iter,
            learning_rate=0.05,
            num_leaves=16,
            min_child_samples=5,
            random_state=seed,
            verbose=-1,
        )
    else:
        from sklearn.ensemble import HistGradientBoostingRegressor

        est = HistGradientBoostingRegressor(
            loss="quantile",
            quantile=tau,
            max_iter=max_iter,
            learning_rate=0.05,
            max_depth=5,
            random_state=seed,
        )
    est.fit(X_train, y_train)
    return est


def predict_cases(model, X, target):
    """Predict on the response scale.

    Size is fitted on log1p, so the prediction is mapped back with expm1.
    Because expm1 is monotone, the log-space tau-quantile becomes the
    case-space tau-quantile; no Jensen factor is applied (Section 2.8).
    """
    pred = model.predict(X)
    if target == "total_cases":
        pred = np.expm1(pred)
    return np.clip(np.asarray(pred, dtype=float), 0.0, None)


# --------------------------------------------------------------------------
# Uncertainty of a paired loss differential
# --------------------------------------------------------------------------
def paired_bootstrap(diff, units, n_boot=2000, seed=SEED, chunk=200):
    """Percentile bootstrap interval for mean(diff), resampling units.

    ``diff`` is the per-row paired loss differential L_reference - L_rung, so a
    positive mean means the rung has the lower loss. Resampling *units* rather
    than rows preserves the within-unit correlation across years, which is the
    dependence the naive interval would pretend away.
    """
    diff = np.asarray(diff, dtype=float)
    units = np.asarray(units)
    ok = np.isfinite(diff)
    diff, units = diff[ok], units[ok]
    if diff.size < 3:
        return {"mean": np.nan, "ci_low": np.nan, "ci_high": np.nan, "se": np.nan,
                "p_value": np.nan, "n": int(diff.size)}

    uniq, inv = np.unique(units, return_inverse=True)
    sums = np.bincount(inv, weights=diff, minlength=uniq.size)
    cnts = np.bincount(inv, minlength=uniq.size).astype(float)
    k = uniq.size
    if k < 2:
        return {"mean": float(diff.mean()), "ci_low": np.nan, "ci_high": np.nan,
                "se": np.nan, "p_value": np.nan, "n": int(diff.size)}

    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    done = 0
    while done < n_boot:
        b = min(chunk, n_boot - done)
        idx = rng.integers(0, k, size=(b, k))
        num = sums[idx].sum(axis=1)
        den = cnts[idx].sum(axis=1)
        means[done:done + b] = num / den
        done += b

    lo, hi = np.percentile(means, [2.5, 97.5])
    p = 2.0 * min(float((means <= 0).mean()), float((means >= 0).mean()))
    return {
        "mean": float(diff.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "se": float(means.std(ddof=1)),
        "p_value": min(1.0, p),
        "n": int(diff.size),
        "n_units": int(k),
    }


def diebold_mariano(series, max_lag=None):
    """Diebold-Mariano statistic with a Newey-West long-run variance.

    Applied to the per-origin mean differentials, which is the only pooling of
    origins that keeps the units of the statistic interpretable. With five
    origins the Newey-West lag truncation is zero and the statistic degenerates
    to a one-sample t-test, so it is reported as a descriptive complement to
    the bootstrap interval, never as the basis of the decision rule.
    """
    d = np.asarray(series, dtype=float)
    d = d[np.isfinite(d)]
    n = d.size
    if n < 2:
        return {"dm_stat": np.nan, "dm_pvalue": np.nan, "dm_n": int(n), "dm_lag": 0}

    if max_lag is None:
        max_lag = int(math.floor(4 * (n / 100.0) ** (2.0 / 9.0)))
    max_lag = max(0, min(max_lag, n - 1))

    dbar = d.mean()
    e = d - dbar
    gamma0 = float(np.dot(e, e) / n)
    lrv = gamma0
    for lag in range(1, max_lag + 1):
        w = 1.0 - lag / (max_lag + 1.0)  # Bartlett
        g = float(np.dot(e[lag:], e[:-lag]) / n)
        lrv += 2.0 * w * g
    if lrv <= 0:
        return {"dm_stat": np.nan, "dm_pvalue": np.nan, "dm_n": int(n), "dm_lag": int(max_lag)}

    stat = dbar / math.sqrt(lrv / n)
    try:
        from scipy import stats as _st

        p = float(2 * (1 - _st.norm.cdf(abs(stat))))
    except Exception:
        p = np.nan
    return {"dm_stat": float(stat), "dm_pvalue": p, "dm_n": int(n), "dm_lag": int(max_lag)}


COLUMNS = [
    "record_type", "level", "task", "target", "origin", "rung", "contrast",
    "model", "loss", "tau", "value", "n_train", "n_val", "n_units",
    "ci_low", "ci_high", "se", "p_value", "ci_excludes_zero", "improves",
    "verdict", "dm_stat", "dm_pvalue", "dm_n", "features", "note",
]


# --------------------------------------------------------------------------
# The regression ladder
# --------------------------------------------------------------------------
ADJACENT = [("M1", "M0"), ("M2", "M1"), ("M3", "M2"), ("M3dur", "M3")]


def _row_key(frame, unit_col):
    return frame[unit_col].astype(str) + "|" + frame["year"].astype(str)


def run_regression_ablation(df, ladder, unit_col, origins, tau_grid, model, seed,
                            n_boot, level, verbose=True, max_iter=300):
    """Evaluate M0..M3 on size, duration and peak week over the origins."""
    rows = []
    rung_order = [r for r in ("M0", "M1", "M2", "M3", "M3dur") if r in ladder or r == "M0"]
    #: per-origin mean differentials, for the Diebold-Mariano summary
    origin_diffs = {}

    for origin in origins:
        train = df[df["year"] < origin]
        val = df[df["year"] == origin]
        if train.empty or val.empty:
            if verbose:
                print(f"[origin {origin}] skipped: n_train={len(train)} n_val={len(val)}")
            continue
        units_val = _row_key(val, unit_col)
        if verbose:
            print(f"[origin {origin}] n_train={len(train)} n_val={len(val)} "
                  f"units={val[unit_col].nunique()}")

        for task, target in TARGETS.items():
            kind = task
            y_train_all = train[target].to_numpy(dtype=float)
            if np.isfinite(y_train_all).sum() < 10:
                continue

            for tau in tau_grid:
                losses = {}
                notes = {}
                preds_half = {}

                # --- M0: persistence / climatology, no fitting ---
                try:
                    pred0 = persistence_predictions(train, val, unit_col, target, tau, kind)
                    yv = val[target].to_numpy(dtype=float)
                    ok = np.isfinite(yv) & np.isfinite(pred0)
                    m0 = pd.Series(np.nan, index=units_val.values)
                    m0.iloc[np.flatnonzero(ok)] = np.maximum(
                        tau * (yv[ok] - pred0[ok]), (tau - 1.0) * (yv[ok] - pred0[ok])
                    )
                    losses["M0"] = m0
                    if abs(tau - 0.5) < 1e-9:
                        preds_half["M0"] = pred0
                    notes["M0"] = ("persistence scaled by the training "
                                   "tau-quantile of the ratio y_t/y_{t-1}"
                                   if kind != "peak_week"
                                   else "climatological training tau-quantile of peak week")
                except Exception as exc:  # noqa: BLE001
                    notes["M0"] = f"baseline failed: {type(exc).__name__}: {exc}"

                # --- M1..M3: quantile models ---
                for rung in rung_order:
                    if rung == "M0" or rung not in ladder:
                        continue
                    feats = ladder[rung]
                    if not feats:
                        notes[rung] = "no feature available in the panel"
                        continue
                    sub = train.dropna(subset=[target])
                    if len(sub) < 10:
                        notes[rung] = f"only {len(sub)} training rows with a target"
                        continue
                    X_train = sub[feats]
                    y_train = sub[target].to_numpy(dtype=float)
                    if target == "total_cases":
                        y_train = np.log1p(y_train)
                    try:
                        est = fit_quantile_model(X_train, y_train, tau, model=model, seed=seed,
                                                 max_iter=max_iter)
                        pred = predict_cases(est, val[feats], target)
                        yv = val[target].to_numpy(dtype=float)
                        ok = np.isfinite(yv) & np.isfinite(pred)
                        s = pd.Series(np.nan, index=units_val.values)
                        s.iloc[np.flatnonzero(ok)] = np.maximum(
                            tau * (yv[ok] - pred[ok]), (tau - 1.0) * (yv[ok] - pred[ok])
                        )
                        losses[rung] = s
                        if abs(tau - 0.5) < 1e-9:
                            preds_half[rung] = pred
                        notes[rung] = ""
                    except Exception as exc:  # noqa: BLE001
                        notes[rung] = f"fit failed: {type(exc).__name__}: {exc}"

                if "M0" not in losses:
                    continue
                L = pd.DataFrame(losses)

                # A rung that could not be evaluated gets a row with a null
                # value and the reason, so that a missing number is visible in
                # the table instead of silently absent from it.
                for rung in rung_order:
                    if rung != "M0" and rung in ladder and rung not in L.columns:
                        rows.append({
                            "record_type": "loss", "level": level, "task": task,
                            "target": target, "origin": int(origin), "rung": rung,
                            "contrast": "", "model": model, "loss": "pinball",
                            "tau": float(tau), "value": np.nan,
                            "n_train": int(len(train)), "n_val": 0,
                            "n_units": int(val[unit_col].nunique()),
                            "features": "|".join(ladder.get(rung, [])),
                            "note": notes.get(rung, "rung could not be evaluated"),
                        })

                L = L.reindex(columns=[r for r in rung_order if r in L.columns])
                L0 = L["M0"]

                # -- loss and skill rows --
                for rung in L.columns:
                    col = L[rung]
                    n_val = int(col.notna().sum())
                    val_mean = float(col.mean()) if n_val else np.nan
                    rows.append({
                        "record_type": "loss", "level": level, "task": task,
                        "target": target, "origin": int(origin), "rung": rung,
                        "contrast": "", "model": ("persistence" if rung == "M0" else model),
                        "loss": "pinball", "tau": float(tau), "value": val_mean,
                        "n_train": int(len(train)), "n_val": n_val,
                        "n_units": int(val[unit_col].nunique()),
                        "features": "|".join(ladder.get(rung, [])) if rung != "M0" else "prev_total_cases",
                        "note": notes.get(rung, ""),
                    })
                    if rung != "M0" and n_val:
                        # Skill is a ratio of two means, so both must be taken
                        # over exactly the same rows; otherwise a rung that
                        # happened to predict fewer rows would be scored on a
                        # different sample from its own benchmark.
                        pair = L[[rung, "M0"]].dropna()
                        if len(pair) >= 3 and pair["M0"].mean() > 0:
                            rows.append({
                                "record_type": "skill", "level": level, "task": task,
                                "target": target, "origin": int(origin), "rung": rung,
                                "contrast": f"{rung}-M0", "model": "",
                                "loss": "pinball", "tau": float(tau),
                                "value": float(1.0 - pair[rung].mean() / pair["M0"].mean()),
                                "n_train": int(len(train)), "n_val": int(len(pair)),
                                "n_units": int(val[unit_col].nunique()),
                                "note": "skill score SS = 1 - L(rung)/L(M0) on pinball loss, "
                                        "computed on the rows where both are defined",
                            })

                # -- paired contrasts against the previous rung --
                for rung, reference in ADJACENT:
                    if rung not in L.columns or reference not in L.columns:
                        continue
                    if set(ladder.get(rung, [])) == set(ladder.get(reference, [])):
                        # Unreachable while build_ladder refuses such a ladder.
                        # Kept as a second line of defence: this is the exact
                        # shape of the bug that wrote 95 zeros, and a missing
                        # number must never be rendered as "no difference".
                        rows.append({
                            "record_type": "contrast", "level": level, "task": task,
                            "target": target, "origin": int(origin), "rung": rung,
                            "contrast": f"{rung}-{reference}", "model": "",
                            "loss": "pinball", "tau": float(tau), "value": np.nan,
                            "n_train": int(len(train)),
                            "n_val": int(len(L[[reference, rung]].dropna())),
                            "n_units": int(val[unit_col].nunique()),
                            "ci_low": np.nan, "ci_high": np.nan, "se": np.nan,
                            "p_value": np.nan, "ci_excludes_zero": False,
                            "improves": False, "features": "",
                            "note": (f"undefined: {rung} and {reference} carry the "
                                     "same feature set, so the differential is "
                                     "identically zero by construction and is not "
                                     "a result"),
                        })
                        continue
                    pair = L[[reference, rung]].dropna()
                    if len(pair) < 3:
                        continue
                    diff = pair[reference].to_numpy() - pair[rung].to_numpy()
                    u = pair.index.to_numpy()
                    st = paired_bootstrap(diff, u, n_boot=n_boot, seed=seed)
                    excludes = (np.isfinite(st["ci_low"]) and np.isfinite(st["ci_high"])
                                and (st["ci_low"] > 0 or st["ci_high"] < 0))
                    improves = bool(excludes and st["mean"] > 0)
                    rows.append({
                        "record_type": "contrast", "level": level, "task": task,
                        "target": target, "origin": int(origin), "rung": rung,
                        "contrast": f"{rung}-{reference}", "model": "",
                        "loss": "pinball", "tau": float(tau), "value": st["mean"],
                        "n_train": int(len(train)), "n_val": st["n"],
                        "n_units": st.get("n_units", np.nan),
                        "ci_low": st["ci_low"], "ci_high": st["ci_high"],
                        "se": st["se"], "p_value": st["p_value"],
                        "ci_excludes_zero": bool(excludes), "improves": improves,
                        "features": "|".join(sorted(set(ladder.get(rung, [])) - set(ladder.get(reference, []))))
                                    if rung != "M0" else "",
                        "note": "positive value = the later rung has the lower loss; "
                                "features = columns added by this rung",
                    })
                    origin_diffs.setdefault(
                        (task, f"{rung}-{reference}", float(tau)), {}
                    )[int(origin)] = st["mean"]

                # -- secondary losses, on the median forecast only --
                # The tau=0.5 predictions are reused here rather than
                # refitted: refitting would silently double the cost of the
                # run and, worse, could evaluate a different model from the
                # one whose pinball loss was just recorded.
                if abs(tau - 0.5) < 1e-9:
                    yv = val[target].to_numpy(dtype=float)
                    denom = mase_denominators(train, unit_col, target)
                    for rung in L.columns:
                        pred = preds_half.get(rung)
                        if pred is None:
                            continue
                        ok = np.isfinite(yv) & np.isfinite(pred)
                        if not ok.any():
                            continue
                        err = np.abs(yv[ok] - pred[ok])
                        base = {"record_type": "loss", "level": level, "task": task,
                                "target": target, "origin": int(origin), "rung": rung,
                                "contrast": "", "model": ("persistence" if rung == "M0" else model),
                                "tau": 0.5, "n_train": int(len(train)),
                                "n_val": int(ok.sum()), "n_units": int(val[unit_col].nunique()),
                                "features": "|".join(ladder.get(rung, [])) if rung != "M0" else "prev_total_cases",
                                "note": ""}
                        for loss_name, value in (
                            ("mae", float(err.mean())),
                            ("rmse", float(np.sqrt(np.mean((yv[ok] - pred[ok]) ** 2)))),
                        ):
                            rows.append({**base, "loss": loss_name, "value": value})
                        unit_ok = val[unit_col].to_numpy()[ok]
                        scaled = np.array([
                            e / denom[u] for e, u in zip(err, unit_ok)
                            if u in denom and denom[u] > 0
                        ])
                        if scaled.size:
                            rows.append({**base, "loss": "mase",
                                         "value": float(np.mean(scaled)),
                                         "n_val": int(scaled.size),
                                         "note": "absolute error / in-sample naive MAE of the same unit"})

    return rows, origin_diffs


def decision_rows(rows, origin_diffs, origins, level, min_origins=3):
    """Apply the pre-registered rule to each adjacent contrast, per task.

    Section 2.10: the rung is credited only if it improves the *primary* loss
    with a 95% bootstrap interval excluding zero, in the same direction, in at
    least three of the five forecast origins. The rule is evaluated here
    mechanically, on the primary loss only -- a verdict computed at every tau
    would multiply the decision without adding evidence, and the primary loss
    is pinball at tau=0.5 for the regressions and the Brier score for the
    classifier.

    Diebold-Mariano is attached to the same row as a descriptive complement.
    """
    out = []
    for (task, contrast, tau_key), per_origin in sorted(
        origin_diffs.items(), key=lambda kv: (kv[0][0], kv[0][1], str(kv[0][2]))
    ):
        if contrast not in ("M1-M0", "M2-M1", "M3-M2", "M3dur-M3"):
            continue
        # only the primary loss of each task enters the decision
        if tau_key is None:
            if task != "exceedance":
                continue
        elif not (np.isfinite(tau_key) and abs(float(tau_key) - 0.5) < 1e-9):
            continue

        series = [per_origin[y] for y in sorted(per_origin)]
        improving = [y for y, m in per_origin.items()
                     if np.isfinite(m) and m > 0
                     and _ci_excluded(rows, level, task, contrast, tau_key, y)]
        n_eval = len(per_origin)
        if n_eval < min_origins:
            verdict = "underpowered"
        elif len(improving) >= min_origins:
            verdict = "supported"
        else:
            verdict = "not_supported"

        dm = diebold_mariano(series)
        out.append({
            "record_type": "decision", "level": level, "task": task,
            "target": TARGETS.get(task, "total_cases"), "origin": -1,
            "rung": contrast.split("-")[0], "contrast": contrast, "model": "",
            "loss": ("brier" if task == "exceedance" else "pinball"),
            "tau": (np.nan if tau_key is None else float(tau_key)),
            "value": int(len(improving)), "n_val": int(n_eval),
            "verdict": verdict,
            "dm_stat": dm["dm_stat"], "dm_pvalue": dm["dm_pvalue"],
            "dm_n": dm["dm_n"],
            "note": (f"{len(improving)}/{n_eval} origins show a same-direction "
                     f"improvement on the primary loss with a 95% bootstrap "
                     f"interval excluding zero; the rule requires "
                     f"{min_origins}/{len(origins)}. Diebold-Mariano over the "
                     f"{dm['dm_n']} per-origin means (NW lag {dm['dm_lag']}) is "
                     "reported as description only: with so few origins it is "
                     "low-powered by construction."),
        })
    return out


def _ci_excluded(rows, level, task, contrast, tau, origin):
    """True if the contrast row for this origin had a CI excluding zero."""
    for r in rows:
        if (r["record_type"] == "contrast" and r["level"] == level
                and r["task"] == task and r["contrast"] == contrast
                and int(r["origin"]) == int(origin)):
            r_tau = r["tau"]
            if tau is None:
                if not (isinstance(r_tau, float) and np.isnan(r_tau)):
                    continue
            else:
                if not (r_tau is not None and np.isfinite(float(r_tau))
                        and abs(float(r_tau) - float(tau)) < 1e-9):
                    continue
            return bool(r.get("improves"))
    return False


# --------------------------------------------------------------------------
# The classifier rung (rarity of the coming season)
# --------------------------------------------------------------------------
def _threshold(train_cases, percentile, unit=None, scope="pooled", per_unit=None):
    if scope == "per_unit" and per_unit is not None and unit in per_unit:
        return per_unit[unit]
    return float(np.percentile(np.asarray(train_cases, dtype=float), percentile * 100.0))


def run_classifier_ablation(df, ladder, unit_col, origins, level, seed, n_boot,
                            percentile=0.75, threshold_scope="pooled", verbose=True):
    """Brier-score ablation for Y = 1[S_t > Q_p], Q_p computed inside each fold.

    The threshold is estimated from the training years only (Section 2.11,
    leakage channel 2). M0 is the training base rate, which is exactly the
    benchmark Table 4 specifies for the classifier.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier

    rows = []
    origin_diffs = {}
    rung_order = [r for r in ("M0", "M1", "M2", "M3", "M3dur") if r in ladder or r == "M0"]

    for origin in origins:
        train = df[df["year"] < origin]
        val = df[df["year"] == origin]
        if train.empty or val.empty:
            continue

        per_unit = None
        if threshold_scope == "per_unit":
            per_unit = {}
            for unit, grp in train.groupby(unit_col):
                cases = grp["total_cases"].to_numpy(dtype=float)
                cases = cases[np.isfinite(cases)]
                if cases.size >= 3:
                    per_unit[unit] = float(np.percentile(cases, percentile * 100.0))
        thr_pooled = _threshold(train["total_cases"].dropna(), percentile)
        thr = _threshold(train["total_cases"].dropna(), percentile,
                         scope=threshold_scope, per_unit=per_unit)

        if threshold_scope == "per_unit" and per_unit:
            thr_val = val[unit_col].map(per_unit).fillna(thr_pooled).to_numpy(dtype=float)
        else:
            thr_val = np.full(len(val), thr_pooled)

        y_tr = (train["total_cases"].to_numpy(dtype=float) > thr).astype(int)
        y_val = (val["total_cases"].to_numpy(dtype=float) > thr_val).astype(int)
        keep_tr = np.isfinite(train["total_cases"].to_numpy(dtype=float))
        y_tr, train_ok = y_tr[keep_tr], train[keep_tr]
        keep_val = np.isfinite(val["total_cases"].to_numpy(dtype=float))
        y_val, val_ok = y_val[keep_val], val[keep_val]

        if y_tr.sum() < 5 or (len(y_tr) - y_tr.sum()) < 5:
            if verbose:
                print(f"[classifier {origin}] skipped: degenerate training labels "
                      f"({int(y_tr.sum())} positives of {len(y_tr)})")
            continue

        units_val = _row_key(val_ok, unit_col)
        losses = {}

        # M0: training base rate
        base = float(y_tr.mean())
        losses["M0"] = pd.Series(
            (np.clip(np.full(len(y_val), base), 0, 1) - y_val) ** 2,
            index=units_val.values,
        )

        for rung in rung_order:
            if rung == "M0" or rung not in ladder:
                continue
            feats = ladder[rung]
            if not feats:
                continue
            X_tr = train_ok[feats]
            X_val = val_ok[feats]
            if X_tr.dropna(how="all").empty:
                continue
            try:
                clf = HistGradientBoostingClassifier(
                    max_iter=300, learning_rate=0.05, max_depth=5, random_state=seed
                )
                clf.fit(X_tr, y_tr)
                p = clf.predict_proba(X_val)[:, 1]
            except Exception as exc:  # noqa: BLE001
                if verbose:
                    print(f"[classifier {origin}] {rung} failed: {exc}")
                continue
            losses[rung] = pd.Series((p - y_val) ** 2, index=units_val.values)

        L = pd.DataFrame(losses)
        L = L.reindex(columns=[r for r in rung_order if r in L.columns])
        for rung in L.columns:
            n_val = int(L[rung].notna().sum())
            rows.append({
                "record_type": "loss", "level": level, "task": "exceedance",
                "target": "total_cases", "origin": int(origin), "rung": rung,
                "contrast": "", "model": ("base_rate" if rung == "M0" else "hist_gradient_boosting"),
                "loss": "brier", "tau": np.nan,
                "value": float(L[rung].mean()) if n_val else np.nan,
                "n_train": int(len(train_ok)), "n_val": n_val,
                "n_units": int(val_ok[unit_col].nunique()),
                "features": "|".join(ladder.get(rung, [])) if rung != "M0" else "",
                "note": f"threshold Q{int(percentile*100)} from training years only "
                        f"({threshold_scope}); positives in validation: {int(y_val.sum())}/{len(y_val)}",
            })

        for rung, reference in ADJACENT:
            if rung not in L.columns or reference not in L.columns:
                continue
            pair = L[[reference, rung]].dropna()
            if len(pair) < 3:
                continue
            diff = pair[reference].to_numpy() - pair[rung].to_numpy()
            st = paired_bootstrap(diff, pair.index.to_numpy(), n_boot=n_boot, seed=seed)
            excludes = (np.isfinite(st["ci_low"]) and np.isfinite(st["ci_high"])
                        and (st["ci_low"] > 0 or st["ci_high"] < 0))
            rows.append({
                "record_type": "contrast", "level": level, "task": "exceedance",
                "target": "total_cases", "origin": int(origin), "rung": rung,
                "contrast": f"{rung}-{reference}", "model": "", "loss": "brier",
                "tau": np.nan, "value": st["mean"], "n_train": int(len(train_ok)),
                "n_val": st["n"], "n_units": st.get("n_units", np.nan),
                "ci_low": st["ci_low"], "ci_high": st["ci_high"], "se": st["se"],
                "p_value": st["p_value"], "ci_excludes_zero": bool(excludes),
                "improves": bool(excludes and st["mean"] > 0),
                "note": "positive value = the later rung has the lower Brier score",
            })
            origin_diffs.setdefault(
                ("exceedance", f"{rung}-{reference}", None), {}
            )[int(origin)] = st["mean"]
    return rows, origin_diffs


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def default_tau_grid():
    """The grid of Section 2.8: tau in {0.05, 0.10, ..., 0.95}."""
    return [round(0.05 * i, 2) for i in range(1, 20)]


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Rolling-origin ablation ladder (M0..M3) for the dengue "
                    "power-law manuscript.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--level", choices=["state", "city"], default="state")
    ap.add_argument("--region", default="BR")
    ap.add_argument("--origins", default="2021,2022,2023,2024,2025",
                    help="comma-separated forecast origins")
    ap.add_argument("--tau-grid", default="default",
                    help="'default' for {0.05..0.95}, 'none' for tau=0.5 only, "
                         "or an explicit comma-separated list")
    ap.add_argument("--model", choices=["hist_gradient_boosting", "lightgbm"],
                    default="hist_gradient_boosting")
    ap.add_argument("--percentile", type=float, default=0.75,
                    help="rarity threshold for the classifier task")
    ap.add_argument("--threshold-scope", choices=["pooled", "per_unit"],
                    default="pooled",
                    help="how Q_p is estimated inside each fold. 'pooled' "
                         "matches binary_classifier.compute_percentile_threshold; "
                         "'per_unit' matches the wording of the Limitations "
                         "section ('75th percentile of a unit's own history')")
    ap.add_argument("--include-classifier", action="store_true",
                    help="add the exceedance (Brier) task")
    ap.add_argument("--include-dur-scaling", action="store_true",
                    help="add the diagnostic rung M3dur (duration scaling descriptors)")
    ap.add_argument("--panel", default=None, help="reuse a cached panel parquet")
    ap.add_argument("--attach-climate", action="store_true",
                    help="build the climate block from data/[0-9]*.parquet when the "
                         "panel has none, and lag it to year-1. Without this the run "
                         "refuses to proceed when M2 would equal M1")
    ap.add_argument("--climate-parquet", default=None,
                    help="pre-aggregated climate file (<unit>, year, "
                         "avg_temp_Q1..Q4, avg_umid_Q1..Q4) merged as year-1 "
                         "predictors")
    ap.add_argument("--climate-block", default=None,
                    help="override the eight column names that define M2 "
                         "(comma-separated); default is the declared CLIMATE_BLOCK")
    ap.add_argument("--save-panel", default=None, help="write the panel here for reuse")
    ap.add_argument("--output", default="ablation_results.csv")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--min-origins", type=int, default=3,
                    help="origins required by the pre-registered decision rule")
    ap.add_argument("--max-iter", type=int, default=300,
                    help="boosting iterations. The manuscript specifies 300; lower "
                         "it only for a pilot run, and note that the value used is "
                         "recorded in the run metadata")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the panel, the rungs and the number of model fits, "
                         "then exit without fitting or writing anything")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--quiet", action="store_true")
    return ap.parse_args(argv)


def parse_tau_grid(spec):
    """Parse --tau-grid. 'default' -> the full grid; 'none' -> tau=0.5 only."""
    spec = (spec or "").strip()
    if spec in ("default", ""):
        return default_tau_grid()
    if spec in ("none", "0.5"):
        return [0.5]
    taus = []
    for token in spec.split(","):
        token = token.strip()
        if not token or token == "none":
            continue
        try:
            tau = float(token)
        except ValueError:
            raise SystemExit(
                f"--tau-grid: {token!r} is not a number. Use 'default', 'none', "
                "or a comma-separated list of quantile levels in (0, 1)."
            ) from None
        if not 0.0 < tau < 1.0:
            raise SystemExit(f"--tau-grid: {tau} is outside (0, 1)")
        taus.append(round(tau, 4))
    taus = sorted(set(taus))
    if 0.5 not in taus:
        # The primary loss is pinball at tau=0.5; without it the decision rule
        # has no input, so the level is added rather than silently missing.
        taus = sorted(set(taus + [0.5]))
    return taus


def main(argv=None):
    args = parse_args(argv)
    verbose = not args.quiet
    t0 = time.time()
    warnings.filterwarnings("ignore", category=FutureWarning)

    # -- panel -----------------------------------------------------------
    df = load_panel(args.level, args.region, args.panel, verbose=verbose)
    df, panel_notes = validate_panel(df, args.level, verbose=verbose)
    if args.save_panel:
        df.to_parquet(args.save_panel, index=False)
        if verbose:
            print(f"[panel] cached to {args.save_panel}")

    unit_col = "state" if args.level == "state" else "geocode"
    years = sorted(int(y) for y in df["year"].unique())
    origins = [int(o) for o in args.origins.split(",") if o.strip()]

    ready, skipped = [], []
    for o in origins:
        has_train = bool((df["year"] < o).any())
        has_val = bool((df["year"] == o).any())
        (ready if (has_train and has_val) else skipped).append(o)
    if skipped and verbose:
        print(f"[origins] no evaluable data, skipped: {skipped}")
    if not ready:
        raise SystemExit(
            f"No forecast origin can be evaluated. Panel years: {years}. "
            f"Requested origins: {origins}. Each origin needs at least one "
            "row with year < origin and at least one with year == origin."
        )

    # -- ladder ----------------------------------------------------------
    climate_block = None
    if args.climate_block:
        climate_block = [c.strip() for c in args.climate_block.split(",") if c.strip()]

    if args.attach_climate or args.climate_parquet:
        df, climate_notes = attach_climate(
            df, args.level, args.region, parquet=args.climate_parquet,
            block=climate_block, verbose=verbose,
        )
        panel_notes = panel_notes + climate_notes
        if args.save_panel:
            df.to_parquet(args.save_panel, index=False)
            if verbose:
                print(f"[panel] re-cached with climate to {args.save_panel}")

    ladder, dropped, climate_info = build_ladder(
        df, include_dur_scaling=args.include_dur_scaling,
        climate_block=climate_block,
    )
    assert_ladder_is_nested(ladder, climate_block=climate_block)
    assert_no_contemporaneous_leak(ladder, df)

    if verbose:
        print("\n[rungs]")
        for rung in [r for r in ("M1", "M2", "M3", "M3dur") if r in ladder]:
            print(f"  {rung:6s} {len(ladder[rung]):2d} features  {ladder[rung]}")
        for rung, gone in dropped.items():
            print(f"  {rung:6s} unavailable in this panel: {gone}")
        if climate_info["missing"]:
            print(f"[climate] {len(climate_info['available'])}/"
                  f"{len(climate_info['block'])} block columns available; missing "
                  f"{climate_info['missing']}")

    # -- runs ------------------------------------------------------------
    tau_grid = parse_tau_grid(args.tau_grid)
    if verbose:
        print(f"\n[run] {len(ready)} origin(s) x {len(tau_grid)} tau x "
              f"{len(TARGETS)} target(s); model={args.model}")

    # --dry-run: show the plan, fit nothing -------------------------------
    if args.dry_run:
        n_fits = len(ready) * len(tau_grid) * len(TARGETS) * len(ladder)
        if args.include_classifier:
            n_fits += len(ready) * len(ladder)
        print(f"\n[dry-run] would fit {n_fits} model(s) "
              f"({len(ready)} origin(s) x {len(tau_grid)} tau x {len(TARGETS)} target(s) "
              f"x {len(ladder)} rung(s)"
              + (" + classifier" if args.include_classifier else "") + ")")
        print(f"[dry-run] origins: {ready}; panel {len(df)} rows, "
              f"{df[unit_col].nunique()} units")
        print("[dry-run] nothing was fitted and nothing was written")
        return 0

    rows, origin_diffs = run_regression_ablation(
        df, ladder, unit_col, ready, tau_grid, args.model, args.seed,
        args.bootstrap, args.level, verbose=verbose, max_iter=args.max_iter,
    )

    if args.include_classifier:
        if verbose:
            if args.level == "state":
                print(f"\n[warning] the classifier is evaluated on a "
                      f"{df[unit_col].nunique()}-unit cross-section; a Brier "
                      "comparison at this size is very low-powered. The "
                      "manuscript's rarity task is a municipality-level one.")
            print("\n[run] classifier task (Brier)")
        clf_rows, clf_diffs = run_classifier_ablation(
            df, ladder, unit_col, ready, args.level, args.seed, args.bootstrap,
            percentile=args.percentile, threshold_scope=args.threshold_scope,
            verbose=verbose,
        )
        rows += clf_rows
        origin_diffs.update(clf_diffs)

    rows += decision_rows(rows, origin_diffs, ready, args.level,
                          min_origins=args.min_origins)

    if not rows:
        raise SystemExit(
            "Nothing was evaluated: every origin/target/rung combination "
            "failed. The CSV was not written, because an empty or all-null "
            "table is indistinguishable from a run that produced no skill."
        )

    out = pd.DataFrame(rows).reindex(columns=COLUMNS)
    out = out.sort_values(
        ["record_type", "task", "loss", "tau", "rung", "origin"],
        na_position="first",
    )
    # Decision rows carry no origin/size; keep those columns integral rather
    # than letting one sentinel turn 2021 into 2021.0 in the CSV.
    for col in ("origin", "n_train", "n_val", "n_units"):
        out[col] = pd.array(pd.to_numeric(out[col], errors="coerce"), dtype="Int64")
    out.to_csv(args.output, index=False)

    # -- metadata --------------------------------------------------------
    import sklearn

    meta = {
        "argv": sys.argv[1:] if argv is None else list(argv),
        "level": args.level,
        "region": args.region,
        "panel_rows": int(len(df)),
        "panel_units": int(df[unit_col].nunique()),
        "panel_years": years,
        "origins_requested": origins,
        "origins_evaluated": ready,
        "origins_skipped": skipped,
        "tau_grid": tau_grid,
        "model": args.model,
        "max_iter": args.max_iter,
        "seed": args.seed,
        "bootstrap_replicates": args.bootstrap,
        "bootstrap_unit": unit_col,
        "decision_rule": ("improvement on the primary loss with a 95% bootstrap "
                          f"interval excluding zero, same direction, in at least "
                          f"{args.min_origins}/{len(ready)} origins"),
        "ladder": {k: v for k, v in ladder.items()},
        "unavailable_columns": dropped,
        "climate": climate_info,
        "climate_columns_in_panel": sorted(
            c for c in df.columns if CLIMATE_RE.match(c)
        ),
        "panel_notes": panel_notes,
        "classifier_threshold_scope": args.threshold_scope,
        "classifier_percentile": args.percentile,
        "versions": {
            "python": sys.version.split()[0],
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
        },
        "elapsed_seconds": round(time.time() - t0, 1),
        "output": str(Path(args.output).resolve()),
    }
    meta_path = Path(args.output).with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, default=str))

    if verbose:
        print()
        _print_summary(out, args.level)
        print(f"\n[out] {args.output}  ({len(out)} rows)")
        print(f"[out] {meta_path}")
        print(f"[time] {meta['elapsed_seconds']}s")
    return 0


def _print_summary(out, level):
    """Human-readable summary of the primary contrast and the verdicts."""
    dec = out[out["record_type"] == "decision"]
    if not dec.empty:
        print("-" * 78)
        print("DECISION RULE -- pre-registered, Section 2.10")
        print("-" * 78)
        for _, r in dec.iterrows():
            print(f"  {r['task']:10s} {r['contrast']:9s} tau={r['tau']:<5} "
                  f"{r['value']}/{r['n_val']} origins -> {r['verdict']}")

    prim = out[(out["record_type"] == "contrast") & (out["loss"] == "pinball")
               & (np.isclose(out["tau"].astype(float), 0.5))]
    if prim.empty:
        return
    print()
    print("-" * 78)
    print("PRIMARY LOSS (pinball, tau=0.5) -- paired contrasts by origin")
    print("-" * 78)
    print(f"  {'task':10s} {'contrast':9s} {'origin':6s} {'diff':>10s} "
          f"{'95% CI':>22s} {'n':>5s}")
    for _, r in prim.sort_values(["task", "contrast", "origin"]).iterrows():
        ci = (f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]"
              if pd.notna(r["ci_low"]) else "n/a")
        print(f"  {r['task']:10s} {r['contrast']:9s} {str(r['origin']):6s} "
              f"{r['value']:+10.4f} {ci:>22s} {int(r['n_val']):5d}")


if __name__ == "__main__":
    sys.exit(main())
