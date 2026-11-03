"""Recompute the headline predictive metrics and compare them with main.tex.

Usage:  python Manuscript/verification/verify_predictive.py [--strict]

The descriptive results have `verify_results.py`; until this script existed
the predictive numbers quoted in Manuscript/main.tex section "Predictive
skill" could not be audited the same way. It re-trains the exact models the
Methods section describes -- random forest, 150 trees, OOB score, state level,
test years 2022-2024 -- from the current cached artefacts, prints each
manuscript-quoted quantity next to the value recomputed now, and flags drift.

The manuscript marks these figures as coming from "an earlier full-pipeline
run", so DRIFT is the expected verdict whenever the database has been
refreshed since they were written; the script surfaces the new numbers so the
text can be regenerated deliberately. With --strict, any drift exits nonzero.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import predictive_model  # noqa: E402

# Quantities quoted in Manuscript/main.tex, section "Predictive skill"
# (marked TODO: "numbers below must be regenerated from the ablation runs
# before submission"). Source: earlier full-pipeline run, state level, all
# predictor blocks, years 2022-2024. OOB R2 is on the log scale for size.
QUOTED = {
    ("total_cases", 2022): 0.345,
    ("total_cases", 2023): 0.348,
    ("total_cases", 2024): 0.374,
    ("ep_dur", 2022): 0.165,
    ("ep_dur", 2023): 0.165,
    ("ep_dur", 2024): 0.165,
    ("peak_week", 2022): 0.068,
    ("peak_week", 2023): 0.068,
    ("peak_week", 2024): 0.068,
}
TOLERANCE = 0.02  # absolute; the manuscript quotes 3 decimals of an OOB score


def _oob_for_year(df_state, year):
    """Train the manuscript's RF on years < year, return per-target OOB R2."""
    models, metrics, *_ = predictive_model.train_predictive_models(
        df_state[df_state["year"] < year], test_year=year
    )
    return {
        target: (m or {}).get("OOB_R2", np.nan)
        for target, m in metrics.items()
    }, models, metrics


def main(strict=False):
    print("=" * 100)
    print("PREDICTIVE-LAYER VERIFICATION -- recompute headline metrics from cached artefacts")
    print("=" * 100)
    print("Panel: prepare_lagged_data('BR', level='state')  |  models: "
          "RandomForestRegressor(150 trees, OOB)  |  split: year forward chain")

    df_state = predictive_model.prepare_lagged_data("BR", target_year=None, level="state")
    if df_state.empty:
        print("FATAL: state-level panel is empty; cannot verify anything.")
        return 1
    print(f"Panel rows: {len(df_state)}  |  units: {df_state['state'].nunique()}  |  "
          f"years: {sorted(df_state['year'].unique())}")

    rows = []
    n_ok = n_drift = n_missing = 0
    for year in (2022, 2023, 2024):
        per_target, _models, metrics = _oob_for_year(df_state, year)
        for target in ("total_cases", "ep_dur", "peak_week"):
            quoted = QUOTED.get((target, year))
            recomputed = per_target.get(target, np.nan)
            if quoted is None:
                verdict = "NOT-QUOTED"
            elif recomputed is None or pd.isna(recomputed):
                verdict, n_missing = "MISSING", n_missing + 1
            elif abs(recomputed - quoted) <= TOLERANCE:
                verdict, n_ok = "OK", n_ok + 1
            else:
                verdict, n_drift = "DRIFT", n_drift + 1
            rows.append({
                "target": target,
                "test_year": year,
                "quoted_OOB_R2": quoted,
                "recomputed_OOB_R2": None if recomputed is None or pd.isna(recomputed)
                else round(float(recomputed), 3),
                "verdict": verdict,
            })

    print()
    print("-" * 100)
    print("Manuscript-quoted OOB R2 (log scale for size) vs recomputed now")
    print("-" * 100)
    print(pd.DataFrame(rows).to_string(index=False))

    # Binary layer headline metrics (no concrete numbers are quoted in the
    # manuscript yet -- printed here so the audit table exists and the
    # per-fold claim in section 2.9 is checkable).
    print()
    print("-" * 100)
    print("Binary classifier (XGBoost, year forward-chain CV) -- current audit table")
    print("-" * 100)
    try:
        from binary_classifier import train_binary_model

        df_city = predictive_model.prepare_lagged_data(
            "BR", target_year=None, level="city"
        )
        result = train_binary_model(df_city, test_year=2024, percentile=0.75)
        m = result["metrics"]
        print(f"rows={len(df_city)}  |  folds={m.get('n_folds')}  |  "
              f"threshold source={m.get('cv_threshold_source')}")
        print(f"CV AUC-ROC={m.get('cv_auc_roc'):.4f}  CV AUC-PR={m.get('cv_auc_pr'):.4f}  "
              f"CV F1@0.5={m.get('cv_f1_at_0.5'):.4f}  CV Brier={m.get('cv_brier'):.4f}")
        if "f1_at_youden" in m:
            print(f"TEST (2024+): AUC-ROC={m['auc_roc']:.4f}  AUC-PR={m['auc_pr']:.4f}  "
                  f"F1@Youden={m['f1_at_youden']:.4f}  F1@0.5={m['f1_at_0.5']:.4f}")
        summary = result.get("cv_fold_summary")
        if summary is not None:
            print(summary.to_string(index=False))
    except Exception as exc:  # noqa: BLE001 - report, don't crash the audit
        print(f"binary layer could not be verified: {type(exc).__name__}: {exc}")
        if strict:
            return 1

    print()
    n_drift_rows = sum(1 for r in rows if r["verdict"] == "DRIFT")
    n_missing_rows = sum(1 for r in rows if r["verdict"] == "MISSING")
    print(f"SUMMARY: {n_ok} OK, {n_drift_rows} DRIFT, {n_missing_rows} MISSING "
          f"(tolerance +/-{TOLERANCE} on OOB R2)")
    if n_drift or n_missing:
        print("The quoted figures no longer match the current artefacts. The "
              "manuscript flags them as pending regeneration (see the TODO in "
              "section 'Predictive skill'); update them from the recomputed "
              "values above before submission.")
    if strict and (n_drift or n_missing):
        return 1
    return 0


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    sys.exit(main(strict="--strict" in sys.argv))
