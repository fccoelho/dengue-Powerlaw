"""Tests for run_ablation.py -- the M0..M3 ladder and its guarantees.

The point of these tests is the sentence the manuscript relies on: "a
contrast between consecutive rungs is attributable to the block that was
added". That sentence is only true if M2 and M3 differ *only* in the scaling
columns, so the assertion that enforces it is pinned here rather than left to
review.
"""
import numpy as np
import pandas as pd
import pytest

import run_ablation as ra


# ---------------------------------------------------------------------------
# Synthetic panel with the schema prepare_lagged_data() returns
# ---------------------------------------------------------------------------
def _panel(n_units=8, y0=2012, y1=2020, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_units):
        for y in range(y0, y1 + 1):
            rows.append({
                "state": f"S{u:02d}", "year": y,
                "total_cases": float(rng.gamma(2.0, 500)),
                "ep_dur": float(rng.gamma(4.0, 3.0)),
                "peak_week": float(rng.integers(5, 30)),
                "R0": float(rng.normal(1.3, 0.2)),
                "log_pop": float(rng.uniform(5, 7)),
                "avg_temp_Q1": float(rng.normal(24, 2)),  # contemporaneous
                "prev_total_cases": float(rng.gamma(2.0, 500)),
                "prev_ep_dur": float(rng.gamma(4.0, 3.0)),
                "prev_peak_week": float(rng.integers(5, 30)),
                "prev_R0": float(rng.normal(1.3, 0.2)),
                "prev_log_pop": float(rng.uniform(5, 7)),
                "prev_avg_temp_Q1": float(rng.normal(24, 2)),
                "prev_avg_umid_Q1": float(rng.normal(70, 5)),
                "prev_alpha_size": float(rng.normal(2.0, 0.2)),
                "prev_xmin_size": float(rng.uniform(50, 500)),
                "prev_alpha_dur": float(rng.normal(2.9, 0.05)),
                "prev_xmin_dur": float(rng.uniform(10, 30)),
                "prev_alpha_trend": float(rng.normal(0, 0.05)),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Ladder structure
# ---------------------------------------------------------------------------
def test_m2_adds_only_climate_and_m3_adds_only_scaling():
    df = _panel()
    ladder, dropped, climate = ra.build_ladder(df)

    # The synthetic panel carries Q1 only. The other six are *expected* and
    # absent, so they must be reported rather than silently omitted: that is
    # exactly what made M2 collapse into M1 when the real panel had no climate.
    assert set(climate["available"]) == {"prev_avg_temp_Q1", "prev_avg_umid_Q1"}
    assert set(climate["missing"]) == set(ra.CLIMATE_BLOCK) - set(climate["available"])
    assert set(dropped["M2"]) == set(climate["missing"])

    added_m2 = set(ladder["M2"]) - set(ladder["M1"])
    assert added_m2 == {"prev_avg_temp_Q1", "prev_avg_umid_Q1"}
    for col in added_m2:
        assert ra.CLIMATE_RE.match(col)

    added_m3 = set(ladder["M3"]) - set(ladder["M2"])
    assert added_m3 == set(ra.SCALING_SIZE)
    assert added_m3 == {"prev_alpha_size", "prev_xmin_size", "prev_alpha_trend"}

    # the rungs are nested, not merely different
    assert set(ladder["M1"]) < set(ladder["M2"]) < set(ladder["M3"])


def test_assert_ladder_is_nested_rejects_a_foreign_column_in_m3():
    df = _panel()
    ladder, _, _ = ra.build_ladder(df)
    ra.assert_ladder_is_nested(ladder)

    broken = {k: list(v) for k, v in ladder.items()}
    broken["M3"] = broken["M3"] + ["prev_R0"]  # already in M1: not scaling
    with pytest.raises(AssertionError, match="prev_R0"):
        ra.assert_ladder_is_nested(broken)


def test_assert_ladder_is_nested_rejects_a_shifted_rung():
    df = _panel()
    ladder, _, _ = ra.build_ladder(df)
    broken = {k: list(v) for k, v in ladder.items()}
    broken["M2"] = broken["M2"] + ["prev_alpha_size"]  # scaling inside M2
    with pytest.raises(AssertionError, match="climate"):
        ra.assert_ladder_is_nested(broken)


def test_contemporaneous_climate_is_rejected_by_the_leak_guard():
    df = _panel()
    ladder, _, _ = ra.build_ladder(df)
    ra.assert_no_contemporaneous_leak(ladder, df)

    broken = {k: list(v) for k, v in ladder.items()}
    broken["M2"] = broken["M2"] + ["avg_temp_Q1"]
    with pytest.raises(AssertionError, match="leak|sanctioned"):
        ra.assert_no_contemporaneous_leak(broken, df)


def test_all_null_column_is_dropped_and_reported():
    df = _panel()
    df["prev_R0"] = np.nan
    ladder, dropped, _ = ra.build_ladder(df)
    assert "prev_R0" not in ladder["M1"]
    assert "prev_R0" in dropped["M1"]


def test_duration_rung_is_optional_and_adds_only_duration_scaling():
    df = _panel()
    assert "M3dur" not in ra.build_ladder(df)[0]
    ladder, _, _ = ra.build_ladder(df, include_dur_scaling=True)
    added = set(ladder["M3dur"]) - set(ladder["M3"])
    assert added == set(ra.SCALING_DUR)


# ---------------------------------------------------------------------------
# Panel validation
# ---------------------------------------------------------------------------
def test_validate_panel_drops_pooled_year_zero_rows():
    df = _panel()
    pooled = df.iloc[[0]].copy()
    pooled["year"] = 0
    df = pd.concat([df, pooled], ignore_index=True)
    out, notes = ra.validate_panel(df, "state", verbose=False)
    assert 0 not in set(out["year"])
    assert any("pooled" in n for n in notes)


def test_validate_panel_rejects_duplicate_unit_years():
    df = _panel()
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    with pytest.raises(RuntimeError, match="duplicated"):
        ra.validate_panel(df, "state", verbose=False)


# ---------------------------------------------------------------------------
# Losses and baselines
# ---------------------------------------------------------------------------
def test_pinball_at_the_median_is_half_the_mae():
    y = np.array([1.0, 2.0, 7.0])
    q = np.array([2.0, 2.0, 3.0])
    assert ra.pinball_loss(y, q, 0.5) == pytest.approx(np.mean(np.abs(y - q)) / 2)


def test_pinball_is_asymmetric_away_from_the_median():
    # under-prediction is penalised more than over-prediction at tau=0.9
    assert ra.pinball_loss(np.array([10.0]), np.array([0.0]), 0.9) > \
           ra.pinball_loss(np.array([0.0]), np.array([10.0]), 0.9)


def test_persistence_at_the_median_reproduces_the_previous_value():
    df = _panel(n_units=4)
    df["prev_total_cases"] = 100.0
    df["total_cases"] = 100.0
    train, val = df.iloc[:-4], df.iloc[-4:]
    pred = ra.persistence_predictions(train, val, "state", "total_cases", 0.5, "size")
    assert np.allclose(pred, 100.0)


def test_paired_bootstrap_interval_excludes_zero_for_a_consistent_gain():
    diff = np.full(40, 0.5)
    units = np.repeat(["a", "b", "c", "d"], 10)
    st = ra.paired_bootstrap(diff, units, n_boot=200)
    assert st["mean"] == pytest.approx(0.5)
    assert st["ci_low"] > 0 and st["ci_high"] > 0


def test_paired_bootstrap_interval_straddles_zero_for_noise():
    rng = np.random.default_rng(1)
    diff = rng.normal(0, 1, 200)
    units = np.repeat([f"u{i}" for i in range(20)], 10)
    st = ra.paired_bootstrap(diff, units, n_boot=500)
    assert st["ci_low"] < 0 < st["ci_high"]


def test_diebold_mariano_flags_a_constant_gain():
    st = ra.diebold_mariano([1.0, 1.0, 1.0, 1.0, 1.0])
    # zero variance -> long-run variance is 0 and the statistic is undefined,
    # which must be reported as NaN rather than as a spurious infinity
    assert st["dm_n"] == 5


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------
def test_parse_tau_grid_modes():
    assert ra.parse_tau_grid("none") == [0.5]
    assert 0.5 in ra.parse_tau_grid("0.1,0.9")      # median always included
    assert ra.parse_tau_grid("default")[0] == 0.05
    with pytest.raises(SystemExit):
        ra.parse_tau_grid("0.5,banana")


# ---------------------------------------------------------------------------
# End-to-end, with a stub estimator so the test stays fast
# ---------------------------------------------------------------------------
class _Constant:
    """Cheap stand-in for the quantile model: predicts the training mean."""

    def fit(self, X, y):
        self.c = float(np.nanmean(np.asarray(y, dtype=float)))
        return self

    def predict(self, X):
        return np.full(len(X), self.c, dtype=float)


def test_run_regression_ablation_emits_one_loss_row_per_rung_and_origin(monkeypatch):
    monkeypatch.setattr(ra, "fit_quantile_model",
                        lambda Xtr, ytr, tau, model=None, seed=42, max_iter=300:
                        _Constant().fit(Xtr, ytr))
    df = _panel(n_units=6, y0=2015, y1=2019)
    ladder, _, _ = ra.build_ladder(df)

    rows, origin_diffs = ra.run_regression_ablation(
        df, ladder, "state", [2019], [0.5], "hist_gradient_boosting",
        42, 100, "state", verbose=False,
    )

    loss = pd.DataFrame([r for r in rows if r["record_type"] == "loss"])
    assert set(loss["rung"]) == {"M0", "M1", "M2", "M3"}
    # three targets x four rungs
    assert len(loss[loss["loss"] == "pinball"]) == 12
    assert len(loss[loss["loss"] == "mase"]) == 12
    assert len(loss[(loss["loss"] == "pinball") & (loss["task"] == "size")]) == 4
    # no fabricated zeros: a missing metric is NaN, never 0
    assert not (loss["value"] == 0).all()

    # the M2->M3 contrast carries exactly the scaling columns
    contrast = [r for r in rows
                if r["record_type"] == "contrast" and r["contrast"] == "M3-M2"]
    assert contrast, "the M3-M2 contrast must be present"
    added = set(contrast[0]["features"].split("|"))
    assert added == set(ra.SCALING_SIZE)


def test_decision_is_not_supported_when_no_origin_improves():
    rows = [{"record_type": "contrast", "level": "state", "task": "size",
             "contrast": "M3-M2", "tau": 0.5, "origin": y, "improves": False}
            for y in (2021, 2022, 2023, 2024, 2025)]
    diffs = {("size", "M3-M2", 0.5): {y: -0.01 for y in (2021, 2022, 2023, 2024, 2025)}}
    dec = ra.decision_rows(rows, diffs, [2021, 2022, 2023, 2024, 2025], "state")
    assert len(dec) == 1
    assert dec[0]["verdict"] == "not_supported"
    assert dec[0]["value"] == 0


def test_decision_is_supported_at_three_of_five_origins():
    origins = (2021, 2022, 2023, 2024, 2025)
    good = {2021, 2022, 2023}
    rows = [{"record_type": "contrast", "level": "state", "task": "size",
             "contrast": "M3-M2", "tau": 0.5, "origin": y, "improves": y in good}
            for y in origins]
    diffs = {("size", "M3-M2", 0.5): {y: (0.1 if y in good else -0.1) for y in origins}}
    dec = ra.decision_rows(rows, diffs, list(origins), "state")
    assert dec[0]["verdict"] == "supported"
    assert dec[0]["value"] == 3
