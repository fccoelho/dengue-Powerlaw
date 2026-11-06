"""Tests for the climate block that defines the M2 rung.

The bug these pin down: ``build_ladder`` discovered the climate columns by
scanning the panel for ``prev_avg_*``. When the panel had none -- which is what
``prepare_lagged_data`` returns whenever ``get_quarterly_climate`` fails --
``climate_cols`` came back empty, M2 was built equal to M1, and the M2-M1
contrast was written as 0.0 with a [0, 0] interval in all 19 tau x 5 origins.
The table then asserted "climate adds nothing" when the truth was "climate was
never entered".

A rung that adds no column must be refused, and a column that was expected and
did not arrive must be reported.
"""
import numpy as np
import pandas as pd
import pytest

import run_ablation as ra


def _panel(n_units=6, y0=2015, y1=2019, climate="none", seed=0):
    """Panel shaped like prepare_lagged_data()'s output.

    ``climate`` is "none" (the failure case), "full" (all eight columns) or
    "partial" (Q1 only).
    """
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_units):
        for y in range(y0, y1 + 1):
            row = {
                "state": f"S{u:02d}", "year": y,
                "total_cases": float(rng.gamma(2.0, 500)),
                "ep_dur": float(rng.gamma(4.0, 3.0)),
                "peak_week": float(rng.integers(5, 30)),
                "R0": float(rng.normal(1.3, 0.2)),
                "log_pop": float(rng.uniform(5, 7)),
                "prev_total_cases": float(rng.gamma(2.0, 500)),
                "prev_ep_dur": float(rng.gamma(4.0, 3.0)),
                "prev_peak_week": float(rng.integers(5, 30)),
                "prev_R0": float(rng.normal(1.3, 0.2)),
                "prev_log_pop": float(rng.uniform(5, 7)),
                "prev_alpha_size": float(rng.normal(2.0, 0.2)),
                "prev_xmin_size": float(rng.uniform(50, 500)),
                "prev_alpha_trend": float(rng.normal(0, 0.05)),
            }
            if climate in ("full", "partial"):
                # the contemporaneous aggregates, which must never be used
                row["avg_temp_Q1"] = float(rng.normal(24, 2))
                row["prev_avg_temp_Q1"] = float(rng.normal(24, 2))
                row["prev_avg_umid_Q1"] = float(rng.normal(70, 5))
            if climate == "full":
                for var, mu, sd in (("temp", 24, 2), ("umid", 70, 5)):
                    for q in (1, 2, 3, 4):
                        row[f"prev_avg_{var}_Q{q}"] = float(rng.normal(mu, sd))
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# The failure that produced 95 zeros
# ---------------------------------------------------------------------------
def test_a_panel_without_climate_is_refused_not_scored():
    df = _panel(climate="none")
    with pytest.raises(SystemExit) as exc:
        ra.build_ladder(df)
    msg = str(exc.value)
    assert "M2" in msg
    assert "prev_avg_temp_Q1" in msg          # names what is missing
    assert "--attach-climate" in msg          # and what to do about it


def test_a_panel_without_climate_reports_which_columns_are_absent():
    df = _panel(climate="none")
    available, missing = ra.climate_availability(df)
    assert available == []
    assert list(missing) == list(ra.CLIMATE_BLOCK)
    assert len(missing) == 8


def test_require_climate_false_still_refuses_to_build_an_empty_rung():
    """Skipping the climate *check* must not resurrect a silent M2 == M1."""
    df = _panel(climate="none")
    with pytest.raises(SystemExit, match="identically zero"):
        ra.build_ladder(df, require_climate=False)


def test_hand_built_empty_rung_is_caught_by_the_nesting_assertion():
    df = _panel(climate="none")
    m1 = (ra.LAGGED_SIZE + ra.COVARIATE_DUR + ra.COVARIATE_PEAK
          + ra.EPI + ra.DEMO)
    hand = {"M1": m1, "M2": list(m1), "M3": m1 + ra.SCALING_SIZE}
    with pytest.raises(AssertionError, match="adds no column"):
        ra.assert_ladder_is_nested(hand)


# ---------------------------------------------------------------------------
# The correct case
# ---------------------------------------------------------------------------
def test_full_climate_gives_a_strictly_larger_m2():
    df = _panel(climate="full")
    ladder, dropped, climate = ra.build_ladder(df)
    assert climate["missing"] == []
    assert dropped == {}
    added = set(ladder["M2"]) - set(ladder["M1"])
    assert added == set(ra.CLIMATE_BLOCK)
    assert set(ladder["M1"]) < set(ladder["M2"]) < set(ladder["M3"])
    # and no contemporaneous aggregate ever appears in a rung
    for rung in ladder.values():
        assert not [c for c in rung if c.startswith("avg_")]


def test_partial_climate_is_reported_rather_than_hidden():
    df = _panel(climate="partial")
    ladder, dropped, climate = ra.build_ladder(df)
    assert set(climate["available"]) == {"prev_avg_temp_Q1", "prev_avg_umid_Q1"}
    assert len(climate["missing"]) == 6
    assert set(dropped["M2"]) == set(climate["missing"])
    assert set(ladder["M2"]) - set(ladder["M1"]) == set(climate["available"])


# ---------------------------------------------------------------------------
# Nesting is not enough; the added block must be non-empty
# ---------------------------------------------------------------------------
def test_nesting_alone_does_not_pass_when_m2_adds_nothing():
    df = _panel(climate="full")
    ladder, _, _ = ra.build_ladder(df)
    broken = {k: list(v) for k, v in ladder.items()}
    broken["M2"] = list(broken["M1"])          # drops the climate block
    with pytest.raises(AssertionError, match="adds no column"):
        ra.assert_ladder_is_nested(broken)


def test_m2_may_not_add_a_column_outside_the_declared_block():
    df = _panel(climate="full")
    ladder, _, _ = ra.build_ladder(df)
    broken = {k: list(v) for k, v in ladder.items()}
    broken["M2"] = broken["M2"] + ["prev_R0_extra"]
    with pytest.raises(AssertionError, match="declared climate block"):
        ra.assert_ladder_is_nested(broken)


# ---------------------------------------------------------------------------
# Rebuilding the block from an explicit file
# ---------------------------------------------------------------------------
def _climate_file(tmp_path, years, units):
    rng = np.random.default_rng(7)
    rows = []
    for u in units:
        for y in years:
            row = {"state": u, "year": y}
            for var, mu, sd in (("temp", 24, 2), ("umid", 70, 5)):
                for q in (1, 2, 3, 4):
                    row[f"avg_{var}_Q{q}"] = float(rng.normal(mu, sd))
            rows.append(row)
    path = tmp_path / "climate.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def test_attach_climate_lags_the_block_by_one_year(tmp_path):
    df = _panel(climate="none", y0=2015, y1=2019)
    units = sorted(df["state"].unique())
    path = _climate_file(tmp_path, range(2014, 2020), units)

    out, notes = ra.attach_climate(df, "state", "BR", parquet=path, verbose=False)

    for col in ra.CLIMATE_BLOCK:
        assert col in out.columns
    # the value attached to year Y must be the file's value for year Y-1
    source = pd.read_parquet(path)
    for unit in units[:2]:
        for year in (2016, 2017, 2018):
            got = out.loc[(out["state"] == unit) & (out["year"] == year),
                          "prev_avg_temp_Q1"]
            want = source.loc[(source["state"] == unit) & (source["year"] == year - 1),
                              "avg_temp_Q1"]
            assert len(got) == 1 and len(want) == 1
            assert float(got.iloc[0]) == pytest.approx(float(want.iloc[0]))
    assert any("climate block attached" in n for n in notes)


def test_attach_climate_is_a_noop_when_the_panel_already_has_the_block():
    df = _panel(climate="full")
    out, notes = ra.attach_climate(df, "state", "BR", verbose=False)
    assert notes == []
    assert out is df


def test_attach_climate_names_the_missing_columns_of_a_bad_file(tmp_path):
    df = _panel(climate="none", y0=2015, y1=2019)
    path = tmp_path / "bad.parquet"
    pd.DataFrame({"state": ["S00"], "year": [2016], "avg_temp_Q1": [24.0]}).to_parquet(
        path, index=False
    )
    with pytest.raises(SystemExit, match="is missing"):
        ra.attach_climate(df, "state", "BR", parquet=path, verbose=False)


# ---------------------------------------------------------------------------
# A zero must never stand in for an undefined contrast
# ---------------------------------------------------------------------------
def test_identical_rungs_produce_a_null_contrast_not_a_zero(monkeypatch):
    class _Constant:
        def fit(self, X, y):
            self.c = float(np.nanmean(np.asarray(y, dtype=float)))
            return self

        def predict(self, X):
            return np.full(len(X), self.c, dtype=float)

    monkeypatch.setattr(
        ra, "fit_quantile_model",
        lambda Xtr, ytr, tau, model=None, seed=42, max_iter=300: _Constant().fit(Xtr, ytr),
    )
    df = _panel(climate="none", y0=2015, y1=2019)
    m1 = (ra.LAGGED_SIZE + ra.COVARIATE_DUR + ra.COVARIATE_PEAK
          + ra.EPI + ra.DEMO)
    hand_built = {"M1": m1, "M2": list(m1), "M3": m1 + ra.SCALING_SIZE}

    rows, _ = ra.run_regression_ablation(
        df, hand_built, "state", [2019], [0.5], "hist_gradient_boosting",
        42, 100, "state", verbose=False,
    )
    m2m1 = [r for r in rows
            if r["record_type"] == "contrast" and r["contrast"] == "M2-M1"]
    assert m2m1, "the undefined contrast must still be visible in the table"
    for r in m2m1:
        assert pd.isna(r["value"]), "an undefined contrast must be null, never 0.0"
        assert "same feature set" in r["note"]
