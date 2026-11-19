"""The size-error audit: the reduction, the rule, and the failure it detects.

The decisive case is the last one. On the real panel the two medians are the same
number (62.7% vs 62.7%) and the third quartiles differ by 5%, and a previous
caption asserted that the median was larger below the threshold. A test that only
checked the arithmetic would have passed that caption; the rule test is what
makes the claim falsifiable.
"""
import numpy as np
import pandas as pd
import pytest

import size_error as se


def _unit(n_small=400, n_large=40, small_sigma=1.0, large_sigma=0.6, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "geocode": [f"{i:07d}" for i in range(n_small + n_large)],
        "population": np.concatenate([rng.uniform(1e3, 9e4, n_small),
                                      rng.uniform(1e5, 4e6, n_large)]),
        "ape": np.concatenate([
            rng.lognormal(np.log(60), small_sigma, n_small),
            rng.lognormal(np.log(60), large_sigma, n_large)]),
        "n_origins": 3,
    })


def _oof(unit, origins=(2023, 2024, 2025), seed=1):
    """Inflate a unit table back into row-level predictions for `unit_errors`."""
    rng = np.random.default_rng(seed)
    rows = []
    for _, r in unit.iterrows():
        for o in origins:
            # y_pred chosen so that |pred - true| / true == ape / 100
            true = float(rng.uniform(50, 5000))
            pred = true * (1 + r["ape"] / 100.0)
            rows.append({"origin": o, "geocode": r["geocode"], "y_true": true,
                         "y_pred": pred, "population": r["population"]})
    return pd.DataFrame(rows)


def test_unit_errors_collapses_origins_to_one_row_per_municipality():
    unit = _unit()
    oof = _oof(unit)
    assert len(oof) == 3 * len(unit)
    back = se.unit_errors(oof)
    assert len(back) == len(unit)
    assert set(back["geocode"]) == set(unit["geocode"])
    assert (back["n_origins"] == 3).all()
    # Population is a constant of the unit, so the collapse cannot scramble it.
    merged = back.merge(unit, on="geocode", suffixes=("", "_in"))
    assert np.allclose(merged["population"], merged["population_in"])


def test_unit_errors_drops_predictions_without_a_relative_error():
    unit = _unit(n_small=10, n_large=5)
    oof = _oof(unit, origins=(2024,))
    bad = pd.DataFrame([
        {"origin": 2024, "geocode": "0000000", "y_true": 0, "y_pred": 10.0,
         "population": 5e3},      # no season -> no relative error
        {"origin": 2024, "geocode": "0000001", "y_true": 100, "y_pred": 0.0,
         "population": 5e3},      # zero forecast -> no ratio
        {"origin": 2024, "geocode": "0000002", "y_true": 100, "y_pred": 120.0,
         "population": 0.0},      # no population -> not attributable
    ])
    back = se.unit_errors(pd.concat([oof, bad], ignore_index=True))
    assert "0000000" not in set(back["geocode"])
    assert "0000001" not in set(back["geocode"])
    assert "0000002" not in set(back["geocode"])


def test_class_stats_reports_the_median_and_the_tail():
    s = se.class_stats(_unit())
    assert s["below"]["n"] + s["above"]["n"] == s["n_units"]
    assert 0 < s["share_below"] < 1
    for side in ("below", "above"):
        assert s[side]["q1"] <= s[side]["median"] <= s[side]["q3"] <= s[side]["p90"]
    assert s["q3_ratio"] > 0


def test_the_two_medians_are_the_same_number_when_the_cut_is_lopsided():
    """Why the claim cannot be stated about the median.

    With 91% of the units below the cut, the small class *is* the sample: its
    median is the overall median to within the 9% above it. Two medians that
    agree is therefore not evidence of no effect -- it is a property of the
    design.
    """
    unit = _unit(n_small=9100, n_large=900)
    assert se.class_stats(unit)["share_below"] > 0.9
    overall = float(np.median(unit["ape"]))
    s = se.class_stats(unit)
    assert abs(s["below"]["median"] - overall) < 0.05 * overall


def test_the_rule_drops_an_effect_that_is_only_in_the_median_scale():
    """A shift that moves both classes together has no Q3 ratio to report."""
    unit = _unit(small_sigma=0.7, large_sigma=0.7)
    s = se.class_stats(unit)
    boot = se.bootstrap_q3_ratio(unit, n_boot=300)
    d = se.decide(s, boot)
    assert d["verdict"] == "drop", d["reason"]
    assert d["q3_ratio"] < se.DECISION_Q3_RATIO


def test_the_rule_keeps_a_heavy_tail_below_the_cut():
    unit = _unit(small_sigma=1.25, large_sigma=0.45)
    s = se.class_stats(unit)
    boot = se.bootstrap_q3_ratio(unit, n_boot=300)
    d = se.decide(s, boot)
    assert d["verdict"] == "keep", d["reason"]
    assert d["q3_ratio"] >= se.DECISION_Q3_RATIO


def test_the_rule_drops_a_ratio_whose_interval_straddles_one():
    """A big point ratio on a handful of units is not an effect."""
    s = {"below": {"median": 60.0, "q3": 100.0}, "above": {"median": 60.0,
                                                          "q3": 70.0},
         "q3_ratio": 1.43, "median_ratio": 1.0, "share_over_big_ratio": 1.2,
         "cut": se.SIZE_CUT}
    d = se.decide(s, {"ci_low": 0.9, "ci_high": 2.2, "excludes_one": False,
                      "n_boot": 10})
    assert d["verdict"] == "drop"
    assert "does not exclude 1" in d["reason"]


def test_class_stats_refuses_an_empty_class():
    unit = _unit(n_large=1)
    unit.loc[unit["population"] >= se.SIZE_CUT, "population"] = 1e3
    with pytest.raises(ValueError, match="both size classes"):
        se.class_stats(unit)


def test_draw_ecdf_returns_a_figure_with_two_curves():
    import matplotlib
    matplotlib.use("Agg")
    unit = _unit()
    fig, ax = se.draw_ecdf(unit)
    lines = [ln for ln in ax.get_lines() if ln.get_drawstyle() == "steps"]
    assert len(lines) == 2
    assert ax.get_xscale() == "log"
    assert ax.get_ylim()[1] <= 1.1
