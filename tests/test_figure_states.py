"""Figures 10: the state ranking behind the panels, and the panels.

The figure selects the four states whose municipalities the forecast predicts
best and draws one panel each. Two things make that selection defensible and
both are tested here: the state is read from the geocode rather than typed, and
a state with too few municipalities cannot win on one lucky forecast.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)


@pytest.fixture(scope="module")
def gf():
    """The figure module, imported once, with the working directory restored."""
    cwd = os.getcwd()
    sys.path.insert(0, os.path.join(HERE, "Manuscript", "figures"))
    try:
        import generate_figures
    finally:
        os.chdir(cwd)
    return generate_figures


def test_uf_is_read_from_the_leading_ibge_digits(gf):
    assert gf._uf_of(3550308) == "SP"
    assert gf._uf_of(1100015) == "RO"
    assert gf._uf_of(5300108) == "DF"
    # six-digit exports keep the same prefix
    assert gf._uf_of(355030) == "SP"


def test_unknown_geocode_is_not_folded_into_a_state(gf):
    assert gf._uf_of(None) == "??"
    assert gf._uf_of("not a code") == "??"
    assert gf._uf_of(99) == "??"


def _frame():
    rows = []
    for uf, apes in {"SP": [10, 12, 14], "MG": [30, 34, 36],
                     "RJ": [20, 22, 24], "RR": [1]}.items():
        for ape in apes * 8:
            rows.append({"uf": uf, "ape": float(ape),
                         "y_true": 100.0,
                         "y_pred": 100.0 / (1.0 + ape / 100.0)})
    return pd.DataFrame(rows)


def test_ranking_orders_by_median_ape(gf):
    r = gf._rank_states(_frame(), min_units=20)
    assert list(r.index) == ["SP", "RJ", "MG"]


def test_thin_states_cannot_win_on_one_forecast(gf):
    d = _frame()
    assert "RR" not in gf._rank_states(d, min_units=20).index
    # with no floor the state of one municipality tops the ranking
    assert gf._rank_states(d, min_units=1).index[0] == "RR"


def test_panels_report_the_shape_they_saved(gf, tmp_path):
    d = _frame()
    d["ape"] = gf._ape_pct(d["y_true"], d["y_pred"])
    ranking = gf._rank_states(d, min_units=20)
    chosen = list(ranking.index)
    shape = gf._draw_size_panels(d, ranking, chosen, 2025, 90.0,
                                 "fig10_pred_vs_obs", out_dir=str(tmp_path))
    from matplotlib import image as mpimg
    saved = mpimg.imread(str(tmp_path / "fig10_pred_vs_obs.png")).shape[:2]
    assert shape == [int(saved[1]), int(saved[0])]
    assert (tmp_path / "fig10_pred_vs_obs.pdf").exists()


def test_ape_is_percentage_and_ignores_sign(gf):
    assert gf._ape_pct([100.0], [50.0])[0] == pytest.approx(50.0)
    assert gf._ape_pct([100.0], [150.0])[0] == pytest.approx(50.0)
