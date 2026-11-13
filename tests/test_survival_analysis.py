# tests/test_survival_analysis.py

"""Regression tests for the survival machinery in ``survival_analysis.py``.

The script compares a fitted power law to the empirical exceedance curve and
tests whether the pair estimated in one season describes the next. Both rest on
two claims that are cheap to state and easy to get wrong: that the product-limit
estimator collapses to a count ratio under a single truncation point, and that
the DKW half-width really is a simultaneous band. They are pinned here so a
future edit cannot quietly change either.
"""

import numpy as np
import pytest

import survival_analysis as sa


def _discrete_pareto(alpha, xmin, n, seed=0):
    """Draw from zeta(alpha) truncated at xmin by inverting its CDF."""
    rng = np.random.default_rng(seed)
    ks = np.arange(xmin, 300000)
    w = ks.astype(float) ** (-alpha)
    w /= w.sum()
    return ks[np.searchsorted(np.cumsum(w), rng.uniform(size=n))]


def test_ccdf_equals_the_count_ratio():
    """The risk fraction at s is exactly the share of units still above s."""
    s = np.array([5, 5, 7, 9, 9, 9, 12])
    values, emp, at_risk = sa.empirical_ccdf(s)
    np.testing.assert_allclose(emp, at_risk / s.size)
    np.testing.assert_allclose(at_risk, [7, 5, 4, 1])
    np.testing.assert_array_equal(values, [5, 7, 9, 12])


def test_ccdf_convention_is_inclusive():
    """``Pr(S >= s)``, not ``Pr(S > s)``: the convention the zeta formula uses.

    The product-limit estimator as usually written returns ``Pr(S > s)``. The
    count ratio returns ``Pr(S >= s)``. They differ by the mass at ``s``, and
    ``zeta(alpha, s)/zeta(alpha, x_min)`` is the inclusive one, so the
    comparison in ``_departure`` must use the inclusive one. Getting this
    backwards shifts the whole curve by one lattice point -- at ``x_min`` near
    700 that is a jump of 1/700 of the curve where the empirical support is
    densest, and it would read as a systematic misfit that is not there.

    Pinned as: inclusive at ``v_k`` equals exclusive at ``v_{k-1}``.
    """
    s = np.sort(_discrete_pareto(2.0, 50, 2000, seed=3)).astype(np.int64)
    values, emp, at_risk = sa.empirical_ccdf(s)

    survival = 1.0
    exclusive = []
    for v in values:
        survival *= 1.0 - np.sum(s == v) / np.sum(s >= v)
        exclusive.append(survival)
    exclusive = np.array(exclusive)

    np.testing.assert_allclose(emp[1:], exclusive[:-1])
    assert emp[0] == 1.0
    assert exclusive[-1] == 0.0
    assert not np.allclose(emp, exclusive)


def test_dkw_halfwidth_shrinks_with_n():
    widths = [sa.dkw_epsilon(n) for n in (100, 1000, 10000)]
    assert widths[0] > widths[1] > widths[2]
    # At 95% and n=4000 the analytic value is sqrt(log(40)/8000) ~= 0.0215.
    assert sa.dkw_epsilon(4000) == pytest.approx(0.0215, abs=5e-4)


def test_pareto_ccdf_is_decreasing_and_bounded():
    grid = np.array([100, 1000, 10000, 100000])
    ccdf = sa.pareto_ccdf(grid, alpha=2.03, xmin=100)
    assert np.all(np.diff(ccdf) < 0)
    assert np.all(ccdf <= 1.0)
    assert ccdf[0] == pytest.approx(1.0)


def test_pareto_hazard_matches_the_continuous_limit():
    """The discrete hazard approaches (alpha-1)/s far above the bound."""
    alpha, s = 2.03, 1_000_000.0
    assert float(sa.pareto_hazard(s, alpha)) == pytest.approx(
        (alpha - 1) / s, rel=1e-4)


def test_departure_has_power_against_a_wrong_xmin():
    """The statistic must separate a right fit from a badly wrong one.

    A diagnostic that cannot fail is not a diagnostic. Changing x_min by a
    third has to move the departure by far more than the band width; that is
    also why the in-season departure is reported as a magnitude and not a
    p-value, since x_min was chosen to minimise it.
    """
    s = _discrete_pareto(2.026, 730, 4000, seed=7)
    right = sa._departure(s, 2.026, 730)
    wrong = sa._departure(s, 2.026, 500)
    assert right is not None and wrong is not None
    assert right["departure_in_bands"] < 1.0
    assert wrong["departure_in_bands"] > 5.0


def test_integer_counts_refuses_a_continuous_series():
    sa._integer_counts([1.0, 2.0, 3.0])  # exact integers are fine
    with pytest.raises(ValueError):
        sa._integer_counts([1.0, 2.5, 3.0])


if __name__ == "__main__":
    import sys

    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"PASS {name}")
        except Exception as exc:  # pragma: no cover - manual runner
            failures += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    sys.exit(1 if failures else 0)
