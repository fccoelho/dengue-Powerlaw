"""Municipality-level behaviour of the rarity classifier.

These tests pin the design decision that makes a municipal classifier mean
something. With a Q_p pooled across every municipality in the country, the
label ``Y = 1[S_t > Q_p]`` is a statement about municipality size: a small
municipality is labelled 0 in every year it exists, and a classifier that only
sees ``log_pop`` reaches an AUC above 0.9. With a per-unit Q_p the same
population-only classifier sits at chance, and the label is a statement about
the season.

They also pin the effect-size floor: the original ``verdict`` must not be
rewritten by it, because a rule fixed after seeing the numbers is not a
pre-registered rule.
"""

import numpy as np
import pandas as pd
import pytest

import run_ablation as ra


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
def _panel(n_units=400, y0=2010, y1=2021, seed=0):
    """A panel where size and epidemic intensity are independent.

    Size is drawn log-normally over several orders of magnitude, so if the
    threshold is pooled the label becomes a size indicator by construction.
    """
    rng = np.random.default_rng(seed)
    log_pop = np.clip(rng.normal(9.6, 1.0, n_units), 6.8, 14.0)
    pop = 10 ** log_pop
    rows = []
    u = rng.normal(0, 1, n_units)
    for year in range(y0, y1 + 1):
        u = 0.7 * u + rng.normal(0, 0.9, n_units)
        cases = pop * 0.0015 * np.exp(u)
        alpha = 2.0 + 0.2 * rng.normal(0, 1, n_units)
        rows.append(pd.DataFrame({
            "geocode": np.arange(n_units) + 1100000,
            "year": year,
            "total_cases": cases,
            "log_pop": log_pop,
            "prev_log_pop": log_pop,
            "prev_alpha_size": alpha,
            "prev_xmin_size": np.maximum(1.0, np.round(3 * rng.lognormal(0, .3, n_units))),
            "prev_alpha_trend": rng.normal(0, .1, n_units),
        }))
    df = pd.concat(rows, ignore_index=True).sort_values(["geocode", "year"])
    df["prev_total_cases"] = df.groupby("geocode")["total_cases"].shift(1)
    return df.dropna(subset=["prev_total_cases"]).reset_index(drop=True)


def _ladder():
    return {
        "M1": ["prev_total_cases", "prev_log_pop"],
        "M2": ["prev_total_cases", "prev_log_pop"],
        "M3": ["prev_total_cases", "prev_log_pop",
               "prev_alpha_size", "prev_xmin_size", "prev_alpha_trend"],
    }


def _args(**kw):
    base = dict(level="city", threshold_scope=None, min_effect_rel=None,
                tasks="size,duration,peak_week", include_classifier=False)
    base.update(kw)
    return type("A", (), base)()


# ---------------------------------------------------------------------------
# the threshold scope
# ---------------------------------------------------------------------------
def test_per_unit_thresholds_exclude_short_history_instead_of_pooling():
    train = pd.DataFrame({
        "geocode": [1, 1, 1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 3],
        "total_cases": [10, 20, 30, 40, 50, 5, 6, 1, 2, 3, 4, 100, 8],
    })
    thr, short = ra._per_unit_thresholds(train, "geocode", 0.75, min_unit_years=5)
    assert short == 1, "unit 2 has two seasons and must be counted as too short"
    assert set(thr) == {1, 3}
    assert 2 not in thr, "a short-history unit must never receive a pooled threshold"
    assert thr[3] == pytest.approx(float(np.percentile([1, 2, 3, 4, 100, 8], 75)))


def test_pooled_scope_makes_the_label_a_population_indicator_but_per_unit_does_not():
    df = _panel()
    ladder = _ladder()

    def pop_auc(scope):
        fold = ra.classifier_fold(
            df, ladder, "geocode", 2021, threshold_scope=scope,
            min_unit_years=5, max_iter=25, seed=0, verbose=False)
        assert fold is not None
        assert fold["pop_prob"] is not None
        return ra._auc(fold["y_val"], fold["pop_prob"])

    auc_pooled = pop_auc("pooled")
    auc_per_unit = pop_auc("per_unit")

    assert auc_pooled > 0.85, f"pooled label is answerable from size (AUC {auc_pooled:.3f})"
    assert abs(auc_per_unit - 0.5) < 0.10, (
        f"per-unit label should not be answerable from size (AUC {auc_per_unit:.3f})")


def test_default_scope_follows_the_aggregation_level():
    assert ra.resolve_threshold_scope(_args(level="city")) == "per_unit"
    assert ra.resolve_threshold_scope(_args(level="state")) == "pooled"
    assert ra.resolve_threshold_scope(
        _args(level="city", threshold_scope="pooled")) == "pooled"


def test_default_effect_floor_follows_the_aggregation_level():
    assert ra.resolve_min_effect_rel(_args(level="city")) == pytest.approx(0.02)
    assert ra.resolve_min_effect_rel(_args(level="state")) == pytest.approx(0.0)
    assert ra.resolve_min_effect_rel(_args(level="city", min_effect_rel=0.0)) == 0.0


# ---------------------------------------------------------------------------
# tasks
# ---------------------------------------------------------------------------
def test_tasks_are_selectable_and_include_classifier_is_a_shorthand():
    assert ra.resolve_tasks(_args(tasks="exceedance")) == ["exceedance"]
    assert ra.resolve_tasks(_args(include_classifier=True)) == [
        "size", "duration", "peak_week", "exceedance"]
    assert ra.resolve_tasks(
        _args(tasks="size,exceedance", include_classifier=True)) == ["size", "exceedance"]


def test_unknown_task_is_refused():
    with pytest.raises(SystemExit):
        ra.resolve_tasks(_args(tasks="size,banana"))
    with pytest.raises(SystemExit):
        ra.resolve_tasks(_args(tasks=""))


# ---------------------------------------------------------------------------
# the ablation rows
# ---------------------------------------------------------------------------
def test_classifier_ablation_emits_a_population_only_diagnostic():
    df = _panel(n_units=200)
    rows, diffs = ra.run_classifier_ablation(
        df, _ladder(), "geocode", [2021], "city", seed=0, n_boot=100,
        percentile=0.75, threshold_scope="per_unit", min_unit_years=5,
        verbose=False, max_iter=25)
    diag = [r for r in rows if r["record_type"] == "diagnostic"]
    assert len(diag) == 1
    assert diag[0]["rung"] == "POP"
    assert diag[0]["features"] == "prev_log_pop"
    assert "population-only floor" in diag[0]["note"]
    losses = [r for r in rows if r["record_type"] == "loss"]
    assert losses and all("auc" in r for r in losses)


def test_contrast_rows_carry_the_relative_improvement():
    df = _panel(n_units=200)
    rows, _ = ra.run_classifier_ablation(
        df, _ladder(), "geocode", [2021, 2022], "city", seed=0, n_boot=100,
        percentile=0.75, threshold_scope="per_unit", min_unit_years=5,
        verbose=False, max_iter=25)
    contrasts = [r for r in rows if r["record_type"] == "contrast"]
    assert contrasts
    for c in contrasts:
        assert "rel_improvement" in c
        assert np.isfinite(c["rel_improvement"])


# ---------------------------------------------------------------------------
# the effect floor must not rewrite the pre-registered verdict
# ---------------------------------------------------------------------------
def test_effect_floor_is_reported_separately_and_never_overwrites_the_verdict():
    origins = [2021, 2022, 2023, 2024, 2025]
    rows = [{
        "record_type": "contrast", "level": "city", "task": "exceedance",
        "contrast": "M3-M2", "origin": o, "tau": np.nan,
        "ci_excludes_zero": True, "improves": True,
        "rel_improvement": 0.001,   # a 0.1% gain: real but operationally nil
        "value": 0.0001,
    } for o in origins]
    diffs = {("exceedance", "M3-M2", None): {o: 0.0001 for o in origins}}

    dec = ra.decision_rows(rows, diffs, origins, "city", min_origins=3,
                           min_effect_rel=0.02)
    assert len(dec) == 1
    d = dec[0]
    assert d["verdict"] == "supported", "the original rule saw 5/5 intervals excluding zero"
    assert d["verdict_effect_aware"] == "not_supported", "0.1% is below the 2% floor"
    assert d["value"] == 5

    dec0 = ra.decision_rows(rows, diffs, origins, "city", min_origins=3,
                            min_effect_rel=0.0)
    assert dec0[0]["verdict_effect_aware"] == "supported"


def test_a_large_effect_passes_both_criteria():
    origins = [2021, 2022, 2023, 2024, 2025]
    rows = [{
        "record_type": "contrast", "level": "city", "task": "exceedance",
        "contrast": "M1-M0", "origin": o, "tau": np.nan,
        "ci_excludes_zero": True, "improves": True,
        "rel_improvement": 0.08, "value": 0.02,
    } for o in origins]
    diffs = {("exceedance", "M1-M0", None): {o: 0.02 for o in origins}}
    d = ra.decision_rows(rows, diffs, origins, "city", min_origins=3,
                         min_effect_rel=0.02)[0]
    assert d["verdict"] == d["verdict_effect_aware"] == "supported"
