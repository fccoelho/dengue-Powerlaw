"""Regression tests for the year-based forward-chaining splitter.

The bug these tests pin down
----------------------------
``train_binary_model()`` used to validate with
``sklearn.model_selection.TimeSeriesSplit``, which splits a matrix by *row
position*. Its input frame comes from ``prepare_lagged_data()``, which sorts on
``[group_col, "year"]``, so contiguous row blocks are blocks of
**municipalities**, not of time. On the real training panel (32,824 rows,
4,725 municipalities, 2010-2023) each of the five folds trained on 2010-2023
and validated on 2010-2023: 100% of validation rows carried a year already
present in training. The reported "temporal cross-validation" was a
grouped-by-municipality holdout.

These tests assert what the design actually requires:

* a fold never trains on a year at or after its validation year;
* the training window expands and is nested across folds;
* each validation year is used exactly once;
* the folds do not depend on the order of the rows;
* the old splitter is caught by ``assert_forward_in_time``;
* ``train_binary_model()`` recomputes the percentile threshold, the labels and
  the class weight inside each fold, from that fold's training years only.
"""

import numpy as np
import pandas as pd
import pytest

import binary_classifier as bc
from temporal_split import (
    YearForwardChainSplit,
    assert_forward_in_time,
    summarize_folds,
    year_forward_chain_folds,
)

YEARS = np.arange(2010, 2024)


def _panel(n_units=30, years=YEARS):
    """Year vector with one block per year.

    Only the *values* matter for the splitter tests, which are order-invariant
    by construction. ``_unit_sorted_panel`` reproduces the order that
    ``prepare_lagged_data()`` actually returns.
    """
    return np.repeat(np.asarray(years), n_units)


def _unit_sorted_panel(n_units=30, years=YEARS):
    """Frame order emitted by prepare_lagged_data(): unit-major, then year.

    ``sort_values([group_col, "year"])`` lays the panel out as
    ``[unit1: 2010..2023][unit2: 2010..2023]...``, so a contiguous row block is
    a block of municipalities over the whole calendar.
    """
    return np.tile(np.asarray(years), n_units)


# ---------------------------------------------------------------------------
# Splitter behaviour
# ---------------------------------------------------------------------------


def test_split_never_trains_on_the_validation_year_or_later():
    years = _panel()
    folds = year_forward_chain_folds(years)
    assert len(folds) == 5
    for train_idx, val_idx in folds:
        train_years = np.unique(years[train_idx])
        val_years = np.unique(years[val_idx])
        assert val_years.size == 1, "a fold must validate on exactly one year"
        assert val_years.min() > train_years.max()
        assert np.intersect1d(val_years, train_years).size == 0


def test_forward_in_time_guard_accepts_the_year_split():
    years = _panel()
    assert assert_forward_in_time(years, year_forward_chain_folds(years)) is True


def test_expanding_window_grows_and_is_nested():
    years = _panel()
    folds = year_forward_chain_folds(years)
    sizes = [len(tr) for tr, _ in folds]
    assert sizes == sorted(sizes) and sizes[0] < sizes[-1]
    for (prev_tr, _), (next_tr, _) in zip(folds, folds[1:]):
        assert set(prev_tr).issubset(set(next_tr)), (
            "an expanding window must contain every earlier training row"
        )


def test_each_validation_year_is_used_exactly_once():
    years = _panel()
    folds = year_forward_chain_folds(years)
    val_years = [int(np.unique(years[v])[0]) for _, v in folds]
    assert val_years == [2019, 2020, 2021, 2022, 2023]
    assert len(val_years) == len(set(val_years))
    # the most recent year is always held out last
    assert val_years[-1] == int(np.max(years))


def test_n_splits_caps_the_number_of_validation_years():
    years = _panel()
    assert len(year_forward_chain_folds(years, n_splits=3)) == 3
    # never more than n_years - min_train_years
    assert len(year_forward_chain_folds(_panel(years=[2010, 2011, 2012]), n_splits=5)) == 1


def test_min_train_years_is_respected():
    years = _panel(years=[2010, 2011, 2012, 2013])
    folds = year_forward_chain_folds(years, n_splits=5, min_train_years=3)
    assert len(folds) == 1
    train_idx, val_idx = folds[0]
    assert np.unique(years[train_idx]).size == 3
    assert np.unique(years[val_idx])[0] == 2013


def test_too_few_years_raises_rather_than_falls_back_to_row_order():
    years = np.array([2010, 2010, 2011, 2011])
    with pytest.raises(ValueError, match="Not enough distinct years"):
        year_forward_chain_folds(years)


def test_groups_are_mandatory():
    splitter = YearForwardChainSplit()
    with pytest.raises(ValueError, match="requires `groups`"):
        list(splitter.split(np.zeros((10, 2))))


def test_get_n_splits_uses_the_year_vector():
    splitter = YearForwardChainSplit(n_splits=5, min_train_years=2)
    assert splitter.get_n_splits() == 5
    assert splitter.get_n_splits(groups=_panel()) == 5
    assert splitter.get_n_splits(groups=np.array([2010, 2011, 2012])) == 1


# ---------------------------------------------------------------------------
# Order invariance: the property TimeSeriesSplit does not have
# ---------------------------------------------------------------------------


def _as_year_sets(folds, years):
    return [
        (tuple(np.unique(years[tr])), tuple(np.unique(years[v])))
        for tr, v in folds
    ]


def test_year_split_is_invariant_to_row_order():
    years = _panel()
    rng = np.random.default_rng(7)
    perm = rng.permutation(years.size)
    straight = _as_year_sets(year_forward_chain_folds(years), years)
    shuffled = _as_year_sets(year_forward_chain_folds(years[perm]), years[perm])
    assert straight == shuffled, "folds must not depend on the order of the rows"

    # sanity: the row-position splitter does depend on it
    from sklearn.model_selection import TimeSeriesSplit

    X = np.zeros((years.size, 1))
    pos_straight = _as_year_sets(TimeSeriesSplit(n_splits=5).split(X), years)
    pos_shuffled = _as_year_sets(TimeSeriesSplit(n_splits=5).split(X[perm]), years[perm])
    assert pos_straight != pos_shuffled


def _shared_year_fraction(years, folds):
    """Share of validation rows whose year also appears in their training fold."""
    return [
        float(np.isin(years[val_idx], np.unique(years[train_idx])).mean())
        for train_idx, val_idx in folds
    ]


def test_row_position_split_leaks_the_calendar_on_the_real_frame_order():
    """The bug, pinned: on a unit-sorted panel TimeSeriesSplit holds out units."""
    from sklearn.model_selection import TimeSeriesSplit

    years = _unit_sorted_panel()
    position_folds = list(TimeSeriesSplit(n_splits=5).split(np.zeros((years.size, 1))))

    assert min(_shared_year_fraction(years, position_folds)) == 1.0, (
        "on a unit-sorted panel every validation row must share a year with "
        "its training fold -- that is what makes the split non-temporal"
    )
    summary = summarize_folds(years, position_folds)
    assert (summary["shared_years_with_train"] > 0).all()
    for row in summary.itertuples():
        assert row.train_years == f"{years.min()}-{years.max()}"
        assert row.n_train_years == np.unique(years).size

    # the guard that exists to prevent exactly this rejects those folds
    with pytest.raises(AssertionError, match="not forward in time"):
        assert_forward_in_time(years, position_folds)


def test_row_position_split_is_only_accidentally_correct_on_a_year_sorted_frame():
    """Why the sort order cannot be left implicit.

    If the very same rows happen to be laid out year-major, the row-position
    split looks temporal -- but it still cuts through calendar years, so a
    validation year is still partly present in training. The scheme is correct
    only for a sort order nobody declares, which is why the split here is
    defined on the year index instead.
    """
    from sklearn.model_selection import TimeSeriesSplit

    years = _panel()  # year-major
    position_folds = list(TimeSeriesSplit(n_splits=5).split(np.zeros((years.size, 1))))
    shared = _shared_year_fraction(years, position_folds)

    assert min(shared) < 0.5, "year-major ordering hides the problem"
    assert max(shared) > 0.0, (
        "even then, contiguous blocks cut through calendar years, so no fold "
        "is cleanly forward in time"
    )


def test_summarize_folds_reports_zero_shared_years_for_the_year_split():
    years = _panel()
    summary = summarize_folds(years, year_forward_chain_folds(years))
    assert list(summary.columns) == [
        "fold", "n_train", "n_val", "train_years",
        "n_train_years", "val_year", "shared_years_with_train",
    ]
    assert (summary["shared_years_with_train"] == 0).all()
    assert summary["n_train_years"].tolist() == sorted(summary["n_train_years"].tolist())


# ---------------------------------------------------------------------------
# Integration with train_binary_model()
# ---------------------------------------------------------------------------

def _classifier_frame(n_units=60, years=range(2010, 2024), seed=0):
    """Minimal frame with the columns prepare_lagged_data() emits."""
    rng = np.random.default_rng(seed)
    season = {y: rng.lognormal(0.0, 0.5) for y in years}
    long_term = {y: 1.0 + 0.04 * (y - min(years)) for y in years}
    rows = []
    for unit in range(n_units):
        level = rng.lognormal(7.0, 1.0)
        previous = level
        for year in years:
            cases = level * season[year] * long_term[year] * rng.lognormal(0.0, 0.4)
            rows.append(
                {
                    "geocode": 1_100_000 + unit,
                    "year": year,
                    "total_cases": cases,
                    "prev_total_cases": previous,
                    "prev_ep_dur": rng.uniform(5, 30),
                    "prev_alpha_size": rng.uniform(1.6, 2.6),
                    "prev_xmin_size": rng.uniform(1, 500),
                    "log_pop": np.log10(level * 100),
                }
            )
            previous = cases
    return pd.DataFrame(rows).sort_values(["geocode", "year"]).reset_index(drop=True)


def test_train_binary_model_validates_on_future_years_only():
    pytest.importorskip("xgboost")
    df = _classifier_frame()
    result = bc.train_binary_model(
        df, test_year=2022, percentile=0.75, model_type="xgboost", random_state=42
    )

    assert result["metrics"]["cv_scheme"] == "year_forward_chain"
    assert result["metrics"]["cv_threshold_source"] == "out_of_fold"

    summary = result["cv_fold_summary"]
    assert len(summary) == 5
    assert (summary["shared_years_with_train"] == 0).all()
    assert summary["val_year"].tolist() == [2017, 2018, 2019, 2020, 2021]
    for row in summary.itertuples():
        assert int(row.train_years.split("-")[1]) < row.val_year

    # out-of-fold predictions cover exactly the validation folds that were fitted
    assert len(result["cv_y_true"]) == sum(
        r["n_val_fold"] for r in result["cv_results"]
    )


def test_threshold_and_class_weight_are_recomputed_within_each_fold():
    pytest.importorskip("xgboost")
    df = _classifier_frame()
    result = bc.train_binary_model(
        df, test_year=2022, percentile=0.75, model_type="xgboost", random_state=42
    )

    global_threshold = result["threshold"]
    per_fold = [r["threshold_case_count_fold"] for r in result["cv_results"]]

    # the fold threshold equals the percentile of its OWN training years
    for row in result["cv_results"]:
        val_year = row["val_year"]
        expected = bc.compute_percentile_threshold(
            df.loc[df["year"] < val_year, "total_cases"].to_numpy(dtype=float),
            percentile=0.75,
        )
        assert row["threshold_case_count_fold"] == pytest.approx(expected)

    # and it is not the full-training threshold reused five times
    assert len(set(np.round(per_fold, 6))) == len(per_fold)
    assert any(abs(t - global_threshold) > 1e-6 for t in per_fold)
    assert all(r["scale_pos_weight_fold"] > 0 for r in result["cv_results"])


def test_too_few_training_years_skips_cv_without_crashing():
    pytest.importorskip("xgboost")
    # Two training years cannot support a forward-chaining fold (it needs one
    # year to train on and a later one to validate on).
    df = _classifier_frame(years=range(2020, 2024))
    result = bc.train_binary_model(
        df, test_year=2022, percentile=0.75, model_type="xgboost", random_state=42
    )
    assert result["cv_results"] == []
    assert result["metrics"]["cv_threshold_source"] == "in_sample_fallback"
    assert result["metrics"]["cv_n_skipped_folds"] == 1
    assert "insufficient_years" in result["cv_skipped_folds"][0]["reason"]
    assert result["model"] is not None

    # boundary: three training years yield exactly one fold, not zero
    df3 = _classifier_frame(years=range(2020, 2024))
    result3 = bc.train_binary_model(
        df3, test_year=2023, percentile=0.75, model_type="xgboost", random_state=42
    )
    assert len(result3["cv_results"]) == 1
    assert result3["cv_fold_summary"]["shared_years_with_train"].tolist() == [0]
    assert result3["metrics"]["cv_threshold_source"] == "out_of_fold"


if __name__ == "__main__":
    import traceback
    failures = []
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception:
                failures.append(name)
                print(f"FAIL {name}")
                traceback.print_exc()
    print(f"\n{len(failures)} failure(s)")
    raise SystemExit(1 if failures else 0)
