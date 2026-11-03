"""
Year-based forward-chaining cross-validation for panel data.

Why this module exists
----------------------
``sklearn.model_selection.TimeSeriesSplit`` splits a matrix by *row position*.
On a panel whose rows are sorted by unit -- which is exactly what
``predictive_model.prepare_lagged_data()`` returns, because it sorts on
``[group_col, "year"]`` -- contiguous row blocks are blocks of *units*, not of
time. Every fold then trains on some municipalities across the whole calendar
and validates on other municipalities across the same calendar.

That is a grouped-by-municipality split. It does not leak the outcome column in
the narrow sense, but it is not a test of temporal generalisation: the
validation fold contains no year that is absent from training, so a model can
memorise year-specific signatures (a disrupted season, a record season) and
still look skilful. Reported skill then does not survive a change of forecast
origin.

This module splits on the year index instead. Each fold validates on one
calendar year and trains on every observation from strictly earlier years --
the expanding-window rolling origin described in the manuscript.

The splitter is dependency-free apart from numpy and implements the
scikit-learn splitter protocol (``split`` / ``get_n_splits``), so it can be
dropped wherever ``TimeSeriesSplit`` was used.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "YearForwardChainSplit",
    "year_forward_chain_folds",
    "assert_forward_in_time",
    "summarize_folds",
]


class YearForwardChainSplit:
    """Expanding-window cross-validator that holds out whole calendar years.

    Parameters
    ----------
    n_splits : int, default 5
        Maximum number of validation years. The effective number is
        ``min(n_splits, n_years - min_train_years)``.
    min_train_years : int, default 2
        Minimum number of distinct years in a training window. The first fold
        therefore trains on the earliest ``min_train_years`` years.

    Notes
    -----
    ``groups`` (the year vector) is mandatory. Unlike ``TimeSeriesSplit`` the
    split is not a function of row order, so shuffling the frame beforehand
    does not change the folds -- which is the property that makes the design
    auditable.
    """

    def __init__(self, n_splits: int = 5, min_train_years: int = 2) -> None:
        if int(n_splits) < 1:
            raise ValueError(f"n_splits must be >= 1, got {n_splits!r}")
        if int(min_train_years) < 1:
            raise ValueError(
                f"min_train_years must be >= 1, got {min_train_years!r}"
            )
        self.n_splits = int(n_splits)
        self.min_train_years = int(min_train_years)

    # -- helpers ----------------------------------------------------------
    def _years(self, groups):
        if groups is None:
            raise ValueError(
                "YearForwardChainSplit requires `groups` = the year of each "
                "row. Without it the split would fall back to row order, "
                "which is the failure mode this splitter exists to avoid."
            )
        years = np.asarray(groups)
        if years.ndim != 1:
            raise ValueError(f"groups must be 1-D, got shape {years.shape}")
        return years

    def _n_validation_years(self, years) -> int:
        return max(0, min(self.n_splits, np.unique(years).size - self.min_train_years))

    # -- sklearn protocol -------------------------------------------------
    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Number of folds, from the year vector when it is supplied."""
        if groups is None:
            return self.n_splits
        return self._n_validation_years(self._years(groups))

    def split(self, X, y=None, groups=None):
        """Yield ``(train_index, validation_index)`` pairs, oldest year first."""
        years = self._years(groups)
        n_val = self._n_validation_years(years)
        if n_val < 1:
            raise ValueError(
                "Not enough distinct years for a forward-chaining split: "
                f"need at least {self.min_train_years + 1}, got "
                f"{np.unique(years).size}."
            )
        unique_years = np.unique(years)
        for val_year in unique_years[-n_val:]:
            train_idx = np.flatnonzero(years < val_year)
            val_idx = np.flatnonzero(years == val_year)
            if train_idx.size == 0 or val_idx.size == 0:  # pragma: no cover
                continue
            yield train_idx, val_idx


def year_forward_chain_folds(years, n_splits: int = 5, min_train_years: int = 2):
    """Materialise the folds of :class:`YearForwardChainSplit` into a list.

    Parameters
    ----------
    years : array-like
        Year of each row, aligned with the feature matrix.
    n_splits, min_train_years :
        See :class:`YearForwardChainSplit`.

    Returns
    -------
    list of (np.ndarray, np.ndarray)
        ``(train_index, validation_index)`` per fold, oldest validation year
        first.

    Raises
    ------
    ValueError
        If fewer than ``min_train_years + 1`` distinct years are available.
    """
    splitter = YearForwardChainSplit(
        n_splits=n_splits, min_train_years=min_train_years
    )
    years = splitter._years(years)
    return list(splitter.split(np.empty((years.size, 1)), groups=years))


def assert_forward_in_time(years, folds) -> bool:
    """Verify that no fold trains on a year at or after its validation year.

    Cheap structural guard, intended to be called right after building the
    folds. It converts a silent leakage bug into an exception.

    Parameters
    ----------
    years : array-like
        Year of each row, aligned with the indices in ``folds``.
    folds : sequence of (array-like, array-like)
        ``(train_index, validation_index)`` pairs.

    Returns
    -------
    bool
        True when every fold is strictly forward in time.

    Raises
    ------
    AssertionError
        If any validation year is <= the maximum training year of its fold, or
        if a fold is empty.
    """
    years = np.asarray(years)
    for position, (train_idx, val_idx) in enumerate(folds, start=1):
        train_years = np.unique(years[np.asarray(train_idx)])
        val_years = np.unique(years[np.asarray(val_idx)])
        if train_years.size == 0 or val_years.size == 0:
            raise AssertionError(f"Fold {position} is empty.")
        if val_years.min() <= train_years.max():
            raise AssertionError(
                f"Fold {position} is not forward in time: validation years "
                f"{val_years.min()}-{val_years.max()} overlap or precede "
                f"training years {train_years.min()}-{train_years.max()}. "
                "The split is holding out units rather than years."
            )
    return True


def summarize_folds(years, folds):
    """Per-fold diagnostics as a :class:`pandas.DataFrame`.

    Useful both in tests and as the audit table of a run. Columns record the
    training window, the validation year, the fold sizes, and the number of
    validation rows whose year also appears in training -- which is zero for a
    correct forward-chaining split and large for a row-position split.
    """
    import pandas as pd

    columns = [
        "fold",
        "n_train",
        "n_val",
        "train_years",
        "n_train_years",
        "val_year",
        "shared_years_with_train",
    ]
    years = np.asarray(years)
    records = []
    for position, (train_idx, val_idx) in enumerate(folds, start=1):
        train_years = np.unique(years[np.asarray(train_idx)])
        val_years = np.unique(years[np.asarray(val_idx)])
        records.append(
            {
                "fold": position,
                "n_train": int(np.size(train_idx)),
                "n_val": int(np.size(val_idx)),
                "train_years": f"{train_years.min()}-{train_years.max()}",
                "n_train_years": int(train_years.size),
                "val_year": int(val_years.max()),
                "shared_years_with_train": int(
                    np.intersect1d(val_years, train_years).size
                ),
            }
        )
    return pd.DataFrame.from_records(records, columns=columns)


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    from sklearn.model_selection import TimeSeriesSplit

    n_units, years_grid = 40, np.arange(2010, 2024)
    # The layout prepare_lagged_data() returns: unit-major, then year.
    frame_years = np.tile(years_grid, n_units)
    print("rows:", frame_years.size, "| distinct years:", np.unique(frame_years).size)

    position_folds = list(TimeSeriesSplit(n_splits=5).split(np.empty((frame_years.size, 1))))
    year_folds = year_forward_chain_folds(frame_years)

    print("\n-- row-position split (TimeSeriesSplit) --")
    print(summarize_folds(frame_years, position_folds).to_string(index=False))
    print("\n-- year forward-chaining split --")
    print(summarize_folds(frame_years, year_folds).to_string(index=False))
