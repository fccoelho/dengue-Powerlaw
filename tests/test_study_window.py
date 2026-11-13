"""The study window is declared once, and the database agrees with it.

The regression this guards against is not a wrong number in the text. It is a
text, a prose gate and a script that agree with each other and disagree with the
database, so that only a clean rebuild reveals the split: the script stopped at
2025, the database carried a 2026 cross-section written by an earlier run, and
``INSERT OR REPLACE`` kept it alive through every rebuild after that.
"""

import os
import sqlite3

import pytest

from study_window import (
    FIRST_SEASON_YEAR,
    LAST_SEASON_YEAR,
    POOLED_YEAR,
    SEASON_YEARS,
    WEEKLY_YEARS,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(ROOT, "powerlaw_results.db")


def test_season_window_is_contiguous_and_inclusive():
    """``range(2011, 2026)`` excludes 2026; a window that means 2011--2026 must not."""
    assert SEASON_YEARS[0] == FIRST_SEASON_YEAR
    assert SEASON_YEARS[-1] == LAST_SEASON_YEAR
    assert SEASON_YEARS == list(range(FIRST_SEASON_YEAR, LAST_SEASON_YEAR + 1))
    assert LAST_SEASON_YEAR in SEASON_YEARS


def test_pooled_sentinel_is_not_a_calendar_year():
    assert POOLED_YEAR not in SEASON_YEARS


def test_weekly_window_starts_before_the_first_season():
    """A season has to be observed beginning; the decomposition needs the weeks before."""
    assert WEEKLY_YEARS[0] < FIRST_SEASON_YEAR
    assert WEEKLY_YEARS[-1] == LAST_SEASON_YEAR


def test_fitting_script_uses_the_declared_window():
    pytest.importorskip("powerlaw", reason="fitting script needs powerlaw")
    pytest.importorskip("dotenv", reason="fitting script needs python-dotenv")
    import fit_episcanner_pl

    assert list(fit_episcanner_pl.YEARS) == SEASON_YEARS


def test_prune_keeps_the_window_and_the_pooled_fit():
    """Pruning is what makes one run equal one database."""
    pytest.importorskip("powerlaw", reason="fitting script needs powerlaw")
    pytest.importorskip("dotenv", reason="fitting script needs python-dotenv")
    import fit_episcanner_pl as fep

    con = sqlite3.connect(":memory:")
    cur = con.cursor()
    cur.execute("CREATE TABLE episcanner_fits (metric TEXT, year INT)")
    cur.execute(
        "CREATE TABLE episcanner_state_fits (state TEXT, metric TEXT, year INT)")
    outside = (FIRST_SEASON_YEAR - 1, LAST_SEASON_YEAR + 1)
    for year in (POOLED_YEAR, FIRST_SEASON_YEAR, LAST_SEASON_YEAR) + outside:
        cur.execute(
            "INSERT INTO episcanner_fits VALUES ('total_cases', ?)", (year,))
        cur.execute(
            "INSERT INTO episcanner_state_fits VALUES ('SP', 'total_cases', ?)",
            (year,))

    fep.prune_out_of_window(cur)

    expected = [POOLED_YEAR, FIRST_SEASON_YEAR, LAST_SEASON_YEAR]
    for table in ("episcanner_fits", "episcanner_state_fits"):
        kept = sorted(r[0] for r in cur.execute(f"SELECT year FROM {table}"))
        assert kept == expected
    con.close()


def test_committed_database_matches_the_declared_window():
    """The manuscript counts these cross-sections, so the rows have to be there."""
    if not os.path.exists(DB):
        pytest.skip("powerlaw_results.db not present")
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    try:
        got = sorted(r[0] for r in con.execute(
            "SELECT DISTINCT year FROM episcanner_fits WHERE year <> 0"))
    finally:
        con.close()
    assert got == SEASON_YEARS
