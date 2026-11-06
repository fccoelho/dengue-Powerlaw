"""Reproduce every quantity reported in Manuscript/main.tex.

Usage:  python Manuscript/verification/verify_results.py [path_to_db]

Prints the main table, the per-state exponents, the national year-by-year
series and the alpha-vs-population correlation, all computed directly from
powerlaw_results.db. Nothing is hard-coded: if the database changes, this
script fails loudly rather than silently reporting stale numbers.
"""
import sqlite3
import sys
import os
import numpy as np
import pandas as pd

DEFAULT_DB = os.path.join(os.path.dirname(__file__), "..", "..", "powerlaw_results.db")

LEVELS = [
    ("L1", "weekly cases, city (full series)", "powerlaw_fits", ""),
    ("L2", "weekly cases, city-year", "powerlaw_fits_yearly", ""),
    ("L3", "annual total, city (across years)", "powerlaw_fits_total_cases", ""),
    ("L4", "epidemic size, state-year",
     "episcanner_state_fits", "metric='total_cases' AND year > 0"),
    ("L6", "epidemic size, national-year",
     "episcanner_fits", "metric='total_cases' AND year > 0"),
]

#: The duration constructions (former L5 and L7) are not verified here
#: because the manuscript no longer reports them: a season is bounded above
#: by the length of the year, so a scale-free law cannot apply to duration a
#: priori. ``fit_episcanner_pl.py`` no longer fits ``metric='ep_dur'``.


def summarise(con, table, where=""):
    """Compute the summary statistics for one aggregation level."""
    sql = "SELECT alpha, xmin, R, p FROM " + table
    if where:
        sql += " WHERE " + where
    d = pd.read_sql(sql, con).dropna()
    r, p = d["R"], d["p"]
    return dict(
        n=len(d),
        alpha_median=round(d["alpha"].median(), 2),
        q1=round(d["alpha"].quantile(0.25), 2),
        q3=round(d["alpha"].quantile(0.75), 2),
        xmin_median=round(d["xmin"].median(), 0),
        pct_xmin_le5=round(100 * (d["xmin"] <= 5).mean(), 1),
        pl_preferred=round(100 * (r > 0).mean(), 1),
        pl_significant=round(100 * ((r > 0) & (p < 0.05)).mean(), 1),
        exp_significant=round(100 * ((r < 0) & (p < 0.05)).mean(), 1),
    )


def main(db_path=DEFAULT_DB):
    """Print the full verification report for the manuscript."""
    con = sqlite3.connect(db_path)

    rows = []
    for lvl, label, table, where in LEVELS:
        s = summarise(con, table, where)
        s.update(level=lvl, unit=label)
        rows.append(s)
    main_table = pd.DataFrame(rows)[
        ["level", "unit", "n", "alpha_median", "q1", "q3", "xmin_median",
         "pct_xmin_le5", "pl_preferred", "pl_significant", "exp_significant"]
    ]
    print("=" * 100)
    print("TABLE 1 -- power-law fits by aggregation level")
    print("=" * 100)
    print(main_table.to_string(index=False))

    print()
    print("=" * 100)
    print("NATIONAL YEAR-BY-YEAR (year=0 is the pooled fit and is excluded)")
    print("=" * 100)
    nat = pd.read_sql(
        "SELECT year, metric, alpha, xmin, R, p FROM episcanner_fits "
        "WHERE metric='total_cases' AND year > 0 ORDER BY year", con)
    print(nat.round(3).to_string(index=False))

    print()
    print("=" * 100)
    print("PER-STATE EXPONENTS -- epidemic size (L4)")
    print("=" * 100)
    st = pd.read_sql(
        "SELECT state, alpha, R, p FROM episcanner_state_fits "
        "WHERE metric='total_cases' AND year > 0", con)
    per_state = st.groupby("state").agg(
        n_years=("alpha", "size"),
        alpha_median=("alpha", "median"),
        pl_preferred_pct=("R", lambda s: round(100 * (s > 0).mean(), 1)),
    ).round(3).sort_values("alpha_median")
    print(per_state.to_string())
    print("\nalpha across state medians: min=%.2f median=%.2f max=%.2f"
          % (per_state.alpha_median.min(), per_state.alpha_median.median(),
             per_state.alpha_median.max()))

    print()
    print("=" * 100)
    print("CORRELATION OF alpha WITH CITY SIZE (L1)")
    print("=" * 100)
    m = pd.read_sql(
        "SELECT f.alpha, c.population, c.incidence FROM powerlaw_fits f "
        "JOIN city_stats c USING (geocode) WHERE c.population > 0", con)
    m["logpop"] = np.log10(m.population)
    sub = m[["logpop", "alpha"]].dropna()
    print("Pearson  r(alpha, log10 population) = %+.3f  (n=%d)"
          % (sub.corr().iloc[0, 1], len(sub)))
    print("Spearman r(alpha, log10 population) = %+.3f"
          % sub.corr(method="spearman").iloc[0, 1])
    sub2 = m[["incidence", "alpha"]].dropna()
    print("Pearson  r(alpha, incidence)         = %+.3f  (n=%d)"
          % (sub2.corr().iloc[0, 1], len(sub2)))
    print("Spearman r(alpha, incidence)         = %+.3f"
          % sub2.corr(method="spearman").iloc[0, 1])
    con.close()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DB)
