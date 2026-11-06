"""Fit discrete power laws to the EpiScanner annual epidemic descriptors.

Only one descriptor is fitted: the annual epidemic **size** (``total_cases``),
at three scopes -- state-year, national-year, and the pooled all-years
versions of both (stored with the sentinel ``year = 0``).

Epidemic duration (``ep_dur``) is deliberately *not* fitted. A season is
bounded above by the length of the year, so a scale-free law cannot apply to
duration a priori, and the fits confirmed it the only way they could: the
estimated exponent ran into the estimator's ceiling and the exponential won
in 15 of 15 national cross-sections. A near-constant exponent recovered from
every independent cross-section is not a universality result, it is evidence
that the model was asked the wrong question. Duration and peak timing remain
in the analysis as ordinary *covariates* of epidemic size, which is where
they carry information; they are not objects of a scaling claim and they are
not forecasting targets.

Rows previously written with ``metric='ep_dur'`` are left untouched by a
normal run. ``--purge-ep-dur`` removes them, which is what makes the database
match the paper: otherwise a reader querying
``episcanner_state_fits WHERE metric='ep_dur'`` finds fits the manuscript no
longer reports.
"""

import pandas as pd
import sqlite3
import powerlaw
import os
import dotenv
from fitpl import fetch_episcanner

dotenv.load_dotenv()

DB_PATH = "powerlaw_results.db"
STATES = [
    "AC", "AL", "AP", "AM", "BA", "CE", "DF", "ES", "GO", "MA", 
    "MT", "MS", "MG", "PA", "PB", "PR", "PE", "PI", "RJ", "RN", 
    "RS", "RO", "RR", "SC", "SP", "SE", "TO"
]
YEARS = list(range(2011, 2026))

#: The fitted descriptors. Adding a metric here is a claim that the variable
#: has unbounded support and a plausible scale-free regime; it is not a
#: formatting choice. See the module docstring for why ``ep_dur`` is excluded.
FIT_METRICS = ["total_cases"]

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS episcanner_fits (
                metric TEXT,
                year INTEGER,
                alpha REAL,
                xmin REAL,
                xmax REAL,
                R REAL,
                p REAL,
                PRIMARY KEY (metric, year)
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS episcanner_state_fits (
                state TEXT,
                metric TEXT,
                year INTEGER,
                alpha REAL,
                xmin REAL,
                xmax REAL,
                R REAL,
                p REAL,
                PRIMARY KEY (state, metric, year)
            )
        ''')
        conn.commit()

def fit_pl(data):
    try:
        # Filter for positive cases
        data = data[data > 0]
        if len(data) < 10:
            return None
        
        # Power law fitting
        fit = powerlaw.Fit(data, verbose=False, discrete=True)
        alpha = fit.power_law.alpha
        xmin = fit.power_law.xmin
        xmax = fit.power_law.xmax
        
        # Comparison with exponential distribution
        R, p = fit.distribution_compare('power_law', 'exponential', normalized_ratio=True)
        
        return alpha, xmin, xmax, R, p
    except Exception as e:
        # Some fits might fail if data is too sparse or non-compliant
        return None

def purge_ep_dur(db_path=DB_PATH):
    """Delete every ``ep_dur`` fit, so the database matches the paper.

    Run after ``--purge-ep-dur``. Reports the counts removed rather than
    deleting silently: a database that shrinks without saying so is the
    failure mode this whole pipeline is built to avoid.
    """
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        for table, key in (
            ("episcanner_state_fits", "state"),
            ("episcanner_fits", "metric"),
        ):
            try:
                n = cursor.execute(
                    f"SELECT COUNT(*) FROM {table} WHERE metric='ep_dur'"
                ).fetchone()[0]
            except sqlite3.OperationalError:
                continue
            cursor.execute(f"DELETE FROM {table} WHERE metric='ep_dur'")
            print(f"[purge] {table}: removed {n} ep_dur fits")
        conn.commit()


def report_stale_ep_dur(db_path=DB_PATH):
    """Warn when duration fits survive in the database."""
    with sqlite3.connect(db_path) as conn:
        left = {}
        for table in ("episcanner_state_fits", "episcanner_fits"):
            try:
                n = conn.execute(
                    f"SELECT COUNT(*) FROM {table} WHERE metric='ep_dur'"
                ).fetchone()[0]
            except sqlite3.OperationalError:
                continue
            if n:
                left[table] = n
    if left:
        total = sum(left.values())
        print(
            f"\n[WARNING] {total} ep_dur fits are still stored ({left}).\n"
            "          The manuscript no longer reports any duration "
            "aggregation level, so these rows are stale. Remove them with:\n"
            "              python fit_episcanner_pl.py --purge-ep-dur"
        )


def main():
    init_db()
    
    state_all_years_data = {state: [] for state in STATES}
    national_yearly_data = {year: [] for year in YEARS}
    
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        
        for year in YEARS:
            print(f"--- Processing Year {year} ---")
            for state in STATES:
                df = fetch_episcanner(state=state, year=year)
                if df is not None and not df.empty:
                    # Save for aggregations
                    state_all_years_data[state].append(df)
                    national_yearly_data[year].append(df)
                    
                    # Individual state/year fits
                    for metric in FIT_METRICS:
                        results = fit_pl(df[metric].values)
                        if results:
                            alpha, xmin, xmax, R, p = results
                            cursor.execute('''
                                INSERT OR REPLACE INTO episcanner_state_fits (state, metric, year, alpha, xmin, xmax, R, p)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            ''', (state, metric, year, alpha, xmin, xmax, R, p))
            
            # National fits for the current year
            if national_yearly_data[year]:
                br_year_df = pd.concat(national_yearly_data[year])
                for metric in FIT_METRICS:
                    results = fit_pl(br_year_df[metric].values)
                    if results:
                        alpha, xmin, xmax, R, p = results
                        cursor.execute('''
                            INSERT OR REPLACE INTO episcanner_fits (metric, year, alpha, xmin, xmax, R, p)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        ''', (metric, year, alpha, xmin, xmax, R, p))
                        print(f"National {metric} fit for {year}: alpha={alpha:.2f}")
            
            conn.commit()
        
        print("\n--- Processing Combined 'All Years' Fits (year=0) ---")
        
        # 1. State/All Years Combined
        for state in STATES:
            if state_all_years_data[state]:
                state_combined_df = pd.concat(state_all_years_data[state])
                for metric in FIT_METRICS:
                    results = fit_pl(state_combined_df[metric].values)
                    if results:
                        alpha, xmin, xmax, R, p = results
                        cursor.execute('''
                            INSERT OR REPLACE INTO episcanner_state_fits (state, metric, year, alpha, xmin, xmax, R, p)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (state, metric, 0, alpha, xmin, xmax, R, p))
                        print(f"Combined {metric} fit for {state}: alpha={alpha:.2f}")
        
        # 2. National/All Years Combined
        all_national_dfs = []
        for year in YEARS:
            if national_yearly_data[year]:
                all_national_dfs.extend(national_yearly_data[year])
        
        if all_national_dfs:
            br_combined_df = pd.concat(all_national_dfs)
            for metric in FIT_METRICS:
                results = fit_pl(br_combined_df[metric].values)
                if results:
                    alpha, xmin, xmax, R, p = results
                    cursor.execute('''
                        INSERT OR REPLACE INTO episcanner_fits (metric, year, alpha, xmin, xmax, R, p)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (metric, 0, alpha, xmin, xmax, R, p))
                    print(f"National combined {metric} fit (All Years): alpha={alpha:.2f}")
        
        conn.commit()

    print("\nFinished.")
    report_stale_ep_dur()


if __name__ == "__main__":
    import sys

    if "--purge-ep-dur" in sys.argv[1:]:
        purge_ep_dur()
    else:
        main()
        print(
            "\nReminder: only "
            f"{FIT_METRICS} are fitted. Duration and peak timing enter the "
            "analysis as covariates of size, never as scaling objects."
        )
