"""
Fit power laws o episcanner data
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
                    for metric in ['total_cases', 'ep_dur']:
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
                for metric in ['total_cases', 'ep_dur']:
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
                for metric in ['total_cases', 'ep_dur']:
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
            for metric in ['total_cases', 'ep_dur']:
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

if __name__ == "__main__":
    main()
