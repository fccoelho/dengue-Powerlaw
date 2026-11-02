import predictive_model
import pandas as pd
import numpy as np
import data_utils
import sqlite3

def debug_rs_city():
    print("--- Debugging RS (city) Step by Step ---")
    region = "RS"
    years = list(range(2010, 2026))
    
    # 1. Metrics
    df_metrics = data_utils.get_duckdb_episcanner_data(region, years)
    print(f"Metrics rows: {len(df_metrics)}")
    
    # 2. Fits
    with sqlite3.connect("powerlaw_results.db") as conn:
        df_fits = pd.read_sql_query("SELECT geocode, year, alpha, xmin, xmax FROM powerlaw_fits_yearly", conn)
    
    # Filter fits for RS
    df_fits = df_fits[(df_fits['geocode'] >= 4300000) & (df_fits['geocode'] < 4400000)]
    print(f"Fits rows for RS: {len(df_fits)}")
    
    # 3. Merge
    df_fits = df_fits.rename(columns={'alpha': 'alpha_size', 'xmin': 'xmin_size'})
    df_features = df_metrics.merge(df_fits, on=['geocode', 'year'], how='left')
    print(f"Merged features rows: {len(df_features)}")
    print(f"Rows with alpha_size: {df_features['alpha_size'].notnull().sum()}")
    
    # 4. Preparing model data (using the function logic)
    df_model = predictive_model.prepare_lagged_data(region, level="city")
    print(f"Final model rows: {len(df_model)}")
    if not df_model.empty:
        print(f"Year distribution:\n{df_model['year'].value_counts().sort_index()}")
        print(f"Rows with prev_alpha_trend: {df_model['prev_alpha_trend'].notnull().sum()}")
        print(f"Sample columns with null counts:\n{df_model.isnull().sum().to_dict()}")

if __name__ == "__main__":
    debug_rs_city()
