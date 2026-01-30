import time
import pandas as pd
import duckdb
import sqlite3
import os
from dashboard import fetch_infodengue, fetch_episcanner, get_yearly_db_data, GEO_STATE_MAP, get_combined_indicator_data
import plotly.graph_objects as go
from concurrent.futures import ThreadPoolExecutor

def benchmark_br_aggregation_current():
    start = time.time()
    def fetch_and_extract(state_uf):
        data = fetch_infodengue(state_uf)
        if data is not None:
            return data[['data_iniSE', 'casos_est']]
        return None

    with ThreadPoolExecutor(max_workers=10) as executor:
        all_dfs = list(executor.map(fetch_and_extract, GEO_STATE_MAP.values()))
    
    all_dfs = [df for df in all_dfs if df is not None]
    if all_dfs:
        combined = pd.concat(all_dfs).groupby('data_iniSE').sum(numeric_only=True).reset_index()
    end = time.time()
    return end - start

def benchmark_br_aggregation_duckdb():
    start = time.time()
    # Ensure state cache exists or simulate the search
    # We use data/[A-Z][A-Z].parquet for state files
    con = duckdb.connect()
    # Check if files exist
    import glob
    files = glob.glob('data/[A-Z][A-Z].parquet')
    if not files:
        return 0
    
    df = con.query("""
        SELECT data_iniSE, SUM(casos_est) as casos_est 
        FROM read_parquet('data/[A-Z][A-Z].parquet') 
        GROUP BY data_iniSE 
        ORDER BY data_iniSE
    """).df()
    end = time.time()
    return end - start

def benchmark_city_indicator_current(geocode):
    start = time.time()
    # Already defined in dashboard.py, but we want to measure it without cache
    # We'll just call the core logic if possible or clear cache
    get_combined_indicator_data.cache_clear()
    df = get_combined_indicator_data(geocode)
    end = time.time()
    return end - start

def benchmark_city_indicator_duckdb(geocode):
    start = time.time()
    # Get state and years first (fast from DB)
    df_yearly = get_yearly_db_data(geocode)
    state_prefix = int(str(geocode)[:2])
    state = GEO_STATE_MAP.get(state_prefix)
    
    con = duckdb.connect()
    # Query all years for this state/city at once
    df_epi_combined = con.query(f"""
        SELECT * FROM read_parquet('data/episcanner_{state}_*.parquet')
        WHERE geocode = {geocode}
    """).df()
    
    if not df_epi_combined.empty:
        merged = df_yearly.merge(df_epi_combined, on='year', suffixes=('', '_epi'))
    end = time.time()
    return end - start

if __name__ == "__main__":
    print("Benchmarking Brazil Aggregation (State Timeseries)...")
    t_curr = benchmark_br_aggregation_current()
    print(f"Current method: {t_curr:.4f}s")
    t_duck = benchmark_br_aggregation_duckdb()
    print(f"DuckDB method: {t_duck:.4f}s")
    if t_duck > 0:
        print(f"Speedup: {t_curr/t_duck:.2f}x")
    
    print("\nBenchmarking City Indicator Aggregation (All Years)...")
    # Use a city geocode known to have data
    sample_geocode = 4314902 # Porto Alegre
    t_curr = benchmark_city_indicator_current(sample_geocode)
    print(f"Current method: {t_curr:.4f}s")
    t_duck = benchmark_city_indicator_duckdb(sample_geocode)
    print(f"DuckDB method: {t_duck:.4f}s")
    if t_duck > 0:
        print(f"Speedup: {t_curr/t_duck:.2f}x")
