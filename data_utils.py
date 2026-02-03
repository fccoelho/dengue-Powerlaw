import pandas as pd
import sqlite3
import geopandas as gpd
import numpy as np
from functools import lru_cache
from fitpl import FitPL, fetch_infodengue as _fetch_infodengue, fetch_episcanner as _fetch_episcanner
from scipy import stats
import duckdb
import glob
import os

GEO_STATE_MAP = {
    11: "RO", 12: "AC", 13: "AM", 14: "RR", 15: "PA", 16: "AP", 17: "TO",
    21: "MA", 22: "PI", 23: "CE", 24: "RN", 25: "PB", 26: "PE", 27: "AL",
    28: "SE", 29: "BA", 31: "MG", 32: "ES", 33: "RJ", 35: "SP", 41: "PR",
    42: "SC", 43: "RS", 50: "MS", 51: "MT", 52: "GO", 53: "DF"
}

STATE_TO_GEO = {v: k for k, v in GEO_STATE_MAP.items()}

@lru_cache(maxsize=128)
def fetch_infodengue(geocode, **kwargs):
    return _fetch_infodengue(geocode, **kwargs)

@lru_cache(maxsize=128)
def fetch_episcanner(disease="dengue", state="RS", year=2024):
    return _fetch_episcanner(disease=disease, state=state, year=year)

def get_db_data(db_path="powerlaw_results.db"):
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query("SELECT * FROM powerlaw_fits", conn)
    return df

def get_city_stats(db_path="powerlaw_results.db"):
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='city_stats'")
        if cursor.fetchone():
            df = pd.read_sql_query("SELECT * FROM city_stats", conn)
            return df
    return pd.DataFrame()

@lru_cache(maxsize=1)
def get_alpha_trends(db_path="powerlaw_results.db"):
    with sqlite3.connect(db_path) as conn:
        df_yearly = pd.read_sql_query("SELECT geocode, year, alpha FROM powerlaw_fits_yearly", conn)
    
    df_yearly['geocode'] = pd.to_numeric(df_yearly['geocode'], errors='coerce')
    df_yearly['year'] = pd.to_numeric(df_yearly['year'], errors='coerce')
    df_yearly['alpha'] = pd.to_numeric(df_yearly['alpha'], errors='coerce')
    df_yearly = df_yearly.dropna(subset=['geocode', 'year', 'alpha'])
    
    trends = []
    for geocode, group in df_yearly.groupby('geocode'):
        group = group.dropna(subset=['alpha', 'year'])
        if len(group) >= 2:
            try:
                res = stats.linregress(group['year'], group['alpha'])
                trends.append({
                    'geocode': geocode, 
                    'alpha_trend': res.slope,
                    'p_value': res.pvalue
                })
            except:
                pass
    return pd.DataFrame(trends, columns=['geocode', 'alpha_trend', 'p_value'])

def get_yearly_db_data(geocode, db_path="powerlaw_results.db"):
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query("SELECT * FROM powerlaw_fits_yearly WHERE geocode=?", conn, params=(geocode,))
    return df

def get_yearly_state_data(region, db_path="powerlaw_results.db"):
    """Fetch all yearly fits for all cities in a region (state or BR)."""
    with sqlite3.connect(db_path) as conn:
        if region == "BR":
            return pd.read_sql_query("SELECT * FROM powerlaw_fits_yearly", conn)
        
        state_code = None
        for k, v in GEO_STATE_MAP.items():
            if v == region:
                state_code = k
                break
        
        if state_code:
            min_geo = state_code * 100000
            max_geo = (state_code + 1) * 100000
            return pd.read_sql_query("SELECT * FROM powerlaw_fits_yearly WHERE geocode >= ? AND geocode < ?", conn, params=(min_geo, max_geo))
            
    return pd.DataFrame()

def get_episcanner_fit_results(region, metric="total_cases", db_path="powerlaw_results.db"):
    """Fetch Episcanner fit results for a region (state or 'BR') for all years."""
    with sqlite3.connect(db_path) as conn:
        if region == "BR":
            df = pd.read_sql_query("SELECT * FROM episcanner_fits WHERE metric=? ORDER BY year ASC", conn, params=(metric,))
        else:
            df = pd.read_sql_query("SELECT * FROM episcanner_state_fits WHERE state=? AND metric=? ORDER BY year ASC", conn, params=(region, metric))
    
    if not df.empty:
        df['sort_year'] = df['year'].apply(lambda x: -1 if x == 0 else x)
        df = df.sort_values('sort_year').drop(columns=['sort_year'])
        
    return df

def ensure_episcanner_files(region, years=None):
    """Ensures that parquet files exist for the given region and years. Fetches if missing."""
    if years is None:
        years = list(range(2011, 2026))
        
    states = [region] if region != "BR" else list(GEO_STATE_MAP.values())
    
    missing_found = False
    for year in years:
        for state in states:
            file_path = f"data/episcanner_{state}_{year}.parquet"
            if not os.path.exists(file_path):
                fetch_episcanner(state=state, year=year)
                missing_found = True
    return missing_found

def get_duckdb_episcanner_data(region, years=None, columns="*"):
    """Aggregates Episcanner data using DuckDB."""
    if years is None:
        years = list(range(2011, 2026))
        
    try:
        con = duckdb.connect()
        min_y, max_y = min(years), max(years)
        
        if region == "BR":
            query = f"""
                SELECT {columns} FROM read_parquet('data/episcanner_*.parquet')
                WHERE year BETWEEN {min_y} AND {max_y}
            """
        else:
            if not glob.glob(f"data/episcanner_{region}_*.parquet"):
                return pd.DataFrame()
            query = f"""
                SELECT {columns} FROM read_parquet('data/episcanner_{region}_*.parquet')
                WHERE year BETWEEN {min_y} AND {max_y}
            """
            
        return con.query(query).df()
    except Exception as e:
        print(f"DuckDB query failed: {e}")
        return pd.DataFrame()

def extract_geocode(geocode_str):
    """Robustly extract geocode from strings like 'City (Geocode) - alpha: 1.23'"""
    if not geocode_str:
        return None
    try:
        start = geocode_str.find('(') + 1
        end = geocode_str.find(')')
        if start > 0 and end > start:
            return int(geocode_str[start:end])
    except:
        pass
    return None

@lru_cache(maxsize=1)
def _load_full_geography(gpkg_path="muni_br.gpkg"):
    if not os.path.exists(gpkg_path):
        return gpd.GeoDataFrame()
    gdf = gpd.read_file(gpkg_path)
    gdf['code_muni'] = gdf['code_muni'].astype(int)
    return gdf

def load_geography(gpkg_path="muni_br.gpkg", state_abbrev=None):
    gdf = _load_full_geography(gpkg_path)
    if gdf.empty:
        return gdf
    
    if state_abbrev:
        return gdf[gdf['abbrev_state'] == state_abbrev].copy()
        
    return gdf.copy()

@lru_cache(maxsize=32)
def get_merged_data(state_abbrev=None, include_trends=False):
    df_results = get_db_data()
    gdf = load_geography(state_abbrev=state_abbrev)
    
    if gdf.empty:
        return gpd.GeoDataFrame(), df_results

    merged = gdf.merge(df_results, left_on='code_muni', right_on='geocode', how='inner' if state_abbrev else 'right')
    
    df_stats = get_city_stats()
    if not df_stats.empty:
        merged = merged.merge(df_stats, left_on='code_muni', right_on='geocode', how='left', suffixes=('', '_stats_dup'))
        if 'geocode_stats_dup' in merged.columns:
            merged.drop(columns=['geocode_stats_dup'], inplace=True)

    if include_trends:
        df_trends = get_alpha_trends()
        merged = merged.merge(df_trends, left_on='code_muni', right_on='geocode', how='left', suffixes=('', '_trend_dup'))
        if 'geocode_trend_dup' in merged.columns:
            merged.drop(columns=['geocode_trend_dup'], inplace=True)
            
    return merged, df_results

@lru_cache(maxsize=100)
def get_combined_indicator_data(geocode):
    if geocode is None:
        return pd.DataFrame()
        
    df_yearly = get_yearly_db_data(geocode)
    if df_yearly.empty:
        return pd.DataFrame()
        
    state_prefix = int(str(geocode)[:2])
    state = GEO_STATE_MAP.get(state_prefix)
    if not state:
        return pd.DataFrame()
        
    try:
        con = duckdb.connect()
        pattern = f"data/episcanner_{state}_*.parquet"
        files = glob.glob(pattern)
        
        if not files:
            return pd.DataFrame()
            
        df_epi_combined = con.query(f"""
            SELECT * FROM read_parquet('{pattern}')
            WHERE geocode = {geocode}
        """).df()
        
        if df_epi_combined.empty:
            return pd.DataFrame()
            
        merged = df_yearly.merge(df_epi_combined, on='year', suffixes=('', '_epi'))
        return merged
        
    except Exception as e:
        print(f"DuckDB indicator fetch failed: {e}")
        return pd.DataFrame()

def get_city_details(geocode_str):
    geocode = extract_geocode(geocode_str)
    if geocode is None:
        return None
    
    with sqlite3.connect("powerlaw_results.db") as conn:
        df = pd.read_sql_query("SELECT * FROM powerlaw_fits WHERE geocode=?", conn, params=(geocode,))
    return df

def get_yearly_details(geocode_str):
    geocode = extract_geocode(geocode_str)
    if geocode is None:
        return None
    
    df = get_yearly_db_data(geocode)
    if df.empty:
        return pd.DataFrame(columns=["year", "alpha", "xmin", "xmax", "R", "p"])
    
    return df[["year", "alpha", "xmin", "xmax", "R", "p"]].sort_values(by='year', ascending=False)
