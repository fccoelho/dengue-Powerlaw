import gradio as gr
import pandas as pd
import sqlite3
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from functools import lru_cache
from fitpl import FitPL, fetch_infodengue as _fetch_infodengue, fetch_episcanner as _fetch_episcanner
from concurrent.futures import ThreadPoolExecutor
import asyncio
from scipy import stats
import pygeoda
import duckdb
import glob
import os

@lru_cache(maxsize=128)
def fetch_infodengue(geocode, **kwargs):
    return _fetch_infodengue(geocode, **kwargs)

@lru_cache(maxsize=128)
def fetch_episcanner(disease="dengue", state="RS", year=2024):
    return _fetch_episcanner(disease=disease, state=state, year=year)

GEO_STATE_MAP = {
    11: "RO", 12: "AC", 13: "AM", 14: "RR", 15: "PA", 16: "AP", 17: "TO",
    21: "MA", 22: "PI", 23: "CE", 24: "RN", 25: "PB", 26: "PE", 27: "AL",
    28: "SE", 29: "BA", 31: "MG", 32: "ES", 33: "RJ", 35: "SP", 41: "PR",
    42: "SC", 43: "RS", 50: "MS", 51: "MT", 52: "GO", 53: "DF"
}
import numpy as np

STATE_TO_GEO = {v: k for k, v in GEO_STATE_MAP.items()}


# Connect to DB and fetch results
def get_db_data(db_path="powerlaw_results.db"):
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query("SELECT * FROM powerlaw_fits", conn)
    return df

def get_city_stats(db_path="powerlaw_results.db"):
    with sqlite3.connect(db_path) as conn:
        # Check if table exists
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
    
    # Ensure numeric types (sometimes SQLite returns bytes or objects)
    df_yearly['geocode'] = pd.to_numeric(df_yearly['geocode'], errors='coerce')
    df_yearly['year'] = pd.to_numeric(df_yearly['year'], errors='coerce')
    df_yearly['alpha'] = pd.to_numeric(df_yearly['alpha'], errors='coerce')
    df_yearly = df_yearly.dropna(subset=['geocode', 'year', 'alpha'])
    
    trends = []
    for geocode, group in df_yearly.groupby('geocode'):
        group = group.dropna(subset=['alpha', 'year'])
        if len(group) >= 2:
            try:
                # Use linregress to get slope and p-value
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

def get_episcanner_fit_results(region, metric="total_cases", db_path="powerlaw_results.db"):
    """Fetch Episcanner fit results for a region (state or 'BR') for all years."""
    with sqlite3.connect(db_path) as conn:
        if region == "BR":
            df = pd.read_sql_query("SELECT * FROM episcanner_fits WHERE metric=? ORDER BY year ASC", conn, params=(metric,))
        else:
            df = pd.read_sql_query("SELECT * FROM episcanner_state_fits WHERE state=? AND metric=? ORDER BY year ASC", conn, params=(region, metric))
    
    # Sort with year=0 at the top
    if not df.empty:
        df['sort_year'] = df['year'].apply(lambda x: -1 if x == 0 else x)
        df = df.sort_values('sort_year').drop(columns=['sort_year'])
        
    return df
    return _fetch_episcanner(disease=disease, state=state, year=year)

def ensure_episcanner_files(region, years=None):
    """Ensures that parquet files exist for the given region and years. Fetches if missing."""
    if years is None:
        years = list(range(2011, 2026))
        
    states = [region] if region != "BR" else [
        "AC", "AL", "AP", "AM", "BA", "CE", "DF", "ES", "GO", "MA", 
        "MT", "MS", "MG", "PA", "PB", "PR", "PE", "PI", "RJ", "RN", 
        "RS", "RO", "RR", "SC", "SP", "SE", "TO"
    ]
    
    missing_found = False
    for year in years:
        for state in states:
            file_path = f"data/episcanner_{state}_{year}.parquet"
            if not os.path.exists(file_path):
                # print(f"Fetching missing data for {state} {year}")
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

def plot_combined_episcanner_fit(region, metric="total_cases"):
    """
    Plots the combined (all years) CCDF for a given metric with year-colored points 
    and the power law fit line.
    """
    if not region:
        return None
        
    metric_label = "Size (cases)" if metric == "total_cases" else "Duration (weeks)"
    
    # 1. Fetch data for all available years
    years = list(range(2011, 2026))
    
    # Ensure files exist
    ensure_episcanner_files(region, years)

    # Use DuckDB to aggregate
    combined_df = get_duckdb_episcanner_data(region, years)
        
    if combined_df.empty:
        return go.Figure().update_layout(title=f"No Episcanner data found for {region}")
    
    # 2. Fetch parameters from DB (year=0)
    fit_df = get_episcanner_fit_results(region, metric=metric)
    if fit_df.empty:
        return go.Figure().update_layout(title=f"No fitted parameters found in DB for {region} (Combined {metric_label})")
    
    # Filter year=0 row
    year_0_fit = fit_df[fit_df['year'] == 0]
    if year_0_fit.empty:
        return go.Figure().update_layout(title=f"No combined (Year 0) parameters found for {region}")
        
    alpha = year_0_fit.iloc[0]['alpha']
    xmin = year_0_fit.iloc[0]['xmin']
    
    # 3. Prepare CCDF data for Plotly (Year-Colored)
    combined_df = combined_df[combined_df[metric] > 0].sort_values(metric)
    n = len(combined_df)
    combined_df['ccdf'] = np.arange(n, 0, -1) / n
    
    # Plotly Scatter (Log-Log)
    fig = px.scatter(
        combined_df, 
        x=metric, 
        y='ccdf', 
        color='year',
        color_continuous_scale="Viridis",
        log_x=True, 
        log_y=True,
        hover_data=['year', 'muni_name'],
        title=f"Epidemic {metric_label} Power Law Fit (All Years Combined) - {region}",
        labels={metric: metric_label, 'ccdf': 'P(X >= x)', 'year': 'Year'}
    )
    
    # 4. Add Theoretical Line
    if xmin and alpha:
        x_max = combined_df[metric].max()
        x_theoretical = np.logspace(np.log10(xmin), np.log10(x_max), 100)
        
        # Scaling to match empirical CCDF at xmin
        proportion_tail = np.sum(combined_df[metric] >= xmin) / n
        y_theoretical = (x_theoretical / xmin) ** -(alpha - 1) * proportion_tail
        
        theoretical_df = pd.DataFrame({'x': x_theoretical, 'y': y_theoretical})
        
        # Get all fit parameters for hover from the year=0 row
        xmax = year_0_fit.iloc[0].get('xmax')
        R = year_0_fit.iloc[0].get('R')
        p = year_0_fit.iloc[0].get('p')
        
        hover_text = (
            f"alpha: {alpha:.4f}<br>"
            f"xmin: {xmin:.4f}<br>"
            f"xmax: {xmax if pd.notnull(xmax) else 'None'}<br>"
            f"R: {R:.4f}<br>"
            f"p: {p:.4g}"
        )
        
        fig.add_trace(go.Scatter(
            x=theoretical_df['x'], 
            y=theoretical_df['y'], 
            mode='lines',
            name=f'Fit (alpha={alpha:.2f})',
            line=dict(color='red', dash='dash', width=3),
            hovertemplate=hover_text + "<extra></extra>"
        ))
        
        # Add xmin vertical line
        fig.add_vline(x=xmin, line_dash="dot", line_color="gray", annotation_text=f"xmin={xmin:.1f}")

    fig.update_layout(
        template="plotly_white", 
        height=600,
        xaxis=dict(type='log', autorange=True),
        yaxis=dict(type='log', autorange=True)
    )
    return fig

def plot_episcanner_state_map(metric="total_cases"):
    """Plots a choropleth map of Brazil using state-level combined alpha results (year=0)."""
    # 1. Fetch data from DB
    metric_label = "Epidemic Size's alpha" if metric == "total_cases" else "Epidemic Duration's alpha"
    
    with sqlite3.connect("powerlaw_results.db") as conn:
        df_states = pd.read_sql_query("SELECT state, alpha FROM episcanner_state_fits WHERE year=0 AND metric=?", conn, params=(metric,))
    
    if df_states.empty:
        return go.Figure().update_layout(title=f"No state-level fit data found for {metric_label}")
    
    # 2. Load and dissolve geography to state level
    gdf = load_geography()
    gdf_states_geom = gdf[['abbrev_state', 'geometry']].dissolve(by='abbrev_state').reset_index()
    
    # 3. Merge
    merged = gdf_states_geom.merge(df_states, left_on='abbrev_state', right_on='state')
    
    # Ensure WGS84 for Plotly maps
    if merged.crs is None or merged.crs.to_epsg() != 4326:
        merged = merged.to_crs(epsg=4326)
    
    fig = px.choropleth_map(
        merged,
        geojson=merged.__geo_interface__,
        locations="abbrev_state",
        featureidkey="properties.abbrev_state",
        color="alpha",
        hover_name="abbrev_state",
        hover_data={"alpha": ":.4f", "abbrev_state": False},
        map_style="carto-positron",
        center={"lat": -15.793889, "lon": -47.882778},
        zoom=3,
        opacity=0.7,
        title=f"All-Time {metric_label} by State"
    )
    
    fig.update_layout(
        margin={"r":0,"t":40,"l":0,"b":0},
        height=600,
        autosize=True
    )
    return fig

def plot_episcanner_region_timeseries(region):
    """Plots the weekly estimated cases for a state or the entire country using DuckDB."""
    if not region:
        return None
        
    con = duckdb.connect()  # in-memory mostly
    
    try:
        if region == "BR":
            # Check if we have state files available to aggregate
            # Assuming state files are named like 'data/RJ.parquet', etc.
            # But the fetch_infodengue cache saves as geocode.parquet mostly.
            # State geocodes are 2 digits? Or fitpl saves them as {UF}.parquet?
            # Let's check fitpl.py again. fitpl saves as data/{geocode}.parquet.
            # If geocode is UF (str), it saves as data/{UF}.parquet?
            # fitpl.py: file_path = f"data/{geocode}.parquet"
            # If geocode is "RJ", it's data/RJ.parquet.
            
            # So we can look for data/[A-Z][A-Z].parquet for states
            files = glob.glob('data/[A-Z][A-Z].parquet')
            if not files:
                # Fallback to slow method if no cache files found
                return go.Figure().update_layout(title="No state cache files found for Brazil aggregation")
                
            combined = con.query("""
                SELECT data_iniSE, SUM(casos_est) as casos_est 
                FROM read_parquet('data/[A-Z][A-Z].parquet') 
                GROUP BY data_iniSE 
                ORDER BY data_iniSE
            """).df()
            title = "Weekly Estimated Cases - Brazil (Total)"
            
        else:
            # region is UF (e.g., 'RJ')
            # Check if file exists
            file_path = f"data/{region}.parquet"
            if not os.path.exists(file_path):
                 # Try fetching via API if not cached? 
                 # The original used fetch_infodengue which fetches and caches.
                 # If we use DuckDB we rely on cache.
                 # Let's use fetch_infodengue to ensure it's there, then read?
                 # Or just fallback to fetch_infodengue if file missing.
                 df = fetch_infodengue(region)
                 if df is None or df.empty:
                    return go.Figure().update_layout(title=f"No timeseries data found for {region}")
                 combined = df
            else:
                 combined = con.query(f"""
                    SELECT data_iniSE, casos_est 
                    FROM read_parquet('{file_path}')
                    ORDER BY data_iniSE
                """).df()
            title = f"Weekly Estimated Cases - {region}"

        if combined.empty:
             return go.Figure().update_layout(title=f"No data for {region}")

        fig = px.line(
            combined, 
            x='data_iniSE', 
            y='casos_est', 
            title=title,
            labels={'data_iniSE': 'Date', 'casos_est': 'Estimated weekly cases'}
        )
        fig.update_layout(template="plotly_white")
        return fig
        
    except Exception as e:
        print(f"DuckDB aggregation failed: {e}")
        return go.Figure().update_layout(title=f"Error plotting timeseries: {e}")

def plot_episcanner_dispersion_alpha(region, metric="total_cases"):
    """
    Plots a boxplot of a metric per year and overlays the alpha fit for each year.
    Uses a secondary Y-axis for alpha.
    """
    if not region:
        return None

    metric_label = "Size (cases)" if metric == "total_cases" else "Duration (weeks)"
    
    # 1. Fetch RAW data for all years to build the boxplot
    years = list(range(2011, 2026))
    
    # Ensure files exist
    ensure_episcanner_files(region, years)
    
    # Use DuckDB to aggregate
    # We only need specific columns: year, metric
    # But wait, we need to construct a dataframe with 'year' and 'metric' 
    combined_df = get_duckdb_episcanner_data(region, years, columns=f"year, {metric}")
    
    if combined_df.empty:
        return go.Figure().update_layout(title=f"No Episcanner data found for {region}")
        
    # combined_raw = combined_df
    combined_raw = combined_df
    # Filter > 0 for log plot
    combined_raw = combined_raw[combined_raw[metric] > 0]

    # 2. Fetch ALPHA values for all years from DB
    fit_df = get_episcanner_fit_results(region, metric=metric)
    # Filter out year=0 (combined) for the trend line
    yearly_alphas = fit_df[fit_df['year'] > 0].sort_values('year')

    # 3. Create the figure with secondary Y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add Boxplot
    for yr in sorted(combined_raw['year'].unique()):
        yr_data = combined_raw[combined_raw['year'] == yr]
        fig.add_trace(
            go.Box(
                y=yr_data[metric],
                name=str(yr),
                marker_color='lightblue',
                showlegend=False,
                boxpoints=False,
            ),
            secondary_y=False,
        )

    # Add Scatter/Line for alpha
    if not yearly_alphas.empty:
        fig.add_trace(
            go.Scatter(
                x=yearly_alphas['year'].astype(str), 
                y=yearly_alphas['alpha'],
                mode='lines+markers',
                name='Alpha Fit',
                line=dict(color='red', width=3),
                marker=dict(size=8)
            ),
            secondary_y=True,
        )

    fig.update_layout(
        title=f"{metric_label} Dispersion vs Alpha by Year - {region}",
        xaxis_title="Year",
        yaxis_title=f"{metric_label} (Linear Scale)",
        yaxis_type="linear",
        template="plotly_white",
        height=600,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    fig.update_yaxes(title_text="Alpha Value", secondary_y=True, range=[1, 4])

    return fig

def extract_geocode(geocode_str):
    """Robustly extract geocode from strings like 'City (Geocode) - alpha: 1.23'"""
    if not geocode_str:
        return None
    try:
        # Extract content inside parentheses
        start = geocode_str.find('(') + 1
        end = geocode_str.find(')')
        if start > 0 and end > start:
            return int(geocode_str[start:end])
    except:
        pass
    return None

# Load Geography
@lru_cache(maxsize=1)
def _load_full_geography(gpkg_path="muni_br.gpkg"):
    gdf = gpd.read_file(gpkg_path)
    # Ensure code_muni is integer for merging
    gdf['code_muni'] = gdf['code_muni'].astype(int)
    return gdf

def load_geography(gpkg_path="muni_br.gpkg", state_abbrev=None):
    gdf = _load_full_geography(gpkg_path)
    
    if state_abbrev:
        # Return a copy to avoid mutating the cached object
        return gdf[gdf['abbrev_state'] == state_abbrev].copy()
        
    return gdf.copy()

# Merge data
@lru_cache(maxsize=32)
def get_merged_data(state_abbrev=None, include_trends=False):
    df_results = get_db_data()
    gdf = load_geography(state_abbrev=state_abbrev)
    
    merged = gdf.merge(df_results, left_on='code_muni', right_on='geocode', how='inner' if state_abbrev else 'right')
    
    # Merge with city stats
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

# Plot Map
def plot_map(state_abbrev=None):
    merged, _ = get_merged_data(state_abbrev=state_abbrev)
    
    if merged.empty:
        return go.Figure().update_layout(title="No data for selected state")

    if state_abbrev is None:
        # State-level aggregation
        state_data = merged.groupby('abbrev_state').agg({
            'alpha': 'mean',
            'city_name': 'count' # Number of cities in the fit
        }).reset_index()
        state_data.rename(columns={'city_name': 'mun_count'}, inplace=True)
        
        # Dissolve geometries to state level (without aggregating non-numeric columns)
        gdf_states_geom = merged[['abbrev_state', 'geometry']].dissolve(by='abbrev_state').reset_index()
        # Merge with aggregated state data
        gdf_states = gdf_states_geom.merge(state_data, on='abbrev_state')
        
        # Ensure WGS84 for Plotly maps
        if gdf_states.crs is None or gdf_states.crs.to_epsg() != 4326:
            gdf_states = gdf_states.to_crs(epsg=4326)
            
        fig = px.choropleth_map(
            gdf_states,
            geojson=gdf_states.__geo_interface__,
            locations="abbrev_state",
            featureidkey="properties.abbrev_state",
            color="alpha",
            hover_name="abbrev_state",
            hover_data={"alpha": ":.4f", "mun_count": True, "abbrev_state": False} if "mun_count" in gdf_states.columns else ["alpha"],
            map_style="carto-positron",
            center={"lat": -15.793889, "lon": -47.882778},
            zoom=3,
            opacity=0.7,
            title="Average Power Law Alpha by State"
        )
    else:
        # Municipality-level view for selected state
        # Ensure WGS84 for Plotly maps
        if merged.crs is None or merged.crs.to_epsg() != 4326:
            merged = merged.to_crs(epsg=4326)
            
        fig = px.choropleth_map(
            merged,
            geojson=merged.__geo_interface__,
            locations="code_muni",
            featureidkey="properties.code_muni",
            color="alpha",
            hover_name="city_name",
            hover_data=["alpha", "xmin", "xmax"],
            custom_data=["city_name", "geocode"],
            map_style="carto-positron",
            center={"lat": merged.geometry.centroid.y.mean(), "lon": merged.geometry.centroid.x.mean()},
            zoom=6,
            opacity=0.7,
            title=f"Power Law Alpha Parameter - {state_abbrev}"
        )

    # Add dots for significant municipalities (p <= 0.05 and R > 0)
    significant = merged[(merged['p'] <= 0.05) & (merged['R'] > 0)].copy()
    if not significant.empty:
        fig.add_trace(go.Scattermap(
            lat=significant.geometry.centroid.y,
            lon=significant.geometry.centroid.x,
            mode='markers',
            marker=dict(size=6, color='gray', opacity=0.6),
            name='Significant (p<=0.05 & R>0)',
            hoverinfo='skip'
        ))

    fig.update_layout(
        margin={"r":0,"t":40,"l":0,"b":0},
        width=800,
        height=600,
        autosize=True
    )
    return fig

def plot_trend_map(state_abbrev=None):
    merged, _ = get_merged_data(state_abbrev=state_abbrev, include_trends=True)
    
    if merged.empty or 'alpha_trend' not in merged.columns or merged['alpha_trend'].isna().all():
        msg = "No trend data available" if merged.empty else "Insufficient data to calculate trends (need at least 2 years per city)"
        return go.Figure().update_layout(title=msg)

    if state_abbrev is None:
        # State-level aggregation of trends
        # Only aggregate cities that have a trend
        trend_cities = merged.dropna(subset=['alpha_trend'])
        if trend_cities.empty:
            return go.Figure().update_layout(title="No trend data available for any city")

        state_data = trend_cities.groupby('abbrev_state').agg({
            'alpha_trend': 'mean',
            'p_value': 'mean' 
        }).reset_index()
        
        gdf_states_geom = load_geography()[['abbrev_state', 'geometry']].dissolve(by='abbrev_state').reset_index()
        gdf_states = gdf_states_geom.merge(state_data, on='abbrev_state', how='left')
        
        # Ensure WGS84 for Plotly maps
        if gdf_states.crs is None or gdf_states.crs.to_epsg() != 4326:
            gdf_states = gdf_states.to_crs(epsg=4326)
            
        fig = px.choropleth_map(
            gdf_states,
            geojson=gdf_states.__geo_interface__,
            locations="abbrev_state",
            featureidkey="properties.abbrev_state",
            color="alpha_trend",
            hover_name="abbrev_state",
            hover_data={"alpha_trend": ":.4f", "p_value": ":.4f", "abbrev_state": False},
            color_continuous_scale="RdBu",
            color_continuous_midpoint=0,
            map_style="carto-positron",
            center={"lat": -15.793889, "lon": -47.882778},
            zoom=3,
            opacity=0.7,
            title="Avg Yearly Alpha Trend by State"
        )
    else:
        # Municipality-level view for selected state
        # Ensure WGS84 for Plotly maps
        if merged.crs is None or merged.crs.to_epsg() != 4326:
            merged = merged.to_crs(epsg=4326)
            
        fig = px.choropleth_map(
            merged,
            geojson=merged.__geo_interface__,
            locations="code_muni",
            featureidkey="properties.code_muni",
            color="alpha_trend",
            hover_name="city_name",
            hover_data=["alpha_trend", "p_value"],
            color_continuous_scale="RdBu",
            color_continuous_midpoint=0,
            map_style="carto-positron",
            center={"lat": merged.geometry.centroid.y.mean(), "lon": merged.geometry.centroid.x.mean()},
            zoom=6,
            opacity=0.7,
            title=f"Yearly Alpha Trend - {state_abbrev}"
        )
        
        # Add hatching for significant trends (p < 0.05)
        # Note: Plotly choropleth_map doesn't support a simple "hatching" argument like px.bar.
        # We can add a second layer with a different opacity or a pattern if we use go.Choroplethmapbox, 
        # but let's stick to adding a trace for significant cities.
        significant = merged[merged['p_value'] < 0.05].copy()
        if not significant.empty:
            # We can use go.Choroplethmap for a custom pattern if available in this plotly version,
            # or add a scatter layer with symbols. 
            # In many plotly versions, choropleth patterns are only for px.bar.
            # A common workaround is to overlay a semi-transparent layer or markers.
            # Let's try adding markers to indicate significance.
            fig.add_trace(go.Scattermap(
                lat=significant.geometry.centroid.y,
                lon=significant.geometry.centroid.x,
                mode='markers',
                marker=dict(size=8, color='black', symbol='circle', opacity=0.5),
                name='p < 0.05',
                hoverinfo='skip'
            ))

    fig.update_layout(
        margin={"r":0,"t":40,"l":0,"b":0},
        width=800,
        height=600,
        autosize=True
    )
    return fig

def plot_local_moran(state_abbrev=None):
    if state_abbrev is None:
        return go.Figure().update_layout(title="Select a state to see Local Moran clustering")
        
    merged, _ = get_merged_data(state_abbrev=state_abbrev)
    
    if merged.empty or 'alpha' not in merged.columns or merged['alpha'].isna().all():
        return go.Figure().update_layout(title="Insufficient data for Local Moran clustering")

    # Filter out NaNs for clustering
    df_clean = merged.dropna(subset=['alpha']).copy()
    if len(df_clean) < 3: # Need at least some neighbors
        return go.Figure().update_layout(title="Insufficient data for Local Moran clustering")

    try:
        # 1. Create Geoda object
        gda = pygeoda.open(df_clean)
        
        # 2. Create Weights
        w = pygeoda.queen_weights(gda)
        
        # 3. Local Moran
        lisa = pygeoda.local_moran(w, gda['alpha'])
        
        # 4. Get clusters
        clusters = lisa.lisa_clusters()
        labels = lisa.lisa_labels()
        colors = lisa.lisa_colors()
        
        # Map clusters to labels
        df_clean['cluster_label'] = [labels[c] for c in clusters]
        
        # Color mapping for Plotly
        color_map = {labels[i]: colors[i] for i in range(len(labels))}
        
        # Ensure WGS84 for Plotly maps
        if df_clean.crs is None or df_clean.crs.to_epsg() != 4326:
            df_clean = df_clean.to_crs(epsg=4326)
            
        fig = px.choropleth_map(
            df_clean,
            geojson=df_clean.__geo_interface__,
            locations="code_muni",
            featureidkey="properties.code_muni",
            color="cluster_label",
            color_discrete_map=color_map,
            hover_name="city_name",
            hover_data=["alpha"],
            map_style="carto-positron",
            center={"lat": df_clean.geometry.centroid.y.mean(), "lon": df_clean.geometry.centroid.x.mean()},
            zoom=6,
            opacity=0.7,
            title=f"Local Moran Clustering (Alpha) - {state_abbrev}"
        )
        
        fig.update_layout(
            margin={"r":0,"t":40,"l":0,"b":0},
            width=800,
            height=600,
            autosize=True,
            legend_title_text='Local Moran Cluster'
        )
        return fig
        
    except Exception as e:
        return go.Figure().update_layout(title=f"Error computing Local Moran: {str(e)}")

def plot_alpha_histogram(state_abbrev=None):
    merged, _ = get_merged_data(state_abbrev=state_abbrev)
    
    if merged.empty or "alpha" not in merged.columns:
        return go.Figure().update_layout(title="No data available for histogram")
    
    if state_abbrev is None:
        # Use state averages for the overall histogram
        plot_df = merged.groupby('abbrev_state')['alpha'].mean().reset_index()
        title = "Distribution of State-Average Alpha Values"
        label = "State Avg Alpha"
    else:
        plot_df = merged.dropna(subset=['alpha'])
        title = f"Alpha Distribution - {state_abbrev}"
        label = "Municipality Alpha"

    if plot_df.empty:
        return go.Figure().update_layout(title="No alpha values found")

    fig = px.histogram(
        plot_df, 
        x="alpha", 
        nbins=15, 
        title=title,
        labels={"alpha": label}
    )
    fig.update_layout(
        margin={"r":20,"t":40,"l":20,"b":40},
        height=400,
        template="plotly_white"
    )
    return fig

def plot_alpha_vs_incidence(state_abbrev=None):
    merged, _ = get_merged_data(state_abbrev=state_abbrev)
    
    if merged.empty or "alpha" not in merged.columns or "incidence" not in merged.columns:
        return go.Figure().update_layout(title="No data available for Alpha vs Incidence plot")
    
    plot_df = merged.dropna(subset=['alpha', 'incidence'])
    
    if plot_df.empty:
        return go.Figure().update_layout(title="No cities with both alpha and incidence found")

    fig = px.scatter(
        plot_df, 
        x="incidence", 
        y="alpha", 
        color="abbrev_state" if state_abbrev is None else None,
        hover_name="city_name",
        hover_data=["total_cases", "population", "alpha", "incidence"],
        title=f"Alpha vs Accumulated Cases/Population {'(Brazil)' if state_abbrev is None else f'({state_abbrev})'}",
        labels={"incidence": "Total Cases / Population * 10^5", "alpha": "Alpha Value"},
        trendline="ols" if len(plot_df) > 2 else None
    )
    
    if len(plot_df) > 2:
        fig = add_pvalue_to_trendline(fig, len(plot_df))
        
    fig.update_layout(
        margin={"r":20,"t":40,"l":20,"b":40},
        height=400,
        template="plotly_white"
    )
    return fig

# Plot Fit
def plot_fit(geocode_str):
    # extract geocode from string
    geocode = extract_geocode(geocode_str)
    if geocode is None:
        return None

    # Fetch parameters from DB
    with sqlite3.connect("powerlaw_results.db") as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT alpha, xmin, xmax FROM powerlaw_fits WHERE geocode=?", (geocode,))
        row = cursor.fetchone()
        
    if not row:
        return None
        
    alpha, xmin, xmax = row


    # Force usage of cached data ideally, or fetch if missing
    df = fetch_infodengue(geocode)
    
    if df is None or df.empty:
        return None
        
    data = df.casos_est.values
    data = data[data > 0]
    
    # Empirical CCDF
    # Sort data
    sorted_data = np.sort(data)
    # Calculate P(X >= x)
    # y values are len(data) down to 1, divided by len(data)
    n = len(sorted_data)
    y_empirical = np.arange(n, 0, -1) / n
    
    # Theoretical Power Law
    # Defined for x >= xmin
    # formula: P(X >= x) = (x / xmin) ** -(alpha - 1)
    # Generate x values for the theoretical line
    if xmin:
        x_theoretical = np.linspace(xmin, max(sorted_data), 100)
        y_theoretical = (x_theoretical / xmin) ** -(alpha - 1)
        # Adjust for the portion of data that is >= xmin? 
        # Usually powerlaw fits conditional P(X>=x | X>=xmin).
        # The empirical CCDF includes all data. The fit is usually plotted starting at xmin.
        # But we need to scale it to match the empirical CCDF at xmin.
        # P_empirical(X >= xmin) is approximately (number of obs >= xmin) / n
        proportion_tail = np.sum(sorted_data >= xmin) / n
        y_theoretical = y_theoretical * proportion_tail
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Log-Log Plot
    ax.loglog(sorted_data, y_empirical, marker='.', linestyle='none', color='blue', alpha=0.5, label='Empirical CCDF')
    
    if xmin:
        ax.loglog(x_theoretical, y_theoretical, color='red', linestyle='--', linewidth=2, label=f'Power Law Fit (alpha={alpha:.2f})')
        ax.axvline(xmin, color='gray', linestyle=':', label=f'xmin={xmin}')
    
    ax.set_ylabel("P(X >= x)")
    ax.set_xlabel("Weekly reported cases (X)")
    ax.legend()
    ax.set_title(f"Power Law Fit for {geocode_str}")
    ax.grid(True, which="both", ls="-", alpha=0.2)
    
    return fig

def add_pvalue_to_trendline(fig, data_len):
    """Extract p-value from OLS trendline and add it to hover text."""
    if data_len > 1:
        try:
            results = px.get_trendline_results(fig)
            if not results.empty:
                ols_res = results.px_fit_results.iloc[0]
                p_val = ols_res.pvalues[1] # p-value for slope
                
                # The trendline is typically the second trace (index 1)
                if len(fig.data) > 1:
                    fig.data[1].hovertemplate += f"<br>p-value: {p_val:.4f}"
        except:
            pass
    return fig

def plot_yearly_trend(geocode_str):
    geocode = extract_geocode(geocode_str)
    if geocode is None:
        return None
        
    df_yearly = get_yearly_db_data(geocode)
    
    if df_yearly.empty:
        return go.Figure().update_layout(title=f"No yearly trend data for {geocode_str}")
        
    fig = px.scatter(
        df_yearly, 
        x="year", 
        y="alpha", 
        title=f"Yearly Alpha Trend - {geocode_str}",
        labels={"year": "Year", "alpha": "Alpha Value"},
        trendline="ols" if len(df_yearly) >= 2 else None
    )
    
    if len(df_yearly) < 2:
        fig.add_annotation(
            text="Insufficient data for trend line (need >= 2 years)",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="red")
        )
    
    fig = add_pvalue_to_trendline(fig, len(df_yearly))
    fig.update_layout(template="plotly_white")
    return fig

def plot_timeseries(geocode_str):
    geocode = extract_geocode(geocode_str)
    if geocode is None:
        return None
    
    df = fetch_infodengue(geocode)
    if df is None or df.empty:
        return go.Figure().update_layout(title=f"No timeseries data for {geocode_str}")
    
    # Sort by date
    df = df.sort_values('data_iniSE')
    df.reset_index(inplace=True)
    
    fig = px.line(
        df, 
        x='data_iniSE', 
        y='casos_est', 
        title=f"Weekly Estimated Cases - {geocode_str}",
        labels={'data_iniSE': 'Date', 'casos_est': 'Estimated weekly cases'}
    )
    fig.update_layout(template="plotly_white")
    return fig

@lru_cache(maxsize=100)
def get_combined_indicator_data(geocode):
    if geocode is None:
        return pd.DataFrame()
        
    # Get yearly alphas
    df_yearly = get_yearly_db_data(geocode)
    if df_yearly.empty:
        return pd.DataFrame()
        
    # Get state from geocode
    state_prefix = int(str(geocode)[:2])
    state = GEO_STATE_MAP.get(state_prefix)
    if not state:
        return pd.DataFrame()
        
    # Use DuckDB to read all yearly episcanner files for this state
    try:
        con = duckdb.connect()
        # Pattern match all years for this state: data/episcanner_{state}_*.parquet
        pattern = f"data/episcanner_{state}_*.parquet"
        files = glob.glob(pattern)
        
        if not files:
            return pd.DataFrame()
            
        # Select only this geocode
        df_epi_combined = con.query(f"""
            SELECT * FROM read_parquet('{pattern}')
            WHERE geocode = {geocode}
        """).df()
        
        if df_epi_combined.empty:
            return pd.DataFrame()
            
        # Merge on year
        # df_epi_combined should have 'year' column from the parquet usually?
        # fitpl.py saves it with 'year' column.
        merged = df_yearly.merge(df_epi_combined, on='year', suffixes=('', '_epi'))
        return merged
        
    except Exception as e:
        print(f"DuckDB indicator fetch failed: {e}")
        return pd.DataFrame()

def plot_indicator_plots(geocode_str):
    geocode = extract_geocode(geocode_str)
    if geocode is None:
        return [None]*5
        
    df = get_combined_indicator_data(geocode)
    
    if df.empty:
        return [go.Figure().update_layout(title="No episcanner data available for correlation")]*5
        
    # Alpha must be on Y axis as per user request
    plots = []
    indicators = [
        ('R0', 'R0 Value'),
        ('total_cases', 'Total Cases'),
        ('ep_ini', 'Epidemic Start (Week)'),
        ('ep_dur', 'Epidemic Duration (Weeks)'),
        ('sum_res', 'Sum of Residuals')
    ]
    
    for col, label in indicators:
        if col in df.columns:
            fig = px.scatter(
                df, x=col, y='alpha', 
                title=f"Alpha vs {label}",
                labels={col: label, 'alpha': 'Alpha Value'},
                hover_data=['year'],
                trendline="ols" if len(df) > 1 else None
            )
            fig = add_pvalue_to_trendline(fig, len(df))
            fig.update_layout(template="plotly_white")
            plots.append(fig)
        else:
            plots.append(go.Figure().update_layout(title=f"Column {col} missing"))
            
    return plots

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
    
    # Sort by year descending for the table
    return df[["year", "alpha", "xmin", "xmax", "R", "p"]].sort_values(by='year', ascending=False)

def on_map_select(evt: gr.SelectData):
    # Plotly's custom_data is passed through evt.value.
    # It's usually in a 'customdata' key within the point info.
    # The structure depends on Gradio version, but evt.value['point_number'] is common.
    # For index-based lookup:
    if isinstance(evt.index, (list, tuple)):
        # For choropleth, index might be [curve_num, point_num]
        return None # Fallback or handle specific index
    
    # Let's try to get it from customdata if available in value
    try:
        # custom_data was set to ["city_name", "geocode"]
        # In Gradio select events for Plotly, the value is the point dict
        data = evt.value
        city_name = data.get("customdata", [None, None])[0]
        geocode = data.get("customdata", [None, None])[1]
        if city_name and geocode:
            return f"{city_name} ({geocode})"
    except:
        pass
    return None

# Dashboard UI
def create_dashboard():
    # Initial data for setup
    with sqlite3.connect("powerlaw_results.db") as conn:
        # Check if table exists before querying
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='powerlaw_fits'")
        if cursor.fetchone():
            df_results = pd.read_sql_query("SELECT city_name, geocode FROM powerlaw_fits", conn)
        else:
            df_results = pd.DataFrame(columns=["city_name", "geocode"])
    
    gdf_all = load_geography()
    states = sorted(gdf_all['abbrev_state'].unique().tolist())
    
    def get_city_options(state_abbrev):
        merged, _ = get_merged_data(state_abbrev=state_abbrev)
        if merged.empty:
            return [], None
        
        # Format choices to include alpha
        choices = []
        for _, row in merged.iterrows():
            alpha_str = f" - alpha: {row['alpha']:.2f}" if pd.notnull(row['alpha']) else ""
            choices.append(f"{row['city_name']} ({row['geocode']}){alpha_str}")
        
        choices = sorted(choices)
        
        # Find city with highest alpha for default value
        if 'alpha' in merged.columns and not merged['alpha'].dropna().empty:
            best_id = merged['alpha'].idxmax()
            best_city = merged.loc[best_id]
            best_alpha = best_city['alpha']
            best_value = f"{best_city['city_name']} ({best_city['geocode']}) - alpha: {best_alpha:.2f}"
        else:
            best_value = choices[0] if choices else None
            
        return choices, best_value

    def update_city_dropdown(state_abbrev):
        choices, best_value = get_city_options(state_abbrev)
        return gr.Dropdown(choices=choices, value=best_value, filterable=True)

    with gr.Blocks(title="Power Law Dashboard") as demo:
        gr.Markdown("# Power Law Estimation Dashboard - Dengue Cases")
        
        with gr.Tabs():
            with gr.TabItem("General Overview"):
                with gr.Row():
                    with gr.Column(scale=1):
                        state_dropdown_overview = gr.Dropdown(choices=states, label="Select State", filterable=True)
                        alpha_hist = gr.Plot(label="Alpha Distribution", min_width=300)
                        load_map_btn = gr.Button("Refresh Overview")
                    with gr.Column(scale=4):
                        map_plot = gr.Plot(label="Alpha Parameter Map")
                        alpha_incidence_plot = gr.Plot(label="Alpha vs Incidence (accumulated)")
                        trend_map_plot = gr.Plot(label="Alpha Trend Map")
                        moran_plot = gr.Plot(label="Local Moran Clustering Map")
                

            with gr.TabItem("City Details & Yearly Trends"):
                with gr.Row():
                    with gr.Column(scale=1):
                        state_dropdown_city = gr.Dropdown(choices=states, label="Filter by State", filterable=True)
                        city_dropdown = gr.Dropdown(choices=[], label="Select City", filterable=True)
                        plot_btn = gr.Button("Refresh City Data")
                        city_details_table = gr.Dataframe(label="General Stats", interactive=False)
                        yearly_fits_table = gr.Dataframe(label="Yearly Fit Parameters", interactive=False)
                    
                    with gr.Column(scale=2):
                        fit_plot = gr.Plot(label="Power Law Fit (Overall)")
                        timeseries_plot = gr.Plot(label="Timeseries of Cases")
                    
                    with gr.Column(scale=2):
                        trend_plot = gr.Plot(label="Yearly Alpha Trend")
                    
                with gr.Row():
                    r0_plot = gr.Plot(label="Alpha vs R0")
                    cases_plot = gr.Plot(label="Alpha vs Total Cases")
                
                with gr.Row():
                    ini_plot = gr.Plot(label="Alpha vs Ep Start")
                    dur_plot = gr.Plot(label="Alpha vs Ep Duration")
                
                with gr.Row():
                    res_plot = gr.Plot(label="Alpha vs Sum of Residuals")
                

            with gr.TabItem("Epidemic Metrics Power Laws"):
                with gr.Row():
                    with gr.Column(scale=1):
                        epi_region_dropdown = gr.Dropdown(
                            choices=["BR"] + states, 
                            value="BR", 
                            label="Select Region (State or BR)", 
                            filterable=True
                        )
                        epi_refresh_btn = gr.Button("Refresh Episcanner Plots")
                        with gr.Row():
                            epi_details_table_size = gr.Dataframe(label="Fit Details (Size)", interactive=False)
                            epi_details_table_dur = gr.Dataframe(label="Fit Details (Duration)", interactive=False)
                    
                    with gr.Column(scale=4):
                        with gr.Row():
                            epi_combined_plot_size = gr.Plot(label="Size: Combined Fit")
                            epi_combined_plot_dur = gr.Plot(label="Duration: Combined Fit")
                        
                        with gr.Row():
                            epi_dispersion_plot_size = gr.Plot(label="Size: Dispersion vs Alpha")
                            epi_dispersion_plot_dur = gr.Plot(label="Duration: Dispersion vs Alpha")
                        
                        with gr.Row():
                            epi_state_map_size = gr.Plot(label="Size: State Alpha Map")
                            epi_state_map_dur = gr.Plot(label="Duration: State Alpha Map")
                        
                        epi_timeseries_plot = gr.Plot(label="Regional Cases Timeseries")

        # Dashboard Logic / Event Handlers
        def sync_state(state):
            # If state is "BR", treat it as None for the other dropdowns that don't have "BR"
            base_state = None if state == "BR" else state
            
            map_fig = plot_map(base_state)
            trend_map_fig = plot_trend_map(base_state)
            moran_fig = plot_local_moran(base_state)
            hist_fig = plot_alpha_histogram(base_state)
            incidence_fig = plot_alpha_vs_incidence(base_state)
            city_choices, best_city = get_city_options(base_state)
            
            # Sync all three dropdowns
            return (
                base_state, # state_dropdown_city
                state if state in ["BR"] + states else base_state, # epi_region_dropdown (can be "BR")
                map_fig, 
                incidence_fig,
                trend_map_fig, 
                moran_fig,
                hist_fig, 
                gr.Dropdown(choices=city_choices, value=best_city)
            )

        state_dropdown_overview.change(
            fn=sync_state, 
            inputs=[state_dropdown_overview], 
            outputs=[state_dropdown_city, epi_region_dropdown, map_plot, alpha_incidence_plot, trend_map_plot, moran_plot, alpha_hist, city_dropdown]
        )
        
        state_dropdown_city.change(
            fn=sync_state, 
            inputs=[state_dropdown_city], 
            outputs=[state_dropdown_overview, epi_region_dropdown, map_plot, alpha_incidence_plot, trend_map_plot, moran_plot, alpha_hist, city_dropdown]
        )

        epi_region_dropdown.change(
            fn=sync_state,
            inputs=[epi_region_dropdown],
            outputs=[state_dropdown_overview, state_dropdown_city, map_plot, alpha_incidence_plot, trend_map_plot, moran_plot, alpha_hist, city_dropdown]
        )

        # Separate handler for episcanner-specific plots when region changes
        def update_episcanner(region):
            with ThreadPoolExecutor(max_workers=10) as executor:
                f1 = executor.submit(plot_combined_episcanner_fit, region, "total_cases")
                f2 = executor.submit(plot_combined_episcanner_fit, region, "ep_dur")
                f3 = executor.submit(plot_episcanner_dispersion_alpha, region, "total_cases")
                f4 = executor.submit(plot_episcanner_dispersion_alpha, region, "ep_dur")
                f5 = executor.submit(plot_episcanner_state_map, "total_cases")
                f6 = executor.submit(plot_episcanner_state_map, "ep_dur")
                f7 = executor.submit(get_episcanner_fit_results, region, "total_cases")
                f8 = executor.submit(get_episcanner_fit_results, region, "ep_dur")
                f9 = executor.submit(plot_episcanner_region_timeseries, region)

                def get_res(f, default=None):
                    try:
                        return f.result()
                    except Exception as e:
                        print(f"Task failed: {e}")
                        return default or go.Figure()

                return (
                    get_res(f1), get_res(f2), get_res(f3), get_res(f4), 
                    get_res(f5), get_res(f6), get_res(f7, pd.DataFrame()), 
                    get_res(f8, pd.DataFrame()), get_res(f9)
                )

        epi_region_dropdown.change(
            fn=update_episcanner, 
            inputs=[epi_region_dropdown], 
            outputs=[
                epi_combined_plot_size, epi_combined_plot_dur, 
                epi_dispersion_plot_size, epi_dispersion_plot_dur, 
                epi_state_map_size, epi_state_map_dur,
                epi_details_table_size, epi_details_table_dur, 
                epi_timeseries_plot
            ]
        )

        city_dropdown.change(fn=plot_fit, inputs=[city_dropdown], outputs=fit_plot)
        city_dropdown.change(fn=plot_yearly_trend, inputs=[city_dropdown], outputs=trend_plot)
        city_dropdown.change(fn=plot_timeseries, inputs=[city_dropdown], outputs=timeseries_plot)
        city_dropdown.change(fn=get_city_details, inputs=[city_dropdown], outputs=city_details_table)
        city_dropdown.change(fn=get_yearly_details, inputs=[city_dropdown], outputs=yearly_fits_table)
        city_dropdown.change(fn=plot_indicator_plots, inputs=[city_dropdown], outputs=[r0_plot, cases_plot, ini_plot, dur_plot, res_plot])
        epi_refresh_btn.click(
            fn=update_episcanner, 
            inputs=[epi_region_dropdown], 
            outputs=[
                epi_combined_plot_size, epi_combined_plot_dur, 
                epi_dispersion_plot_size, epi_dispersion_plot_dur, 
                epi_state_map_size, epi_state_map_dur,
                epi_details_table_size, epi_details_table_dur, 
                epi_timeseries_plot
            ]
        )

        plot_btn.click(fn=plot_fit, inputs=[city_dropdown], outputs=fit_plot)
        plot_btn.click(fn=plot_yearly_trend, inputs=[city_dropdown], outputs=trend_plot)
        plot_btn.click(fn=plot_timeseries, inputs=[city_dropdown], outputs=timeseries_plot)
        plot_btn.click(fn=get_city_details, inputs=[city_dropdown], outputs=city_details_table)
        plot_btn.click(fn=get_yearly_details, inputs=[city_dropdown], outputs=yearly_fits_table)
        plot_btn.click(fn=plot_indicator_plots, inputs=[city_dropdown], outputs=[r0_plot, cases_plot, ini_plot, dur_plot, res_plot])

        def initial_load():
            with ThreadPoolExecutor(max_workers=10) as executor:
                # 1. Overview data
                f_map = executor.submit(plot_map, None)
                f_tmap = executor.submit(plot_trend_map, None)
                f_moran = executor.submit(plot_local_moran, None)
                f_hist = executor.submit(plot_alpha_histogram, None)
                f_incidence = executor.submit(plot_alpha_vs_incidence, None)
                
                # 2. Get best city
                choices, best_value = get_city_options(None)
                
                # 3. City details data
                f_fit = executor.submit(plot_fit, best_value)
                f_ytrend = executor.submit(plot_yearly_trend, best_value)
                f_ts = executor.submit(plot_timeseries, best_value)
                f_details = executor.submit(get_city_details, best_value)
                f_yearly = executor.submit(get_yearly_details, best_value)
                f_indicator = executor.submit(plot_indicator_plots, best_value)
                
                # 4. Episcanner data (Size)
                f_epi_size = executor.submit(plot_combined_episcanner_fit, "BR", "total_cases")
                f_edisp_size = executor.submit(plot_episcanner_dispersion_alpha, "BR", "total_cases")
                f_edetails_size = executor.submit(get_episcanner_fit_results, "BR", "total_cases")
                f_emap_size = executor.submit(plot_episcanner_state_map, "total_cases")
                
                # 5. Episcanner data (Duration)
                f_epi_dur = executor.submit(plot_combined_episcanner_fit, "BR", "ep_dur")
                f_edisp_dur = executor.submit(plot_episcanner_dispersion_alpha, "BR", "ep_dur")
                f_edetails_dur = executor.submit(get_episcanner_fit_results, "BR", "ep_dur")
                f_emap_dur = executor.submit(plot_episcanner_state_map, "ep_dur")
                
                f_ets = executor.submit(plot_episcanner_region_timeseries, "BR")

                def safe_result(future, default=None):
                    try:
                        return future.result()
                    except Exception as e:
                        print(f"Task failed: {e}")
                        return default or go.Figure().update_layout(title=f"Error loading plot: {e}")

                map_fig = safe_result(f_map)
                trend_map_fig = safe_result(f_tmap)
                moran_fig = safe_result(f_moran)
                hist_fig = safe_result(f_hist)
                incidence_fig = safe_result(f_incidence)
                city_upd = gr.Dropdown(choices=choices, value=best_value, filterable=True)
                fit_fig = safe_result(f_fit)
                trend_fig = safe_result(f_ytrend)
                ts_fig = safe_result(f_ts)
                details = safe_result(f_details, pd.DataFrame())
                yearly = safe_result(f_yearly, pd.DataFrame())
                indicator_plots = safe_result(f_indicator, [go.Figure()]*5)
                
                epi_fig_size = safe_result(f_epi_size)
                epi_dur_fig = safe_result(f_epi_dur)
                epi_disp_fig_size = safe_result(f_edisp_size)
                epi_disp_fig_dur = safe_result(f_edisp_dur)
                epi_details_size = safe_result(f_edetails_size, pd.DataFrame())
                epi_details_dur = safe_result(f_edetails_dur, pd.DataFrame())
                epi_map_size = safe_result(f_emap_size)
                epi_map_dur = safe_result(f_emap_dur)
                epi_ts = safe_result(f_ets)

            # Determine state of the best city
            best_geocode = extract_geocode(best_value)
            state_val = None
            if best_geocode:
                state_prefix = int(str(best_geocode)[:2])
                state_val = GEO_STATE_MAP.get(state_prefix)
            
            if not state_val and states:
                state_val = states[0]
            
            return [state_val, state_val, state_val, map_fig, incidence_fig, trend_map_fig, moran_fig, hist_fig, city_upd, 
            fit_fig, trend_fig, ts_fig, details, yearly] + indicator_plots + [epi_fig_size, epi_dur_fig, epi_disp_fig_size, 
            epi_disp_fig_dur, epi_details_size, epi_details_dur, epi_map_size, epi_map_dur, epi_ts]

        # Trigger initial load (All Brazil)
        demo.load(fn=initial_load, inputs=[], outputs=[
            state_dropdown_overview, state_dropdown_city, epi_region_dropdown,
            map_plot, alpha_incidence_plot, trend_map_plot, moran_plot, alpha_hist, city_dropdown, fit_plot, trend_plot, timeseries_plot, 
            city_details_table, yearly_fits_table,
            r0_plot, cases_plot, ini_plot, dur_plot, res_plot,
            epi_combined_plot_size, epi_combined_plot_dur, 
            epi_dispersion_plot_size, epi_dispersion_plot_dur, 
            epi_details_table_size, epi_details_table_dur, 
            epi_state_map_size, epi_state_map_dur, 
            epi_timeseries_plot
        ])

    return demo

if __name__ == "__main__":
    demo = create_dashboard()
    import os
    import gradio.networking
    gradio.networking.url_ok = lambda x: True
    os.environ["NO_PROXY"] = "localhost,127.0.0.1"
    demo.launch(
        debug=True,
        server_name="127.0.0.1",
        server_port=7861
    )
