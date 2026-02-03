import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import pygeoda
import sqlite3
import data_utils
from data_utils import (
    load_geography, get_merged_data, get_city_stats, get_alpha_trends,
    get_yearly_db_data, get_yearly_state_data, get_episcanner_fit_results,
    ensure_episcanner_files, get_duckdb_episcanner_data, extract_geocode,
    fetch_infodengue, fetch_episcanner, get_combined_indicator_data
)

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
        return go.Figure().update_layout(title=f"No combined (year=0) fit found for {region}")
        
    alpha = year_0_fit.iloc[0]['alpha']
    xmin = year_0_fit.iloc[0]['xmin']
    
    # 3. Calculate CCDFs for each year
    fig = go.Figure()
    
    # Add empirical dots for each year
    colors = px.colors.qualitative.Plotly + px.colors.qualitative.Set2
    for i, year in enumerate(sorted(combined_df['year'].unique())):
        df_year = combined_df[combined_df['year'] == year]
        data = df_year[metric].values
        data = data[data > 0]
        if len(data) == 0: continue
        
        sorted_data = np.sort(data)
        n = len(sorted_data)
        y_empirical = np.arange(n, 0, -1) / n
        
        fig.add_trace(go.Scatter(
            x=sorted_data, y=y_empirical,
            mode='markers',
            marker=dict(size=4, opacity=0.5, color=colors[i % len(colors)]),
            name=str(year)
        ))
    
    # Add the combined fit line
    # Scale fit line: P(X>=x) = (x/x_min)**-(alpha-1) * P(X>=xmin)
    # We need to find P(X>=xmin) for the COMBINED data
    all_data = combined_df[metric].values
    all_data = all_data[all_data > 0]
    n_total = len(all_data)
    proportion_tail = np.sum(all_data >= xmin) / n_total if n_total > 0 else 1.0
    
    x_theoretical = np.logspace(np.log10(xmin), np.log10(max(all_data)), 100)
    y_theoretical = (x_theoretical / xmin) ** -(alpha - 1) * proportion_tail
    
    fig.add_trace(go.Scatter(
        x=x_theoretical, y=y_theoretical,
        mode='lines',
        line=dict(color='black', width=3, dash='dash'),
        name=f'Combined Fit (α={alpha:.2f})'
    ))
    
    fig.update_layout(
        title=f"CCDF for {metric_label} - {region} (All Years)",
        xaxis_type="log",
        yaxis_type="log",
        xaxis_title=metric_label,
        yaxis_title="P(X >= x)",
        template="plotly_white",
        height=600,
        showlegend=True
    )
    
    return fig

def plot_episcanner_state_map(metric="total_cases"):
    """Plots a choropleth map of Brazil using state-level combined alpha results (year=0)."""
    metric_label = "Epidemic Size's alpha" if metric == "total_cases" else "Epidemic Duration's alpha"
    
    with sqlite3.connect("powerlaw_results.db") as conn:
        df_states = pd.read_sql_query("SELECT state, alpha FROM episcanner_state_fits WHERE year=0 AND metric=?", conn, params=(metric,))
    
    if df_states.empty:
        return go.Figure().update_layout(title="No state data available")
        
    gdf_states = load_geography()[['abbrev_state', 'geometry']].dissolve(by='abbrev_state').reset_index()
    gdf_states = gdf_states.merge(df_states, left_on='abbrev_state', right_on='state', how='left')
    
    if gdf_states.crs is None or gdf_states.crs.to_epsg() != 4326:
        gdf_states = gdf_states.to_crs(epsg=4326)
        
    fig = px.choropleth_map(
        gdf_states,
        geojson=gdf_states.__geo_interface__,
        locations="abbrev_state",
        featureidkey="properties.abbrev_state",
        color="alpha",
        hover_name="abbrev_state",
        hover_data={"alpha": ":.2f", "abbrev_state": False},
        map_style="carto-positron",
        center={"lat": -15.793889, "lon": -47.882778},
        zoom=3,
        opacity=0.7,
        title=f"All-Time {metric_label} by State"
    )
    
    fig.update_layout(
        margin={"r":0,"t":40,"l":0,"b":0},
        width=800,
        height=600,
        autosize=True
    )
    return fig

def plot_episcanner_region_timeseries(region):
    """Plots the weekly estimated cases for a state or the entire country using DuckDB."""
    if not region:
        return None
        
    con = duckdb.connect()
    
    try:
        if region == "BR":
            files = glob.glob('data/[A-Z][A-Z].parquet')
            if not files:
                return go.Figure().update_layout(title="No state cache files found for Brazil aggregation")
                
            combined = con.query("""
                SELECT data_iniSE, SUM(casos_est) as casos_est 
                FROM read_parquet('data/[A-Z][A-Z].parquet') 
                GROUP BY data_iniSE 
                ORDER BY data_iniSE
            """).df()
            title = "Weekly Estimated Cases - Brazil (Total)"
            
        else:
            file_path = f"data/{region}.parquet"
            if not os.path.exists(file_path):
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
    
    years = list(range(2011, 2026))
    ensure_episcanner_files(region, years)
    combined_df = get_duckdb_episcanner_data(region, years, columns=f"year, {metric}")
    
    if combined_df.empty:
        return go.Figure().update_layout(title=f"No Episcanner data found for {region}")
        
    combined_raw = combined_df
    combined_raw = combined_raw[combined_raw[metric] > 0]

    fit_df = get_episcanner_fit_results(region, metric=metric)
    yearly_alphas = fit_df[fit_df['year'] > 0].sort_values('year')

    fig = make_subplots(specs=[[{"secondary_y": True}]])

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

def plot_state_residuals_vs_alpha(region):
    """
    Scatter plot of Alpha vs Sum of Residuals for all cities/years in the region.
    """
    if not region:
        return None
        
    df_alphas = get_yearly_state_data(region)
    if df_alphas.empty:
        return go.Figure().update_layout(title=f"No powerlaw fit data for {region}")
        
    ensure_episcanner_files(region)
    df_res = get_duckdb_episcanner_data(region, columns="geocode, year, sum_res")
    
    if df_res.empty:
        return go.Figure().update_layout(title=f"No residuals data found for {region}")
        
    merged = df_alphas.merge(df_res, on=['geocode', 'year'])
    
    if merged.empty:
        return go.Figure().update_layout(title=f"No overlapping data (alpha & residuals) for {region}")
        
    fig = px.scatter(
        merged,
        x="sum_res",
        y="alpha",
        title=f"Alpha vs Sum of Residuals - {region} (All Epidemics)",
        labels={"sum_res": "Sum of Residuals", "alpha": "Alpha Value"},
        hover_data=["city_name", "year"],
        opacity=0.6,
        trendline="ols" if len(merged) > 2 else None
    )
    
    if len(merged) > 2:
        fig = add_pvalue_to_trendline(fig, len(merged))
        
    fig.update_layout(template="plotly_white", height=400)
    return fig

def plot_map(state_abbrev=None):
    merged, _ = get_merged_data(state_abbrev=state_abbrev)
    
    if merged.empty or 'alpha' not in merged.columns or merged['alpha'].isna().all():
        return go.Figure().update_layout(title="No data available for map")

    if state_abbrev is None:
        state_data = merged.groupby('abbrev_state')['alpha'].mean().reset_index()
        gdf_states_geom = load_geography()[['abbrev_state', 'geometry']].dissolve(by='abbrev_state').reset_index()
        gdf_states = gdf_states_geom.merge(state_data, on='abbrev_state', how='left')
        
        if gdf_states.crs is None or gdf_states.crs.to_epsg() != 4326:
            gdf_states = gdf_states.to_crs(epsg=4326)
            
        fig = px.choropleth_map(
            gdf_states,
            geojson=gdf_states.__geo_interface__,
            locations="abbrev_state",
            featureidkey="properties.abbrev_state",
            color="alpha",
            hover_name="abbrev_state",
            hover_data={"alpha": ":.2f", "abbrev_state": False},
            map_style="carto-positron",
            center={"lat": -15.793889, "lon": -47.882778},
            zoom=3,
            opacity=0.7,
            title="Average Power Law Alpha by State"
        )
    else:
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
        return go.Figure().update_layout(title="No trend data available for map")

    if state_abbrev is None:
        state_data = merged.groupby('abbrev_state')['alpha_trend'].mean().reset_index()
        gdf_states_geom = load_geography()[['abbrev_state', 'geometry']].dissolve(by='abbrev_state').reset_index()
        gdf_states = gdf_states_geom.merge(state_data, on='abbrev_state', how='left')
        
        if gdf_states.crs is None or gdf_states.crs.to_epsg() != 4326:
            gdf_states = gdf_states.to_crs(epsg=4326)
            
        fig = px.choropleth_map(
            gdf_states,
            geojson=gdf_states.__geo_interface__,
            locations="abbrev_state",
            featureidkey="properties.abbrev_state",
            color="alpha_trend",
            hover_name="abbrev_state",
            hover_data={"alpha_trend": ":.4f", "abbrev_state": False},
            map_style="carto-positron",
            center={"lat": -15.793889, "lon": -47.882778},
            zoom=3,
            opacity=0.7,
            title="Average Alpha Trend (Slope) by State",
            color_continuous_scale="RdBu",
            color_continuous_midpoint=0
        )
    else:
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
            custom_data=["city_name", "geocode"],
            map_style="carto-positron",
            center={"lat": merged.geometry.centroid.y.mean(), "lon": merged.geometry.centroid.x.mean()},
            zoom=6,
            opacity=0.7,
            title=f"Alpha Trend (Slope) - {state_abbrev}",
            color_continuous_scale="RdBu",
            color_continuous_midpoint=0
        )
    
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

    df_clean = merged.dropna(subset=['alpha']).copy()
    if len(df_clean) < 3:
        return go.Figure().update_layout(title="Insufficient data for Local Moran clustering")

    try:
        gda = pygeoda.open(df_clean)
        w = pygeoda.queen_weights(gda)
        lisa = pygeoda.local_moran(w, gda['alpha'])
        clusters = lisa.lisa_clusters()
        labels = lisa.lisa_labels()
        colors = lisa.lisa_colors()
        
        df_clean['cluster_label'] = [labels[c] for c in clusters]
        color_map = {labels[i]: colors[i] for i in range(len(labels))}
        
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
    
    if merged.empty or 'alpha' not in merged.columns or merged['alpha'].isna().all():
        return go.Figure().update_layout(title="No alpha data for histogram")
        
    fig = px.histogram(
        merged, x="alpha", 
        nbins=30,
        title=f"Distribution of Alpha Parameters - {state_abbrev if state_abbrev else 'Brazil'}",
        labels={'alpha': 'Alpha Value'},
        marginal="box",
        color_discrete_sequence=['#636EFA']
    )
    fig.update_layout(template="plotly_white")
    return fig

def plot_alpha_vs_incidence(state_abbrev=None):
    merged, _ = get_merged_data(state_abbrev=state_abbrev)
    
    if merged.empty or 'alpha' not in merged.columns or 'incidence' not in merged.columns:
        return go.Figure().update_layout(title="Insufficient data for incidence plot")
        
    df_plot = merged.dropna(subset=['alpha', 'incidence'])
    if df_plot.empty:
        return go.Figure().update_layout(title="No overlapping data for incidence plot")

    fig = px.scatter(
        df_plot,
        x="incidence",
        y="alpha",
        hover_name="city_name",
        hover_data=["total_cases", "population", "alpha", "incidence"],
        trendline="ols",
        title=f"Alpha vs. Incidence - {state_abbrev if state_abbrev else 'Brazil'}",
        labels={"incidence": "Incidence (Cases/Population)", "alpha": "Alpha Value"}
    )
    
    fig = add_pvalue_to_trendline(fig, len(df_plot))
    fig.update_layout(template="plotly_white")
    return fig

def plot_fit(geocode_str):
    geocode = extract_geocode(geocode_str)
    if geocode is None:
        return None

    with sqlite3.connect("powerlaw_results.db") as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT alpha, xmin, xmax FROM powerlaw_fits WHERE geocode=?", (geocode,))
        row = cursor.fetchone()
        
    if not row:
        return None
        
    alpha, xmin, xmax = row
    df = fetch_infodengue(geocode)
    
    if df is None or df.empty:
        return None
        
    data = df.casos_est.values
    data = data[data > 0]
    sorted_data = np.sort(data)
    n = len(sorted_data)
    y_empirical = np.arange(n, 0, -1) / n
    
    if xmin:
        x_theoretical = np.linspace(xmin, max(sorted_data), 100)
        y_theoretical = (x_theoretical / xmin) ** -(alpha - 1)
        proportion_tail = np.sum(sorted_data >= xmin) / n
        y_theoretical = y_theoretical * proportion_tail
    
    fig, ax = plt.subplots(figsize=(10, 6))
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
    if data_len > 1:
        try:
            results = px.get_trendline_results(fig)
            if not results.empty:
                ols_res = results.px_fit_results.iloc[0]
                p_val = ols_res.pvalues[1]
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
        df_yearly, x="year", y="alpha", 
        title=f"Yearly Alpha Trend - {geocode_str}",
        labels={"year": "Year", "alpha": "Alpha Value"},
        trendline="ols" if len(df_yearly) >= 2 else None
    )
    
    if len(df_yearly) < 2:
        fig.add_annotation(
            text="Insufficient data for trend line (need >= 2 years)",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
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
    
    df = df.sort_values('data_iniSE')
    df.reset_index(inplace=True)
    
    fig = px.line(
        df, x='data_iniSE', y='casos_est', 
        title=f"Weekly Estimated Cases - {geocode_str}",
        labels={'data_iniSE': 'Date', 'casos_est': 'Estimated weekly cases'}
    )
    fig.update_layout(template="plotly_white")
    return fig

def plot_indicator_plots(geocode_str):
    geocode = extract_geocode(geocode_str)
    if geocode is None:
        return [None]*5
        
    df = get_combined_indicator_data(geocode)
    if df.empty:
        return [go.Figure().update_layout(title="No episcanner data available for correlation")]*5
        
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
