import gradio as gr
import pandas as pd
import sqlite3
import geopandas as gpd
import plotly.express as px
import matplotlib.pyplot as plt
from functools import lru_cache
from fitpl import FitPL, fetch_infodengue, fetch_episcanner

GEO_STATE_MAP = {
    11: "RO", 12: "AC", 13: "AM", 14: "RR", 15: "PA", 16: "AP", 17: "TO",
    21: "MA", 22: "PI", 23: "CE", 24: "RN", 25: "PB", 26: "PE", 27: "AL",
    28: "SE", 29: "BA", 31: "MG", 32: "ES", 33: "RJ", 35: "SP", 41: "PR",
    42: "SC", 43: "RS", 50: "MS", 51: "MT", 52: "GO", 53: "DF"
}
import numpy as np


# Connect to DB and fetch results
def get_db_data(db_path="powerlaw_results.db"):
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query("SELECT * FROM powerlaw_fits", conn)
    return df

@lru_cache(maxsize=1)
def get_alpha_trends(db_path="powerlaw_results.db"):
    with sqlite3.connect(db_path) as conn:
        df_yearly = pd.read_sql_query("SELECT geocode, year, alpha FROM powerlaw_fits_yearly", conn)
    
    trends = []
    for geocode, group in df_yearly.groupby('geocode'):
        if len(group) >= 2:
            # OLS slope
            try:
                # Remove NaNs
                group = group.dropna(subset=['alpha', 'year'])
                if len(group) >= 2:
                    slope, _ = np.polyfit(group['year'], group['alpha'], 1)
                    trends.append({'geocode': geocode, 'alpha_trend': slope})
            except:
                pass
    return pd.DataFrame(trends)

def get_yearly_db_data(geocode, db_path="powerlaw_results.db"):
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query("SELECT * FROM powerlaw_fits_yearly WHERE geocode=?", conn, params=(geocode,))
    return df

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
        fig = px.scatter_map(lat=[-15.79], lon=[-47.88], zoom=3, map_style="carto-positron")
        fig.update_layout(title="No data for selected state")
        return fig

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
        
        fig = px.choropleth_map(
            gdf_states,
            geojson=gdf_states.geometry,
            locations=gdf_states.index,
            color="alpha",
            hover_name="abbrev_state",
            hover_data={"alpha": ":.4f", "mun_count": True} if "mun_count" in gdf_states.columns else ["alpha"],
            map_style="carto-positron",
            center={"lat": -15.793889, "lon": -47.882778},
            zoom=3,
            opacity=0.7,
            title="Average Power Law Alpha by State"
        )
    else:
        # Municipality-level view for selected state
        fig = px.choropleth_map(
            merged,
            geojson=merged.geometry,
            locations=merged.index,
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

    fig.update_layout(
        margin={"r":0,"t":40,"l":0,"b":0},
        width=800,
        height=600,
        autosize=True
    )
    return fig

def plot_trend_map(state_abbrev=None):
    merged, _ = get_merged_data(state_abbrev=state_abbrev, include_trends=True)
    
    if merged.empty or 'alpha_trend' not in merged.columns:
        fig = px.scatter_map(lat=[-15.79], lon=[-47.88], zoom=3, map_style="carto-positron")
        fig.update_layout(title="No trend data available")
        return fig

    if state_abbrev is None:
        # State-level aggregation of trends
        state_data = merged.groupby('abbrev_state').agg({
            'alpha_trend': 'mean'
        }).reset_index()
        
        gdf_states_geom = merged[['abbrev_state', 'geometry']].dissolve(by='abbrev_state').reset_index()
        gdf_states = gdf_states_geom.merge(state_data, on='abbrev_state')
        
        fig = px.choropleth_map(
            gdf_states,
            geojson=gdf_states.geometry,
            locations=gdf_states.index,
            color="alpha_trend",
            hover_name="abbrev_state",
            hover_data={"alpha_trend": ":.4f"},
            color_continuous_scale="RdBu",
            color_continuous_midpoint=0,
            map_style="carto-positron",
            center={"lat": -15.793889, "lon": -47.882778},
            zoom=3,
            opacity=0.7,
            title="Avg Yearly Alpha Trend by State"
        )
    else:
        fig = px.choropleth_map(
            merged,
            geojson=merged.geometry,
            locations=merged.index,
            color="alpha_trend",
            hover_name="city_name",
            hover_data=["alpha_trend"],
            color_continuous_scale="RdBu",
            color_continuous_midpoint=0,
            map_style="carto-positron",
            center={"lat": merged.geometry.centroid.y.mean(), "lon": merged.geometry.centroid.x.mean()},
            zoom=6,
            opacity=0.7,
            title=f"Yearly Alpha Trend - {state_abbrev}"
        )

    fig.update_layout(
        margin={"r":0,"t":40,"l":0,"b":0},
        width=800,
        height=600,
        autosize=True
    )
    return fig

def plot_alpha_histogram(state_abbrev=None):
    merged, _ = get_merged_data(state_abbrev=state_abbrev)
    
    if merged.empty or "alpha" not in merged.columns:
        fig = px.scatter(title="No data available for histogram")
        return fig
    
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
        fig = px.scatter(title="No alpha values found")
        return fig

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

def plot_yearly_trend(geocode_str):
    geocode = extract_geocode(geocode_str)
    if geocode is None:
        return None
        
    df_yearly = get_yearly_db_data(geocode)
    
    if df_yearly.empty:
        fig = px.scatter(title=f"No yearly trend data for {geocode_str}")
        return fig
        
    fig = px.scatter(
        df_yearly, 
        x="year", 
        y="alpha", 
        title=f"Yearly Alpha Trend - {geocode_str}",
        labels={"year": "Year", "alpha": "Alpha Value"},
        trendline="ols" if len(df_yearly) > 1 else None
    )
    
    if len(df_yearly) > 1:
        # Extract p-value from trendline results
        results = px.get_trendline_results(fig)
        if not results.empty:
            ols_res = results.px_fit_results.iloc[0]
            p_val = ols_res.pvalues[1] # p-value for slope
            
            # The trendline is typically the second trace (index 1)
            if len(fig.data) > 1:
                fig.data[1].hovertemplate += f"<br>p-value: {p_val:.4f}"

    fig.update_layout(template="plotly_white")
    return fig

def plot_timeseries(geocode_str):
    geocode = extract_geocode(geocode_str)
    if geocode is None:
        return None
    
    df = fetch_infodengue(geocode)
    if df is None or df.empty:
        return px.line(title=f"No timeseries data for {geocode_str}")
    
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
        
    years = df_yearly['year'].unique()
    epi_dfs = []
    
    for yr in years:
        df_epi = fetch_episcanner(state=state, year=yr)
        if df_epi is not None and not df_epi.empty:
            # Filter for this city
            df_epi_city = df_epi[df_epi['geocode'] == geocode].copy()
            df_epi_city['year'] = yr
            epi_dfs.append(df_epi_city)
            
    if not epi_dfs:
        return pd.DataFrame()
        
    df_epi_combined = pd.concat(epi_dfs)
    
    # Merge on year
    merged = df_yearly.merge(df_epi_combined, on='year', suffixes=('', '_epi'))
    return merged

def plot_indicator_plots(geocode_str):
    geocode = extract_geocode(geocode_str)
    if geocode is None:
        return [None]*5
        
    df = get_combined_indicator_data(geocode)
    
    if df.empty:
        empty_fig = px.scatter(title="No episcanner data available for correlation")
        return [empty_fig]*5
        
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
            fig.update_layout(template="plotly_white")
            plots.append(fig)
        else:
            plots.append(px.scatter(title=f"Column {col} missing"))
            
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
                        trend_map_plot = gr.Plot(label="Alpha Trend Map")
                

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
                

        # Dashboard Logic / Event Handlers
        def sync_state(state):
            return state, plot_map(state), plot_trend_map(state), plot_alpha_histogram(state), update_city_dropdown(state)

        state_dropdown_overview.change(
            fn=sync_state, 
            inputs=[state_dropdown_overview], 
            outputs=[state_dropdown_city, map_plot, trend_map_plot, alpha_hist, city_dropdown]
        )
        load_map_btn.click(fn=plot_map, inputs=[state_dropdown_overview], outputs=map_plot)
        load_map_btn.click(fn=plot_trend_map, inputs=[state_dropdown_overview], outputs=trend_map_plot)
        load_map_btn.click(fn=plot_alpha_histogram, inputs=[state_dropdown_overview], outputs=alpha_hist)

        state_dropdown_city.change(
            fn=sync_state, 
            inputs=[state_dropdown_city], 
            outputs=[state_dropdown_overview, map_plot, trend_map_plot, alpha_hist, city_dropdown]
        )
        
        city_dropdown.change(fn=plot_fit, inputs=[city_dropdown], outputs=fit_plot)
        city_dropdown.change(fn=plot_yearly_trend, inputs=[city_dropdown], outputs=trend_plot)
        city_dropdown.change(fn=plot_timeseries, inputs=[city_dropdown], outputs=timeseries_plot)
        city_dropdown.change(fn=get_city_details, inputs=[city_dropdown], outputs=city_details_table)
        city_dropdown.change(fn=get_yearly_details, inputs=[city_dropdown], outputs=yearly_fits_table)
        city_dropdown.change(fn=plot_indicator_plots, inputs=[city_dropdown], outputs=[r0_plot, cases_plot, ini_plot, dur_plot, res_plot])
        
        plot_btn.click(fn=plot_fit, inputs=[city_dropdown], outputs=fit_plot)
        plot_btn.click(fn=plot_yearly_trend, inputs=[city_dropdown], outputs=trend_plot)
        plot_btn.click(fn=plot_timeseries, inputs=[city_dropdown], outputs=timeseries_plot)
        plot_btn.click(fn=get_city_details, inputs=[city_dropdown], outputs=city_details_table)
        plot_btn.click(fn=get_yearly_details, inputs=[city_dropdown], outputs=yearly_fits_table)
        plot_btn.click(fn=plot_indicator_plots, inputs=[city_dropdown], outputs=[r0_plot, cases_plot, ini_plot, dur_plot, res_plot])

        def initial_load():
            map_fig = plot_map(None)
            trend_map_fig = plot_trend_map(None)
            hist_fig = plot_alpha_histogram(None)
            
            # Find the best city in all results to initialize the details tab
            choices, best_value = get_city_options(None)
            city_upd = gr.Dropdown(choices=choices, value=best_value, filterable=True)
            
            fit_fig = plot_fit(best_value)
            trend_fig = plot_yearly_trend(best_value)
            ts_fig = plot_timeseries(best_value)
            details = get_city_details(best_value)
            yearly = get_yearly_details(best_value)
            indicator_plots = plot_indicator_plots(best_value)
            
            return [map_fig, trend_map_fig, hist_fig, city_upd, fit_fig, trend_fig, ts_fig, details, yearly] + indicator_plots

        # Trigger initial load (All Brazil)
        demo.load(fn=initial_load, inputs=[], outputs=[
            map_plot, trend_map_plot, alpha_hist, city_dropdown, fit_plot, trend_plot, timeseries_plot, 
            city_details_table, yearly_fits_table,
            r0_plot, cases_plot, ini_plot, dur_plot, res_plot
        ])

    return demo

if __name__ == "__main__":
    demo = create_dashboard()
    demo.launch(debug=True)
