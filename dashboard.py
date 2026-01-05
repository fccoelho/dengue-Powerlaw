import gradio as gr
import pandas as pd
import sqlite3
import geopandas as gpd
import plotly.express as px
import matplotlib.pyplot as plt
from fitpl import FitPL, fetch_infodengue
import numpy as np


# Connect to DB and fetch results
def get_db_data(db_path="powerlaw_results.db"):
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query("SELECT * FROM powerlaw_fits", conn)
    return df

# Load Geography
def load_geography(gpkg_path="muni_br.gpkg", state_abbrev=None):
    gdf = gpd.read_file(gpkg_path)
    # Ensure code_muni is integer for merging
    gdf['code_muni'] = gdf['code_muni'].astype(int)
    
    if state_abbrev:
        gdf = gdf[gdf['abbrev_state'] == state_abbrev]
        
    return gdf

# Merge data
def get_merged_data(state_abbrev=None):
    df_results = get_db_data()
    gdf = load_geography(state_abbrev=state_abbrev)
    
    merged = gdf.merge(df_results, left_on='code_muni', right_on='geocode', how='inner' if state_abbrev else 'right')
    return merged, df_results

# Plot Map
def plot_map(state_abbrev=None):
    merged, _ = get_merged_data(state_abbrev=state_abbrev)
    
    if merged.empty:
        # Return an empty figure with a message if no data for the state
        fig = px.scatter_map(lat=[-15.79], lon=[-47.88], zoom=3, map_style="carto-positron")
        fig.update_layout(title="No data for selected state")
        return fig

    # Plotly Choropleth
    fig = px.choropleth_map(
        merged,
        geojson=merged.geometry,
        locations=merged.index,
        color="alpha",
        hover_name="city_name",
        hover_data=["alpha", "xmin", "xmax"],
        custom_data=["city_name", "geocode"],
        map_style="carto-positron",
        center={"lat": merged.geometry.centroid.y.mean(), "lon": merged.geometry.centroid.x.mean()} if state_abbrev else {"lat": -15.793889, "lon": -47.882778}, 
        zoom=6 if state_abbrev else 3,
        opacity=0.7,
        title=f"Power Law Alpha Parameter - {state_abbrev}" if state_abbrev else "Power Law Alpha Parameter by Municipality"
    )
    fig.update_layout(
        margin={"r":0,"t":40,"l":0,"b":0},
        width=1200,
        height=800,
        autosize=False
    )
    return fig

def plot_alpha_histogram(state_abbrev=None):
    merged, _ = get_merged_data(state_abbrev=state_abbrev)
    
    if merged.empty or "alpha" not in merged.columns:
        fig = px.scatter(title="No data available for histogram")
        return fig
    
    data = merged["alpha"].dropna()
    if data.empty:
        fig = px.scatter(title="No alpha values found in selected state")
        return fig

    fig = px.histogram(
        merged[["alpha"]].copy(), 
        x="alpha", 
        nbins=20, 
        title=f"Alpha Distribution - {state_abbrev}" if state_abbrev else "Alpha Distribution (All)",
        labels={"alpha": "Alpha Value"}
    )
    fig.update_layout(
        margin={"r":20,"t":40,"l":20,"b":40},
        height=400,
        template="plotly_white"
    )
    return fig

# Plot Fit
def plot_fit(geocode_str):
    if not geocode_str:
        return None
        
    # extract geocode from string "Name (Geocode)"
    try:
        geocode = int(geocode_str.split('(')[-1].strip(')'))
    except:
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

def get_city_details(geocode_str):
    if not geocode_str:
        return None
    try:
        geocode = int(geocode_str.split('(')[-1].strip(')'))
    except:
        return None
    
    with sqlite3.connect("powerlaw_results.db") as conn:
        df = pd.read_sql_query("SELECT * FROM powerlaw_fits WHERE geocode=?", conn, params=(geocode,))
    return df

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
        df_results = pd.read_sql_query("SELECT city_name, geocode FROM powerlaw_fits", conn)
    
    gdf_all = load_geography()
    states = sorted(gdf_all['abbrev_state'].unique().tolist())
    
    def update_city_dropdown(state_abbrev):
        merged, _ = get_merged_data(state_abbrev=state_abbrev)
        choices = sorted([f"{row['city_name']} ({row['geocode']})" for _, row in merged.iterrows()])
        return gr.Dropdown(choices=choices, value=None, filterable=True)

    with gr.Blocks(title="Power Law Dashboard") as demo:
        gr.Markdown("# Power Law Estimation Dashboard - Dengue Cases")
        
        with gr.Row():
            with gr.Column(scale=1):
                state_dropdown = gr.Dropdown(choices=states, label="Select State", filterable=True)
                alpha_hist = gr.Plot(label="Alpha Distribution", min_width=300)
                load_map_btn = gr.Button("Refresh Map")
            with gr.Column(scale=4):
                map_plot = gr.Plot(label="Alpha Parameter Map")
            
        with gr.Row():
            with gr.Column(scale=1):
                city_dropdown = gr.Dropdown(choices=[], label="Select City to View Fit", filterable=True)
                plot_btn = gr.Button("Plot Fit")
                city_details_table = gr.Dataframe(label="City Details", interactive=False)
            
            with gr.Column(scale=2):
                fit_plot = gr.Plot(label="Fit Visualization")
        
        # Event Handlers
        state_dropdown.change(fn=update_city_dropdown, inputs=[state_dropdown], outputs=city_dropdown)
        state_dropdown.change(fn=plot_map, inputs=[state_dropdown], outputs=map_plot)
        state_dropdown.change(fn=plot_alpha_histogram, inputs=[state_dropdown], outputs=alpha_hist)
        
        load_map_btn.click(fn=plot_map, inputs=[state_dropdown], outputs=map_plot)
        load_map_btn.click(fn=plot_alpha_histogram, inputs=[state_dropdown], outputs=alpha_hist)
        
        # Map click interaction is currently not supported by gr.Plot in this Gradio version.
        # map_plot.select(fn=on_map_select, inputs=[], outputs=city_dropdown)
        
        # City selection triggers plot and table update
        city_dropdown.change(fn=plot_fit, inputs=[city_dropdown], outputs=fit_plot)
        city_dropdown.change(fn=get_city_details, inputs=[city_dropdown], outputs=city_details_table)
        
        plot_btn.click(fn=plot_fit, inputs=[city_dropdown], outputs=fit_plot)
        plot_btn.click(fn=get_city_details, inputs=[city_dropdown], outputs=city_details_table)
        
        def initial_load(state):
            return plot_map(state), plot_alpha_histogram(state)

        # Trigger initial load (All Brazil)
        demo.load(fn=initial_load, inputs=[state_dropdown], outputs=[map_plot, alpha_hist])

    return demo

if __name__ == "__main__":
    demo = create_dashboard()
    demo.launch(debug=True)
