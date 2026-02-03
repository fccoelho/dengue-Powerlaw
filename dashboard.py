import gradio as gr
import pandas as pd
import sqlite3
from concurrent.futures import ThreadPoolExecutor
import plotly.graph_objects as go
import os

import data_utils
from data_utils import (
    GEO_STATE_MAP, load_geography, get_merged_data, get_city_details, 
    get_yearly_details, extract_geocode, get_episcanner_fit_results
)
import viz_utils
from viz_utils import (
    plot_map, plot_trend_map, plot_local_moran, plot_alpha_histogram, 
    plot_alpha_vs_incidence, plot_fit, plot_yearly_trend, plot_timeseries, 
    plot_indicator_plots, plot_combined_episcanner_fit, 
    plot_episcanner_dispersion_alpha, plot_episcanner_state_map, 
    plot_episcanner_region_timeseries, plot_state_residuals_vs_alpha
)

def on_map_select(evt: gr.SelectData):
    """Event handler for map selection."""
    if isinstance(evt.index, (list, tuple)):
        return None
    try:
        data = evt.value
        city_name = data.get("customdata", [None, None])[0]
        geocode = data.get("customdata", [None, None])[1]
        if city_name and geocode:
            return f"{city_name} ({geocode})"
    except:
        pass
    return None

def create_dashboard():
    """Builds the Gradio dashboard blocks."""
    # Initial data for setup
    with sqlite3.connect("powerlaw_results.db") as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='powerlaw_fits'")
        if cursor.fetchone():
            df_results = pd.read_sql_query("SELECT city_name, geocode FROM powerlaw_fits", conn)
        else:
            df_results = pd.DataFrame(columns=["city_name", "geocode"])
    
    gdf_all = load_geography()
    states = sorted(gdf_all['abbrev_state'].unique().tolist()) if not gdf_all.empty else []
    
    def get_city_options(state_abbrev):
        merged, _ = get_merged_data(state_abbrev=state_abbrev)
        if merged.empty:
            return [], None
        
        choices = []
        for _, row in merged.iterrows():
            alpha_str = f" - alpha: {row['alpha']:.2f}" if pd.notnull(row['alpha']) else ""
            choices.append(f"{row['city_name']} ({row['geocode']}){alpha_str}")
        
        choices = sorted(choices)
        
        if 'alpha' in merged.columns and not merged['alpha'].dropna().empty:
            best_id = merged['alpha'].idxmax()
            best_city = merged.loc[best_id]
            best_alpha = best_city['alpha']
            best_value = f"{best_city['city_name']} ({best_city['geocode']}) - alpha: {best_alpha:.2f}"
        else:
            best_value = choices[0] if choices else None
            
        return choices, best_value

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
                            epi_state_map_size = gr.Plot(label="Epidemic Size Alpha Map")
                            epi_state_map_dur = gr.Plot(label="Epidemic Duration Alpha Map")
                        
                        with gr.Row():
                            epi_residuals_plot = gr.Plot(label="Alpha vs Residuals (Statewide)")
                            
                        with gr.Row():
                            epi_timeseries_plot = gr.Plot(label="Regional Timeseries")

        def sync_state(state):
            base_state = None if state == "BR" else state
            map_fig = plot_map(base_state)
            trend_map_fig = plot_trend_map(base_state)
            moran_fig = plot_local_moran(base_state)
            hist_fig = plot_alpha_histogram(base_state)
            incidence_fig = plot_alpha_vs_incidence(base_state)
            city_choices, best_city = get_city_options(base_state)
            
            return (
                base_state,
                state if state in ["BR"] + states else base_state,
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
                f10 = executor.submit(plot_state_residuals_vs_alpha, region)

                def get_res(f, default=None):
                    try:
                        return f.result()
                    except Exception as e:
                        print(f"Task failed: {e}")
                        return default or go.Figure()

                return (
                    get_res(f1), get_res(f2), get_res(f3), get_res(f4), 
                    get_res(f5), get_res(f6), get_res(f7, pd.DataFrame()), 
                    get_res(f8, pd.DataFrame()), get_res(f9), get_res(f10)
                )

        epi_region_dropdown.change(
            fn=update_episcanner, 
            inputs=[epi_region_dropdown], 
            outputs=[
                epi_combined_plot_size, epi_combined_plot_dur, 
                epi_dispersion_plot_size, epi_dispersion_plot_dur, 
                epi_state_map_size, epi_state_map_dur,
                epi_details_table_size, epi_details_table_dur, 
                epi_timeseries_plot, epi_residuals_plot
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
                epi_timeseries_plot, epi_residuals_plot
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
                f_map = executor.submit(plot_map, None)
                f_tmap = executor.submit(plot_trend_map, None)
                f_moran = executor.submit(plot_local_moran, None)
                f_hist = executor.submit(plot_alpha_histogram, None)
                f_incidence = executor.submit(plot_alpha_vs_incidence, None)
                
                choices, best_value = get_city_options(None)
                
                f_fit = executor.submit(plot_fit, best_value)
                f_ytrend = executor.submit(plot_yearly_trend, best_value)
                f_ts = executor.submit(plot_timeseries, best_value)
                f_details = executor.submit(get_city_details, best_value)
                f_yearly = executor.submit(get_yearly_details, best_value)
                f_indicator = executor.submit(plot_indicator_plots, best_value)
                
                f_epi_size = executor.submit(plot_combined_episcanner_fit, "BR", "total_cases")
                f_edisp_size = executor.submit(plot_episcanner_dispersion_alpha, "BR", "total_cases")
                f_edetails_size = executor.submit(get_episcanner_fit_results, "BR", "total_cases")
                f_emap_size = executor.submit(plot_episcanner_state_map, "total_cases")
                
                f_epi_dur = executor.submit(plot_combined_episcanner_fit, "BR", "ep_dur")
                f_edisp_dur = executor.submit(plot_episcanner_dispersion_alpha, "BR", "ep_dur")
                f_edetails_dur = executor.submit(get_episcanner_fit_results, "BR", "ep_dur")
                f_emap_dur = executor.submit(plot_episcanner_state_map, "ep_dur")
                
                f_ets = executor.submit(plot_episcanner_region_timeseries, "BR")
                f_eres = executor.submit(plot_state_residuals_vs_alpha, "BR")

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
                epi_res = safe_result(f_eres)

            best_geocode = extract_geocode(best_value)
            state_val = None
            if best_geocode:
                state_prefix = int(str(best_geocode)[:2])
                state_val = GEO_STATE_MAP.get(state_prefix)
            
            if not state_val and states:
                state_val = states[0]
            
            return [state_val, state_val, state_val, map_fig, incidence_fig, trend_map_fig, moran_fig, hist_fig, city_upd, 
            fit_fig, trend_fig, ts_fig, details, yearly] + indicator_plots + [epi_fig_size, epi_dur_fig, epi_disp_fig_size, 
            epi_disp_fig_dur, epi_details_size, epi_details_dur, epi_map_size, epi_map_dur, epi_ts, epi_res]

        demo.load(fn=initial_load, inputs=[], outputs=[
            state_dropdown_overview, state_dropdown_city, epi_region_dropdown,
            map_plot, alpha_incidence_plot, trend_map_plot, moran_plot, alpha_hist, city_dropdown, fit_plot, trend_plot, timeseries_plot, 
            city_details_table, yearly_fits_table,
            r0_plot, cases_plot, ini_plot, dur_plot, res_plot,
            epi_combined_plot_size, epi_combined_plot_dur, 
            epi_dispersion_plot_size, epi_dispersion_plot_dur, 
            epi_details_table_size, epi_details_table_dur, 
            epi_state_map_size, epi_state_map_dur, 
            epi_timeseries_plot, epi_residuals_plot
        ])

    return demo

if __name__ == "__main__":
    demo = create_dashboard()
    import gradio.networking
    gradio.networking.url_ok = lambda x: True
    os.environ["NO_PROXY"] = "localhost,127.0.0.1"
    demo.launch(
        debug=True,
        server_name="127.0.0.1",
        server_port=7861
    )
