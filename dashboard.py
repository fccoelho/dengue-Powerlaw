import gradio as gr
import pandas as pd
import numpy as np
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
import predictive_model

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

            with gr.TabItem("Predictive Modeling"):
                gr.Markdown("""
                ## Forecast Epidemic Metrics (Beta)

                ### Model Description
                The predictive model forecasts epidemic characteristics for year $t$ using data from year $t-1$.
                # """)
                gr.Markdown("""
                **Model Equation:**

                $$Y_t = f_{RF}(X_{t-1})$$
                """)
                gr.Markdown(r"""
                > **Tip**: For improved accuracy, we recommend incorporating climate data (e.g., ERA5 temperature/precipitation) from the [Mosqlimate](https://mosqlimate.org) platform.
                
                
                

                where $f_{RF}$ is a Random Forest Regressor.
                
                **Target Variables ($Y_t$):**

                $S_t$: Epidemic Size (Total Cases)
                
                $D_t$: Epidemic Duration (Weeks)
                
                $P_t$: Peak Week
                
                **Features ($X_{t-1}$):**

                1. **Lagged Epidemic Metrics**: $S_{t-1}, D_{t-1}, P_{t-1}$
                2. **Scaling Factors**: $\alpha_{S, t-1}, x_{min, S, t-1}$ (Power Law fit for Size)
                3. **Historical Trend**: $\dot{\alpha}_{S, t-1}$ (Slope of alpha over past years)
                4. **Epidemiological Driver**: $R_{0, t-1}$ (Lagged Basic Reproduction Number)
                5. **Demographics**: $\log_{10}(Pop)$ (Logarithm of total population)
                6. **Climate (Quarterly)**: Mean quarterly minimum temperature and minimum humidity ($\bar{T}_{Q1...Q4}, \bar{H}_{Q1...Q4}$) from Year $t-1$.
                """, latex_delimiters=[{ "left": "$", "right": "$", "display": False }])
                
                with gr.Row():
                    with gr.Column(scale=1):
                        pred_level_radio = gr.Radio(choices=["State", "City"], value="State", label="Prediction Level")
                        
                        pred_state_dropdown = gr.Dropdown(
                            choices=["BR"] + states, 
                            value="BR", 
                            label="Region/State", 
                            filterable=True
                        )
                        
                        pred_city_dropdown = gr.Dropdown(
                            choices=[], 
                            label="City (Select State first)", 
                            visible=False,
                            filterable=True
                        )
                        
                        pred_target_year = gr.Slider(
                            minimum=2012, maximum=2025, step=1, value=2024, label="Target Prediction Year"
                        )
                        pred_train_btn = gr.Button("Train & Predict", variant="primary")
                        
                        gr.Markdown("### Predicted Values")
                        pred_values_output = gr.Markdown("Run prediction to see values.")
                        
                        pred_metrics_table = gr.Dataframe(
                            headers=["Metric", "MAE", "MAPE", "R2"], 
                            label="Model Performance (Test Set)",
                            interactive=False
                        )
                        
                    with gr.Column(scale=3):
                        with gr.Row():
                            pred_plot_size = gr.Plot(label="Predicted vs Actual: Total Cases")
                            pred_plot_dur = gr.Plot(label="Predicted vs Actual: Duration")
                            pred_plot_peak = gr.Plot(label="Predicted vs Actual: Peak Week")
                        
                        with gr.Row():
                            pred_feat_imp_plot = gr.Plot(label="Feature Importance")
                        
                        gr.Markdown("### Error Analysis: Population Distribution (>20% absolute error)")
                        with gr.Row():
                            pred_hist_size = gr.Plot(label="Pop. Histogram: Cases Error > 20%")
                            pred_hist_dur = gr.Plot(label="Pop. Histogram: Duration Error > 20%")
                            pred_hist_peak = gr.Plot(label="Pop. Histogram: Peak Week Error > 20%")

        # Update City Dropdown visibility based on Radio
        def update_pred_inputs(level, state):
            city_visible = (level == "City")
            choices = []
            if city_visible and state and state != "BR":
                merged, _ = get_merged_data(state) # Reuse helper
                choices = [f"{row['city_name']} ({row['geocode']})" for _, row in merged.iterrows()]
            
            return gr.update(visible=city_visible, choices=choices, value=None)

        pred_level_radio.change(fn=update_pred_inputs, inputs=[pred_level_radio, pred_state_dropdown], outputs=[pred_city_dropdown])
        pred_state_dropdown.change(fn=update_pred_inputs, inputs=[pred_level_radio, pred_state_dropdown], outputs=[pred_city_dropdown])

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

        def run_prediction(level, region, city_str, year):
            try:
                # Determine aggregation level
                # Force 'city' level training if we want state aggregate from city observations
                agg_level = 'city' if (level == 'City' or (level == 'State' and region != 'BR')) else 'state'
                
                # 1. Prepare data
                # If City level, we fetch data for the STATE to train the model on many cities
                # But if region is BR, handle carefully. BR + City level -> All cities in Brazil? (Too big/slow?)
                # Warn user? For now assume Region is State if Level is City.
                
                if agg_level == 'city' and region == 'BR':
                     # Default to RS or warn? Or just try?
                     pass
                
                feature_level_arg = agg_level
                
                df_all = predictive_model.prepare_lagged_data(region, target_year=None, level=feature_level_arg)
                
                # Critical: Reset index to ensure positional alignment with numpy arrays from model.predict
                if not df_all.empty:
                    df_all = df_all.reset_index(drop=True)
                
                if df_all.empty:
                    return [go.Figure().update_layout(title="No data for training")]*4 + ["No data", pd.DataFrame()]

                # 2. Train
                models, metrics, feature_cols = predictive_model.train_predictive_models(df_all, test_year=year)
                
                # 3. Plots (Validation on whole dataset)
                val_preds = predictive_model.predict_future(models, df_all, feature_cols)
                
                plots = []
                hist_plots = []
                for target in ['total_cases', 'ep_dur', 'peak_week']:
                    if target not in val_preds:
                        plots.append(go.Figure().update_layout(title=f"{target}: Model not trained (Insufficient data)"))
                        hist_plots.append(go.Figure().update_layout(title=f"{target}: No Error Data"))
                        continue
                    
                    y_true = df_all[target]
                    y_pred = val_preds[target]
                    
                    # High Error Calculation (> 20% relative error)
                    # We use relative error: abs(true - pred) / true
                    # Adding small epsilon to avoid div by zero
                    rel_error = np.abs(y_true - y_pred) / (y_true.replace(0, 1e-5))
                    high_error_mask = rel_error > 0.2
                    
                    fig = go.Figure()
                    
                    # Create hover text
                    hover_text = []
                    for i, row in df_all.iterrows():
                        txt = f"Year: {int(row['year'])}<br>"
                        if 'city_name' in df_all.columns:
                            txt += f"City: {row['city_name']} ({int(row['geocode'])})"
                        elif 'state' in df_all.columns:
                            txt += f"State: {row['state']}"
                        
                        # Add error info to hover
                        err_val = rel_error.iloc[i] * 100
                        txt += f"<br>Error: {err_val:.1f}%"
                        hover_text.append(txt)
                        
                    fig.add_trace(go.Scatter(
                        x=y_true, 
                        y=y_pred, 
                        mode='markers', 
                        name='Data Points', 
                        marker=dict(color=high_error_mask.map({True: 'red', False: 'blue'}), opacity=0.6),
                        hovertext=hover_text,
                        hoverinfo='text+x+y'
                    ))
                    min_val = min(y_true.min(), y_pred.min())
                    max_val = max(y_true.max(), y_pred.max())
                    fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode='lines', line=dict(dash='dash', color='gray'), name='Ideal'))
                    fig.update_layout(title=f"{target}: Actual vs Predicted", xaxis_title="Actual", yaxis_title="Predicted", template="plotly_white")
                    plots.append(fig)
                    
                    # Population Histogram for High Error cities
                    if 'population' in df_all.columns:
                        high_error_pop = df_all.loc[high_error_mask, 'population']
                        fig_hist = go.Figure()
                        if not high_error_pop.empty:
                             fig_hist.add_trace(go.Histogram(x=high_error_pop, name='Pop. Size', marker_color='red'))
                             fig_hist.update_layout(
                                 title=f"Pop. Distribution of Cities with >20% Error in {target}",
                                 xaxis_title="Population Size",
                                 yaxis_title="Count",
                                 template="plotly_white"
                             )
                        else:
                             fig_hist.update_layout(title=f"No cities with >20% error in {target}")
                        hist_plots.append(fig_hist)
                    else:
                        hist_plots.append(go.Figure().update_layout(title="Population data not available"))
                    
                # 4. Feature Importance
                imp_df = predictive_model.get_variable_importance(models, feature_cols)
                fig_imp = go.Figure()
                for col in imp_df.columns:
                    if col != "Feature":
                        fig_imp.add_trace(go.Bar(name=col, y=imp_df['Feature'], x=imp_df[col], orientation='h'))
                fig_imp.update_layout(title="Feature Importance", barmode='group', template="plotly_white", height=500)
                
                # 5. Metrics Table
                metrics_data = []
                for t, m in metrics.items():
                    metrics_data.append([t, f"{m['MAE']:.2f}", f"{m['MAPE']:.2f}", f"{m['R2']:.2f}"])
                metrics_df = pd.DataFrame(metrics_data, columns=["Metric", "MAE", "MAPE", "R2"])
                
                # 6. Specific Prediction & Aggregation
                target_pred_text = f"### Forecast for Year {year}\n"
                
                # We always want city-level data if region != BR to perform aggregation
                df_target_year = df_all[df_all['year'] == year].copy()
                
                if not df_target_year.empty:
                    # Predict for all cities in target year
                    X_target_all = df_target_year[feature_cols]
                    
                    # Initialize with zeros to avoid KeyError if models weren't trained
                    targets = ['total_cases', 'ep_dur', 'peak_week']
                    preds_all = {t: np.zeros(len(df_target_year)) for t in targets}
                    
                    for t in targets:
                        model = models.get(t)
                        if model:
                            preds_all[t] = model.predict(X_target_all)
                    
                    # 6a. State Aggregation (if applicable)
                    if region != "BR":
                        # Predicted Aggregates
                        pred_cases_total = np.sum(preds_all['total_cases'])
                        
                        # Weighted means for duration and peak (weighted by predicted cases)
                        weights = preds_all['total_cases']
                        if np.sum(weights) > 0:
                            pred_dur_agg = np.sum(preds_all['ep_dur'] * weights) / np.sum(weights)
                            pred_peak_agg = np.sum(preds_all['peak_week'] * weights) / np.sum(weights)
                        else:
                            pred_dur_agg = np.mean(preds_all['ep_dur'])
                            pred_peak_agg = np.mean(preds_all['peak_week'])
                            
                        # Observed Aggregates
                        obs_cases_total = df_target_year['total_cases'].sum()
                        obs_weights = df_target_year['total_cases']
                        if obs_cases_total > 0:
                            obs_dur_agg = np.sum(df_target_year['ep_dur'] * obs_weights) / obs_cases_total
                            obs_peak_agg = np.sum(df_target_year['peak_week'] * obs_weights) / obs_cases_total
                        else:
                            obs_dur_agg = df_target_year['ep_dur'].mean()
                            obs_peak_agg = df_target_year['peak_week'].mean()
                            
                        # Errors
                        err_cases = pred_cases_total - obs_cases_total
                        err_dur = pred_dur_agg - obs_dur_agg
                        err_peak = pred_peak_agg - obs_peak_agg
                        
                        target_pred_text += f"#### State Aggregate ({region})\n"
                        target_pred_text += f"- **Total Cases**: {pred_cases_total:.0f} (Observed: {obs_cases_total:.0f}, Error: {err_cases:+.0f})\n"
                        target_pred_text += f"- **Duration (Weighted)**: {pred_dur_agg:.1f} weeks (Observed: {obs_dur_agg:.1f}, Error: {err_dur:+.1f})\n"
                        target_pred_text += f"- **Peak Week (Weighted)**: {pred_peak_agg:.1f} (Observed: {obs_peak_agg:.1f}, Error: {err_peak:+.1f})\n\n"

                    # 6b. Individual City Prediction (if Level=City)
                    if level == 'City' and city_str:
                        geocode = extract_geocode(city_str)
                        if geocode:
                            city_row_idx = df_target_year.index[df_target_year['geocode'] == int(geocode)]
                            if not city_row_idx.empty:
                                pos = df_target_year.index.get_loc(city_row_idx[0])
                                target_pred_text += f"#### Individual City: {city_str}\n"
                                for t in ['total_cases', 'ep_dur', 'peak_week']:
                                    val = preds_all[t][pos]
                                    obs = df_target_year.loc[city_row_idx[0], t]
                                    err = val - obs
                                    label = t.replace('_', ' ').title()
                                    if t == 'ep_dur': label = "Duration"
                                    target_pred_text += f"- **{label}**: {val:.1f} (Observed: {obs:.1f}, Error: {err:+.1f})\n"
                            else:
                                target_pred_text += f"*City {city_str} data not available for {year}.*\n"
                else:
                    target_pred_text += f"*No data available for year {year} in {region}.*\n"
                
                return plots + [fig_imp, target_pred_text, metrics_df] + hist_plots
                
            except Exception as e:
                print(f"Prediction failed: {e}")
                import traceback
                traceback.print_exc()
                empty = go.Figure().update_layout(title=f"Error: {str(e)}")
                return [empty]*4 + [f"Error: {str(e)}", pd.DataFrame()] + [empty]*3

        pred_train_btn.click(
            fn=run_prediction,
            inputs=[pred_level_radio, pred_state_dropdown, pred_city_dropdown, pred_target_year],
            outputs=[
                pred_plot_size, pred_plot_dur, pred_plot_peak, 
                pred_feat_imp_plot, pred_values_output, pred_metrics_table,
                pred_hist_size, pred_hist_dur, pred_hist_peak
            ]
        )

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
        server_port=7860
    )
