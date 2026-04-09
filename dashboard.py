import gradio as gr
import pandas as pd
import numpy as np
import sqlite3
import plotly.graph_objects as go
import os
from sklearn.metrics import roc_auc_score, average_precision_score

import data_utils
from data_utils import (
    GEO_STATE_MAP,
    load_geography,
    get_merged_data,
    get_city_details,
    get_yearly_details,
    extract_geocode,
    get_episcanner_fit_results,
)
import viz_utils
from viz_utils import (
    plot_map,
    plot_trend_map,
    plot_local_moran,
    plot_alpha_histogram,
    plot_alpha_vs_incidence,
    plot_fit,
    plot_yearly_trend,
    plot_timeseries,
    plot_indicator_plots,
    plot_combined_episcanner_fit,
    plot_episcanner_dispersion_alpha,
    plot_episcanner_state_map,
    plot_episcanner_region_timeseries,
    plot_state_residuals_vs_alpha,
)
import predictive_model
import binary_classifier
from binary_classifier import (
    prepare_binary_dataset,
    train_binary_model,
    get_binary_feature_importance,
    get_shap_values,
    predict_future_binary,
)
import binary_viz_utils
from binary_viz_utils import (
    plot_confusion_matrix_binary,
    plot_roc_curve_binary,
    plot_precision_recall_curve_binary,
    plot_calibration_curve_binary,
    plot_feature_importance_binary,
    plot_shap_summary_binary,
    plot_threshold_tradeoff_binary,
    plot_prediction_map_binary,
    plot_cv_folds_metrics_binary,
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
    with sqlite3.connect("powerlaw_results.db") as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='powerlaw_fits'"
        )
        if cursor.fetchone():
            df_results = pd.read_sql_query(
                "SELECT city_name, geocode FROM powerlaw_fits", conn
            )
        else:
            df_results = pd.DataFrame(columns=["city_name", "geocode"])

    gdf_all = load_geography()
    states = (
        sorted(gdf_all["abbrev_state"].unique().tolist()) if not gdf_all.empty else []
    )

    def get_city_options(state_abbrev):
        merged, _ = get_merged_data(state_abbrev=state_abbrev)
        if merged.empty:
            return [], None

        choices = []
        for _, row in merged.iterrows():
            alpha_str = (
                f" - alpha: {row['alpha']:.2f}" if pd.notnull(row["alpha"]) else ""
            )
            choices.append((f"{row['city_name']} ({row['geocode']}){alpha_str}", row['alpha'] if pd.notnull(row['alpha']) else -float('inf')))

        choices = sorted(choices, key=lambda x: x[1], reverse=True)
        choices = [c[0] for c in choices]

        if "alpha" in merged.columns and not merged["alpha"].dropna().empty:
            best_id = merged["alpha"].idxmax()
            best_city = merged.loc[best_id]
            best_alpha = best_city["alpha"]
            best_value = f"{best_city['city_name']} ({best_city['geocode']}) - alpha: {best_alpha:.2f}"
        else:
            best_value = choices[0] if choices else None

        return choices, best_value

    with gr.Blocks(title="Power Law Dashboard") as demo:
        gr.Markdown("# Power Law Estimation Dashboard - Dengue Cases")

        with gr.Tabs():
            tab_overview = gr.TabItem("General Overview", id="overview")
            with tab_overview:
                with gr.Row():
                    with gr.Column(scale=1):
                        state_dropdown_overview = gr.Dropdown(
                            choices=states, label="Select State", filterable=True
                        )
                        alpha_hist = gr.Plot(
                            label="Alpha Distribution (Weekly Cases)", min_width=300
                        )
                    with gr.Column(scale=4):
                        map_plot = gr.Plot(label="Alpha Parameter Map (Weekly Cases)")
                        alpha_incidence_plot = gr.Plot(
                            label="Alpha vs Incidence (Weekly Cases)"
                        )
                        trend_map_plot = gr.Plot(label="Alpha Trend Map (Weekly Cases)")
                        moran_plot = gr.Plot(
                            label="Local Moran Clustering Map (Weekly Cases)"
                        )

            tab_city = gr.TabItem("City Details & Yearly Trends", id="city")
            with tab_city:
                with gr.Row():
                    with gr.Column(scale=1):
                        state_dropdown_city = gr.Dropdown(
                            choices=states,
                            label="Filter by State",
                            filterable=True,
                            value=states[0] if states else None,
                        )
                        city_dropdown = gr.Dropdown(
                            choices=[], label="Select City", filterable=True
                        )
                        plot_btn = gr.Button("Refresh City Data")
                        city_details_table = gr.Dataframe(
                            label="General Stats", interactive=False
                        )
                        yearly_fits_table = gr.Dataframe(
                            label="Yearly Fit Parameters", interactive=False
                        )

                    with gr.Column(scale=2):
                        fit_plot = gr.Plot(
                            label="Power Law Fit (Weekly Cases - Overall)"
                        )
                        timeseries_plot = gr.Plot(label="Timeseries of Weekly Cases")

                    with gr.Column(scale=2):
                        trend_plot = gr.Plot(label="Yearly Alpha Trend (Weekly Cases)")

                with gr.Row():
                    r0_plot = gr.Plot(label="Alpha (Weekly) vs R0")
                    cases_plot = gr.Plot(label="Alpha (Weekly) vs Total Cases (Annual)")

                with gr.Row():
                    ini_plot = gr.Plot(label="Alpha (Weekly) vs Ep Start")
                    dur_plot = gr.Plot(label="Alpha (Weekly) vs Ep Duration")

                with gr.Row():
                    res_plot = gr.Plot(label="Alpha (Weekly) vs Sum of Residuals")

            tab_episcanner = gr.TabItem(
                "Epidemic Metrics Power Laws (Total Annual Cases)", id="episcanner"
            )
            with tab_episcanner:
                with gr.Row():
                    with gr.Column(scale=1):
                        epi_region_dropdown = gr.Dropdown(
                            choices=["BR"] + states,
                            value="BR",
                            label="Select Region (State or BR)",
                            filterable=True,
                        )
                        epi_refresh_btn = gr.Button("Refresh Episcanner Plots")
                        with gr.Row():
                            epi_details_table_size = gr.Dataframe(
                                label="Fit Details (Size)", interactive=False
                            )
                            epi_details_table_dur = gr.Dataframe(
                                label="Fit Details (Duration)", interactive=False
                            )

                    with gr.Column(scale=4):
                        with gr.Row():
                            epi_combined_plot_size = gr.Plot(
                                label="Size: Combined Fit (Total Cases)"
                            )
                            epi_combined_plot_dur = gr.Plot(
                                label="Duration: Combined Fit (Total Cases)"
                            )

                        with gr.Row():
                            epi_dispersion_plot_size = gr.Plot(
                                label="Size: Dispersion vs Alpha (Total Cases)"
                            )
                            epi_dispersion_plot_dur = gr.Plot(
                                label="Duration: Dispersion vs Alpha (Total Cases)"
                            )

                        with gr.Row():
                            epi_state_map_size = gr.Plot(
                                label="Epidemic Size Alpha Map (Total Cases)"
                            )
                            epi_state_map_dur = gr.Plot(
                                label="Epidemic Duration Alpha Map (Total Cases)"
                            )

                        with gr.Row():
                            epi_residuals_plot = gr.Plot(
                                label="Alpha vs Residuals (Total Cases - Statewide)"
                            )

                        with gr.Row():
                            epi_timeseries_plot = gr.Plot(
                                label="Regional Timeseries (Weekly Cases)"
                            )

            tab_predictive = gr.TabItem("Predictive Modeling", id="predictive")
            with tab_predictive:
                gr.Markdown("""
                ## Forecast Epidemic Metrics (Beta)

                ### Model Description
                The predictive model forecasts epidemic characteristics for year $t$ using data from year $t-1$.
                # """)
                gr.Markdown("""
                **Model Equation:**

                $$Y_t = f(X_{t-1})$$
                """)
                gr.Markdown(
                    r"""
                > **Tip**: For improved accuracy, we recommend incorporating climate data (e.g., ERA5 temperature/precipitation) from the [Mosqlimate](https://mosqlimate.org) platform.




                where $f$ is the selected regression model (Random Forest, LightGBM, or HistGradientBoosting).

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
                """,
                    latex_delimiters=[{"left": "$", "right": "$", "display": False}],
                )

                with gr.Row():
                    with gr.Column(scale=1):
                        pred_state_dropdown = gr.Dropdown(
                            choices=["BR"] + states,
                            value="BR",
                            label="Region/State",
                            filterable=True,
                        )

                        pred_city_dropdown = gr.Dropdown(
                            choices=[],
                            label="City (Select State first)",
                            visible=True,
                            filterable=True,
                        )

                        pred_target_year = gr.Slider(
                            minimum=2012,
                            maximum=2025,
                            step=1,
                            value=2024,
                            label="Target Prediction Year",
                        )

                        pred_model_type = gr.Dropdown(
                            choices=[
                                ("Random Forest", "random_forest"),
                                ("LightGBM (Quantile)", "lightgbm_quantile"),
                                ("HistGradientBoosting (Quantile)", "hist_gradient_boosting_quantile"),
                            ],
                            value="random_forest",
                            label="Model Type",
                        )

                        pred_quantile_input = gr.Slider(
                            minimum=0.05,
                            maximum=0.95,
                            step=0.05,
                            value=0.5,
                            label="Quantile Level (for Quantile Models)",
                            visible=False,
                        )

                        pred_min_pop = gr.Slider(
                            minimum=0,
                            maximum=500000,
                            step=5000,
                            value=0,
                            label="Minimum City Population (filter)",
                        )

                        pred_train_btn = gr.Button("Train & Predict", variant="primary")

                        gr.Markdown("### Predicted Values")
                        pred_values_output = gr.Markdown(
                            "Run prediction to see values."
                        )

                        pred_metrics_table = gr.Dataframe(
                            label="Model Performance (Test Set)",
                            interactive=False,
                        )

                    with gr.Column(scale=3):
                        with gr.Row():
                            pred_plot_size = gr.Plot(
                                label="Predicted vs Actual: Total Cases (Annual)"
                            )
                            pred_plot_dur = gr.Plot(
                                label="Predicted vs Actual: Duration (Annual)"
                            )
                            pred_plot_peak = gr.Plot(
                                label="Predicted vs Actual: Peak Week (Annual)"
                            )

                        with gr.Row():
                            pred_feat_imp_plot = gr.Plot(label="Feature Importance")

                        pred_feat_imp_explanation = gr.Markdown(
                            value="Train a model to see how feature importance is calculated.",
                            label="Feature Importance Explanation",
                        )

                        gr.Markdown(
                            "### Error Analysis: Population Distribution (>20% absolute error)"
                        )
                        with gr.Row():
                            pred_hist_size = gr.Plot(
                                label="Pop. Histogram: Total Cases Error > 20%"
                            )
                            pred_hist_dur = gr.Plot(
                                label="Pop. Histogram: Duration Error > 20%"
                            )
                            pred_hist_peak = gr.Plot(
                                label="Pop. Histogram: Peak Week Error > 20%"
                            )

                        gr.Markdown("### Residual Analysis")
                        with gr.Row():
                            pred_resid_size = gr.Plot(
                                label="Residuals vs Predicted: Total Cases"
                            )
                            pred_resid_dur = gr.Plot(
                                label="Residuals vs Predicted: Duration"
                            )
                            pred_resid_peak = gr.Plot(
                                label="Residuals vs Predicted: Peak Week"
                            )

                        with gr.Row():
                            pred_resid_hist_size = gr.Plot(
                                label="Residual Distribution: Total Cases"
                            )
                            pred_resid_hist_dur = gr.Plot(
                                label="Residual Distribution: Duration"
                            )
                            pred_resid_hist_peak = gr.Plot(
                                label="Residual Distribution: Peak Week"
                            )

            tab_binary_predictive = gr.TabItem(
                "Predicting Large Epidemics", id="binary_predictive"
            )
            with tab_binary_predictive:
                gr.Markdown("""
                ## 🎯 Binary Classification Model — Epidemic Forecasting

                ### Methodology Description

                This model predicts whether the total annual number of dengue cases in a geographic unit **will exceed a configurable historical percentile**.

                #### Target Variable
                - **Binary**: `1` if `cases_year_t > historical_percentile`, `0` otherwise.
                - The percentile is computed **ONLY on training data** from each fold to prevent data leakage.

                #### Features (Predictive Variables)
                1. **Lagged Epidemic Metrics**: Total cases, duration, and peak week from the previous year (t−1)
                2. **Scaling Factors**: α (alpha) and x_min from the Power Law fit of the previous year
                3. **Historical Trend**: Slope of α over preceding years
                4. **Epidemiological Driver**: R₀ (Basic Reproduction Number) from the previous year
                5. **Demographics**: log₁₀(Population)
                6. **Quarterly Climate**: Mean minimum temperature and minimum humidity per quarter (Q1–Q4)

                #### Validation & Training
                - **Temporal Validation**: `TimeSeriesSplit` (no shuffle) to preserve temporal ordering
                - **Imbalance Handling**: `scale_pos_weight` (XGBoost) or `is_unbalance=True` (LightGBM)
                - **Optimal Threshold**: Computed via Youden's J-statistic (maximizes sensitivity + specificity − 1)
                - **Reproducibility**: `random_state=42` in all stochastic calls

                #### Evaluation Metrics
                - **AUC-ROC**: Area under the ROC curve (discriminative ability)
                - **AUC-PR**: Area under the Precision-Recall curve (better for imbalanced classes)
                - **F1-Score**: Harmonic mean of precision and recall
                - **Brier Score**: Probabilistic accuracy (lower = better)
                - **Calibration Curve**: Reliability of predicted probabilities
                """)

                with gr.Row():
                    with gr.Column(scale=1):
                        binary_state_dropdown = gr.Dropdown(
                            choices=["BR"] + states,
                            value="BR",
                            label="Region (State or BR)",
                            filterable=True,
                        )

                        binary_percentile_slider = gr.Slider(
                            minimum=0.50,
                            maximum=0.99,
                            step=0.05,
                            value=0.75,
                            label="Historical Percentile (Target Threshold)",
                        )

                        binary_model_type = gr.Dropdown(
                            choices=[
                                ("XGBoost Classifier", "xgboost"),
                                ("LightGBM Classifier", "lightgbm"),
                            ],
                            value="xgboost",
                            label="Model Type",
                        )

                        binary_min_pop = gr.Slider(
                            minimum=0,
                            maximum=500000,
                            step=5000,
                            value=0,
                            label="Minimum City Population (filter)",
                        )

                        binary_test_year = gr.Slider(
                            minimum=2015,
                            maximum=2025,
                            step=1,
                            value=2024,
                            label="Test Year (training = prior years)",
                        )

                        binary_train_btn = gr.Button(
                            "Train Binary Model", variant="primary"
                        )

                        gr.Markdown("### 📊 Model Metrics")
                        binary_metrics_table = gr.Dataframe(
                            label="Model Performance", interactive=False
                        )

                        binary_threshold_info = gr.Markdown(
                            value="Train the model to see threshold information."
                        )

                        binary_cv_table = gr.Dataframe(
                            label="Per-Fold Results (TimeSeriesSplit)",
                            interactive=False,
                        )

                    with gr.Column(scale=3):
                        with gr.Row():
                            binary_conf_matrix = gr.Plot(
                                label="Confusion Matrix"
                            )
                            binary_roc_plot = gr.Plot(
                                label="ROC Curve"
                            )

                        with gr.Row():
                            binary_pr_plot = gr.Plot(
                                label="Precision-Recall Curve"
                            )
                            binary_calibration_plot = gr.Plot(
                                label="Calibration Curve"
                            )

                        with gr.Row():
                            binary_feat_imp_plot = gr.Plot(
                                label="Feature Importance"
                            )
                            binary_shap_plot = gr.Plot(
                                label="SHAP Summary Values"
                            )

                        binary_shap_btn = gr.Button(
                            "Compute SHAP Values", variant="secondary"
                        )
                        binary_shap_status = gr.Markdown(
                            value="Click the button above to compute SHAP values (may take a few seconds)."
                        )

                        with gr.Row():
                            binary_threshold_plot = gr.Plot(
                                label="Threshold Trade-off"
                            )
                            binary_cv_plot = gr.Plot(
                                label="Per-Fold Metrics (CV)"
                            )

                        with gr.Row():
                            binary_pred_map = gr.Plot(
                                label="Prediction Probabilities by City"
                            )

                # Hidden state to store trained model result for SHAP computation
                binary_model_state = gr.State(value=None)

        def update_pred_inputs(state):
            choices = []
            if state and state != "BR":
                merged, _ = get_merged_data(state)
                choices = [
                    f"{row['city_name']} ({row['geocode']})"
                    for _, row in merged.iterrows()
                ]

            return gr.update(visible=True, choices=choices, value=None)

        pred_state_dropdown.change(
            fn=update_pred_inputs,
            inputs=[pred_state_dropdown],
            outputs=[pred_city_dropdown],
        )

        def update_quantile_visibility(model_type):
            is_quantile = model_type in ("lightgbm_quantile", "hist_gradient_boosting_quantile")
            return gr.update(visible=is_quantile)

        pred_model_type.change(
            fn=update_quantile_visibility,
            inputs=[pred_model_type],
            outputs=[pred_quantile_input],
        )

        def run_prediction(region, city_str, year, model_type, quantile, min_pop):
            try:
                df_all = predictive_model.prepare_lagged_data(
                    region, target_year=None, level="city"
                )

                if not df_all.empty:
                    df_all = df_all.reset_index(drop=True)

                # Filter by minimum population
                if min_pop > 0 and "population" in df_all.columns:
                    count_before = len(df_all)
                    df_all = df_all[df_all["population"] >= min_pop].copy()
                    count_after = len(df_all)
                    print(
                        f"[Pop Filter] {min_pop}: {count_before} -> {count_after} rows "
                        f"({count_before - count_after} removed)"
                    )

                if df_all.empty:
                    return [
                        go.Figure().update_layout(title="No data for training")
                    ] * 4 + ["", "No data available for this selection.", pd.DataFrame()] + [go.Figure().update_layout(title="No data")] * 9

                models, metrics, feature_cols, train_data, bias_correction = (
                    predictive_model.train_predictive_models(
                        df_all, test_year=year, model_type=model_type, quantile=quantile
                    )
                )

                val_preds = predictive_model.predict_future(
                    models, df_all, feature_cols, bias_correction=bias_correction
                )

                plots = []
                hist_plots = []
                for target in ["total_cases", "ep_dur", "peak_week"]:
                    if target not in val_preds:
                        plots.append(
                            go.Figure().update_layout(
                                title=f"{target}: Model not trained (Insufficient data)"
                            )
                        )
                        hist_plots.append(
                            go.Figure().update_layout(title=f"{target}: No Error Data")
                        )
                        continue

                    y_true = df_all[target]
                    y_pred = val_preds[target]

                    rel_error = np.abs(y_true - y_pred) / (y_true.replace(0, 1e-5))
                    high_error_mask = rel_error > 0.2

                    fig = go.Figure()

                    hover_text = []
                    for i, row in df_all.iterrows():
                        txt = f"Year: {int(row['year'])}<br>"
                        if "city_name" in df_all.columns:
                            txt += f"City: {row['city_name']} ({int(row['geocode'])})"
                        elif "state" in df_all.columns:
                            txt += f"State: {row['state']}"

                        err_val = rel_error.loc[i] * 100
                        txt += f"<br>Error: {err_val:.1f}%"
                        hover_text.append(txt)

                    fig.add_trace(
                        go.Scatter(
                            x=y_true,
                            y=y_pred,
                            mode="markers",
                            name="Data Points",
                            marker=dict(
                                color=high_error_mask.map({True: "red", False: "blue"}),
                                opacity=0.6,
                            ),
                            hovertext=hover_text,
                            hoverinfo="text+x+y",
                        )
                    )
                    min_val = min(y_true.min(), y_pred.min())
                    max_val = max(y_true.max(), y_pred.max())
                    fig.add_trace(
                        go.Scatter(
                            x=[min_val, max_val],
                            y=[min_val, max_val],
                            mode="lines",
                            line=dict(dash="dash", color="gray"),
                            name="Ideal",
                        )
                    )
                    fig.update_layout(
                        title=f"{target}: Actual vs Predicted",
                        xaxis_title="Actual",
                        yaxis_title="Predicted",
                        template="plotly_white",
                    )

                    if target == "total_cases":
                        fig.update_layout(
                            xaxis_title="Actual (log scale)",
                            yaxis_title="Predicted (log scale)",
                            title=f"{target}: Actual vs Predicted (Log-Log)",
                        )
                        fig.update_xaxes(type="log")
                        fig.update_yaxes(type="log")

                    plots.append(fig)

                    if "population" in df_all.columns:
                        high_error_pop = df_all.loc[high_error_mask, "population"]
                        fig_hist = go.Figure()
                        if not high_error_pop.empty:
                            fig_hist.add_trace(
                                go.Histogram(
                                    x=high_error_pop,
                                    name="Pop. Size",
                                    marker_color="red",
                                )
                            )
                            fig_hist.update_layout(
                                title=f"Pop. Distribution of Cities with >20% Error in {target}",
                                xaxis_title="Population Size",
                                yaxis_title="Count",
                                template="plotly_white",
                            )
                        else:
                            fig_hist.update_layout(
                                title=f"No cities with >20% error in {target}"
                            )
                        hist_plots.append(fig_hist)
                    else:
                        hist_plots.append(
                            go.Figure().update_layout(
                                title="Population data not available"
                            )
                        )

                # --- Residual analysis plots ---
                resid_plots = []
                resid_hist_plots = []
                for target in ["total_cases", "ep_dur", "peak_week"]:
                    if target not in val_preds:
                        resid_plots.append(
                            go.Figure().update_layout(
                                title=f"{target}: Residuals not available"
                            )
                        )
                        resid_hist_plots.append(
                            go.Figure().update_layout(
                                title=f"{target}: Residual distribution not available"
                            )
                        )
                        continue

                    y_true = df_all[target]
                    y_pred = val_preds[target]
                    residuals = y_true - y_pred

                    # Residuals vs Predicted
                    fig_resid = go.Figure()
                    fig_resid.add_trace(
                        go.Scatter(
                            x=y_pred,
                            y=residuals,
                            mode="markers",
                            marker=dict(color="steelblue", opacity=0.5),
                            hovertext=[
                                f"Year: {int(row['year'])}<br>"
                                + (
                                    f"City: {row['city_name']} ({int(row['geocode'])})"
                                    if "city_name" in df_all.columns
                                    else f"State: {row['state']}"
                                )
                                for _, row in df_all.iterrows()
                            ],
                            hoverinfo="text+x+y",
                        )
                    )
                    fig_resid.add_hline(y=0, line_dash="dash", line_color="red")
                    fig_resid.update_layout(
                        title=f"{target}: Residuals vs Predicted",
                        xaxis_title="Predicted",
                        yaxis_title="Residual (Actual - Predicted)",
                        template="plotly_white",
                    )
                    resid_plots.append(fig_resid)

                    # Residual distribution histogram
                    fig_resid_hist = go.Figure()
                    fig_resid_hist.add_trace(
                        go.Histogram(
                            x=residuals,
                            name="Residuals",
                            marker_color="steelblue",
                            opacity=0.7,
                        )
                    )
                    mean_resid = residuals.mean()
                    fig_resid_hist.add_vline(
                        x=0, line_dash="dash", line_color="green", annotation_text="Zero"
                    )
                    fig_resid_hist.add_vline(
                        x=mean_resid,
                        line_dash="dot",
                        line_color="red",
                        annotation_text=f"Mean: {mean_resid:.1f}",
                    )
                    fig_resid_hist.update_layout(
                        title=f"{target}: Residual Distribution",
                        xaxis_title="Residual",
                        yaxis_title="Count",
                        template="plotly_white",
                    )
                    resid_hist_plots.append(fig_resid_hist)

                # Get X_train from any available target (same for all)
                x_train_ref = None
                y_train_ref = {}
                if train_data:
                    for t, d in train_data.items():
                        y_train_ref[t] = d.get('y_train')
                        if x_train_ref is None and d.get('X_train') is not None:
                            x_train_ref = d['X_train']

                imp_df = predictive_model.get_variable_importance(
                    models, feature_cols,
                    X_train=x_train_ref,
                    y_train_dict=y_train_ref,
                )
                fig_imp = go.Figure()
                for col in imp_df.columns:
                    if col != "Feature":
                        fig_imp.add_trace(
                            go.Bar(
                                name=col,
                                y=imp_df["Feature"],
                                x=imp_df[col],
                                orientation="h",
                            )
                        )
                fig_imp.update_layout(
                    title="Feature Importance",
                    barmode="group",
                    template="plotly_white",
                    height=500,
                )

                # Build feature importance explanation
                if model_type == "random_forest":
                    fi_text = """
### How Feature Importance is Calculated

- **Model**: Random Forest Regressor
- **Method**: Mean Decrease in Impurity (MDI), also known as Gini importance.
  Each tree in the forest tracks how much each feature reduces the variance (impurity) across all its splits. The importances are averaged over all trees and normalized so they sum to 1.
- **Interpretation**: A higher value means the feature contributes more to reducing prediction error across all decision tree splits.
"""
                elif model_type == "lightgbm_quantile":
                    fi_text = f"""
### How Feature Importance is Calculated

- **Model**: LightGBM (Quantile Regression, τ={quantile})
- **Method**: Split-based importance (number of times each feature is used to split nodes across all trees).
  If all features show zero split importance, **Permutation Importance** is used as fallback: each feature's values are randomly shuffled and the resulting drop in prediction accuracy is measured.
- **Interpretation**: Features that are used more frequently for splitting (or cause a larger accuracy drop when permuted) are considered more important.
"""
                elif model_type == "hist_gradient_boosting_quantile":
                    fi_text = f"""
### How Feature Importance is Calculated

- **Model**: HistGradientBoosting Regressor (Quantile Regression, τ={quantile})
- **Method**: Permutation Importance. Since this model does not expose native `feature_importances_`, each feature is randomly shuffled one at a time and the drop in prediction accuracy is measured. The mean decrease across 10 random permutations is reported.
- **Interpretation**: A higher value means that shuffling that feature's values causes a larger drop in model accuracy, indicating the model relies on it more heavily.
"""
                else:
                    fi_text = ""

                is_quantile = model_type in ("lightgbm_quantile", "hist_gradient_boosting_quantile")

                metrics_data = []
                for t, m in metrics.items():
                    if is_quantile:
                        pinball_val = m.get('Pinball_Loss')
                        quantile_val = m.get('Quantile')
                        metrics_data.append(
                            [
                                t,
                                f"{m['MAE']:.2f}",
                                f"{m['RMSE']:.2f}",
                                f"{pinball_val:.4f}" if pinball_val is not None else "N/A",
                                f"{quantile_val:.2f}" if quantile_val is not None else "N/A",
                            ]
                        )
                    else:
                        oob = m.get('OOB_R2')
                        metrics_data.append(
                            [
                                t,
                                f"{m['MAE']:.2f}",
                                f"{m['RMSE']:.2f}",
                                f"{oob:.2f}" if oob is not None else "N/A",
                            ]
                        )

                if is_quantile:
                    metrics_df = pd.DataFrame(
                        metrics_data,
                        columns=["Metric", "MAE", "RMSE", "Pinball Loss", "Quantile"],
                    )
                else:
                    metrics_df = pd.DataFrame(
                        metrics_data, columns=["Metric", "MAE", "RMSE", "OOB R2"]
                    )

                target_pred_text = f"### Forecast for Year {year}\n"

                df_target_year = df_all[df_all["year"] == year].copy()

                if not df_target_year.empty:
                    X_target_all = df_target_year[feature_cols]

                    targets = ["total_cases", "ep_dur", "peak_week"]
                    preds_all = {t: np.zeros(len(df_target_year)) for t in targets}

                    for t in targets:
                        model = models.get(t)
                        if model:
                            preds = model.predict(X_target_all)
                            if t == "total_cases":
                                preds = np.expm1(preds)
                                if bias_correction and t in bias_correction:
                                    preds = preds * bias_correction[t]
                            preds_all[t] = preds

                    if region != "BR":
                        pred_cases_total = np.sum(preds_all["total_cases"])

                        weights = preds_all["total_cases"]
                        if np.sum(weights) > 0:
                            pred_dur_agg = np.sum(
                                preds_all["ep_dur"] * weights
                            ) / np.sum(weights)
                            pred_peak_agg = np.sum(
                                preds_all["peak_week"] * weights
                            ) / np.sum(weights)
                        else:
                            pred_dur_agg = np.mean(preds_all["ep_dur"])
                            pred_peak_agg = np.mean(preds_all["peak_week"])

                        obs_cases_total = df_target_year["total_cases"].sum()
                        obs_weights = df_target_year["total_cases"]
                        if obs_cases_total > 0:
                            obs_dur_agg = (
                                np.sum(df_target_year["ep_dur"] * obs_weights)
                                / obs_cases_total
                            )
                            obs_peak_agg = (
                                np.sum(df_target_year["peak_week"] * obs_weights)
                                / obs_cases_total
                            )
                        else:
                            obs_dur_agg = df_target_year["ep_dur"].mean()
                            obs_peak_agg = df_target_year["peak_week"].mean()

                        err_cases = pred_cases_total - obs_cases_total
                        err_dur = pred_dur_agg - obs_dur_agg
                        err_peak = pred_peak_agg - obs_peak_agg

                        target_pred_text += f"#### State Aggregate ({region})\n"
                        target_pred_text += f"- **Total Cases**: {pred_cases_total:.0f} (Observed: {obs_cases_total:.0f}, Error: {err_cases:+.0f})\n"
                        target_pred_text += f"- **Duration (Weighted)**: {pred_dur_agg:.1f} weeks (Observed: {obs_dur_agg:.1f}, Error: {err_dur:+.1f})\n"
                        target_pred_text += f"- **Peak Week (Weighted)**: {pred_peak_agg:.1f} (Observed: {obs_peak_agg:.1f}, Error: {err_peak:+.1f})\n\n"

                    if city_str:
                        geocode = extract_geocode(city_str)
                        if geocode:
                            city_row_idx = df_target_year.index[
                                df_target_year["geocode"] == int(geocode)
                            ]
                            if not city_row_idx.empty:
                                pos = df_target_year.index.get_loc(city_row_idx[0])
                                target_pred_text += (
                                    f"#### Individual City: {city_str}\n"
                                )
                                for t in ["total_cases", "ep_dur", "peak_week"]:
                                    val = preds_all[t][pos]
                                    obs = df_target_year.loc[city_row_idx[0], t]
                                    err = val - obs
                                    label = t.replace("_", " ").title()
                                    if t == "ep_dur":
                                        label = "Duration"
                                    target_pred_text += f"- **{label}**: {val:.1f} (Observed: {obs:.1f}, Error: {err:+.1f})\n"
                            else:
                                target_pred_text += f"*City {city_str} data not available for {year}.*\n"
                else:
                    target_pred_text += (
                        f"*No data available for year {year} in {region}.*\n"
                    )

                return plots + [fig_imp, fi_text, target_pred_text, metrics_df] + hist_plots + resid_plots + resid_hist_plots

            except Exception as e:
                print(f"Prediction failed: {e}")
                import traceback

                traceback.print_exc()
                empty = go.Figure().update_layout(title=f"Error: {str(e)}")
                return (
                    [empty] * 4
                    + [f"### How Feature Importance is Calculated\n\n*Error occurred: {str(e)}*", "", pd.DataFrame()]
                    + [empty] * 9
                )

        pred_train_btn.click(
            fn=run_prediction,
            inputs=[
                pred_state_dropdown,
                pred_city_dropdown,
                pred_target_year,
                pred_model_type,
                pred_quantile_input,
                pred_min_pop,
            ],
            outputs=[
                pred_plot_size,
                pred_plot_dur,
                pred_plot_peak,
                pred_feat_imp_plot,
                pred_feat_imp_explanation,
                pred_values_output,
                pred_metrics_table,
                pred_hist_size,
                pred_hist_dur,
                pred_hist_peak,
                pred_resid_size,
                pred_resid_dur,
                pred_resid_peak,
                pred_resid_hist_size,
                pred_resid_hist_dur,
                pred_resid_hist_peak,
            ],
        )

        def run_binary_training(region, percentile, model_type, min_pop, test_year):
            """Handler for training the binary classification model."""
            try:
                # Prepare dataset
                df_binary = prepare_binary_dataset(
                    region=region, level="city", min_pop=min_pop
                )

                if df_binary.empty:
                    empty_fig = go.Figure().update_layout(
                        title="Insufficient data for training"
                    )
                    return (
                        None,  # state
                        pd.DataFrame(),
                        "No data available for this selection.",
                        pd.DataFrame(),
                        *[empty_fig] * 9,
                    )

                # Train model
                result = train_binary_model(
                    df=df_binary,
                    test_year=int(test_year),
                    percentile=percentile,
                    model_type=model_type,
                    random_state=42,
                )

                if result["model"] is None:
                    empty_fig = go.Figure().update_layout(
                        title="Training failed (insufficient data)"
                    )
                    return (
                        None,  # state
                        pd.DataFrame(),
                        "Training failed. Insufficient data for the selected test year.",
                        pd.DataFrame(),
                        *[empty_fig] * 9,
                    )

                metrics = result["metrics"]

                # --- Build metrics table ---
                metrics_data = [
                    ["AUC-ROC (CV)", f"{metrics.get('cv_auc_roc', 0):.4f}"],
                    ["AUC-PR (CV)", f"{metrics.get('cv_auc_pr', 0):.4f}"],
                    ["F1-Score (CV)", f"{metrics.get('cv_f1', 0):.4f}"],
                    ["Precision (CV)", f"{metrics.get('cv_precision', 0):.4f}"],
                    ["Recall (CV)", f"{metrics.get('cv_recall', 0):.4f}"],
                    ["Brier Score (CV)", f"{metrics.get('cv_brier', 0):.4f}"],
                    ["Num Folds", f"{metrics.get('n_folds', 0)}"],
                    ["Threshold (cases)", f"{metrics.get('threshold_case_count', 0):.0f}"],
                    ["Optimal threshold (prob)", f"{metrics.get('optimal_prob_threshold', 0):.4f}"],
                    ["Training samples", f"{metrics.get('n_train', 0)}"],
                    ["Positive training", f"{metrics.get('n_positive_train', 0)}"],
                    ["Negative training", f"{metrics.get('n_negative_train', 0)}"],
                    ["scale_pos_weight", f"{metrics.get('scale_pos_weight', 0):.2f}"],
                ]

                if "f1_at_optimal" in metrics:
                    metrics_data.append(["F1 (test, optimal threshold)", f"{metrics['f1_at_optimal']:.4f}"])
                if "auc_roc" in metrics and metrics["auc_roc"] > 0:
                    metrics_data.append(["AUC-ROC (test)", f"{metrics['auc_roc']:.4f}"])
                if "auc_pr" in metrics and metrics["auc_pr"] > 0:
                    metrics_data.append(["AUC-PR (test)", f"{metrics['auc_pr']:.4f}"])
                if "n_test_samples" in metrics:
                    metrics_data.append(["Test samples", f"{metrics['n_test_samples']}"])

                metrics_df = pd.DataFrame(metrics_data, columns=["Metric", "Value"])

                # --- Build threshold info ---
                threshold_text = f"""
                ### Threshold & Balancing

                - **Configured percentile**: {percentile * 100:.0f}th
                - **Case count threshold**: {metrics.get('threshold_case_count', 0):.0f} annual cases
                - **Optimal probability threshold**: {metrics.get('optimal_prob_threshold', 0):.4f}
                - **Balance ratio**: {metrics.get('scale_pos_weight', 0):.2f}

                The optimal threshold was computed via **Youden's J-statistic**, maximizing:
                `J = Sensitivity + Specificity − 1`
                """

                # --- Build CV folds table ---
                cv_results = result.get("cv_results", [])
                if cv_results:
                    cv_df = pd.DataFrame(cv_results)
                    cv_display_cols = ["fold", "auc_roc", "auc_pr", "f1", "precision", "recall", "brier"]
                    cv_display_cols = [c for c in cv_display_cols if c in cv_df.columns]
                    cv_display = cv_df[cv_display_cols].copy()
                else:
                    cv_display = pd.DataFrame({"Info": ["No CV results"]})

                # --- Visualization: Confusion Matrix ---
                if result.get("y_test") is not None and len(result["y_test"]) > 0:
                    y_pred_test = (result["y_prob_test"] >= result["optimal_threshold"]).astype(int)
                    conf_fig = binary_viz_utils.plot_confusion_matrix_binary(
                        result["y_test"], y_pred_test
                    )
                else:
                    conf_fig = go.Figure().update_layout(title="Confusion Matrix (no test data)")

                # --- ROC Curve — use CV out-of-fold predictions with computed AUC ---
                if result.get("fpr") is not None:
                    # Compute AUC from aggregated CV predictions
                    cv_y_true = result.get("cv_y_true")
                    cv_y_prob = result.get("cv_y_prob")
                    if cv_y_true and cv_y_prob and len(np.unique(cv_y_true)) > 1:
                        auc_roc_cv = roc_auc_score(cv_y_true, cv_y_prob)
                    else:
                        auc_roc_cv = metrics.get("cv_auc_roc", 0)

                    roc_fig = binary_viz_utils.plot_roc_curve_binary(
                        result["fpr"], result["tpr"], result["roc_thresholds"],
                        auc_roc=auc_roc_cv,
                        label="CV Out-of-Fold"
                    )
                else:
                    roc_fig = go.Figure().update_layout(title="ROC Curve not available")

                # --- PR Curve — use CV out-of-fold predictions with computed AUC ---
                if result.get("prec") is not None:
                    cv_y_true = result.get("cv_y_true")
                    cv_y_prob = result.get("cv_y_prob")
                    if cv_y_true and cv_y_prob:
                        auc_pr_cv = average_precision_score(cv_y_true, cv_y_prob)
                    else:
                        auc_pr_cv = metrics.get("cv_auc_pr", 0)

                    pr_fig = binary_viz_utils.plot_precision_recall_curve_binary(
                        result["prec"], result["rec"], result["pr_thresholds"],
                        auc_pr=auc_pr_cv,
                        label="CV Out-of-Fold"
                    )
                else:
                    pr_fig = go.Figure().update_layout(title="PR Curve not available")

                # --- Calibration Curve ---
                if result.get("y_train") is not None and result.get("y_prob_train") is not None:
                    cal_fig = binary_viz_utils.plot_calibration_curve_binary(
                        result["y_train"], result["y_prob_train"]
                    )
                else:
                    cal_fig = go.Figure().update_layout(title="Calibration Curve not available")

                # --- Feature Importance ---
                feat_imp_df = get_binary_feature_importance(result)
                if not feat_imp_df.empty:
                    feat_fig = binary_viz_utils.plot_feature_importance_binary(
                        feat_imp_df, top_n=20
                    )
                else:
                    feat_fig = go.Figure().update_layout(title="Feature Importance not available")

                # SHAP is NOT computed here — left to a separate button
                shap_fig = go.Figure().update_layout(
                    title='Click "Compute SHAP Values" button below'
                )

                # --- Threshold Trade-off ---
                if result.get("fpr") is not None:
                    thresh_fig = binary_viz_utils.plot_threshold_tradeoff_binary(
                        result["fpr"], result["tpr"], result["roc_thresholds"],
                        result["optimal_threshold"]
                    )
                else:
                    thresh_fig = go.Figure().update_layout(title="Threshold Trade-off not available")

                # --- CV Folds Metrics ---
                if cv_results:
                    cv_fig = binary_viz_utils.plot_cv_folds_metrics_binary(cv_results)
                else:
                    cv_fig = go.Figure().update_layout(title="Per-Fold Metrics not available")

                # --- Prediction Map ---
                df_test = result.get("df_test")
                if df_test is not None and not df_test.empty and result.get("y_prob_test") is not None:
                    df_pred_map = df_test.copy()
                    df_pred_map["y_prob"] = result["y_prob_test"]
                    pred_map_fig = binary_viz_utils.plot_prediction_map_binary(
                        df_pred_map, threshold=result["optimal_threshold"]
                    )
                else:
                    pred_map_fig = go.Figure().update_layout(
                        title="Prediction Map not available (no test data)"
                    )

                return (
                    result,  # store in state
                    metrics_df,
                    threshold_text,
                    cv_display,
                    conf_fig,
                    roc_fig,
                    pr_fig,
                    cal_fig,
                    feat_fig,
                    shap_fig,
                    thresh_fig,
                    cv_fig,
                    pred_map_fig,
                )

            except Exception as e:
                print(f"Binary training failed: {e}")
                import traceback
                traceback.print_exc()

                empty_fig = go.Figure().update_layout(title=f"Error: {str(e)}")
                return (
                    None,
                    pd.DataFrame(),
                    f"Training error: {str(e)}",
                    pd.DataFrame(),
                    *[empty_fig] * 9,
                )

        binary_train_btn.click(
            fn=run_binary_training,
            inputs=[
                binary_state_dropdown,
                binary_percentile_slider,
                binary_model_type,
                binary_min_pop,
                binary_test_year,
            ],
            outputs=[
                binary_model_state,
                binary_metrics_table,
                binary_threshold_info,
                binary_cv_table,
                binary_conf_matrix,
                binary_roc_plot,
                binary_pr_plot,
                binary_calibration_plot,
                binary_feat_imp_plot,
                binary_shap_plot,
                binary_threshold_plot,
                binary_cv_plot,
                binary_pred_map,
            ],
        )

        def compute_shap_values(model_state):
            """Handler for computing SHAP values separately."""
            if model_state is None:
                empty_fig = go.Figure().update_layout(
                    title='Train a model first, then click "Compute SHAP Values"'
                )
                return empty_fig, "Please train the model before computing SHAP values."

            try:
                shap_data = get_shap_values(model_state)
                if shap_data is not None:
                    shap_fig = binary_viz_utils.plot_shap_summary_binary(
                        shap_data, model_state["feature_cols"], max_display=20
                    )
                    status_text = (
                        f"✅ SHAP values computed successfully.\n\n"
                        f"- **Method**: KernelExplainer (model-agnostic)\n"
                        f"- **Background samples**: {len(shap_data.get('X_background', []))}\n"
                        f"- **Explained samples**: {len(shap_data.get('X_explained', []))}\n"
                        f"- **Features**: {len(shap_data.get('feature_names', []))}"
                    )
                    return shap_fig, status_text
                else:
                    return (
                        go.Figure().update_layout(
                            title="SHAP values could not be computed (check data/model)"
                        ),
                        "❌ SHAP computation failed. Check the console for details.",
                    )
            except Exception as e:
                print(f"SHAP computation failed: {e}")
                import traceback
                traceback.print_exc()
                return (
                    go.Figure().update_layout(title=f"SHAP Error: {str(e)}"),
                    f"❌ SHAP error: {str(e)}",
                )

        binary_shap_btn.click(
            fn=compute_shap_values,
            inputs=[binary_model_state],
            outputs=[binary_shap_plot, binary_shap_status],
        )

        def initial_load():
            default_state = "SP" if "SP" in states else (states[0] if states else None)
            return [
                gr.update(value=default_state),
                gr.update(value="BR"),
            ]

        def load_overview_plots(state):
            base_state = None if state == "BR" else state

            if base_state:
                map_fig = plot_map(base_state)
                hist_fig = plot_alpha_histogram(base_state)
                incidence_fig = plot_alpha_vs_incidence(base_state)
                trend_fig = plot_trend_map(base_state)
                moran_fig = plot_local_moran(base_state)
            else:
                map_fig = plot_map(None)
                hist_fig = plot_alpha_histogram(None)
                incidence_fig = plot_alpha_vs_incidence(None)
                trend_fig = go.Figure().update_layout(
                    title="Select a state for trend map"
                )
                moran_fig = go.Figure().update_layout(
                    title="Select a state for Moran clustering"
                )

            return (
                map_fig,
                incidence_fig,
                trend_fig,
                moran_fig,
                hist_fig,
            )

        def load_city_tab(city_str):
            if not city_str:
                empty_figs = [go.Figure() for _ in range(5)]
                return (
                    *empty_figs,
                    go.Figure(),
                    go.Figure(),
                    go.Figure(),
                    pd.DataFrame(),
                    pd.DataFrame(),
                )

            indicator = plot_indicator_plots(city_str)
            if not isinstance(indicator, list) or len(indicator) != 5:
                indicator = [go.Figure() for _ in range(5)]

            fit_fig = plot_fit(city_str)
            trend_fig = plot_yearly_trend(city_str)
            ts_fig = plot_timeseries(city_str)
            details = get_city_details(city_str)
            yearly = get_yearly_details(city_str)

            return (
                *indicator,
                fit_fig or go.Figure(),
                trend_fig or go.Figure(),
                ts_fig or go.Figure(),
                details if details is not None else pd.DataFrame(),
                yearly if yearly is not None else pd.DataFrame(),
            )

        def load_episcanner_tab(region):
            combined_size = plot_combined_episcanner_fit(region, "total_cases")
            combined_dur = plot_combined_episcanner_fit(region, "ep_dur")
            disp_size = plot_episcanner_dispersion_alpha(region, "total_cases")
            disp_dur = plot_episcanner_dispersion_alpha(region, "ep_dur")
            state_map_size = plot_episcanner_state_map("total_cases")
            state_map_dur = plot_episcanner_state_map("ep_dur")
            details_size = get_episcanner_fit_results(region, "total_cases")
            details_dur = get_episcanner_fit_results(region, "ep_dur")
            timeseries = plot_episcanner_region_timeseries(region)
            residuals = plot_state_residuals_vs_alpha(region)

            return (
                combined_size,
                combined_dur,
                disp_size,
                disp_dur,
                state_map_size,
                state_map_dur,
                details_size if details_size is not None else pd.DataFrame(),
                details_dur if details_dur is not None else pd.DataFrame(),
                timeseries,
                residuals,
            )

        demo.load(
            fn=initial_load,
            inputs=[],
            outputs=[
                state_dropdown_overview,
                epi_region_dropdown,
            ],
        )

        state_dropdown_overview.change(
            fn=load_overview_plots,
            inputs=[state_dropdown_overview],
            outputs=[
                map_plot,
                alpha_incidence_plot,
                trend_map_plot,
                moran_plot,
                alpha_hist,
            ],
        )

        def on_state_city_change(state):
            base_state = None if state == "BR" else state
            choices, best = get_city_options(base_state)
            city_update = gr.update(choices=choices, value=best)
            if best:
                city_data = load_city_tab(best)
            else:
                city_data = [gr.update() for _ in range(10)]
            return [city_update] + list(city_data)

        state_dropdown_city.change(
            fn=on_state_city_change,
            inputs=[state_dropdown_city],
            outputs=[
                city_dropdown,
                r0_plot,
                cases_plot,
                ini_plot,
                dur_plot,
                res_plot,
                fit_plot,
                trend_plot,
                timeseries_plot,
                city_details_table,
                yearly_fits_table,
            ],
        )

        plot_btn.click(
            fn=load_city_tab,
            inputs=[city_dropdown],
            outputs=[
                r0_plot,
                cases_plot,
                ini_plot,
                dur_plot,
                res_plot,
                fit_plot,
                trend_plot,
                timeseries_plot,
                city_details_table,
                yearly_fits_table,
            ],
        )

        epi_refresh_btn.click(
            fn=load_episcanner_tab,
            inputs=[epi_region_dropdown],
            outputs=[
                epi_combined_plot_size,
                epi_combined_plot_dur,
                epi_dispersion_plot_size,
                epi_dispersion_plot_dur,
                epi_state_map_size,
                epi_state_map_dur,
                epi_details_table_size,
                epi_details_table_dur,
                epi_timeseries_plot,
                epi_residuals_plot,
            ],
        )

    return demo


if __name__ == "__main__":
    demo = create_dashboard()
    import gradio.networking

    gradio.networking.url_ok = lambda x: True
    os.environ["NO_PROXY"] = "localhost,127.0.0.1"
    demo.launch(
        debug=True,
        server_name="127.0.0.1",
    )
