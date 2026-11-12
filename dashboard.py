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
    plot_epidemic_years_map,
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
                        with gr.Row():
                            map_plot = gr.Plot(label="Alpha Parameter Map (Weekly Cases)")
                            epi_years_map_plot = gr.Plot(label="Epidemic Years by City")
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

                    with gr.Column(scale=4):
                        with gr.Row():
                            epi_combined_plot_size = gr.Plot(
                                label="Size: Combined Fit (Total Cases)"
                            )

                        with gr.Row():
                            epi_dispersion_plot_size = gr.Plot(
                                label="Size: Dispersion vs Alpha (Total Cases)"
                            )

                        with gr.Row():
                            epi_state_map_size = gr.Plot(
                                label="Epidemic Size Alpha Map (Total Cases)"
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

                $$Q_{\\tau}(Y_t \\mid X_{t-1}) = f(X_{t-1})$$
                """)
                gr.Markdown(
                    r"""
                > **Estimator**: quantile regression on a cubic spline basis with linear
                boundary extrapolation (Koenker & Bassett 1978), fitted on the
                $\log(1+S)$ scale -- so outside the observed range the boundary
                segment continues linearly, which is a multiplicative extrapolation
                on the case scale. Minimises the pinball loss at the chosen
                quantile level $\tau$.

                **Target Variable ($Y_t$):**

                $S_t$: Epidemic Size (Total Cases) -- the only target.

                **Predictors ($X_{t-1}$), all measured in year $t-1$:**

                1. **Lagged Epidemic Metrics**: $S_{t-1}$ (size), $D_{t-1}$ (epidemic duration, weeks), $P_{t-1}$ (peak week)
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
                                ("Quantile regression (spline basis)", "spline_quantile"),
                            ],
                            value="spline_quantile",
                            label="Model Type",
                        )

                        pred_quantile_input = gr.Slider(
                            minimum=0.05,
                            maximum=0.95,
                            step=0.05,
                            value=0.5,
                            label="Quantile Level (τ) -- pinball loss",
                            visible=True,
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

                        gr.Markdown("### Residual Analysis")
                        with gr.Row():
                            pred_resid_size = gr.Plot(
                                label="Residuals vs Predicted: Total Cases"
                            )

                        with gr.Row():
                            pred_resid_hist_size = gr.Plot(
                                label="Residual Distribution: Total Cases"
                            )

            tab_binary_predictive = gr.TabItem(
                "Predicting Large Epidemics", id="binary_predictive"
            )
            with tab_binary_predictive:
                gr.Markdown("""
                ## 🎯 Size Classification Model — Epidemic Forecasting

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
                - **Temporal Validation**: year-based forward chaining (expanding window, one held-out
  calendar year per fold); no fold trains on a year at or after its validation year
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
                            label="Per-Fold Results (held-out year)",
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

        def run_prediction(region, city_str, year, model_type, quantile, min_pop):
            """Train the manuscript's primary estimator and forecast epidemic size.

            Epidemic size is the only target; previous-season epidemic duration
            and peak week enter the model as *predictors* (covariates of the
            coming season's magnitude), alongside the lagged size, R0, the
            scaling block, log-population and the quarterly climate aggregates.
            """
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
                    empty = go.Figure().update_layout(title="No data for training")
                    return (
                        empty, empty, "No data available for this selection.",
                        "Run prediction to see values.", pd.DataFrame(),
                        empty, empty, empty,
                    )

                models, metrics, feature_cols = (
                    predictive_model.train_size_quantile_model(
                        df_all, test_year=year, tau=quantile
                    )
                )

                if not models:
                    empty = go.Figure().update_layout(
                        title="Training failed (no usable rows)"
                    )
                    return (
                        empty, empty, "Training failed: no usable rows.",
                        "Run prediction to see values.", pd.DataFrame(),
                        empty, empty, empty,
                    )

                est = models["total_cases"]
                m = metrics.get("total_cases", {})

                # point predictions on every panel row; the pipeline imputes
                # missing feature medians internally
                from run_ablation import predict_cases

                y_pred = predict_cases(est, df_all[feature_cols], "total_cases")
                y_true = df_all["total_cases"].to_numpy(dtype=float)

                is_test = (df_all["year"] >= year).to_numpy()
                is_train = ~is_test

                rel_error = pd.Series(
                    np.where(
                        y_true > 0,
                        np.abs(y_true - y_pred) / np.where(y_true > 0, y_true, 1),
                        np.nan,
                    ),
                    index=df_all.index,
                )
                high_error_mask = rel_error > 0.20

                # --- Predicted vs Actual (log-log: sizes span three decades) ---
                hover = []
                for i, row in df_all.iterrows():
                    txt = (
                        f"Year: {int(row['year'])} "
                        f"({'Test' if is_test[i] else 'Train'})<br>"
                    )
                    if "city_name" in df_all.columns:
                        txt += f"City: {row['city_name']} ({int(row['geocode'])})"
                    elif "state" in df_all.columns:
                        txt += f"State: {row['state']}"
                    e = rel_error.loc[i]
                    if np.isfinite(e):
                        txt += f"<br>Error: {e * 100:.1f}%"
                    hover.append(txt)

                fig_pred = go.Figure()
                if is_train.any():
                    fig_pred.add_trace(go.Scatter(
                        x=y_true[is_train],
                        y=y_pred[is_train],
                        mode="markers",
                        name="Train",
                        marker=dict(size=8, color="steelblue", opacity=0.6),
                        hovertext=[h for h, t in zip(hover, is_train) if t],
                        hoverinfo="text+x+y",
                    ))
                if is_test.any():
                    fig_pred.add_trace(go.Scatter(
                        x=y_true[is_test],
                        y=y_pred[is_test],
                        mode="markers",
                        name="Test",
                        marker=dict(size=10, symbol="diamond", color="darkgreen",
                                    opacity=0.8),
                        hovertext=[h for h, t in zip(hover, is_test) if t],
                        hoverinfo="text+x+y",
                    ))
                positive = y_true > 0
                lo = min(np.nanmin(y_true[positive]), np.nanmin(y_pred[y_pred > 0]))
                hi = max(np.nanmax(y_true), np.nanmax(y_pred))
                fig_pred.add_trace(go.Scatter(
                    x=[lo, hi], y=[lo, hi], mode="lines",
                    line=dict(dash="dash", color="gray"), name="Ideal",
                ))
                fig_pred.update_xaxes(type="log")
                fig_pred.update_yaxes(type="log")
                fig_pred.update_layout(
                    title="Epidemic size: Actual vs Predicted (Log-Log)",
                    xaxis_title="Actual (cases)",
                    yaxis_title="Predicted (cases)",
                    template="plotly_white",
                )

                # --- Feature importance: permutation on the held-out seasons ---
                df_test_ok = df_all[df_all["year"] >= year].dropna(
                    subset=["total_cases"]
                )
                if len(df_test_ok) >= 5:
                    from sklearn.inspection import permutation_importance

                    y_log = np.log1p(df_test_ok["total_cases"].to_numpy(dtype=float))
                    pi = permutation_importance(
                        est, df_test_ok[feature_cols], y_log,
                        n_repeats=5, random_state=42,
                        scoring="neg_mean_absolute_error",
                    )
                    imp_df = (
                        pd.DataFrame({
                            "Feature": [
                                c.replace("prev_", "") for c in feature_cols
                            ],
                            "Importance": pi.importances_mean,
                        })
                        .sort_values("Importance", ascending=False)
                    )
                    imp_df = imp_df[imp_df["Importance"] > 0]
                else:
                    imp_df = pd.DataFrame({"Feature": [], "Importance": []})

                fig_imp = go.Figure()
                if not imp_df.empty:
                    fig_imp.add_trace(go.Bar(
                        y=imp_df["Feature"],
                        x=imp_df["Importance"],
                        orientation="h",
                        marker_color="#4c78a8",
                    ))
                fig_imp.update_layout(
                    title="Feature Importance (permutation, held-out seasons)",
                    xaxis_title="mean increase in MAE when shuffled (log scale)",
                    template="plotly_white",
                    height=500,
                )

                fi_text = r"""
### How Feature Importance is Calculated

- **Model**: quantile regression on a cubic spline basis with linear boundary
  extrapolation (the manuscript's primary estimator, Section 2.8), fitted on
  $\log(1+S)$ at the chosen $\\tau$; predictions are back-transformed with
  $\exp(\\cdot)-1$.
- **Method**: permutation importance on the held-out seasons. Each predictor's
  column is randomly shuffled and the resulting increase in mean absolute error
  (log scale) is recorded, averaged over 5 repeats.
- **Interpretation**: a larger value means the forecast degrades when the
  predictor carries no information. Previous-season **epidemic duration**
  (`ep_dur`) and **peak week** (`peak_week`) are among these predictors -- they
  are covariates of the coming season's size and are never modelled as outcomes
  themselves.
"""

                def _fmt_reg(val, spec=".2f"):
                    """NaN-safe metric formatting: NaN/None means 'not measured'."""
                    if val is None or (isinstance(val, float) and not np.isfinite(val)):
                        return "N/A"
                    return f"{val:{spec}}"

                metrics_df = pd.DataFrame(
                    [[
                        "Epidemic size (total cases)",
                        _fmt_reg(m.get("MAE"), ".1f"),
                        _fmt_reg(m.get("RMSE"), ".1f"),
                        _fmt_reg(m.get("Pinball_Loss"), ".1f"),
                        f"{quantile:.2f}",
                        f"{m.get('n_train', 0):,}",
                        f"{m.get('n_test', 0):,}",
                    ]],
                    columns=[
                        "Target", "MAE (cases)", "RMSE (cases)",
                        "Pinball Loss (τ)", "τ", "Train rows", "Test rows",
                    ],
                )

                # --- Forecast for the selected year (size only) ---
                target_pred_text = f"### Forecast for Year {year}\n"
                df_target_year = df_all[df_all["year"] == year].copy()

                if not df_target_year.empty:
                    preds_year = predict_cases(
                        est, df_target_year[feature_cols], "total_cases"
                    )
                    pred_cases_total = float(np.sum(preds_year))
                    obs_cases_total = float(df_target_year["total_cases"].sum())
                    target_pred_text += (
                        f"#### State Aggregate ({region})\n"
                        f"- **Total Cases**: {pred_cases_total:.0f} "
                        f"(Observed: {obs_cases_total:.0f}, "
                        f"Error: {pred_cases_total - obs_cases_total:+.0f})\n\n"
                    )

                    if city_str:
                        geocode = extract_geocode(city_str)
                        if geocode:
                            city_idx = df_target_year.index[
                                df_target_year["geocode"] == int(geocode)
                            ]
                            if not city_idx.empty:
                                pos = city_idx[0]
                                p = float(
                                    preds_year[df_target_year.index.get_loc(pos)]
                                )
                                o = float(df_target_year.loc[pos, "total_cases"])
                                target_pred_text += (
                                    f"#### Individual City: {city_str}\n"
                                    f"- **Epidemic Size**: {p:.1f} "
                                    f"(Observed: {o:.1f}, Error: {p - o:+.1f})\n"
                                )
                            else:
                                target_pred_text += (
                                    f"*City {city_str} data not available for {year}.*\n"
                                )
                else:
                    target_pred_text += (
                        f"*No data available for year {year} in {region}.*\n"
                    )

                # --- Error histogram: population distribution of >20% errors ---
                if "population" in df_all.columns:
                    high_error_pop = df_all.loc[
                        high_error_mask.fillna(False), "population"
                    ]
                    fig_hist = go.Figure()
                    if not high_error_pop.empty:
                        fig_hist.add_trace(go.Histogram(
                            x=high_error_pop,
                            name="Pop. Size",
                            marker_color="red",
                        ))
                        fig_hist.update_layout(
                            title="Pop. Distribution of Cities with >20% Error",
                            xaxis_title="Population Size",
                            yaxis_title="Count",
                            template="plotly_white",
                        )
                    else:
                        fig_hist.update_layout(
                            title="No cities with >20% error"
                        )
                else:
                    fig_hist = go.Figure().update_layout(
                        title="Population data not available"
                    )

                # --- Residual analysis ---
                residuals = pd.Series(y_true - y_pred, index=df_all.index)

                fig_resid = go.Figure()
                fig_resid.add_trace(go.Scatter(
                    x=y_pred,
                    y=residuals,
                    mode="markers",
                    marker=dict(color="steelblue", opacity=0.5),
                    hovertext=hover,
                    hoverinfo="text+x+y",
                ))
                fig_resid.add_hline(y=0, line_dash="dash", line_color="red")
                mean_resid = float(residuals.mean())
                fig_resid.add_vline(
                    x=0, line_dash="dot", line_color="green",
                    annotation_text="Zero",
                )
                fig_resid.update_layout(
                    title="Residuals vs Predicted: Total Cases",
                    xaxis_title="Predicted",
                    yaxis_title="Residual (Actual - Predicted)",
                    template="plotly_white",
                )

                fig_resid_hist = go.Figure()
                fig_resid_hist.add_trace(go.Histogram(
                    x=residuals,
                    name="Residuals",
                    marker_color="steelblue",
                    opacity=0.7,
                ))
                fig_resid_hist.add_vline(
                    x=0, line_dash="dash", line_color="green",
                    annotation_text="Zero",
                )
                fig_resid_hist.add_vline(
                    x=mean_resid, line_dash="dot", line_color="red",
                    annotation_text=f"Mean: {mean_resid:.1f}",
                )
                fig_resid_hist.update_layout(
                    title="Residual Distribution: Total Cases",
                    xaxis_title="Residual",
                    yaxis_title="Count",
                    template="plotly_white",
                )

                return (
                    fig_pred,
                    fig_imp,
                    fi_text,
                    target_pred_text,
                    metrics_df,
                    fig_hist,
                    fig_resid,
                    fig_resid_hist,
                )

            except Exception as e:
                print(f"Prediction failed: {e}")
                import traceback

                traceback.print_exc()
                empty = go.Figure().update_layout(title=f"Error: {str(e)}")
                return (
                    empty, empty,
                    f"### Error\n\n*{str(e)}*",
                    "Prediction failed.", pd.DataFrame(),
                    empty, empty, empty,
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
                pred_feat_imp_plot,
                pred_feat_imp_explanation,
                pred_values_output,
                pred_metrics_table,
                pred_hist_size,
                pred_resid_size,
                pred_resid_hist_size,
            ],
        )

        def run_binary_training(region, percentile, model_type, min_pop, test_year):
            """Handler for training the size classification model."""
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

                # NaN-safe formatting: a missing metric is reported as an
                # em dash, never as 0 (absence must stay distinguishable from
                # a terrible score).
                def _fmt(key, spec=".4f", default=None):
                    val = metrics.get(key, default)
                    if val is None or (isinstance(val, float) and not np.isfinite(val)):
                        return "—"
                    return f"{val:{spec}}"

                # --- Build metrics table ---
                metrics_data = [
                    ["AUC-ROC (CV)", _fmt("cv_auc_roc")],
                    ["AUC-PR (CV)", _fmt("cv_auc_pr")],
                    ["F1-Score (CV, 0.5 rule)", _fmt("cv_f1_at_0.5")],
                    ["Precision (CV)", _fmt("cv_precision")],
                    ["Recall (CV)", _fmt("cv_recall")],
                    ["Brier Score (CV)", _fmt("cv_brier")],
                    ["Num Folds", f"{metrics.get('n_folds', 0)}"],
                    ["Threshold (cases)", _fmt("threshold_case_count", ".0f", 0)],
                    ["Optimal threshold (prob)", _fmt("optimal_prob_threshold")],
                    ["Training samples", f"{metrics.get('n_train', 0)}"],
                    ["Positive training", f"{metrics.get('n_positive_train', 0)}"],
                    ["Negative training", f"{metrics.get('n_negative_train', 0)}"],
                    ["scale_pos_weight", _fmt("scale_pos_weight", ".2f")],
                ]

                if "f1_at_youden" in metrics:
                    metrics_data.append(["F1 (test, Youden threshold)", _fmt("f1_at_youden")])
                    metrics_data.append(["F1 (test, 0.5 rule)", _fmt("f1_at_0.5")])
                if "auc_roc" in metrics and pd.notna(metrics["auc_roc"]):
                    metrics_data.append(["AUC-ROC (test)", _fmt("auc_roc")])
                if "auc_pr" in metrics and pd.notna(metrics["auc_pr"]):
                    metrics_data.append(["AUC-PR (test)", _fmt("auc_pr")])
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
                    cv_display_cols = [
        "fold", "val_year", "auc_roc", "auc_pr",
        "f1_at_0.5", "precision", "recall", "brier",
    ]
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
                epi_years_fig = plot_epidemic_years_map(base_state)
                hist_fig = plot_alpha_histogram(base_state)
                incidence_fig = plot_alpha_vs_incidence(base_state)
                trend_fig = plot_trend_map(base_state)
                moran_fig = plot_local_moran(base_state)
            else:
                map_fig = plot_map(None)
                epi_years_fig = plot_epidemic_years_map(None)
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
                epi_years_fig,
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
            # Only epidemic size is fitted as a power law. Duration was
            # removed as a scaling object -- a season is bounded by the length
            # of the year, so a scale-free law cannot apply to it a priori --
            # and the database no longer stores metric='ep_dur' fits.
            combined_size = plot_combined_episcanner_fit(region, "total_cases")
            disp_size = plot_episcanner_dispersion_alpha(region, "total_cases")
            state_map_size = plot_episcanner_state_map("total_cases")
            details_size = get_episcanner_fit_results(region, "total_cases")
            timeseries = plot_episcanner_region_timeseries(region)
            residuals = plot_state_residuals_vs_alpha(region)

            return (
                combined_size,
                disp_size,
                state_map_size,
                details_size if details_size is not None else pd.DataFrame(),
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
                epi_years_map_plot,
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
                epi_dispersion_plot_size,
                epi_state_map_size,
                epi_details_table_size,
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
