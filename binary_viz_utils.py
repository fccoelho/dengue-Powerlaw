"""
Visualization utilities for binary classification model evaluation.

All functions return Plotly figures for use in Gradio dashboards.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve


def plot_confusion_matrix_binary(y_true, y_pred, labels=["Epidemic", "Big Epidemic"]):
    """
    Plot confusion matrix as a heatmap.

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_pred : array-like
        Predicted binary labels.
    labels : list
        Class label names.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    cm = confusion_matrix(y_true, y_pred)

    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 20},
        colorscale="Blues",
        colorbar=dict(title="Count"),
    ))

    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        template="plotly_white",
        height=400,
    )

    return fig


def plot_roc_curve_binary(fpr, tpr, thresholds, auc_roc, label="Model"):
    """
    Plot ROC curve with AUC annotation.

    Parameters
    ----------
    fpr, tpr, thresholds : array-like
        From sklearn roc_curve.
    auc_roc : float
        AUC-ROC score.
    label : str
        Label for the curve.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=fpr,
        y=tpr,
        mode="lines",
        name=f"{label} (AUC = {auc_roc:.3f})",
        line=dict(color="darkblue", width=2),
        fill="tozeroy",
        fillcolor="rgba(0, 100, 200, 0.1)",
    ))

    # Diagonal line (random classifier)
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        name="Random Classifier",
        line=dict(dash="dash", color="gray", width=1),
    ))

    fig.update_layout(
        title="ROC Curve (Receiver Operating Characteristic)",
        xaxis_title="False Positive Rate (1 − Specificity)",
        yaxis_title="True Positive Rate (Sensitivity)",
        template="plotly_white",
        height=450,
        showlegend=True,
    )

    return fig


def plot_precision_recall_curve_binary(prec, rec, thresholds, auc_pr, label="Model"):
    """
    Plot Precision-Recall curve with AUC-PR annotation.

    Parameters
    ----------
    prec, rec, thresholds : array-like
        From sklearn precision_recall_curve.
    auc_pr : float
        Average Precision Score (AUC-PR).
    label : str
        Label for the curve.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=rec,
        y=prec,
        mode="lines",
        name=f"{label} (AUC-PR = {auc_pr:.3f})",
        line=dict(color="darkgreen", width=2),
        fill="tozeroy",
        fillcolor="rgba(0, 150, 50, 0.1)",
    ))

    fig.update_layout(
        title="Precision-Recall Curve",
        xaxis_title="Recall (Sensitivity)",
        yaxis_title="Precision (Positive Predictive Value)",
        template="plotly_white",
        height=450,
        showlegend=True,
    )

    return fig


def plot_calibration_curve_binary(y_true, y_prob, n_bins=10):
    """
    Plot calibration curve (reliability diagram).

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_prob : array-like
        Predicted probabilities.
    n_bins : int
        Number of bins for calibration.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = []
    bin_frac_pos = []
    bin_counts = []

    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        mask = (y_prob >= low) & (y_prob < high)
        if i == n_bins - 1:  # Include right edge for last bin
            mask = (y_prob >= low) & (y_prob <= high)

        if mask.sum() > 0:
            bin_centers.append((low + high) / 2)
            bin_frac_pos.append(y_true[mask].mean())
            bin_counts.append(mask.sum())
        else:
            bin_centers.append((low + high) / 2)
            bin_frac_pos.append(np.nan)
            bin_counts.append(0)

    fig = go.Figure()

    # Perfect calibration line
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        name="Perfect Calibration",
        line=dict(dash="dash", color="gray", width=2),
    ))

    # Actual calibration
    fig.add_trace(go.Scatter(
        x=bin_centers,
        y=bin_frac_pos,
        mode="lines+markers",
        name="Model Calibration",
        line=dict(color="darkorange", width=2),
        marker=dict(size=8, color="darkorange"),
    ))

    fig.update_layout(
        title="Calibration Curve (Reliability Diagram)",
        xaxis_title="Mean Predicted Probability",
        yaxis_title="Fraction of Positives",
        template="plotly_white",
        height=450,
        showlegend=True,
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
    )

    return fig


def plot_feature_importance_binary(importance_df, top_n=20):
    """
    Plot horizontal bar chart of feature importance.

    Parameters
    ----------
    importance_df : pd.DataFrame
        DataFrame with 'Feature' and 'Importance' columns.
    top_n : int
        Number of top features to display.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if importance_df.empty:
        return go.Figure().update_layout(title="Feature Importance not available")

    df_sorted = importance_df.head(top_n).sort_values("Importance", ascending=True)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=df_sorted["Feature"],
        x=df_sorted["Importance"],
        orientation="h",
        marker_color="steelblue",
        name="Importance",
    ))

    fig.update_layout(
        title=f"Top {top_n} Features — Importance",
        xaxis_title="Importance (Gini / Split-based)",
        yaxis_title="",
        template="plotly_white",
        height=max(400, len(df_sorted) * 25),
        showlegend=False,
    )

    return fig


def plot_shap_summary_binary(shap_data, feature_names, max_display=20):
    """
    Plot SHAP summary as a horizontal bar chart with color-coded values.

    Parameters
    ----------
    shap_data : dict
        Output of get_shap_values().
    feature_names : list
        Feature column names.
    max_display : int
        Max features to display.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if shap_data is None:
        return go.Figure().update_layout(title="SHAP values not available")

    shap_values = shap_data["shap_values"]
    X_explained = shap_data["X_explained"]

    # Compute mean absolute SHAP value per feature
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    # Create DataFrame and sort
    df_shap = pd.DataFrame({
        "Feature": feature_names,
        "MeanAbsSHAP": mean_abs_shap,
    }).sort_values("MeanAbsSHAP", ascending=True).tail(max_display)

    top_features = df_shap["Feature"].tolist()
    top_indices = [feature_names.index(f) for f in top_features]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=df_shap["Feature"],
        x=df_shap["MeanAbsSHAP"],
        orientation="h",
        marker_color="indianred",
        name="|SHAP| médio",
    ))

    fig.update_layout(
        title=f"SHAP Summary — Top {len(top_features)} Features",
        xaxis_title="Mean |SHAP| Value (Model Impact)",
        yaxis_title="",
        template="plotly_white",
        height=max(400, len(top_features) * 25),
        showlegend=False,
    )

    return fig


def plot_threshold_tradeoff_binary(fpr, tpr, thresholds, optimal_threshold):
    """
    Plot sensitivity and specificity vs threshold to visualize trade-off.

    Parameters
    ----------
    fpr, tpr, thresholds : array-like
        From sklearn roc_curve.
    optimal_threshold : float
        Optimal threshold from Youden's J.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    specificity = 1 - fpr

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=thresholds,
        y=tpr,
        mode="lines",
        name="Sensitivity (TPR)",
        line=dict(color="green", width=2),
    ))

    fig.add_trace(go.Scatter(
        x=thresholds,
        y=specificity,
        mode="lines",
        name="Specificity (TNR)",
        line=dict(color="blue", width=2),
    ))

    # Mark optimal threshold
    fig.add_vline(
        x=optimal_threshold,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Optimal: {optimal_threshold:.3f}",
        annotation_position="top right",
    )

    fig.update_layout(
        title="Sensitivity vs Specificity Trade-off by Threshold",
        xaxis_title="Classification Threshold",
        yaxis_title="Rate",
        template="plotly_white",
        height=400,
        showlegend=True,
    )

    return fig


def plot_prediction_map_binary(df_predictions, geocode_col="geocode", prob_col="y_prob",
                                threshold=0.5, city_name_col="city_name"):
    """
    Plot a bar chart of predicted probabilities per city, sorted descending.

    Parameters
    ----------
    df_predictions : pd.DataFrame
        DataFrame with geocode and prediction columns.
    geocode_col : str
        Column name for geocode.
    prob_col : str
        Column name for predicted probabilities.
    threshold : float
        Classification threshold.
    city_name_col : str, optional
        Column name for city names (for x-axis and hover).

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if df_predictions.empty or prob_col not in df_predictions.columns:
        return go.Figure().update_layout(title="Prediction data not available")

    df = df_predictions.copy()

    # Sort by probability descending
    df = df.sort_values(prob_col, ascending=False).reset_index(drop=True)

    df["class"] = (df[prob_col] >= threshold).astype(int)
    df["class_label"] = df["class"].map({0: "Epidemic", 1: "Big Epidemic"})

    # Use city name if available, otherwise fall back to geocode
    if city_name_col and city_name_col in df.columns:
        df["display_name"] = df[city_name_col].astype(str)
    else:
        df["display_name"] = df[geocode_col].astype(str)

    # Build hover text
    has_year = "year" in df.columns

    def _build_hover(row):
        txt = f"City: {row['display_name']}<br>"
        if has_year:
            txt += f"Year: {int(row['year'])}<br>"
        txt += f"Geocode: {row[geocode_col]}<br>"
        txt += f"Probability: {row[prob_col]:.3f}<br>"
        txt += f"Class: {row['class_label']}"
        return txt

    df["hover_text"] = df.apply(_build_hover, axis=1)

    colors = df["class"].map({0: "steelblue", 1: "indianred"})

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=df["display_name"],
        y=df[prob_col],
        marker_color=colors,
        hovertext=df["hover_text"],
        hoverinfo="text",
        name="Prediction",
    ))

    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color="gray",
        annotation_text=f"Threshold = {threshold:.2f}",
        annotation_position="right",
    )

    fig.update_layout(
        title="Prediction Probabilities by City (sorted descending)",
        xaxis_title="City",
        yaxis_title="Probability of Being Epidemic",
        template="plotly_white",
        height=450,
        showlegend=False,
        yaxis=dict(range=[0, 1.05]),
        xaxis_tickangle=-45,
    )

    return fig


def plot_cv_folds_metrics_binary(cv_results):
    """
    Plot per-fold metrics as grouped bar chart.

    Parameters
    ----------
    cv_results : list of dict
        Output cv_results from train_binary_model().

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if not cv_results:
        return go.Figure().update_layout(title="CV results not available")

    df_cv = pd.DataFrame(cv_results)
    folds = df_cv["fold"].astype(str).tolist()

    metrics_to_plot = ["auc_roc", "auc_pr", "f1", "precision", "recall", "brier"]
    labels = ["AUC-ROC", "AUC-PR", "F1-Score", "Precision", "Recall", "Brier Score"]

    fig = go.Figure()

    for metric, label in zip(metrics_to_plot, labels):
        if metric in df_cv.columns:
            fig.add_trace(go.Bar(
                name=label,
                x=folds,
                y=df_cv[metric].values,
            ))

    fig.update_layout(
        title="Per-Fold Metrics (TimeSeriesSplit)",
        xaxis_title="Fold",
        yaxis_title="Metric Value",
        template="plotly_white",
        height=400,
        barmode="group",
        showlegend=True,
    )

    return fig
