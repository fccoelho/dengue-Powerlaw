"""
Binary classification pipeline for predicting epidemic threshold exceedance.

Predicts whether total annual dengue cases in a geographic unit will exceed
a configurable historical percentile (e.g., 75th, 90th, 95th).

Uses year-based forward-chaining cross-validation (see ``temporal_split``) so
that no fold trains on a year at or after its validation year, with threshold
optimization via Youden's J-statistic.
"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    brier_score_loss,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

from predictive_model import prepare_lagged_data
from temporal_split import (
    assert_forward_in_time,
    summarize_folds,
    year_forward_chain_folds,
)
from data_utils import GEO_STATE_MAP
try:
    from _mun_by_geocode import NAME_BY_GEOCODE
except ImportError:
    NAME_BY_GEOCODE = {}


def compute_percentile_threshold(train_cases, percentile=0.75):
    """
    Compute the case count threshold from training data at the given percentile.

    Parameters
    ----------
    train_cases : array-like
        Historical total cases from training folds only.
    percentile : float
        Percentile threshold (0-1), default 0.75.

    Returns
    -------
    float
        Case count threshold.
    """
    return float(np.percentile(train_cases, percentile * 100))


def create_binary_target(cases, threshold):
    """
    Create binary target: 1 if cases > threshold, 0 otherwise.
    """
    return (np.asarray(cases) > threshold).astype(int)


def prepare_binary_dataset(region="BR", level="city", min_pop=0):
    """
    Prepare dataset for binary classification.

    Reuses prepare_lagged_data() and adds binary target column.
    The threshold is NOT computed here — it is computed per-fold
    during cross-validation to prevent leakage.

    Parameters
    ----------
    region : str
        "BR" for all Brazil or state abbreviation (e.g., "SP").
    level : str
        "city" or "state".
    min_pop : int
        Minimum city population filter.

    Returns
    -------
    pd.DataFrame
        Dataset with features + 'total_cases' + metadata columns.
    """
    df = prepare_lagged_data(region, target_year=None, level=level)

    if df.empty:
        return df

    # Filter by minimum population
    if min_pop > 0 and "population" in df.columns:
        df = df[df["population"] >= min_pop].copy()

    return df


# ---------------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------------
# predictive_model.get_quarterly_climate() emits avg_temp_Q1..Q4 and
# avg_umid_Q1..Q4 for the *target* year, which prepare_lagged_data() then
# shifts into prev_avg_* columns. Only the shifted columns are knowable when
# the prediction is made; the unshifted ones are contemporaneous with the
# outcome and must never enter the feature matrix.
_CONTEMPORANEOUS_CLIMATE_PREFIXES = ("avg_temp", "avg_umid")

# Non-lagged columns that are safe to use: static, or derived from years < T.
_STATIC_FEATURES = ("log_pop",)


def _is_contemporaneous_climate(col):
    """True for target-year climate aggregates such as 'avg_temp_Q3'."""
    return col.startswith(_CONTEMPORANEOUS_CLIMATE_PREFIXES)


def _get_feature_columns(df, expected=None):
    """
    Identify the feature columns of the lagged dataset, in a stable order.

    Only lagged (``prev_*``) and static (``log_pop``) columns are eligible.
    The unshifted climate aggregates describe the target year and would leak
    the outcome, so they are excluded; duplicates are dropped preserving
    order. This is the single source of truth for the feature matrix — both
    training and scoring go through it, so the column order cannot drift.

    Parameters
    ----------
    df : pd.DataFrame
        Lagged dataset from prepare_lagged_data().
    expected : sequence of str, optional
        Authoritative feature list, e.g. the one stored in the training
        result. When supplied it is returned verbatim after checking that
        every column is available, which guarantees scoring uses the same
        columns *in the same order* as training.

    Returns
    -------
    list of str

    Raises
    ------
    KeyError
        If ``expected`` is given and some of its columns are missing from df.
    """
    if expected is not None:
        missing = [c for c in expected if c not in df.columns]
        if missing:
            raise KeyError(
                f"Cannot build the feature matrix: {len(missing)} training "
                f"feature(s) absent from the frame: {missing}"
            )
        return list(expected)

    candidates = [c for c in df.columns if c.startswith("prev_")]
    candidates += [c for c in _STATIC_FEATURES if c in df.columns]

    feature_cols = []
    seen = set()
    for col in candidates:
        if col in seen or _is_contemporaneous_climate(col):
            continue
        if df[col].isnull().all():
            continue
        seen.add(col)
        feature_cols.append(col)
    return feature_cols


def _get_numerical_and_categorical_features(feature_cols, df):
    """
    Separate features into numerical and categorical lists.

    Note: currently unused by the training pipeline (all features are
    numerical lags and reach the classifiers raw -- see build_model_pipeline).
    Kept for future categorical support.
    """
    # In current pipeline, all features are numerical (lags of metrics)
    # If categorical features are added later, this handles them.
    numerical = []
    categorical = []
    for col in feature_cols:
        if col in df.columns:
            if df[col].dtype in ["object", "category", "bool"]:
                categorical.append(col)
            else:
                numerical.append(col)
    return numerical, categorical


def compute_optimal_threshold(fpr, tpr, thresholds):
    """
    Compute optimal threshold using Youden's J-statistic.

    J = sensitivity + specificity - 1 = tpr - fpr

    Parameters
    ----------
    fpr, tpr, thresholds : array-like
        From sklearn roc_curve.

    Returns
    -------
    float
        Optimal threshold maximizing Youden's J.
    """
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    return float(thresholds[optimal_idx])


def build_model_pipeline(model_type="xgboost", scale_pos_weight=1.0, random_state=42):
    """
    Build a scikit-learn Pipeline with a single classifier step.

    No preprocessing/scaling is applied: tree ensembles are scale-invariant,
    and the fold-level median imputation of NaNs happens explicitly in
    train_binary_model() before this pipeline is fitted, so a
    ColumnTransformer + StandardScaler step here would be dead weight that
    advertises preprocessing the model never receives.

    Parameters
    ----------
    model_type : str
        "xgboost" or "lightgbm".
    scale_pos_weight : float
        Weight for positive class to handle imbalance.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Unfitted pipeline with one step: ("classifier", classifier).
    """
    if model_type == "xgboost" and HAS_XGBOOST:
        classifier = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=random_state,
            use_label_encoder=False,
            tree_method="hist",
            n_jobs=-1,
        )
    elif model_type == "lightgbm" and HAS_LIGHTGBM:
        classifier = lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            is_unbalance=True,  # handles class imbalance
            random_state=random_state,
            verbose=-1,
            n_jobs=-1,
        )
    else:
        # Fallback: LightGBM if XGBoost not available
        if HAS_LIGHTGBM:
            classifier = lgb.LGBMClassifier(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                is_unbalance=True,
                random_state=random_state,
                verbose=-1,
                n_jobs=-1,
            )
        else:
            raise ValueError(
                "Neither XGBoost nor LightGBM is available. "
                "Install one of them: pip install xgboost lightgbm"
            )

    pipeline = Pipeline([
        ("classifier", classifier),
    ])

    return pipeline


def train_binary_model(
    df, test_year=2024, percentile=0.75, model_type="xgboost", random_state=42
):
    """
    Train binary classification model with temporal cross-validation.

    Parameters
    ----------
    df : pd.DataFrame
        Output of prepare_binary_dataset().
    test_year : int
        Year to use as test set (data >= test_year).
        Training uses data < test_year.
    percentile : float
        Percentile threshold for binary target (0-1).
    model_type : str
        "xgboost" or "lightgbm".
    random_state : int
        Random seed.

    Returns
    -------
    dict with keys:
        - model: trained Pipeline
        - metrics: dict of evaluation metrics
        - feature_cols: list of feature names
        - threshold: computed case count threshold
        - optimal_threshold: optimal prediction threshold (0-1)
        - X_train, X_test, y_train, y_test: data splits
        - y_prob_train, y_prob_test: predicted probabilities
        - fpr, tpr, roc_thresholds: ROC curve data
        - prec, rec, pr_thresholds: PR curve data
        - cv_results: list of per-fold CV metrics
    """
    if df.empty:
        return {
            "model": None,
            "metrics": {},
            "feature_cols": [],
            "threshold": None,
            "optimal_threshold": None,
            "X_train": None,
            "X_test": None,
            "y_train": None,
            "y_test": None,
            "y_prob_train": None,
            "y_prob_test": None,
            "fpr": None,
            "tpr": None,
            "roc_thresholds": None,
            "prec": None,
            "rec": None,
            "pr_thresholds": None,
            "cv_results": [],
        }

    feature_cols = _get_feature_columns(df)
    targets = ["total_cases"]

    # Use only rows with non-null features and target
    df_clean = df.dropna(subset=feature_cols + targets).copy()

    if df_clean.empty:
        return {
            "model": None,
            "metrics": {},
            "feature_cols": feature_cols,
            "threshold": None,
            "optimal_threshold": None,
            "X_train": None,
            "X_test": None,
            "y_train": None,
            "y_test": None,
            "y_prob_train": None,
            "y_prob_test": None,
            "fpr": None,
            "tpr": None,
            "roc_thresholds": None,
            "prec": None,
            "rec": None,
            "pr_thresholds": None,
            "cv_results": [],
        }

    # --- Temporal split ---
    mask_train = df_clean["year"] < test_year
    mask_test = df_clean["year"] >= test_year

    df_train = df_clean[mask_train].copy()
    df_test = df_clean[mask_test].copy()

    if df_train.empty:
        return {
            "model": None,
            "metrics": {},
            "feature_cols": feature_cols,
            "threshold": None,
            "optimal_threshold": None,
            "X_train": None,
            "X_test": df_test[feature_cols].values if not df_test.empty else None,
            "y_train": None,
            "y_test": None,
            "y_prob_train": None,
            "y_prob_test": None,
            "fpr": None,
            "tpr": None,
            "roc_thresholds": None,
            "prec": None,
            "rec": None,
            "pr_thresholds": None,
            "cv_results": [],
        }

    # --- Threshold, labels and weights for the held-out test years ---
    # This threshold labels the test years (>= test_year, never used for
    # fitting) and the final model. The CV folds below recompute their own
    # threshold from their own training years, so a fold's validation labels
    # can never depend on the year being validated.
    threshold = compute_percentile_threshold(
        df_train["total_cases"].values, percentile=percentile
    )
    cases_train = df_train["total_cases"].to_numpy(dtype=float)

    y_train = create_binary_target(cases_train, threshold)
    y_test = (
        create_binary_target(df_test["total_cases"].values, threshold)
        if not df_test.empty
        else None
    )

    X_train_raw = df_train[feature_cols].to_numpy(dtype=float)
    X_test_raw = (
        df_test[feature_cols].to_numpy(dtype=float) if not df_test.empty else None
    )

    # --- Class imbalance weight (training years only) ---
    n_negative = np.sum(y_train == 0)
    n_positive = np.sum(y_train == 1)
    scale_pos_weight = float(n_negative / n_positive) if n_positive > 0 else 1.0

    # --- Year-based forward-chaining cross-validation ---
    # Rows arrive sorted by unit, so a row-position split (TimeSeriesSplit)
    # would hold out municipalities over the whole calendar instead of years:
    # every validation row would then carry a year already seen in training.
    # The split is defined on the year index with an expanding window -- fold
    # j trains on year < v_j and validates on year v_j.
    train_years = df_train["year"].to_numpy()
    n_splits = min(5, max(2, len(np.unique(train_years)) - 1))
    cv_skipped = []
    try:
        folds = year_forward_chain_folds(
            train_years, n_splits=n_splits, min_train_years=2
        )
        assert_forward_in_time(train_years, folds)  # structural guard
    except ValueError as exc:
        # Too few training years (a short series, not a coding error). We skip
        # the CV rather than silently fall back to a row-position split, which
        # would be reported as temporal validation while holding out units.
        folds = []
        cv_skipped.append({"fold": 0, "reason": f"insufficient_years: {exc}"})

    cv_results = []
    cv_y_true_all = []
    cv_y_prob_all = []

    for fold_idx, (tr_idx, val_idx) in enumerate(folds, start=1):
        # Everything that defines this fold's labels and class weight is
        # recomputed from the fold's training years only.
        fold_threshold = compute_percentile_threshold(
            cases_train[tr_idx], percentile=percentile
        )
        y_tr = create_binary_target(cases_train[tr_idx], fold_threshold)
        y_val = create_binary_target(cases_train[val_idx], fold_threshold)

        if len(np.unique(y_tr)) < 2:
            cv_skipped.append(
                {"fold": fold_idx, "reason": "single_class_in_training_years"}
            )
            continue

        fold_imputer = SimpleImputer(strategy="median")
        X_tr = fold_imputer.fit_transform(X_train_raw[tr_idx])
        X_val = fold_imputer.transform(X_train_raw[val_idx])

        fold_n_neg = float(np.sum(y_tr == 0))
        fold_n_pos = float(np.sum(y_tr == 1))
        fold_pos_weight = fold_n_neg / fold_n_pos if fold_n_pos > 0 else 1.0

        pipeline = build_model_pipeline(
            model_type=model_type,
            scale_pos_weight=fold_pos_weight,
            random_state=random_state,
        )
        pipeline.fit(X_tr, y_tr)
        y_prob_val = pipeline.predict_proba(X_val)[:, 1]

        fold_metrics = _compute_fold_metrics(y_val, y_prob_val)
        fold_metrics.update(
            {
                "fold": fold_idx,
                "val_year": int(np.max(train_years[val_idx])),
                "train_years": "{}-{}".format(
                    int(np.min(train_years[tr_idx])),
                    int(np.max(train_years[tr_idx])),
                ),
                "n_train_fold": int(np.size(tr_idx)),
                "n_val_fold": int(np.size(val_idx)),
                "n_positive_val": int(np.sum(y_val)),
                "threshold_case_count_fold": float(fold_threshold),
                "scale_pos_weight_fold": float(fold_pos_weight),
            }
        )
        cv_results.append(fold_metrics)

        cv_y_true_all.extend(y_val.tolist())
        cv_y_prob_all.extend(y_prob_val.tolist())

    # --- Imputer for the deployable model (fitted on ALL training rows) ---
    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train_raw)
    X_test = imputer.transform(X_test_raw) if X_test_raw is not None else None

    # --- Train final model on ALL training data ---
    final_pipeline = build_model_pipeline(
        model_type=model_type,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
    )
    final_pipeline.fit(X_train, y_train)

    # Predictions
    y_prob_train = final_pipeline.predict_proba(X_train)[:, 1]
    y_prob_test = final_pipeline.predict_proba(X_test)[:, 1] if X_test is not None else None

    # --- Use CV out-of-fold predictions for ROC/PR curves (honest estimate) ---
    if len(cv_y_true_all) > 0 and len(np.unique(cv_y_true_all)) > 1:
        cv_y_true_arr = np.array(cv_y_true_all)
        cv_y_prob_arr = np.array(cv_y_prob_all)
        fpr_cv, tpr_cv, roc_thresholds_cv = roc_curve(cv_y_true_arr, cv_y_prob_arr)
        prec_cv, rec_cv, pr_thresholds_cv = precision_recall_curve(cv_y_true_arr, cv_y_prob_arr)
        optimal_threshold = compute_optimal_threshold(fpr_cv, tpr_cv, roc_thresholds_cv)
        threshold_source = "out_of_fold"
    else:
        # No usable out-of-fold predictions (too few years, or a single class
        # in every fold). The curves below are then in-sample and optimistic;
        # the flag records that instead of silently reporting them as CV.
        fpr_cv, tpr_cv, roc_thresholds_cv = roc_curve(y_train, y_prob_train)
        prec_cv, rec_cv, pr_thresholds_cv = precision_recall_curve(y_train, y_prob_train)
        optimal_threshold = compute_optimal_threshold(fpr_cv, tpr_cv, roc_thresholds_cv)
        threshold_source = "in_sample_fallback"

    # --- Final metrics on test set ---
    test_metrics = {}
    if y_test is not None and len(y_test) > 0:
        y_pred_test = (y_prob_test >= optimal_threshold).astype(int)
        test_metrics = _compute_fold_metrics(y_test, y_prob_test)
        test_metrics["f1_at_youden"] = f1_score(
            y_test, y_pred_test, zero_division=np.nan
        )
        test_metrics["confusion_matrix"] = confusion_matrix(y_test, y_pred_test).tolist()
        test_metrics["n_test_samples"] = len(y_test)
        test_metrics["n_positive_test"] = int(np.sum(y_test))
        test_metrics["n_negative_test"] = int(np.sum(y_test == 0))

    # CV aggregate metrics. Fold-level AUCs can be NaN for single-class folds,
    # so the aggregate skips NaNs and is NaN itself when nothing measurable
    # remains -- absent evidence must not look like a score of zero.
    def _nanmean(values):
        arr = np.asarray(values, dtype=float)
        return float(np.nanmean(arr)) if np.isfinite(arr).any() else np.nan

    if cv_results:
        cv_metrics = {
            "cv_auc_roc": _nanmean([r["auc_roc"] for r in cv_results]),
            "cv_auc_pr": _nanmean([r["auc_pr"] for r in cv_results]),
            "cv_f1_at_0.5": _nanmean([r["f1_at_0.5"] for r in cv_results]),
            "cv_precision": _nanmean([r["precision"] for r in cv_results]),
            "cv_recall": _nanmean([r["recall"] for r in cv_results]),
            "cv_brier": _nanmean([r["brier"] for r in cv_results]),
            "n_folds": len(cv_results),
        }
    else:
        cv_metrics = {}

    # --- Combine metrics ---
    all_metrics = {
        "threshold_case_count": threshold,
        "percentile": percentile,
        "optimal_prob_threshold": optimal_threshold,
        "n_train": len(y_train),
        "n_positive_train": int(np.sum(y_train)),
        "n_negative_train": int(np.sum(y_train == 0)),
        "scale_pos_weight": scale_pos_weight,
        "cv_scheme": "year_forward_chain",
        "cv_threshold_source": threshold_source,
        "cv_n_skipped_folds": len(cv_skipped),
        **cv_metrics,
        **test_metrics,
    }

    return {
        "model": final_pipeline,
        "metrics": all_metrics,
        "feature_cols": feature_cols,
        "threshold": threshold,
        "optimal_threshold": optimal_threshold,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_prob_train": y_prob_train,
        "y_prob_test": y_prob_test,
        # CV-based curves (honest out-of-fold estimates)
        "fpr": fpr_cv,
        "tpr": tpr_cv,
        "roc_thresholds": roc_thresholds_cv,
        "prec": prec_cv,
        "rec": rec_cv,
        "pr_thresholds": pr_thresholds_cv,
        "cv_results": cv_results,
        "cv_skipped_folds": cv_skipped,
        "cv_fold_summary": summarize_folds(train_years, folds),
        "cv_y_true": cv_y_true_all,
        "cv_y_prob": cv_y_prob_all,
        # Metadata for predictions
        "df_train": df_train,
        "df_test": df_test,
        "imputer": imputer,
    }


def _compute_fold_metrics(y_true, y_prob):
    """
    Compute metrics for a single fold or dataset.

    The F1/precision/recall reported here use the fixed 0.5 probability rule;
    the key names make that explicit (``f1_at_0.5``) so they cannot be
    confused with metrics computed at the Youden threshold
    (``f1_at_youden``, added by the caller for test sets). ``auc_roc`` and
    ``auc_pr`` are threshold-free; when ``y_true`` has a single class they are
    undefined and reported as NaN rather than as a misleading 0.
    """
    y_pred = (y_prob >= 0.5).astype(int)

    single_class = len(np.unique(y_true)) < 2
    metrics = {
        "auc_roc": np.nan if single_class else roc_auc_score(y_true, y_prob),
        "auc_pr": np.nan if single_class else average_precision_score(y_true, y_prob),
        "f1_at_0.5": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "brier": brier_score_loss(y_true, y_prob),
    }

    return metrics


def get_binary_feature_importance(result_dict):
    """
    Extract feature importance from trained binary model.

    Parameters
    ----------
    result_dict : dict
        Output of train_binary_model().

    Returns
    -------
    pd.DataFrame
        DataFrame with 'Feature' and 'Importance' columns.
    """
    model = result_dict.get("model")
    feature_cols = result_dict.get("feature_cols", [])

    if model is None or feature_cols is None:
        return pd.DataFrame(columns=["Feature", "Importance"])

    classifier = model.named_steps["classifier"]

    if hasattr(classifier, "feature_importances_") and classifier.feature_importances_ is not None:
        importances = np.asarray(classifier.feature_importances_, dtype=float)
    else:
        # Fallback: return zeros
        importances = np.zeros(len(feature_cols))

    df_imp = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": importances,
    }).sort_values("Importance", ascending=False).reset_index(drop=True)

    return df_imp


def get_shap_values(result_dict, max_samples=1000):
    """
    Compute SHAP values for the trained model.

    Parameters
    ----------
    result_dict : dict
        Output of train_binary_model().
    max_samples : int
        Max samples to use for SHAP computation (for performance).

    Returns
    -------
    dict with keys:
        - shap_values: np.ndarray
        - expected_value: float
        - X_background: pd.DataFrame (background data)
        - feature_names: list
    """
    try:
        import shap
    except ImportError:
        return None

    model = result_dict.get("model")
    X_train = result_dict.get("X_train")
    feature_cols = result_dict.get("feature_cols", [])

    if model is None or X_train is None:
        return None

    classifier = model.named_steps["classifier"]

    # Sample background data for efficiency
    if len(X_train) > max_samples:
        rng = np.random.RandomState(42)
        indices = rng.choice(len(X_train), max_samples, replace=False)
        X_background = X_train[indices]
    else:
        X_background = X_train

    try:
        # Create a wrapper function that returns probability of class 1
        def model_predict_proba(X):
            return model.predict_proba(X)[:, 1]

        explainer = shap.KernelExplainer(model_predict_proba, X_background)

        # Use a sample of training data for SHAP computation
        n_shap_samples = min(200, len(X_train))
        rng = np.random.RandomState(42)
        shap_indices = rng.choice(len(X_train), n_shap_samples, replace=False)
        X_to_explain = X_train[shap_indices]

        shap_values = explainer.shap_values(X_to_explain)

        return {
            "shap_values": shap_values,
            "expected_value": explainer.expected_value,
            "X_background": X_background,
            "X_explained": X_to_explain,
            "feature_names": feature_cols,
        }
    except Exception as e:
        print(f"SHAP computation failed: {e}")
        return None


def predict_future_binary(model, df_future, imputer=None, feature_cols=None):
    """
    Predict binary outcome for future data.

    The feature matrix is built by _get_feature_columns(), the same function
    used for training, so the column set and its order always match the model.

    Parameters
    ----------
    model : sklearn.Pipeline
        Trained binary classification pipeline.
    df_future : pd.DataFrame
        DataFrame with feature columns for prediction.
    imputer : SimpleImputer, optional
        Fitted imputer from training.
    feature_cols : sequence of str, optional
        Feature list the model was trained on, typically
        ``result["feature_cols"]`` from train_binary_model(). Passing it is
        strongly recommended: it also guarantees that target-year climate
        columns cannot be reintroduced. When omitted the columns are inferred
        from df_future.

    Returns
    -------
    dict with keys:
        - y_prob: predicted probabilities
        - y_pred_default: predictions at default 0.5 threshold
        - feature_cols: the columns actually used, in model order
    """
    feature_cols = _get_feature_columns(df_future, expected=feature_cols)

    X = df_future[feature_cols].values

    if imputer is not None:
        X = imputer.transform(X)

    y_prob = model.predict_proba(X)[:, 1]
    y_pred_default = (y_prob >= 0.5).astype(int)

    return {
        "y_prob": y_prob,
        "y_pred_default": y_pred_default,
        "feature_cols": feature_cols,
    }
