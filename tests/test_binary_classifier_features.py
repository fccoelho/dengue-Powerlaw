"""Regression tests for the binary classifier's feature selection.

``predictive_model.get_quarterly_climate()`` produces ``avg_temp_Q1..Q4`` and
``avg_umid_Q1..Q4`` for the *target* year, which ``prepare_lagged_data()``
then shifts into ``prev_avg_*`` columns. Only the shifted columns are
knowable at prediction time. Before this was pinned down, the selector also
picked up the unshifted columns (leaking the outcome) and listed the shifted
ones twice, so training used 34 columns of which only 26 were distinct.
"""

import numpy as np
import pandas as pd
import pytest

import binary_classifier as bc

TARGET_YEAR_CLIMATE = [
    f"avg_{var}_Q{q}" for var in ("temp", "umid") for q in range(1, 5)
]
LAGGED_CLIMATE = [f"prev_{c}" for c in TARGET_YEAR_CLIMATE]

IDENTIFIER_COLS = ["year", "geocode", "city_name", "population", "log_pop"]
TARGET_YEAR_METRICS = [
    "total_cases",
    "ep_dur",
    "peak_week",
    "R0",
    "alpha_size",
    "xmin_size",
    "alpha_dur",
    "xmin_dur",
    "alpha_trend",
]
LAGGED_METRICS = [f"prev_{c}" for c in TARGET_YEAR_METRICS]


def _lagged_frame(n_rows=40, seed=0):
    """Build a frame with the columns prepare_lagged_data() produces."""
    rng = np.random.default_rng(seed)
    columns = IDENTIFIER_COLS + TARGET_YEAR_METRICS + TARGET_YEAR_CLIMATE
    columns += LAGGED_METRICS + LAGGED_CLIMATE
    df = pd.DataFrame(
        rng.normal(size=(n_rows, len(columns))), columns=columns
    )
    df["year"] = np.repeat(np.arange(2011, 2011 + n_rows // 4 + 1), 4)[:n_rows]
    df["total_cases"] = df["total_cases"].abs() * 100
    return df


def test_target_year_climate_is_excluded():
    """Contemporaneous climate must never enter the feature matrix."""
    df = _lagged_frame()
    features = bc._get_feature_columns(df)

    assert [c for c in features if bc._is_contemporaneous_climate(c)] == []
    for col in TARGET_YEAR_CLIMATE:
        assert col not in features


def test_no_duplicate_features():
    df = _lagged_frame()
    features = bc._get_feature_columns(df)
    assert len(features) == len(set(features))


def test_lagged_climate_kept_exactly_once():
    df = _lagged_frame()
    features = bc._get_feature_columns(df)
    for col in LAGGED_CLIMATE:
        assert features.count(col) == 1


def test_features_are_lagged_or_static_only():
    """Every feature is a lagged column or an explicitly allowed static one."""
    df = _lagged_frame()
    allowed_static = {"log_pop"}
    features = bc._get_feature_columns(df)
    for col in features:
        assert col.startswith("prev_") or col in allowed_static, col


def test_all_null_columns_are_dropped():
    df = _lagged_frame()
    df["prev_alpha_dur"] = np.nan
    features = bc._get_feature_columns(df)
    assert "prev_alpha_dur" not in features


def test_expected_list_is_returned_verbatim():
    """Scoring must reuse the training list, in the training order."""
    df = _lagged_frame()
    expected = bc._get_feature_columns(df)
    shuffled = list(reversed(expected))
    assert bc._get_feature_columns(df, expected=shuffled) == shuffled


def test_expected_list_with_missing_column_raises():
    df = _lagged_frame()
    expected = bc._get_feature_columns(df)
    with pytest.raises(KeyError):
        bc._get_feature_columns(df.drop(columns=["prev_R0"]), expected=expected)


def test_predict_future_binary_uses_training_columns():
    """predict() must see the same columns, in the same order, as fit()."""
    from sklearn.linear_model import LogisticRegression

    df = _lagged_frame()
    features = bc._get_feature_columns(df)
    y = (df["total_cases"] > df["total_cases"].median()).astype(int)

    model = LogisticRegression(max_iter=500).fit(df[features].values, y)

    out = bc.predict_future_binary(model, df, feature_cols=features)
    assert out["feature_cols"] == features
    assert out["y_prob"].shape == (len(df),)
    assert [c for c in out["feature_cols"] if bc._is_contemporaneous_climate(c)] == []


def test_predict_future_binary_inference_matches_training_order():
    """Inferring the columns must give the same result as passing them."""
    from sklearn.linear_model import LogisticRegression

    df = _lagged_frame()
    features = bc._get_feature_columns(df)
    y = (df["total_cases"] > df["total_cases"].median()).astype(int)
    model = LogisticRegression(max_iter=500).fit(df[features].values, y)

    with_list = bc.predict_future_binary(model, df, feature_cols=features)
    inferred = bc.predict_future_binary(model, df)

    assert inferred["feature_cols"] == features
    np.testing.assert_allclose(with_list["y_prob"], inferred["y_prob"])


def test_no_unqualified_f1_keys():
    """B4: F1 must always carry its decision rule in the key name."""
    pytest.importorskip("xgboost")
    df = _lagged_frame()
    probs = 1 / (1 + np.exp(-df["prev_R0"].to_numpy(float)))  # valid probabilities
    metrics = bc._compute_fold_metrics(
        (df["total_cases"] > df["total_cases"].median()).to_numpy(int),
        probs,
    )
    assert "f1" not in metrics
    assert "f1_at_0.5" in metrics

    result = bc.train_binary_model(df, test_year=2019, percentile=0.75)
    all_keys = set(result["metrics"].keys())
    assert "cv_f1" not in all_keys
    assert "f1" not in all_keys
    assert "f1_at_optimal" not in all_keys
    assert "cv_f1_at_0.5" in all_keys
    # test set exists for this frame, so the Youden F1 must be present too
    assert "f1_at_youden" in all_keys


def test_single_class_fold_reports_nan_auc_not_zero():
    """B5: an undefined metric must be NaN, never a misleading 0."""
    y_single_class = np.zeros(20, dtype=int)
    metrics = bc._compute_fold_metrics(y_single_class, np.linspace(0, 1, 20))
    assert np.isnan(metrics["auc_roc"])
    assert np.isnan(metrics["auc_pr"])
    # threshold metrics with zero_division=0 remain defined conventions
    assert metrics["f1_at_0.5"] == 0.0


def test_build_model_pipeline_has_no_dead_preprocessing():
    """B3: the pipeline must not advertise a ColumnTransformer it never runs."""
    pytest.importorskip("xgboost")
    pipeline = bc.build_model_pipeline(model_type="xgboost")
    assert list(pipeline.named_steps.keys()) == ["classifier"]


if __name__ == "__main__":
    import sys

    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"PASS {name}")
        except Exception as exc:  # pragma: no cover - manual runner
            failures += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    sys.exit(1 if failures else 0)
