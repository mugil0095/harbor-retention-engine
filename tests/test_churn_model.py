"""F: statistical calc sanity (indirectly). G: model training. H: predictions.
I: evaluation metrics."""

from __future__ import annotations

import numpy as np
import pytest

from harbor.features.build_features import (
    CATEGORICAL_FEATURE_COLUMNS,
    NUMERIC_FEATURE_COLUMNS,
    build_feature_store,
)
from harbor.models.churn_model import (
    LABEL_COLUMN,
    STATIC_FEATURE_COLUMNS,
    calibration_curve_points,
    evaluate_model,
    expected_calibration_error,
    rolling_origin_splits,
    segment_performance,
    train_baseline_model,
    train_calibrated_advanced_model,
    train_classical_model,
    tune_advanced_model,
)
from harbor.nlp.ticket_analysis import compare_nlp_approaches


@pytest.fixture(scope="module")
def feature_store(synthetic_tables, test_config):
    _, clf = compare_nlp_approaches(synthetic_tables["support_tickets"], random_seed=123)
    data_cfg = test_config.section("data")
    return build_feature_store(
        synthetic_tables["accounts"], synthetic_tables["monthly_usage"], synthetic_tables["support_tickets"],
        clf, data_cfg["snapshot_months"], data_cfg["churn_horizon_months"],
    )


@pytest.fixture(scope="module")
def train_test_split(feature_store, test_config):
    test_month = test_config.section("data")["test_snapshot_month"]
    train_df = feature_store[feature_store["snapshot_month"] != test_month]
    test_df = feature_store[feature_store["snapshot_month"] == test_month]
    return train_df, test_df


def test_rolling_origin_splits_never_uses_future_month_to_predict_past():
    folds = rolling_origin_splits([9, 12, 15, 18, 21], test_month=21)
    for train_months, val_month in folds:
        assert all(m < val_month for m in train_months)


def test_rolling_origin_splits_excludes_test_month():
    folds = rolling_origin_splits([9, 12, 15, 18, 21], test_month=21)
    all_months_used = {m for train_months, val_month in folds for m in train_months + [val_month]}
    assert 21 not in all_months_used


def test_baseline_model_trains_and_predicts_valid_probabilities(train_test_split):
    train_df, test_df = train_test_split
    if train_df[LABEL_COLUMN].nunique() < 2:
        pytest.skip("degenerate label distribution in tiny test fixture")
    model = train_baseline_model(train_df, random_seed=123)
    proba = model.predict_proba(test_df[STATIC_FEATURE_COLUMNS])[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()
    assert len(proba) == len(test_df)


def test_classical_model_trains_and_predicts(train_test_split):
    train_df, test_df = train_test_split
    if train_df[LABEL_COLUMN].nunique() < 2:
        pytest.skip("degenerate label distribution in tiny test fixture")
    model = train_classical_model(train_df, NUMERIC_FEATURE_COLUMNS, CATEGORICAL_FEATURE_COLUMNS, random_seed=123)
    proba = model.predict_proba(test_df[NUMERIC_FEATURE_COLUMNS + CATEGORICAL_FEATURE_COLUMNS])[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()


def test_evaluate_model_returns_expected_metric_keys(train_test_split):
    train_df, test_df = train_test_split
    if train_df[LABEL_COLUMN].nunique() < 2 or test_df[LABEL_COLUMN].nunique() < 2:
        pytest.skip("degenerate label distribution in tiny test fixture")
    model = train_baseline_model(train_df, random_seed=123)
    metrics = evaluate_model(model, test_df, STATIC_FEATURE_COLUMNS)
    for key in ["roc_auc", "pr_auc", "brier_score", "best_f1", "best_f1_threshold", "top_decile_lift", "n_samples"]:
        assert key in metrics


def test_evaluate_model_perfect_predictions_give_auc_near_one(train_test_split):
    _, test_df = train_test_split
    if test_df[LABEL_COLUMN].nunique() < 2:
        pytest.skip("degenerate label distribution in tiny test fixture")

    class PerfectModel:
        def predict_proba(self, X):
            y = test_df.loc[X.index, LABEL_COLUMN].to_numpy()
            return np.column_stack([1 - y, y])

    metrics = evaluate_model(PerfectModel(), test_df, STATIC_FEATURE_COLUMNS)
    assert metrics["roc_auc"] > 0.99


def test_segment_performance_returns_one_row_per_segment_with_both_classes(train_test_split):
    train_df, test_df = train_test_split
    if train_df[LABEL_COLUMN].nunique() < 2:
        pytest.skip("degenerate label distribution in tiny test fixture")
    model = train_baseline_model(train_df, random_seed=123)
    seg_perf = segment_performance(model, test_df, STATIC_FEATURE_COLUMNS, "plan_tier")
    assert "plan_tier" in seg_perf.columns
    assert len(seg_perf) <= test_df["plan_tier"].nunique()


def test_advanced_model_tuning_picks_from_grid(feature_store, test_config):
    data_cfg = test_config.section("data")
    if feature_store[LABEL_COLUMN].nunique() < 2:
        pytest.skip("degenerate label distribution in tiny test fixture")
    result = tune_advanced_model(
        feature_store, NUMERIC_FEATURE_COLUMNS, CATEGORICAL_FEATURE_COLUMNS,
        data_cfg["snapshot_months"], data_cfg["test_snapshot_month"], random_seed=123,
    )
    assert result.best_params in [r["params"] for r in result.all_results]
    assert 0.0 <= result.best_score <= 1.0


def test_expected_calibration_error_is_near_zero_for_well_calibrated_predictions():
    rng = np.random.default_rng(0)
    proba = rng.uniform(0, 1, 5000)
    y_true = (rng.random(5000) < proba).astype(int)  # outcomes literally drawn from the predicted probabilities
    ece = expected_calibration_error(y_true, proba, n_bins=10)
    assert ece < 0.05


def test_expected_calibration_error_is_high_for_badly_miscalibrated_predictions():
    rng = np.random.default_rng(0)
    proba = np.full(2000, 0.9)  # model always says "90% likely to churn"
    y_true = (rng.random(2000) < 0.1).astype(int)  # true rate is actually 10%
    ece = expected_calibration_error(y_true, proba, n_bins=10)
    assert ece > 0.5


def test_calibration_curve_points_returns_matching_length_lists():
    rng = np.random.default_rng(0)
    proba = rng.uniform(0, 1, 500)
    y_true = (rng.random(500) < proba).astype(int)
    predicted, observed = calibration_curve_points(y_true, proba, n_bins=5)
    assert len(predicted) == len(observed)
    assert len(predicted) > 0
    assert all(0.0 <= p <= 1.0 for p in predicted)
    assert all(0.0 <= o <= 1.0 for o in observed)


def test_calibrated_model_predict_proba_is_valid(train_test_split):
    train_df, test_df = train_test_split
    if train_df[LABEL_COLUMN].nunique() < 2:
        pytest.skip("degenerate label distribution in tiny test fixture")
    calibrated = train_calibrated_advanced_model(
        train_df, NUMERIC_FEATURE_COLUMNS, CATEGORICAL_FEATURE_COLUMNS,
        best_params={"max_iter": 100, "max_depth": 3, "learning_rate": 0.1, "l2_regularization": 0.0},
        random_seed=123, cv=2,
    )
    proba = calibrated.predict_proba(test_df[NUMERIC_FEATURE_COLUMNS + CATEGORICAL_FEATURE_COLUMNS])[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()
    assert len(proba) == len(test_df)
