"""SHAP explainability: correctness and agreement with permutation importance."""

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
    top_feature_importances,
    train_advanced_model,
    tune_advanced_model,
)
from harbor.models.explainability import (
    _normalize_shap_output,
    _raw_feature_name,
    compute_shap_importance,
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
def advanced_pipe_and_test_df(feature_store, test_config):
    data_cfg = test_config.section("data")
    test_month = data_cfg["test_snapshot_month"]
    train_df = feature_store[feature_store["snapshot_month"] != test_month]
    test_df = feature_store[feature_store["snapshot_month"] == test_month]
    if train_df[LABEL_COLUMN].nunique() < 2 or test_df[LABEL_COLUMN].nunique() < 2:
        pytest.skip("degenerate label distribution in tiny test fixture")
    tune_result = tune_advanced_model(
        feature_store, NUMERIC_FEATURE_COLUMNS, CATEGORICAL_FEATURE_COLUMNS,
        data_cfg["snapshot_months"], test_month, random_seed=123,
    )
    pipe = train_advanced_model(
        train_df, NUMERIC_FEATURE_COLUMNS, CATEGORICAL_FEATURE_COLUMNS, tune_result.best_params, random_seed=123,
    )
    return pipe, test_df


def test_normalize_shap_output_passes_through_plain_2d_array():
    """The only shape this project's own pinned shap+sklearn combination
    actually produces -- verified directly in compute_shap_importance's own
    tests, not just assumed here."""
    arr = np.array([[0.1, 0.2], [0.3, 0.4]])
    result = _normalize_shap_output(arr)
    np.testing.assert_array_equal(result, arr)


def test_normalize_shap_output_unwraps_list_of_per_class_arrays():
    """Older shap versions return [array_for_class_0, array_for_class_1]
    for a binary classifier -- this can't be produced by a real call with
    the pinned shap version, so it's tested directly against a synthetic
    input rather than left unverified."""
    class_0 = np.array([[0.9, 0.8]])
    class_1 = np.array([[0.1, 0.2]])
    result = _normalize_shap_output([class_0, class_1])
    np.testing.assert_array_equal(result, class_1)  # positive-class contributions


def test_normalize_shap_output_collapses_3d_array_to_positive_class():
    """Some shap/model combinations return (n_samples, n_features,
    n_classes) directly instead of a list -- same rationale as above."""
    arr = np.zeros((2, 3, 2))
    arr[:, :, 1] = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    result = _normalize_shap_output(arr)
    assert result.shape == (2, 3)
    np.testing.assert_array_equal(result, arr[:, :, 1])


def test_raw_feature_name_strips_numeric_prefix():
    assert _raw_feature_name("num__tenure_months", categorical_cols=["plan_tier"]) == "tenure_months"


def test_raw_feature_name_collapses_onehot_categorical_to_source_column():
    assert _raw_feature_name("cat__plan_tier_growth", categorical_cols=["plan_tier"]) == "plan_tier"
    assert _raw_feature_name("cat__industry_fintech", categorical_cols=["industry", "plan_tier"]) == "industry"


def test_compute_shap_importance_returns_valid_dataframe(advanced_pipe_and_test_df):
    pipe, test_df = advanced_pipe_and_test_df
    full_cols = NUMERIC_FEATURE_COLUMNS + CATEGORICAL_FEATURE_COLUMNS
    result = compute_shap_importance(
        pipe, test_df, full_cols, categorical_cols=CATEGORICAL_FEATURE_COLUMNS, max_samples=100, random_seed=123
    )
    assert "feature" in result.columns and "mean_abs_shap" in result.columns
    assert (result["mean_abs_shap"] >= 0).all()
    assert result["mean_abs_shap"].is_monotonic_decreasing
    # every reported feature must be a real raw column, not a leftover encoded name
    assert set(result["feature"]).issubset(set(full_cols))


def test_shap_importance_ranking_overlaps_with_permutation_importance(advanced_pipe_and_test_df):
    """The two explainability methods measure different things (marginal
    prediction contribution vs. performance drop under shuffling) and can
    legitimately disagree, but for a model with a handful of clearly
    dominant behavioral features, they should agree on at least some of
    the top features -- if they shared zero features, that would suggest a
    bug in one of the two implementations rather than a genuine methodological
    difference."""
    pipe, test_df = advanced_pipe_and_test_df
    full_cols = NUMERIC_FEATURE_COLUMNS + CATEGORICAL_FEATURE_COLUMNS
    shap_result = compute_shap_importance(
        pipe, test_df, full_cols, categorical_cols=CATEGORICAL_FEATURE_COLUMNS, max_samples=100, random_seed=123
    )
    perm_result = top_feature_importances(pipe, test_df, full_cols, random_seed=123)
    overlap = set(shap_result["feature"].head(8)) & set(perm_result["feature"].head(8))
    assert len(overlap) >= 1
