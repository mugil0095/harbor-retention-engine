"""T: end-to-end pipeline execution (data -> features -> models -> decisions).
P/Q: model-loading and failure-handling behavior of the batch-scoring guard."""

from __future__ import annotations

import pandas as pd
import pytest

from harbor.decision.decision_engine import (
    UPLIFT_FEATURE_COLUMNS,
    build_recommendations,
)
from harbor.features.build_features import (
    CATEGORICAL_FEATURE_COLUMNS,
    NUMERIC_FEATURE_COLUMNS,
    build_feature_store,
)
from harbor.models.churn_model import (
    LABEL_COLUMN,
    evaluate_model,
    train_advanced_model,
    train_baseline_model,
    tune_advanced_model,
)
from harbor.models.uplift import fit_t_learner, predict_cate_t_learner
from harbor.nlp.ticket_analysis import compare_nlp_approaches
from harbor.utils.artifacts import require_files_exist


def test_require_files_exist_raises_with_clear_message_when_artifact_missing(tmp_path):
    missing_path = tmp_path / "does_not_exist.joblib"
    with pytest.raises(FileNotFoundError, match="missing required artifact"):
        require_files_exist([missing_path], context="Test context")


def test_require_files_exist_passes_when_all_present(tmp_path):
    present_path = tmp_path / "exists.txt"
    present_path.write_text("ok")
    require_files_exist([present_path], context="Test context")  # must not raise


def test_full_pipeline_runs_end_to_end_on_synthetic_data(synthetic_tables, test_config):
    data_cfg = test_config.section("data")

    nlp_result, ticket_classifier = compare_nlp_approaches(synthetic_tables["support_tickets"], random_seed=123)
    assert nlp_result.n_test > 0

    feature_store = build_feature_store(
        synthetic_tables["accounts"], synthetic_tables["monthly_usage"], synthetic_tables["support_tickets"],
        ticket_classifier, data_cfg["snapshot_months"], data_cfg["churn_horizon_months"],
    )
    assert len(feature_store) > 0
    if feature_store[LABEL_COLUMN].nunique() < 2:
        pytest.skip("degenerate label distribution in tiny test fixture")

    test_month = data_cfg["test_snapshot_month"]
    train_df = feature_store[feature_store["snapshot_month"] != test_month]
    test_df = feature_store[feature_store["snapshot_month"] == test_month]

    baseline = train_baseline_model(train_df, random_seed=123)
    baseline_metrics = evaluate_model(baseline, test_df, ["plan_tier", "industry", "employee_size_band", "region", "tenure_months", "base_mrr"])
    assert 0.0 <= baseline_metrics["roc_auc"] <= 1.0

    tune_result = tune_advanced_model(
        feature_store, NUMERIC_FEATURE_COLUMNS, CATEGORICAL_FEATURE_COLUMNS,
        data_cfg["snapshot_months"], test_month, random_seed=123,
    )
    advanced = train_advanced_model(train_df, NUMERIC_FEATURE_COLUMNS, CATEGORICAL_FEATURE_COLUMNS, tune_result.best_params, random_seed=123)
    advanced_metrics = evaluate_model(advanced, test_df, NUMERIC_FEATURE_COLUMNS + CATEGORICAL_FEATURE_COLUMNS)
    assert 0.0 <= advanced_metrics["roc_auc"] <= 1.0

    exp = synthetic_tables["experiment_assignments"]
    if exp["group"].nunique() == 2 and len(exp) > 20:
        exp_features = synthetic_tables["accounts"].merge(exp, on="account_id")
        cutoff = data_cfg["observation_cutoff_month"]
        snap = feature_store[feature_store["snapshot_month"] == cutoff] if cutoff in data_cfg["snapshot_months"] else feature_store[feature_store["snapshot_month"] == test_month]
        exp_features = exp_features.merge(
            snap[["account_id", "tenure_months", "avg_login_trailing_3m", "avg_login_trailing_6m", "login_trend_3v6", "avg_ticket_risk_proba_trailing_3m"]],
            on="account_id", how="left",
        ).fillna(0.0)
        exp_features["treatment"] = (exp_features["group"] == "treatment").astype(int)
        if exp_features["treatment"].nunique() == 2:
            t_treat, t_control, X_t = fit_t_learner(exp_features, UPLIFT_FEATURE_COLUMNS, "treatment", "observed_churn_90d", random_seed=123)
            cate = predict_cate_t_learner(t_treat, t_control, X_t)

            recs = build_recommendations(
                account_ids=exp_features["account_id"],
                churn_probability=pd.Series([0.5] * len(exp_features)),
                predicted_uplift=pd.Series(cate),
                monthly_margin=38.0, months_credited=12, cost_per_account=45.0,
            )
            assert "recommended_action" in recs.columns
            assert len(recs) == len(exp_features)
