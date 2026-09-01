#!/usr/bin/env python
"""Nightly batch scoring: applies the trained churn model and T-learner
uplift model to the latest feature-store snapshot, prices each account's
expected intervention value via the decision engine, and writes
data/processed/batch_decisions.csv -- the table the API's
`/accounts/{id}/decision` endpoint serves from.

Usage:
    python pipelines/run_batch_scoring.py
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib
import numpy as np
import pandas as pd

from harbor.config import load_config
from harbor.decision.decision_engine import (
    UPLIFT_FEATURE_COLUMNS,
    build_recommendations,
    summarize_portfolio_impact,
)
from harbor.features.build_features import (
    CATEGORICAL_FEATURE_COLUMNS,
    NUMERIC_FEATURE_COLUMNS,
)
from harbor.utils.artifacts import require_files_exist
from harbor.utils.logging_config import configure_logging

logger = logging.getLogger(__name__)


def main() -> None:
    configure_logging()
    cfg = load_config()
    data_cfg = cfg.section("data")
    decision_cfg = cfg.section("decision")
    models_dir = cfg.models_dir
    processed_dir = cfg.path(key="processed_dir")

    churn_model_path = models_dir / "churn_advanced_model.joblib"
    t_learner_path = models_dir / "uplift_t_learner.joblib"
    encoded_cols_path = models_dir / "uplift_feature_columns_encoded.joblib"
    require_files_exist(
        [churn_model_path, t_learner_path, encoded_cols_path], context="Batch scoring"
    )

    churn_model = joblib.load(churn_model_path)
    t_learner = joblib.load(t_learner_path)
    encoded_columns = joblib.load(encoded_cols_path)

    feature_store = pd.read_parquet(processed_dir / data_cfg["feature_store_file"])
    current = feature_store[feature_store["snapshot_month"] == data_cfg["test_snapshot_month"]].copy()
    logger.info("Scoring %d accounts from snapshot_month=%d", len(current), data_cfg["test_snapshot_month"])

    full_feature_cols = NUMERIC_FEATURE_COLUMNS + CATEGORICAL_FEATURE_COLUMNS
    churn_proba = churn_model.predict_proba(current[full_feature_cols])[:, 1]

    X_uplift = pd.get_dummies(current[UPLIFT_FEATURE_COLUMNS], drop_first=True)
    X_uplift = X_uplift.reindex(columns=encoded_columns, fill_value=0)
    p_treat = t_learner["treat"].predict_proba(X_uplift)[:, 1]
    p_control = t_learner["control"].predict_proba(X_uplift)[:, 1]
    predicted_uplift = p_control - p_treat

    recommendations = build_recommendations(
        account_ids=current["account_id"],
        churn_probability=pd.Series(churn_proba),
        predicted_uplift=pd.Series(predicted_uplift),
        monthly_margin=decision_cfg["avg_monthly_margin_per_account"],
        months_credited=decision_cfg["ltv_months_saved"],
        cost_per_account=decision_cfg["intervention_cost_per_account"],
    )

    impact = summarize_portfolio_impact(recommendations, decision_cfg["intervention_cost_per_account"])
    logger.info("Portfolio impact: %s", impact)

    # Revenue at risk: ties the churn model's output to a dollar figure a
    # business stakeholder actually cares about, independent of whether any
    # particular intervention is funded. Top-decile threshold matches the
    # convention already used in evaluate_model's top_decile_lift metric.
    decile_cut = np.quantile(churn_proba, 0.9)
    top_decile_mask = churn_proba >= decile_cut
    total_portfolio_mrr = float(current["current_mrr"].sum())
    mrr_at_risk_top_decile = float(current.loc[top_decile_mask, "current_mrr"].sum())
    impact["total_portfolio_mrr"] = total_portfolio_mrr
    impact["mrr_at_risk_top_decile"] = mrr_at_risk_top_decile
    impact["mrr_at_risk_share"] = mrr_at_risk_top_decile / total_portfolio_mrr if total_portfolio_mrr > 0 else float("nan")
    logger.info(
        "Revenue at risk: $%.0f/mo in the top-decile churn-risk accounts (%.1f%% of $%.0f/mo total portfolio MRR)",
        mrr_at_risk_top_decile, impact["mrr_at_risk_share"] * 100, total_portfolio_mrr,
    )

    recommendations["scored_at"] = datetime.now(timezone.utc).isoformat()
    recommendations["model_version"] = f"{churn_model_path.name}@{int(churn_model_path.stat().st_mtime)}"

    # Behavioral segment is optional enrichment: batch scoring works fine
    # without it (segmentation is a separate pipeline), but when available
    # it lets a retention team see *which kind* of account a recommendation
    # is about, instead of scoring being disconnected from the segmentation
    # analysis entirely.
    segmentation_path = processed_dir / "segmentation_projection.csv"
    if segmentation_path.exists():
        segments = pd.read_csv(segmentation_path)[["account_id", "kmeans_label"]].rename(
            columns={"kmeans_label": "behavioral_segment"}
        )
        recommendations = recommendations.merge(segments, on="account_id", how="left")
        logger.info("Enriched batch decisions with behavioral segment from %s", segmentation_path)
    else:
        recommendations["behavioral_segment"] = pd.NA
        logger.info(
            "No segmentation_projection.csv found -- run pipelines/run_segmentation.py first to enrich "
            "batch decisions with behavioral segment membership."
        )

    out_path = processed_dir / "batch_decisions.csv"
    recommendations.to_csv(out_path, index=False)
    logger.info("Wrote %d batch decisions to %s", len(recommendations), out_path)

    # Budget sensitivity: a retention team rarely has an unlimited budget --
    # showing how many accounts get treated and what ROI results at several
    # budget levels (instead of only the unconstrained optimum) is what
    # actually lets someone pick a number to ask for.
    unconstrained_cost = impact["total_intervention_cost"]
    budget_points = sorted(
        {round(unconstrained_cost * frac) for frac in [0.1, 0.25, 0.5, 0.75, 1.0] if unconstrained_cost > 0}
    )
    budget_sensitivity = []
    for budget in budget_points:
        recs_at_budget = build_recommendations(
            account_ids=current["account_id"],
            churn_probability=pd.Series(churn_proba),
            predicted_uplift=pd.Series(predicted_uplift),
            monthly_margin=decision_cfg["avg_monthly_margin_per_account"],
            months_credited=decision_cfg["ltv_months_saved"],
            cost_per_account=decision_cfg["intervention_cost_per_account"],
            budget=budget,
        )
        budget_impact = summarize_portfolio_impact(recs_at_budget, decision_cfg["intervention_cost_per_account"])
        budget_sensitivity.append({"budget": budget, **budget_impact})
    logger.info("Budget sensitivity computed at %d budget levels.", len(budget_sensitivity))

    import json
    with open(processed_dir / "batch_scoring_impact.json", "w", encoding="utf-8") as f:
        json.dump(impact, f, indent=2)
    with open(processed_dir / "budget_sensitivity.json", "w", encoding="utf-8") as f:
        json.dump(budget_sensitivity, f, indent=2)


if __name__ == "__main__":
    main()
