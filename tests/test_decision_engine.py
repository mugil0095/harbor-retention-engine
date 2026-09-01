"""Decision engine: expected-value computation and budget-constrained ranking."""

from __future__ import annotations

import pandas as pd

from harbor.decision.decision_engine import (
    build_recommendations,
    compute_expected_value,
    summarize_portfolio_impact,
)


def test_expected_value_positive_when_uplift_outweighs_cost():
    ev = compute_expected_value(pd.Series([0.10]), monthly_margin=40.0, months_credited=12, cost_per_account=45.0)
    assert (ev > 0).all()  # 0.10 * 40 * 12 = 48 > 45


def test_expected_value_negative_when_cost_outweighs_uplift():
    ev = compute_expected_value(pd.Series([0.01]), monthly_margin=40.0, months_credited=12, cost_per_account=45.0)
    assert (ev < 0).all()  # 0.01 * 40 * 12 = 4.8 < 45


def test_build_recommendations_sorted_by_expected_value_descending():
    df = build_recommendations(
        account_ids=pd.Series([1, 2, 3]),
        churn_probability=pd.Series([0.5, 0.5, 0.5]),
        predicted_uplift=pd.Series([0.01, 0.20, 0.10]),
        monthly_margin=40.0, months_credited=12, cost_per_account=45.0,
    )
    assert df["expected_value"].is_monotonic_decreasing


def test_build_recommendations_only_treats_positive_ev_accounts():
    df = build_recommendations(
        account_ids=pd.Series([1, 2]),
        churn_probability=pd.Series([0.5, 0.5]),
        predicted_uplift=pd.Series([0.001, 0.30]),
        monthly_margin=40.0, months_credited=12, cost_per_account=45.0,
    )
    treated = df[df["treated"]]
    assert (treated["expected_value"] > 0).all()
    untreated = df[~df["treated"]]
    assert (untreated["expected_value"] <= 0).all()


def test_budget_constraint_limits_number_treated():
    df = build_recommendations(
        account_ids=pd.Series(range(10)),
        churn_probability=pd.Series([0.5] * 10),
        predicted_uplift=pd.Series([0.30] * 10),  # all strongly positive EV
        monthly_margin=40.0, months_credited=12, cost_per_account=45.0,
        budget=100.0,  # only affords 2 accounts at $45 each
    )
    assert df["treated"].sum() <= 2


def test_summarize_portfolio_impact_matches_manual_calculation():
    df = build_recommendations(
        account_ids=pd.Series([1, 2]),
        churn_probability=pd.Series([0.5, 0.5]),
        predicted_uplift=pd.Series([0.30, 0.30]),
        monthly_margin=40.0, months_credited=12, cost_per_account=45.0,
    )
    impact = summarize_portfolio_impact(df, cost_per_account=45.0)
    assert impact["n_accounts_recommended"] == 2
    assert impact["total_intervention_cost"] == 90.0
