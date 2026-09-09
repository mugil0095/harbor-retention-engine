"""Decision engine: expected-value computation and budget-constrained ranking."""

from __future__ import annotations

import pandas as pd
import pytest

from harbor.decision.decision_engine import (
    InterventionTier,
    build_recommendations,
    build_tiered_recommendations,
    compute_expected_value,
    compute_tier_expected_value,
    summarize_portfolio_impact,
    summarize_tiered_portfolio_impact,
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


NUDGE = InterventionTier(name="email_nudge", cost_per_account=8.0, effect_multiplier=0.35)
STANDARD = InterventionTier(name="standard_offer", cost_per_account=45.0, effect_multiplier=1.0)
WHITE_GLOVE = InterventionTier(name="white_glove_call", cost_per_account=150.0, effect_multiplier=1.6)
TIERS = [NUDGE, STANDARD, WHITE_GLOVE]


def test_compute_tier_expected_value_scales_by_effect_multiplier():
    uplift = pd.Series([0.10])
    ev_standard = compute_tier_expected_value(uplift, STANDARD, monthly_margin=40.0, months_credited=12)
    ev_white_glove = compute_tier_expected_value(uplift, WHITE_GLOVE, monthly_margin=40.0, months_credited=12)
    # standard: 0.10 * 1.0 * 40 * 12 - 45 = 3.0; white_glove: 0.10 * 1.6 * 40 * 12 - 150 = -73.2
    assert ev_standard.iloc[0] == pytest.approx(3.0)
    assert ev_white_glove.iloc[0] == pytest.approx(-73.2)


def test_build_tiered_recommendations_picks_best_positive_ev_tier_per_account():
    # uplift=0.10: nudge EV=0.10*0.35*40*12-8=8.8, standard EV=3.0, white_glove EV=-3.0 -> nudge wins
    df = build_tiered_recommendations(
        account_ids=pd.Series([1]), churn_probability=pd.Series([0.5]), predicted_uplift=pd.Series([0.10]),
        tiers=TIERS, monthly_margin=40.0, months_credited=12,
    )
    assert df.loc[0, "recommended_action"] == "email_nudge"
    assert df.loc[0, "expected_value"] == pytest.approx(8.8)
    assert df.loc[0, "intervention_cost"] == 8.0


def test_build_tiered_recommendations_assigns_no_action_when_all_tiers_negative_ev():
    df = build_tiered_recommendations(
        account_ids=pd.Series([1]), churn_probability=pd.Series([0.5]), predicted_uplift=pd.Series([0.001]),
        tiers=TIERS, monthly_margin=40.0, months_credited=12,
    )
    assert df.loc[0, "recommended_action"] == "no_action"
    assert df.loc[0, "treated"] == False
    assert df.loc[0, "intervention_cost"] == 0.0


def test_build_tiered_recommendations_tie_broken_by_tier_list_order():
    """Two tiers with identical multipliers and cost produce identical EV --
    the earlier one in the tiers list must win, deterministically."""
    tier_a = InterventionTier(name="tier_a", cost_per_account=45.0, effect_multiplier=1.0)
    tier_b = InterventionTier(name="tier_b", cost_per_account=45.0, effect_multiplier=1.0)
    df = build_tiered_recommendations(
        account_ids=pd.Series([1]), churn_probability=pd.Series([0.5]), predicted_uplift=pd.Series([0.20]),
        tiers=[tier_a, tier_b], monthly_margin=40.0, months_credited=12,
    )
    assert df.loc[0, "recommended_action"] == "tier_a"


def test_build_tiered_recommendations_respects_budget_by_dropping_lowest_ev_first():
    df = build_tiered_recommendations(
        account_ids=pd.Series([1, 2, 3]),
        churn_probability=pd.Series([0.5, 0.5, 0.5]),
        predicted_uplift=pd.Series([0.30, 0.20, 0.10]),  # decreasing EV
        tiers=[STANDARD], monthly_margin=40.0, months_credited=12,
        budget=45.0,  # affords exactly 1 account
    )
    assert df["treated"].sum() == 1
    assert df.loc[df["treated"], "account_id"].iloc[0] == 1  # highest predicted_uplift kept


def test_build_tiered_recommendations_raises_on_empty_tier_list():
    with pytest.raises(ValueError):
        build_tiered_recommendations(
            account_ids=pd.Series([1]), churn_probability=pd.Series([0.5]), predicted_uplift=pd.Series([0.10]),
            tiers=[], monthly_margin=40.0, months_credited=12,
        )


def test_summarize_tiered_portfolio_impact_reports_tier_counts_and_total_cost():
    df = build_tiered_recommendations(
        account_ids=pd.Series([1, 2]),
        churn_probability=pd.Series([0.5, 0.5]),
        predicted_uplift=pd.Series([0.10, 0.30]),
        tiers=TIERS, monthly_margin=40.0, months_credited=12,
    )
    impact = summarize_tiered_portfolio_impact(df)
    assert impact["n_accounts_recommended"] == 2
    assert impact["total_intervention_cost"] == pytest.approx(df["intervention_cost"].sum())
    assert sum(impact["tier_counts"].values()) == 2
