"""The decision layer: turns a churn-risk score and a causal uplift estimate
into a $-denominated recommendation per account. This is the module that
makes the project a *decision engine* rather than two separate models --
neither the churn probability nor the uplift estimate alone tells a retention
team what to do; the combination, priced against the cost of intervening,
does.

expected_value = predicted_uplift (churn-prob reduction) * monthly_margin *
                  months_credited  -  cost_of_intervention

An account is only worth treating if this is positive. Under a budget
constraint, accounts are ranked by expected value and funded top-down until
the budget runs out (a real ops team can't treat everyone even when EV > 0
for all of them).
"""

from __future__ import annotations

import pandas as pd

# Deliberately a *small* numeric-only feature set. The RCT population is a
# few hundred accounts per arm; one-hot-encoding plan_tier/industry/size on
# top of that spreads what little signal exists across many sparse columns
# and the uplift models overfit noise instead of recovering the true
# heterogeneity. These five columns are the features most directly related
# to the engagement decline that the experiment's eligibility and true
# effect were actually built from (see generate_synthetic_data.simulate_experiment).
UPLIFT_FEATURE_COLUMNS = [
    "tenure_months",
    "avg_login_trailing_3m",
    "avg_login_trailing_6m",
    "login_trend_3v6",
    "avg_ticket_risk_proba_trailing_3m",
]


def compute_expected_value(
    predicted_uplift: pd.Series,
    monthly_margin: float,
    months_credited: int,
    cost_per_account: float,
) -> pd.Series:
    expected_months_saved_value = predicted_uplift * monthly_margin * months_credited
    return expected_months_saved_value - cost_per_account


def build_recommendations(
    account_ids: pd.Series,
    churn_probability: pd.Series,
    predicted_uplift: pd.Series,
    monthly_margin: float,
    months_credited: int,
    cost_per_account: float,
    budget: float | None = None,
) -> pd.DataFrame:
    expected_value = compute_expected_value(
        predicted_uplift, monthly_margin, months_credited, cost_per_account
    )
    df = pd.DataFrame(
        {
            "account_id": account_ids.to_numpy(),
            "churn_probability": churn_probability.to_numpy(),
            "predicted_uplift": predicted_uplift.to_numpy(),
            "expected_value": expected_value.to_numpy(),
        }
    ).sort_values("expected_value", ascending=False).reset_index(drop=True)

    df["recommended_action"] = "no_action"
    positive_ev_mask = df["expected_value"] > 0

    if budget is None:
        df.loc[positive_ev_mask, "recommended_action"] = "retention_outreach"
    else:
        cumulative_cost = (positive_ev_mask.cumsum() * cost_per_account)
        within_budget = positive_ev_mask & (cumulative_cost <= budget)
        df.loc[within_budget, "recommended_action"] = "retention_outreach"

    df["treated"] = df["recommended_action"] == "retention_outreach"
    return df


def summarize_portfolio_impact(recommendations: pd.DataFrame, cost_per_account: float) -> dict:
    treated = recommendations[recommendations["treated"]]
    return {
        "n_accounts_scored": len(recommendations),
        "n_accounts_recommended": len(treated),
        "total_intervention_cost": float(len(treated) * cost_per_account),
        "total_expected_value": float(treated["expected_value"].sum()),
        "expected_net_roi": (
            float(treated["expected_value"].sum() / (len(treated) * cost_per_account))
            if len(treated) > 0
            else float("nan")
        ),
    }
