"""Cohort retention curves -- the classic SaaS "layer cake" chart.

Every account in this dataset already existed before `month_index=0` with a
different starting age (`signup_month_offset`, 1-48 months), so there is no
clean set of accounts that all signed up in the same calendar month
observable within the simulated window. What *is* available, and is a
standard, meaningful cut in its own right, is a **maturity/vintage
cohort**: bucket accounts by how old they already were at the start of the
observation window, then track how each bucket's retention decays going
forward. This answers a real question a retention team asks -- "do older
accounts churn differently than newer ones?" -- which a pure behavioral
segmentation (see `harbor.segmentation`) doesn't directly address.
"""

from __future__ import annotations

import pandas as pd


def assign_tenure_cohort(accounts: pd.DataFrame, n_bins: int = 4) -> pd.Series:
    """Quartile (or n_bins-tile) cohort by signup_month_offset -- Q1 is the
    accounts that were newest (smallest offset, least tenure) at
    month_index=0, the last bin the most tenured."""
    labels = [f"Q{i + 1}" for i in range(n_bins)]
    labels[0] += " (newest)"
    labels[-1] += " (oldest)"
    return pd.qcut(accounts["signup_month_offset"], q=n_bins, labels=labels)


def compute_retention_curve(
    accounts: pd.DataFrame, monthly_usage: pd.DataFrame, n_cohort_bins: int = 4
) -> pd.DataFrame:
    """One row per (cohort, month_index): the fraction of that cohort's
    accounts still active at that month, plus the cohort's starting size
    for context. `is_active` can only go True -> False once an account
    churns (never back), so `retention_rate` is guaranteed non-increasing
    within a cohort as month_index advances -- a real invariant of the
    underlying data, not just an expectation about this chart.
    """
    cohorts = assign_tenure_cohort(accounts, n_cohort_bins)
    cohort_map = pd.DataFrame({"account_id": accounts["account_id"], "cohort": cohorts})
    merged = monthly_usage.merge(cohort_map, on="account_id", how="left")

    cohort_sizes = merged[merged["month_index"] == merged["month_index"].min()].groupby("cohort", observed=True)[
        "account_id"
    ].count()

    curve = (
        merged.groupby(["cohort", "month_index"], observed=True)["is_active"]
        .mean()
        .reset_index()
        .rename(columns={"is_active": "retention_rate"})
    )
    curve["cohort_size"] = curve["cohort"].map(cohort_sizes)
    return curve.sort_values(["cohort", "month_index"]).reset_index(drop=True)
