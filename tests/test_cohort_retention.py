"""Cohort retention curves: correctness of the cohort assignment and the
retention-rate computation, including the invariant the whole chart relies
on (retention within a cohort can only go down, never up)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from harbor.analytics.cohort_retention import (
    assign_tenure_cohort,
    compute_retention_curve,
)


def test_assign_tenure_cohort_returns_requested_number_of_bins():
    accounts = pd.DataFrame({"account_id": range(100), "signup_month_offset": np.arange(1, 101)})
    cohorts = assign_tenure_cohort(accounts, n_bins=4)
    assert cohorts.nunique() == 4


def test_assign_tenure_cohort_orders_newest_to_oldest():
    accounts = pd.DataFrame({"account_id": range(100), "signup_month_offset": np.arange(1, 101)})
    cohorts = assign_tenure_cohort(accounts, n_bins=4)
    mean_offset_by_cohort = accounts.groupby(cohorts, observed=True)["signup_month_offset"].mean()
    labels_in_order = sorted(mean_offset_by_cohort.index, key=lambda c: mean_offset_by_cohort[c])
    assert "newest" in labels_in_order[0]
    assert "oldest" in labels_in_order[-1]


def _tiny_lifecycle_data(seed=0):
    """A small, hand-checkable monthly_usage table: 20 accounts, 6 months,
    with a known, deterministic churn pattern so retention rates can be
    verified by direct computation, not just sanity-checked."""
    rng = np.random.default_rng(seed)
    n_accounts = 20
    n_months = 6
    accounts = pd.DataFrame(
        {"account_id": range(n_accounts), "signup_month_offset": rng.integers(1, 49, size=n_accounts)}
    )
    rows = []
    active = np.ones(n_accounts, dtype=bool)
    # deterministic: account i churns at month i (for i < n_months), rest stay active throughout
    churn_month = {i: i for i in range(n_months)}
    for month in range(n_months):
        for i in range(n_accounts):
            if i in churn_month and month >= churn_month[i]:
                active[i] = False
        rows.append(pd.DataFrame({"account_id": range(n_accounts), "month_index": month, "is_active": active.copy()}))
    monthly_usage = pd.concat(rows, ignore_index=True)
    return accounts, monthly_usage


def test_retention_rate_is_never_increasing_within_a_cohort():
    accounts, monthly_usage = _tiny_lifecycle_data()
    curve = compute_retention_curve(accounts, monthly_usage, n_cohort_bins=2)
    for _cohort, group in curve.groupby("cohort", observed=True):
        rates = group.sort_values("month_index")["retention_rate"].to_numpy()
        assert np.all(np.diff(rates) <= 1e-9)  # non-increasing, allowing for float noise


def test_retention_rate_at_first_month_reflects_known_churn():
    accounts, monthly_usage = _tiny_lifecycle_data()
    curve = compute_retention_curve(accounts, monthly_usage, n_cohort_bins=1)
    # by construction, account 0 churns at month 0 -> 19/20 active at month 0
    rate_month_0 = curve[curve["month_index"] == 0]["retention_rate"].iloc[0]
    assert rate_month_0 == pytest.approx(19 / 20)


def test_retention_rate_is_valid_probability_range():
    accounts, monthly_usage = _tiny_lifecycle_data()
    curve = compute_retention_curve(accounts, monthly_usage, n_cohort_bins=3)
    assert (curve["retention_rate"] >= 0).all()
    assert (curve["retention_rate"] <= 1).all()


def test_cohort_sizes_sum_to_total_accounts():
    accounts, monthly_usage = _tiny_lifecycle_data()
    curve = compute_retention_curve(accounts, monthly_usage, n_cohort_bins=4)
    sizes = curve.drop_duplicates("cohort")["cohort_size"]
    assert sizes.sum() == len(accounts)


def test_compute_retention_curve_on_real_synthetic_fixture(synthetic_tables):
    """End-to-end check against the project's own synthetic data generator,
    not just the hand-built fixture above."""
    curve = compute_retention_curve(synthetic_tables["accounts"], synthetic_tables["monthly_usage"], n_cohort_bins=4)
    assert curve["cohort"].nunique() == 4
    # retention should start high and decay -- not flat, not inverted
    first_month = curve["month_index"].min()
    last_month = curve["month_index"].max()
    avg_first = curve[curve["month_index"] == first_month]["retention_rate"].mean()
    avg_last = curve[curve["month_index"] == last_month]["retention_rate"].mean()
    assert avg_first > avg_last
    assert avg_first > 0.9  # near-100% at the very start of the window
