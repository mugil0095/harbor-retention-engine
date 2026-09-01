"""A: data generation. S: reproducibility."""

from __future__ import annotations

import numpy as np

from harbor.data.generate_synthetic_data import generate_all
from tests.conftest import make_test_config


def test_generated_tables_have_expected_row_counts(synthetic_tables, test_config):
    n = test_config.section("data")["n_accounts"]
    n_months = test_config.section("data")["n_months_history"]
    assert len(synthetic_tables["accounts"]) == n
    assert len(synthetic_tables["monthly_usage"]) == n * n_months


def test_generated_accounts_have_unique_ids(synthetic_tables):
    ids = synthetic_tables["accounts"]["account_id"]
    assert ids.is_unique
    assert ids.min() == 0


def test_churn_is_monotonic_once_churned_stays_churned(synthetic_tables):
    usage = synthetic_tables["monthly_usage"].sort_values(["account_id", "month_index"])
    for account_id, group in usage.groupby("account_id"):
        active = group["is_active"].to_numpy()
        # once inactive, must never become active again
        became_inactive = np.where(~active)[0]
        if len(became_inactive) > 0:
            first_inactive = became_inactive[0]
            assert not active[first_inactive:].any()


def test_reproducibility_same_seed_gives_identical_data():
    cfg = make_test_config()
    tables_a = generate_all(cfg)
    tables_b = generate_all(cfg)
    assert tables_a["accounts"].equals(tables_b["accounts"])
    assert tables_a["monthly_usage"].equals(tables_b["monthly_usage"])


def test_reproducibility_different_seed_gives_different_data():
    cfg_a = make_test_config()
    cfg_b = make_test_config(random_seed=999)
    tables_a = generate_all(cfg_a)
    tables_b = generate_all(cfg_b)
    assert not tables_a["monthly_usage"]["mrr"].equals(tables_b["monthly_usage"]["mrr"])


def test_experiment_has_both_arms_with_reasonable_balance(synthetic_tables):
    exp = synthetic_tables["experiment_assignments"]
    counts = exp["group"].value_counts()
    assert "treatment" in counts and "control" in counts
    ratio = counts["treatment"] / counts.sum()
    assert 0.3 < ratio < 0.7  # not a hard 50/50 given randomness, but should not be wildly skewed
