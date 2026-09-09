from __future__ import annotations

import numpy as np
import pytest

from harbor.config import HarborConfig
from harbor.data.generate_synthetic_data import generate_all


def make_test_config(**overrides) -> HarborConfig:
    raw = {
        "random_seed": 123,
        "data": {
            "n_accounts": 250,
            "n_months_history": 15,
            "observation_cutoff_month": 12,
            "churn_horizon_months": 3,
            "snapshot_months": [6, 9, 12],
            "test_snapshot_month": 12,
            "raw_dir": "data/raw",
            "processed_dir": "data/processed",
            "accounts_file": "accounts.csv",
            "usage_file": "monthly_usage.csv",
            "tickets_file": "support_tickets.csv",
            "experiment_file": "experiment_assignments.csv",
            "feature_store_file": "feature_store.parquet",
        },
        "experiment": {
            "eligible_risk_percentile": 0.6,
            "treatment_fraction": 0.5,
            "true_effect_high_engagement": -0.02,
            "true_effect_low_engagement": -0.12,
            "true_effect_default": -0.06,
        },
        "model": {
            "churn": {"test_size_months": 3, "cv_splits": 2},
            "forecasting": {"horizon_months": 3, "min_history_months": 6},
            "uplift": {"learner": "t_learner"},
        },
        "decision": {
            "intervention_cost_per_account": 45.0,
            "avg_monthly_margin_per_account": 38.0,
            "ltv_months_saved": 12,
            "intervention_tiers": [
                {"name": "email_nudge", "cost_per_account": 8.0, "effect_multiplier": 0.35},
                {"name": "standard_offer", "cost_per_account": 45.0, "effect_multiplier": 1.0},
                {"name": "white_glove_call", "cost_per_account": 150.0, "effect_multiplier": 1.6},
            ],
            "use_tiered_decision_engine": False,
        },
        "serving": {"host": "0.0.0.0", "port": 8000},
        "monitoring": {"psi_warning_threshold": 0.1, "psi_critical_threshold": 0.25},
    }
    for key, value in overrides.items():
        raw[key] = value
    return HarborConfig(raw=raw)


@pytest.fixture(scope="session")
def test_config() -> HarborConfig:
    return make_test_config()


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    return np.random.default_rng(123)


@pytest.fixture(scope="session")
def synthetic_tables(test_config):
    return generate_all(test_config)
