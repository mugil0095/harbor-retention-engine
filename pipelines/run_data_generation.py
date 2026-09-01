#!/usr/bin/env python
"""Generate (or regenerate) all synthetic raw data from configs/config.yaml.

Usage:
    python pipelines/run_data_generation.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from harbor.config import load_config
from harbor.data.generate_synthetic_data import generate_all, write_all
from harbor.data.schema import (
    ACCOUNTS,
    EXPERIMENT_ASSIGNMENTS,
    MONTHLY_USAGE,
    SUPPORT_TICKETS,
)
from harbor.data.validate import assert_valid, validate_table
from harbor.utils.logging_config import configure_logging

logger = logging.getLogger(__name__)


def main() -> None:
    configure_logging()
    cfg = load_config()
    logger.info("Generating synthetic data with seed=%d", cfg.random_seed)

    tables = generate_all(cfg)

    reference = {"accounts": tables["accounts"]}
    assert_valid(validate_table(tables["accounts"], ACCOUNTS))
    assert_valid(validate_table(tables["monthly_usage"], MONTHLY_USAGE, reference))
    assert_valid(validate_table(tables["support_tickets"], SUPPORT_TICKETS, reference))
    assert_valid(validate_table(tables["experiment_assignments"], EXPERIMENT_ASSIGNMENTS, reference))
    logger.info("All raw tables passed validation.")

    write_all(tables, cfg)

    logger.info("accounts: %d rows", len(tables["accounts"]))
    logger.info("monthly_usage: %d rows", len(tables["monthly_usage"]))
    logger.info("support_tickets: %d rows", len(tables["support_tickets"]))
    logger.info("experiment_assignments: %d rows", len(tables["experiment_assignments"]))
    overall_churn_rate = 1 - tables["monthly_usage"].groupby("account_id")["is_active"].last().mean()
    logger.info("Overall churn rate by end of history: %.1f%%", overall_churn_rate * 100)


if __name__ == "__main__":
    main()
