#!/usr/bin/env python
"""Cohort retention analysis: buckets accounts by how mature they already
were at the start of the observation window, then tracks each bucket's
retention decay forward -- the classic SaaS "layer cake" chart. Independent
of the churn/uplift/segmentation modeling; reads only raw accounts.csv and
monthly_usage.csv, so it can run right after data generation.

Usage:
    python pipelines/run_cohort_analysis.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from harbor.analytics.cohort_retention import compute_retention_curve
from harbor.config import load_config
from harbor.utils.artifacts import require_files_exist
from harbor.utils.logging_config import configure_logging

logger = logging.getLogger(__name__)


def main() -> None:
    configure_logging()
    cfg = load_config()
    data_cfg = cfg.section("data")
    raw_dir = cfg.path(key="raw_dir")
    processed_dir = cfg.path(key="processed_dir")
    processed_dir.mkdir(parents=True, exist_ok=True)

    accounts_path = raw_dir / data_cfg["accounts_file"]
    usage_path = raw_dir / data_cfg["usage_file"]
    require_files_exist([accounts_path, usage_path], context="Cohort analysis")

    accounts = pd.read_csv(accounts_path)
    monthly_usage = pd.read_csv(usage_path)

    curve = compute_retention_curve(accounts, monthly_usage, n_cohort_bins=4)
    out_path = processed_dir / "cohort_retention.csv"
    curve.to_csv(out_path, index=False)
    logger.info("Wrote %d rows to %s", len(curve), out_path)

    last_month = curve["month_index"].max()
    final = curve[curve["month_index"] == last_month].sort_values("cohort")
    for _, row in final.iterrows():
        logger.info(
            "Cohort %s (n=%d): %.1f%% retained by month %d",
            row["cohort"], row["cohort_size"], row["retention_rate"] * 100, last_month,
        )
    spread = final["retention_rate"].max() - final["retention_rate"].min()
    logger.info("Retention spread across cohorts at month %d: %.1f pp", last_month, spread * 100)


if __name__ == "__main__":
    main()
