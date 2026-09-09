#!/usr/bin/env python
"""Drift monitoring: compares the current feature-store snapshot against the
reference snapshot saved at training time (models_store/feature_reference.parquet,
written by run_training_pipeline.py), computes per-feature PSI, and decides
whether retraining is recommended. Writes data/processed/drift_report.json --
the file the API's /monitoring/drift endpoint and the dashboard's Monitoring
tab read from.

Usage:
    python pipelines/run_monitoring.py
    python pipelines/run_monitoring.py --auto-retrain   # re-runs training if drift is flagged
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from harbor.config import load_config
from harbor.monitoring.drift import (
    DRIFT_MONITORED_FEATURES,
    build_drift_report,
    monitor_features,
)
from harbor.utils.artifacts import require_files_exist
from harbor.utils.logging_config import configure_logging

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--auto-retrain",
        action="store_true",
        help="Re-run the training pipeline if drift monitoring recommends it.",
    )
    args = parser.parse_args()

    configure_logging()
    cfg = load_config()
    data_cfg = cfg.section("data")
    models_dir = cfg.models_dir
    processed_dir = cfg.path(key="processed_dir")

    reference_path = models_dir / "feature_reference.parquet"
    feature_store_path = processed_dir / data_cfg["feature_store_file"]
    require_files_exist([reference_path, feature_store_path], context="Drift monitoring")

    reference_df = pd.read_parquet(reference_path)
    feature_store = pd.read_parquet(feature_store_path)
    current_month = data_cfg["test_snapshot_month"]
    current_df = feature_store[feature_store["snapshot_month"] == current_month]

    monitoring_cfg = cfg.section("monitoring")
    reports = monitor_features(
        reference_df,
        current_df,
        DRIFT_MONITORED_FEATURES,
        warning_threshold=monitoring_cfg["psi_warning_threshold"],
        critical_threshold=monitoring_cfg["psi_critical_threshold"],
    )
    report = build_drift_report(
        reports,
        reference_snapshot_month=int(reference_df["snapshot_month"].iloc[0]),
        current_snapshot_month=int(current_month),
    )

    out_path = processed_dir / "drift_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(
        "Drift report: %d features checked, %d warning, %d critical -> retraining_recommended=%s. Wrote %s",
        report["n_features_checked"], report["n_warning"], report["n_critical"],
        report["retraining_recommended"], out_path,
    )
    for feature in report["features"]:
        if feature["status"] != "stable":
            logger.info("  %-40s psi=%.4f status=%s", feature["feature"], feature["psi"], feature["status"])

    if report["retraining_recommended"] and args.auto_retrain:
        logger.warning("Retraining recommended -- invoking run_training_pipeline.py (--auto-retrain was set).")
        subprocess.run(
            [sys.executable, str(Path(__file__).resolve().parent / "run_training_pipeline.py")],
            check=True,
        )
    elif report["retraining_recommended"]:
        logger.warning("Retraining recommended -- re-run with --auto-retrain to retrain automatically.")


if __name__ == "__main__":
    main()
