"""Data and prediction drift monitoring via Population Stability Index (PSI).

PSI is the standard, dependency-light drift metric used in credit-risk and
churn-model monitoring: bin a reference distribution, compare the proportion
of a new sample falling in each bin, and summarize the divergence in one
number. Thresholds follow the conventional industry rule of thumb
(< 0.1 stable, 0.1-0.25 moderate shift / investigate, > 0.25 significant
shift / retrain).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from harbor.features.build_features import NUMERIC_FEATURE_COLUMNS

# tenure_months is excluded from drift monitoring: it increases
# deterministically for every account, every month, by construction -- it
# "drifts" between any two snapshots regardless of whether anything about
# the business actually changed (verified: on real snapshots it alone
# produces PSI > 0.9 purely from accounts aging). Monitoring it would be
# pure noise, not signal, and would falsely trigger retraining on a
# perfectly healthy model. A real system would either exclude counters like
# this or monitor their *rate* of change instead -- out of scope here.
DRIFT_MONITORED_FEATURES = [c for c in NUMERIC_FEATURE_COLUMNS if c != "tenure_months"]


@dataclass
class DriftReport:
    feature: str
    psi: float
    status: str  # "stable", "warning", "critical"


def _psi_bins(reference: np.ndarray, n_bins: int = 10) -> np.ndarray:
    quantiles = np.linspace(0, 100, n_bins + 1)
    edges = np.unique(np.percentile(reference, quantiles))
    if len(edges) < 3:
        edges = np.linspace(reference.min(), reference.max() + 1e-9, n_bins + 1)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def population_stability_index(
    reference: np.ndarray, current: np.ndarray, n_bins: int = 10, epsilon: float = 1e-4
) -> float:
    edges = _psi_bins(reference, n_bins)
    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)

    ref_pct = ref_counts / max(ref_counts.sum(), 1) + epsilon
    cur_pct = cur_counts / max(cur_counts.sum(), 1) + epsilon

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def classify_psi(psi: float, warning_threshold: float = 0.1, critical_threshold: float = 0.25) -> str:
    if psi >= critical_threshold:
        return "critical"
    if psi >= warning_threshold:
        return "warning"
    return "stable"


def monitor_features(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    feature_columns: list[str],
    warning_threshold: float = 0.1,
    critical_threshold: float = 0.25,
) -> list[DriftReport]:
    reports = []
    for col in feature_columns:
        ref_vals = reference_df[col].dropna().to_numpy(dtype=float)
        cur_vals = current_df[col].dropna().to_numpy(dtype=float)
        if len(ref_vals) < 10 or len(cur_vals) < 10:
            continue
        psi = population_stability_index(ref_vals, cur_vals)
        reports.append(
            DriftReport(
                feature=col,
                psi=psi,
                status=classify_psi(psi, warning_threshold, critical_threshold),
            )
        )
    return reports


def retraining_recommended(reports: list[DriftReport]) -> bool:
    """Retraining trigger: any single critical-drift feature, or three-plus
    features simultaneously in warning, indicates the training distribution
    no longer represents production and the model should be retrained."""
    n_critical = sum(1 for r in reports if r.status == "critical")
    n_warning = sum(1 for r in reports if r.status == "warning")
    return n_critical >= 1 or n_warning >= 3


def build_drift_report(
    reports: list[DriftReport], reference_snapshot_month: int, current_snapshot_month: int
) -> dict:
    """Pure assembly of the JSON payload run_monitoring.py writes to disk --
    kept separate from any file I/O so it's unit-testable on its own."""
    n_critical = sum(1 for r in reports if r.status == "critical")
    n_warning = sum(1 for r in reports if r.status == "warning")
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "reference_snapshot_month": reference_snapshot_month,
        "current_snapshot_month": current_snapshot_month,
        "n_features_checked": len(reports),
        "n_critical": n_critical,
        "n_warning": n_warning,
        "retraining_recommended": retraining_recommended(reports),
        "features": [
            {"feature": r.feature, "psi": r.psi, "status": r.status} for r in reports
        ],
    }
