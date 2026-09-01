"""R: data/prediction drift detection."""

from __future__ import annotations

import numpy as np
import pandas as pd

from harbor.monitoring.drift import (
    classify_psi,
    monitor_features,
    population_stability_index,
    retraining_recommended,
)


def test_psi_is_near_zero_for_identical_distributions():
    rng = np.random.default_rng(0)
    reference = rng.normal(0, 1, 5000)
    current = reference.copy()
    psi = population_stability_index(reference, current)
    assert psi < 0.01


def test_psi_is_large_for_shifted_distribution():
    rng = np.random.default_rng(0)
    reference = rng.normal(0, 1, 5000)
    current = rng.normal(3, 1, 5000)  # large mean shift
    psi = population_stability_index(reference, current)
    assert psi > 0.25


def test_classify_psi_thresholds():
    assert classify_psi(0.05) == "stable"
    assert classify_psi(0.15) == "warning"
    assert classify_psi(0.30) == "critical"


def test_monitor_features_flags_drifted_column():
    rng = np.random.default_rng(0)
    reference_df = pd.DataFrame({"stable_col": rng.normal(0, 1, 2000), "drifted_col": rng.normal(0, 1, 2000)})
    current_df = pd.DataFrame({"stable_col": rng.normal(0, 1, 2000), "drifted_col": rng.normal(5, 1, 2000)})
    reports = monitor_features(reference_df, current_df, ["stable_col", "drifted_col"])
    report_by_col = {r.feature: r for r in reports}
    assert report_by_col["stable_col"].status == "stable"
    assert report_by_col["drifted_col"].status == "critical"


def test_retraining_recommended_triggers_on_single_critical_feature():
    from harbor.monitoring.drift import DriftReport

    reports = [DriftReport("a", 0.3, "critical"), DriftReport("b", 0.02, "stable")]
    assert retraining_recommended(reports)


def test_retraining_not_recommended_when_all_stable():
    from harbor.monitoring.drift import DriftReport

    reports = [DriftReport("a", 0.02, "stable"), DriftReport("b", 0.03, "stable")]
    assert not retraining_recommended(reports)


def test_retraining_recommended_triggers_on_multiple_warnings():
    from harbor.monitoring.drift import DriftReport

    reports = [DriftReport(f"f{i}", 0.15, "warning") for i in range(3)]
    assert retraining_recommended(reports)
