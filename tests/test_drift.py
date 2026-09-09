"""R: data/prediction drift detection."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from harbor.monitoring.drift import (
    DRIFT_MONITORED_FEATURES,
    DriftReport,
    build_drift_report,
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


def test_build_drift_report_includes_expected_fields():
    reports = [DriftReport("stable_col", 0.02, "stable"), DriftReport("warned_col", 0.15, "warning")]
    report = build_drift_report(reports, reference_snapshot_month=24, current_snapshot_month=27)
    assert report["reference_snapshot_month"] == 24
    assert report["current_snapshot_month"] == 27
    assert report["n_features_checked"] == 2
    assert report["n_warning"] == 1
    assert report["n_critical"] == 0
    assert report["retraining_recommended"] is False  # only 1 warning, threshold is >=3
    assert "generated_at" in report
    assert report["features"] == [
        {"feature": "stable_col", "psi": 0.02, "status": "stable"},
        {"feature": "warned_col", "psi": 0.15, "status": "warning"},
    ]


def test_build_drift_report_synthetic_shift_triggers_retraining_flag():
    """Proof the mechanism actually fires, using a hard synthetic shift --
    independent of whatever the real (likely mild) drift between snapshot
    months turns out to be. Mirrors the CUPED/cohort-retention pattern of
    pairing a real-data result with a controlled sanity check."""
    rng = np.random.default_rng(0)
    reference_df = pd.DataFrame({f"col_{i}": rng.normal(0, 1, 2000) for i in range(4)})
    current_df = pd.DataFrame({f"col_{i}": rng.normal(6, 1, 2000) for i in range(4)})  # every column shifted hard
    reports = monitor_features(reference_df, current_df, list(reference_df.columns))
    report = build_drift_report(reports, reference_snapshot_month=24, current_snapshot_month=27)
    assert report["retraining_recommended"] is True
    assert report["n_critical"] >= 1


@pytest.mark.slow
def test_real_synthetic_snapshots_show_no_regime_change():
    """Honest-negative-result check on the actual DGP, at production scale:
    build two real, mature feature-store snapshots 3 months apart (matching
    configs/config.yaml's actual observation_cutoff_month -> test_snapshot_month
    gap) and confirm the drift monitor does NOT flag a regime change, since
    nothing in the synthetic DGP injects one -- only the mechanical effect of
    accounts aging.

    Two things had to be corrected to get an honest, non-flaky version of
    this test, both real findings worth recording rather than papering over:

    1. First attempt reused the shared tiny `test_config`/`synthetic_tables`
       fixtures (250 accounts) and compared the *earliest* snapshot (month 6)
       against the latest -- `tenure_months` (PSI 0.79) and `login_trend_6v12`
       (PSI 2.5!) both went "critical". Root cause: month 6 doesn't have a
       full 12-month trailing window yet, so avg_*_trailing_12m/trend
       features are still ramping up -- an immaturity artifact, not real
       drift. Fixed by comparing two already-mature snapshots (both >=12
       months in) instead, which real training (configs/config.yaml) always
       does.
    2. Even with mature snapshots, the tiny 250-account fixture with a
       6-month gap still tripped the >=3-warnings retraining trigger, driven
       mostly by `tenure_months` (PSI 0.92) and small-sample noise in
       per-account login/adoption fluctuation. `tenure_months` increases
       deterministically for every account every month by construction, so
       it "drifts" between ANY two snapshots regardless of whether the
       business changed -- monitoring it is pure noise. It's now excluded
       via DRIFT_MONITORED_FEATURES in harbor.monitoring.drift. And at
       small N, ordinary per-account behavioral noise (not a real regime
       change) is enough to cross PSI's warning threshold by chance on
       several features at once -- so this test uses the real production
       scale (n_accounts=3000) and the real 3-month gap, both of which make
       PSI far less noisy, matching how the actual pipeline runs. Marked
       slow: full 3000-account generation takes several seconds.
    """
    from harbor.data.generate_synthetic_data import generate_all
    from harbor.features.build_features import build_feature_store
    from harbor.nlp.ticket_analysis import compare_nlp_approaches
    from tests.conftest import make_test_config

    base_data = make_test_config().raw["data"]
    cfg = make_test_config(
        data={
            **base_data,
            "n_accounts": 3000,
            "n_months_history": 24,
            "snapshot_months": [15, 18, 21],
            "observation_cutoff_month": 18,
            "test_snapshot_month": 21,
        }
    )
    tables = generate_all(cfg)
    _, ticket_classifier = compare_nlp_approaches(tables["support_tickets"], random_seed=cfg.random_seed)
    data_cfg = cfg.section("data")
    feature_store = build_feature_store(
        tables["accounts"],
        tables["monthly_usage"],
        tables["support_tickets"],
        ticket_classifier,
        data_cfg["snapshot_months"],
        data_cfg["churn_horizon_months"],
    )
    reference_month = data_cfg["observation_cutoff_month"]
    current_month = data_cfg["test_snapshot_month"]
    reference_df = feature_store[feature_store["snapshot_month"] == reference_month]
    current_df = feature_store[feature_store["snapshot_month"] == current_month]

    monitored_features = [c for c in DRIFT_MONITORED_FEATURES if c in feature_store.columns]
    reports = monitor_features(reference_df, current_df, monitored_features)
    report = build_drift_report(reports, reference_snapshot_month=reference_month, current_snapshot_month=current_month)
    assert report["retraining_recommended"] is False
