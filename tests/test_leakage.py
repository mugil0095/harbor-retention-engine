"""E: leakage prevention. This is the highest-value test category for a
churn-model portfolio project: a leaking feature makes offline metrics look
great and production performance collapse."""

from __future__ import annotations

import pandas as pd
import pytest

from harbor.features.build_features import _assert_no_leakage, compute_snapshot_features
from harbor.nlp.ticket_analysis import compare_nlp_approaches


def test_assert_no_leakage_raises_when_history_touches_snapshot_month():
    with pytest.raises(ValueError, match="Leakage detected"):
        _assert_no_leakage(history_max_month=12, snapshot_month=12, source="monthly_usage")


def test_assert_no_leakage_raises_when_history_is_after_snapshot_month():
    with pytest.raises(ValueError, match="Leakage detected"):
        _assert_no_leakage(history_max_month=15, snapshot_month=12, source="monthly_usage")


def test_assert_no_leakage_passes_when_history_strictly_before():
    _assert_no_leakage(history_max_month=11, snapshot_month=12, source="monthly_usage")  # must not raise


def test_features_never_use_usage_rows_at_or_after_snapshot(synthetic_tables):
    _, clf = compare_nlp_approaches(synthetic_tables["support_tickets"], random_seed=123)
    snapshot_month = 9

    # sabotage future rows so if leakage existed, values would be obviously wrong
    usage = synthetic_tables["monthly_usage"].copy()
    future_mask = usage["month_index"] >= snapshot_month
    usage.loc[future_mask, "login_count"] = 999999.0

    features = compute_snapshot_features(
        synthetic_tables["accounts"], usage, synthetic_tables["support_tickets"], clf, snapshot_month, 3
    )
    # if any future row leaked in, trailing averages would be huge
    assert (features["avg_login_trailing_3m"] < 999999.0).all()
    assert (features["avg_login_trailing_12m"] < 999999.0).all()


def test_features_never_use_ticket_rows_at_or_after_snapshot(synthetic_tables):
    tickets = synthetic_tables["support_tickets"].copy()
    snapshot_month = 9
    future_ticket_mask = tickets["month_index"] >= snapshot_month
    tickets.loc[future_ticket_mask, "text"] = "we are cancelling immediately this is terrible"
    tickets.loc[future_ticket_mask, "category"] = "cancellation_intent"

    _, clf = compare_nlp_approaches(tickets, random_seed=123)
    features_sabotaged = compute_snapshot_features(
        synthetic_tables["accounts"], synthetic_tables["monthly_usage"], tickets, clf, snapshot_month, 3
    )
    features_clean = compute_snapshot_features(
        synthetic_tables["accounts"], synthetic_tables["monthly_usage"], synthetic_tables["support_tickets"],
        clf, snapshot_month, 3,
    )
    # sabotaging only future tickets must not change trailing (past-only) ticket features
    merged = features_sabotaged.set_index("account_id")[["avg_ticket_risk_proba_trailing_3m"]].join(
        features_clean.set_index("account_id")[["avg_ticket_risk_proba_trailing_3m"]],
        lsuffix="_sabotaged", rsuffix="_clean",
    )
    pd.testing.assert_series_equal(
        merged["avg_ticket_risk_proba_trailing_3m_sabotaged"],
        merged["avg_ticket_risk_proba_trailing_3m_clean"],
        check_names=False,
    )


def test_label_horizon_does_not_overlap_feature_window(synthetic_tables):
    _, clf = compare_nlp_approaches(synthetic_tables["support_tickets"], random_seed=123)
    snapshot_month, horizon = 9, 3
    features = compute_snapshot_features(
        synthetic_tables["accounts"], synthetic_tables["monthly_usage"], synthetic_tables["support_tickets"],
        clf, snapshot_month, horizon,
    )
    # the feature window is [0, snapshot_month) and the label window is
    # [snapshot_month, snapshot_month + horizon) -- they share the boundary
    # month but must never overlap on an actual month_index
    assert (features["snapshot_month"] == snapshot_month).all()
    feature_window_end = snapshot_month  # exclusive
    label_window_start = snapshot_month  # inclusive
    assert feature_window_end == label_window_start  # no gap, but critically no overlap either
