"""D: feature engineering correctness."""

from __future__ import annotations

import pytest

from harbor.features.build_features import (
    build_feature_store,
    compute_snapshot_features,
)
from harbor.nlp.ticket_analysis import compare_nlp_approaches


@pytest.fixture(scope="module")
def ticket_classifier(synthetic_tables):
    _, clf = compare_nlp_approaches(synthetic_tables["support_tickets"], random_seed=123)
    return clf


def test_feature_store_only_includes_accounts_active_at_cutoff(synthetic_tables, ticket_classifier):
    snapshot_month = 9
    horizon = 3
    features = compute_snapshot_features(
        synthetic_tables["accounts"], synthetic_tables["monthly_usage"], synthetic_tables["support_tickets"],
        ticket_classifier, snapshot_month, horizon,
    )
    usage = synthetic_tables["monthly_usage"]
    active_ids = set(
        usage[(usage["month_index"] == snapshot_month - 1) & usage["is_active"]]["account_id"]
    )
    assert set(features["account_id"]) == active_ids


def test_feature_store_has_no_missing_values_in_engineered_columns(synthetic_tables, ticket_classifier):
    features = compute_snapshot_features(
        synthetic_tables["accounts"], synthetic_tables["monthly_usage"], synthetic_tables["support_tickets"],
        ticket_classifier, 9, 3,
    )
    engineered_cols = [c for c in features.columns if c.startswith(("avg_", "ticket_count", "login_trend"))]
    assert features[engineered_cols].isna().sum().sum() == 0


def test_login_trend_is_negative_for_declining_accounts(synthetic_tables, ticket_classifier):
    features = compute_snapshot_features(
        synthetic_tables["accounts"], synthetic_tables["monthly_usage"], synthetic_tables["support_tickets"],
        ticket_classifier, 12, 3,
    )
    # accounts with near-zero recent login but nonzero 12m average are declining
    declining = features[
        (features["avg_login_trailing_12m"] > 1.0) & (features["avg_login_trailing_3m"] < 0.1 * features["avg_login_trailing_12m"])
    ]
    if len(declining) > 0:
        assert (declining["login_trend_3v6"] <= 0).mean() > 0.5


def test_label_reflects_future_churn_not_past(synthetic_tables, ticket_classifier):
    snapshot_month, horizon = 9, 3
    features = compute_snapshot_features(
        synthetic_tables["accounts"], synthetic_tables["monthly_usage"], synthetic_tables["support_tickets"],
        ticket_classifier, snapshot_month, horizon,
    )
    usage = synthetic_tables["monthly_usage"]
    future_churn = usage[
        (usage["month_index"] >= snapshot_month) & (usage["month_index"] < snapshot_month + horizon)
    ].groupby("account_id")["churned_this_month"].max()

    merged = features.set_index("account_id")["churn_within_horizon"]
    common = merged.index.intersection(future_churn.index)
    assert (merged.loc[common] == future_churn.loc[common].astype(int)).all()


def test_build_feature_store_stacks_multiple_snapshots(synthetic_tables, ticket_classifier):
    store = build_feature_store(
        synthetic_tables["accounts"], synthetic_tables["monthly_usage"], synthetic_tables["support_tickets"],
        ticket_classifier, [6, 9, 12], 3,
    )
    assert set(store["snapshot_month"].unique()) == {6, 9, 12}
    assert len(store) > 0


def test_ticket_features_are_zero_when_no_tickets_in_window(synthetic_tables, ticket_classifier):
    features = compute_snapshot_features(
        synthetic_tables["accounts"], synthetic_tables["monthly_usage"], synthetic_tables["support_tickets"],
        ticket_classifier, 6, 3,
    )
    no_ticket_accounts = features[features["ticket_count_trailing_3m"] == 0]
    if len(no_ticket_accounts) > 0:
        assert (no_ticket_accounts["avg_ticket_risk_proba_trailing_3m"] == 0.0).all()
