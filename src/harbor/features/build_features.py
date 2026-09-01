"""Feature engineering: raw tables -> one row per (account, snapshot_month).

Every feature for a given snapshot is computed strictly from data with
`month_index < snapshot_month`. The label looks strictly at
`[snapshot_month, snapshot_month + horizon)`. `_assert_no_leakage` enforces
this at runtime rather than trusting it by convention, because feature
leakage is the single most common way churn-model portfolios quietly cheat.

Multiple snapshot months are stacked into one "feature store" so training
can use a genuine time-aware split (train on earlier snapshots, test on the
latest one) instead of a random row split, which would let information from
one account's later snapshot leak alongside an earlier one.
"""

from __future__ import annotations

import pandas as pd

from harbor.nlp.ticket_analysis import TicketRiskClassifier, vader_compound_scores

LEAKAGE_SENTINEL_COLUMNS = {"mrr", "active_users", "login_count", "api_calls", "feature_adoption_count"}


def _assert_no_leakage(history_max_month: int, snapshot_month: int, source: str) -> None:
    if history_max_month >= snapshot_month:
        raise ValueError(
            f"Leakage detected building features from '{source}': history contains "
            f"month_index {history_max_month} >= snapshot_month {snapshot_month}."
        )


def _trailing_mean(usage_hist: pd.DataFrame, col: str, cutoff: int, window: int) -> pd.Series:
    window_df = usage_hist[usage_hist["month_index"] >= cutoff - window]
    return window_df.groupby("account_id")[col].mean()


def compute_snapshot_features(
    accounts: pd.DataFrame,
    monthly_usage: pd.DataFrame,
    tickets: pd.DataFrame,
    ticket_classifier: TicketRiskClassifier,
    snapshot_month: int,
    horizon: int,
) -> pd.DataFrame:
    usage_hist = monthly_usage[monthly_usage["month_index"] < snapshot_month]
    tickets_hist = tickets[tickets["month_index"] < snapshot_month]
    if len(usage_hist) > 0:
        _assert_no_leakage(int(usage_hist["month_index"].max()), snapshot_month, "monthly_usage")
    if len(tickets_hist) > 0:
        _assert_no_leakage(int(tickets_hist["month_index"].max()), snapshot_month, "support_tickets")

    active_flags = usage_hist[usage_hist["month_index"] == snapshot_month - 1].set_index(
        "account_id"
    )["is_active"]
    active_ids = active_flags[active_flags].index.to_numpy()
    if len(active_ids) == 0:
        return pd.DataFrame()

    feat = accounts[accounts["account_id"].isin(active_ids)].copy()
    feat = feat.set_index("account_id")

    for window in (3, 6, 12):
        feat[f"avg_login_trailing_{window}m"] = _trailing_mean(
            usage_hist, "login_count", snapshot_month, window
        )
        feat[f"avg_api_calls_trailing_{window}m"] = _trailing_mean(
            usage_hist, "api_calls", snapshot_month, window
        )
        feat[f"avg_feature_adoption_trailing_{window}m"] = _trailing_mean(
            usage_hist, "feature_adoption_count", snapshot_month, window
        )

    feat["login_trend_3v6"] = (
        feat["avg_login_trailing_3m"] - feat["avg_login_trailing_6m"]
    ) / (feat["avg_login_trailing_6m"].abs() + 1.0)
    feat["login_trend_6v12"] = (
        feat["avg_login_trailing_6m"] - feat["avg_login_trailing_12m"]
    ) / (feat["avg_login_trailing_12m"].abs() + 1.0)

    current_mrr = usage_hist[usage_hist["month_index"] == snapshot_month - 1].set_index(
        "account_id"
    )["mrr"]
    feat["current_mrr"] = current_mrr
    feat["tenure_months"] = feat["signup_month_offset"] + snapshot_month

    for window in (3, 6):
        ticket_window = tickets_hist[tickets_hist["month_index"] >= snapshot_month - window]
        counts = ticket_window.groupby("account_id").size()
        feat[f"ticket_count_trailing_{window}m"] = counts

        if len(ticket_window) > 0:
            risk_proba = ticket_classifier.predict_proba(ticket_window["text"])
            sentiment = vader_compound_scores(ticket_window["text"])
            ticket_window = ticket_window.assign(_risk_proba=risk_proba, _sentiment=sentiment)
            agg = ticket_window.groupby("account_id").agg(
                **{
                    f"avg_ticket_risk_proba_trailing_{window}m": ("_risk_proba", "mean"),
                    f"avg_ticket_sentiment_trailing_{window}m": ("_sentiment", "mean"),
                }
            )
            feat = feat.join(agg)

    ticket_cols = [c for c in feat.columns if c.startswith(("ticket_count_", "avg_ticket_"))]
    feat[ticket_cols] = feat[ticket_cols].fillna(0.0)

    usage_cols = [c for c in feat.columns if c.startswith(("avg_login", "avg_api", "avg_feature"))]
    feat[usage_cols] = feat[usage_cols].fillna(0.0)
    feat["login_trend_3v6"] = feat["login_trend_3v6"].fillna(0.0)
    feat["login_trend_6v12"] = feat["login_trend_6v12"].fillna(0.0)
    feat["current_mrr"] = feat["current_mrr"].fillna(0.0)

    feat["low_plan_and_declining"] = (
        (feat["plan_tier"] == "starter") & (feat["login_trend_3v6"] < -0.1)
    ).astype(int)

    future = monthly_usage[
        (monthly_usage["month_index"] >= snapshot_month)
        & (monthly_usage["month_index"] < snapshot_month + horizon)
    ]
    label = future.groupby("account_id")["churned_this_month"].max()
    feat["churn_within_horizon"] = label.reindex(feat.index).fillna(False).astype(int)

    feat["snapshot_month"] = snapshot_month
    feat = feat.reset_index()
    return feat


def build_feature_store(
    accounts: pd.DataFrame,
    monthly_usage: pd.DataFrame,
    tickets: pd.DataFrame,
    ticket_classifier: TicketRiskClassifier,
    snapshot_months: list[int],
    horizon: int,
) -> pd.DataFrame:
    frames = [
        compute_snapshot_features(accounts, monthly_usage, tickets, ticket_classifier, m, horizon)
        for m in snapshot_months
    ]
    frames = [f for f in frames if len(f) > 0]
    return pd.concat(frames, ignore_index=True)


NUMERIC_FEATURE_COLUMNS = [
    "base_mrr",
    "current_mrr",
    "tenure_months",
    "avg_login_trailing_3m",
    "avg_login_trailing_6m",
    "avg_login_trailing_12m",
    "avg_api_calls_trailing_3m",
    "avg_api_calls_trailing_6m",
    "avg_api_calls_trailing_12m",
    "avg_feature_adoption_trailing_3m",
    "avg_feature_adoption_trailing_6m",
    "avg_feature_adoption_trailing_12m",
    "login_trend_3v6",
    "login_trend_6v12",
    "ticket_count_trailing_3m",
    "ticket_count_trailing_6m",
    "avg_ticket_risk_proba_trailing_3m",
    "avg_ticket_risk_proba_trailing_6m",
    "avg_ticket_sentiment_trailing_3m",
    "avg_ticket_sentiment_trailing_6m",
    "low_plan_and_declining",
]

CATEGORICAL_FEATURE_COLUMNS = ["plan_tier", "industry", "employee_size_band", "region"]
