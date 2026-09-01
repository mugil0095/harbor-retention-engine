"""Canonical schema definitions for every table Harbor produces or consumes.

Keeping these as plain data (not just "whatever pandas infers") is what lets
`validate.py` catch a broken upstream change before it silently corrupts a
downstream model.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TableSchema:
    name: str
    required_columns: dict[str, str]  # column -> pandas dtype kind ("i","f","O","b","M")
    primary_key: tuple[str, ...]
    foreign_keys: dict[str, str]  # column -> referenced table name


ACCOUNTS = TableSchema(
    name="accounts",
    required_columns={
        "account_id": "i",
        "plan_tier": "O",
        "industry": "O",
        "employee_size_band": "O",
        "region": "O",
        "signup_month_offset": "i",
        "base_mrr": "f",
    },
    primary_key=("account_id",),
    foreign_keys={},
)

MONTHLY_USAGE = TableSchema(
    name="monthly_usage",
    required_columns={
        "account_id": "i",
        "month_index": "i",
        "active_users": "f",
        "login_count": "f",
        "api_calls": "f",
        "feature_adoption_count": "f",
        "mrr": "f",
        "is_active": "b",
        "churned_this_month": "b",
    },
    primary_key=("account_id", "month_index"),
    foreign_keys={"account_id": "accounts"},
)

SUPPORT_TICKETS = TableSchema(
    name="support_tickets",
    required_columns={
        "ticket_id": "i",
        "account_id": "i",
        "month_index": "i",
        "category": "O",
        "text": "O",
    },
    primary_key=("ticket_id",),
    foreign_keys={"account_id": "accounts"},
)

EXPERIMENT_ASSIGNMENTS = TableSchema(
    name="experiment_assignments",
    required_columns={
        "account_id": "i",
        "assigned_month": "i",
        "group": "O",
        "engagement_segment": "O",
        "observed_churn_90d": "b",
    },
    primary_key=("account_id",),
    foreign_keys={"account_id": "accounts"},
)

PLAN_TIERS = ("starter", "growth", "enterprise")
INDUSTRIES = ("retail", "fintech", "healthtech", "media", "logistics", "b2b_software")
SIZE_BANDS = ("micro", "small", "mid_market", "enterprise")
REGIONS = ("na", "emea", "apac", "latam")
TICKET_CATEGORIES = (
    "billing",
    "bug_report",
    "feature_request",
    "onboarding",
    "cancellation_intent",
    "general_praise",
)
