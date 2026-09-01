"""Synthetic data generator for Harbor.

Why synthetic: no public dataset combines subscription billing, product
usage, support-ticket text, and a retention experiment for the same accounts.
Rather than stitch together five unrelated public datasets (which is exactly
the "disconnected notebooks" failure mode this project avoids), we simulate
one internally-consistent data-generating process (DGP) with a real latent
"account health" variable driving usage decline, support sentiment, and churn
hazard together. Because we control the DGP, we can also validate that the
causal/uplift models recover the *true* simulated treatment effect — a
sanity check real-world projects rarely get to run.

Reproduction: `python pipelines/run_data_generation.py` regenerates every
file byte-for-byte from `configs/config.yaml`'s random_seed. No external
downloads are involved.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from harbor.config import HarborConfig
from harbor.data.schema import (
    INDUSTRIES,
    PLAN_TIERS,
    REGIONS,
    SIZE_BANDS,
    TICKET_CATEGORIES,
)

logger = logging.getLogger(__name__)

PLAN_BASE_MRR = {"starter": 49.0, "growth": 199.0, "enterprise": 799.0}
SIZE_MULTIPLIER = {"micro": 0.8, "small": 1.0, "mid_market": 1.4, "enterprise": 2.2}

_TICKET_TEMPLATES: dict[str, list[str]] = {
    "billing": [
        "Can you clarify the charge on our latest invoice for {org}?",
        "We were billed twice this month, please refund the duplicate charge.",
        "Requesting an itemized invoice for our {plan} plan.",
        "Our card on file expired, please update payment before renewal.",
    ],
    "bug_report": [
        "The export feature is throwing an error when we try to download reports.",
        "Dashboard is loading very slowly for our team today.",
        "We found a bug where the {feature} tab shows stale data.",
        "Getting a 500 error when saving settings, can someone look into this.",
    ],
    "feature_request": [
        "It would help a lot if you added bulk editing to {feature}.",
        "Any plans to support single sign-on for the {plan} plan?",
        "We'd like an API endpoint for pulling usage metrics automatically.",
    ],
    "onboarding": [
        "New team members are having trouble finding the setup guide.",
        "Could we get a walkthrough of {feature} for our new admin?",
        "Onboarding checklist seems incomplete, several steps are unclear.",
    ],
    "cancellation_intent": [
        "We are evaluating other vendors and considering cancelling next quarter.",
        "This tool no longer fits our budget, please send cancellation steps.",
        "Frustrated with repeated issues, we may not renew our {plan} subscription.",
        "Please confirm how to close our account, we have decided to switch providers.",
    ],
    "general_praise": [
        "Just wanted to say the new {feature} update has been great for our team.",
        "Support resolved our issue quickly, really appreciate the help.",
        "Loving the product so far, {org} team is very happy with results.",
    ],
}

_ORG_WORDS = ["Acme", "Northwind", "Vertex", "Pioneer", "Summit", "Beacon", "Harborlight", "Cobalt"]

# Shared, category-agnostic filler sentences. Real support tickets share a
# lot of boilerplate (greetings, sign-offs, account references) regardless
# of what they're actually about; mixing this in keeps the classification
# task learnable without letting a classifier win on category-specific
# keywords alone -- without it, TF-IDF trivially separates the hand-written
# per-category templates with ~perfect accuracy, which would overstate how
# well this approach works on real, noisier support text.
_SHARED_FILLER = [
    "Hope you are doing well.",
    "Thanks in advance for your help.",
    "Let me know if you need more details from our side.",
    "We have been a customer for a while now.",
    "Appreciate the quick response on previous tickets.",
    "Please loop in the account manager if needed.",
]


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _seasonal_factor(month_index: np.ndarray, industry: np.ndarray) -> np.ndarray:
    """Small, industry-flavored seasonality so forecasting has real signal
    to model instead of a pure trend line."""
    phase = pd.Series(industry).map(
        {
            "retail": 0.0,
            "fintech": 1.0,
            "healthtech": 2.0,
            "media": 3.0,
            "logistics": 4.0,
            "b2b_software": 5.0,
        }
    ).to_numpy()
    return 1.0 + 0.12 * np.sin(2 * np.pi * (month_index + phase * 2) / 12.0)


def generate_accounts(cfg: HarborConfig, rng: np.random.Generator) -> pd.DataFrame:
    n = cfg.section("data")["n_accounts"]
    account_id = np.arange(n, dtype=np.int64)
    plan_tier = rng.choice(PLAN_TIERS, size=n, p=[0.55, 0.35, 0.10])
    industry = rng.choice(INDUSTRIES, size=n)
    employee_size_band = rng.choice(SIZE_BANDS, size=n, p=[0.35, 0.30, 0.25, 0.10])
    region = rng.choice(REGIONS, size=n, p=[0.45, 0.30, 0.18, 0.07])
    signup_month_offset = rng.integers(1, 49, size=n)  # tenure in months before month_index 0

    base_mrr = np.array([PLAN_BASE_MRR[p] for p in plan_tier]) * np.array(
        [SIZE_MULTIPLIER[s] for s in employee_size_band]
    )
    base_mrr = base_mrr * rng.lognormal(mean=0.0, sigma=0.08, size=n)

    return pd.DataFrame(
        {
            "account_id": account_id,
            "plan_tier": plan_tier,
            "industry": industry,
            "employee_size_band": employee_size_band,
            "region": region,
            "signup_month_offset": signup_month_offset,
            "base_mrr": np.round(base_mrr, 2),
        }
    )


def _simulate_latent_health(
    n: int, n_months: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (effective_latent[n, n_months], base_latent[n]).

    base_latent: fixed inherent account-health tendency.
    shocks: ~35% of accounts hit a negative shock at a random month (e.g. a
    champion leaves, a bad support experience) that raises risk from that
    point forward. This is what makes rolling/recency features necessary --
    a model using only static signup-time attributes cannot see the shock.
    """
    base_latent = rng.normal(loc=0.0, scale=0.55, size=n)
    has_shock = rng.random(n) < 0.35
    shock_month = np.where(has_shock, rng.integers(3, n_months - 2, size=n), n_months + 1)
    shock_magnitude = rng.uniform(0.8, 2.2, size=n)

    months = np.arange(n_months)
    effective = base_latent[:, None] + np.where(
        months[None, :] >= shock_month[:, None], shock_magnitude[:, None], 0.0
    )
    # small month-to-month noise on top of the structural latent trajectory
    effective = effective + rng.normal(0.0, 0.12, size=(n, n_months))
    return effective, base_latent


def simulate_lifecycle(
    accounts: pd.DataFrame, cfg: HarborConfig, rng: np.random.Generator
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Simulates monthly usage/billing and support tickets month-by-month,
    applying a churn hazard driven by latent account health.

    Returns (monthly_usage_df, tickets_df, effective_latent_matrix).
    The latent matrix is NOT written to any feature file -- it exists only so
    downstream evaluation code can report "recovered vs. true" effects.
    """
    n = len(accounts)
    n_months = cfg.section("data")["n_months_history"]
    effective_latent, _ = _simulate_latent_health(n, n_months, rng)

    plan_tier = accounts["plan_tier"].to_numpy()
    industry = accounts["industry"].to_numpy()
    base_mrr = accounts["base_mrr"].to_numpy()
    baseline_engagement = rng.lognormal(
        mean=np.where(plan_tier == "enterprise", 3.2, np.where(plan_tier == "growth", 2.6, 2.0)),
        sigma=0.3,
    )

    active = np.ones(n, dtype=bool)
    churn_month = np.full(n, -1, dtype=np.int64)
    base_hazard = 0.010

    usage_rows = []
    ticket_rows = []
    ticket_id_counter = 0
    org_names = rng.choice(_ORG_WORDS, size=n)

    for t in range(n_months):
        latent_t = effective_latent[:, t]
        seasonal = _seasonal_factor(np.full(n, t), industry)
        engagement = (
            baseline_engagement
            * seasonal
            * np.exp(-0.45 * np.clip(latent_t, -2, 4))
            * rng.lognormal(0.0, 0.15, size=n)
        )
        engagement = np.where(active, engagement, 0.0)
        login_count = np.clip(engagement * rng.uniform(3.5, 5.5, size=n), 0, None)
        api_calls = np.clip(engagement * rng.uniform(40, 90, size=n), 0, None)
        feature_adoption = np.clip(
            (engagement / (baseline_engagement + 1e-6)) * rng.uniform(1, 6, size=n), 0, 12
        )
        active_users = np.clip(1 + engagement / 25.0, 0, None)
        mrr = np.where(active, base_mrr, 0.0)

        hazard = base_hazard * np.exp(1.15 * np.clip(latent_t, -2, 4))
        hazard = np.clip(hazard, 0.0, 0.35)
        will_churn_this_month = active & (rng.random(n) < hazard)

        usage_rows.append(
            pd.DataFrame(
                {
                    "account_id": accounts["account_id"].to_numpy(),
                    "month_index": t,
                    "active_users": np.round(active_users, 2),
                    "login_count": np.round(login_count, 1),
                    "api_calls": np.round(api_calls, 0),
                    "feature_adoption_count": np.round(feature_adoption, 1),
                    "mrr": np.round(mrr, 2),
                    "is_active": active.copy(),
                    "churned_this_month": will_churn_this_month,
                }
            )
        )

        # support tickets: rate and sentiment/category mix both driven by latent health
        expected_tickets = np.where(active, 0.18 * np.exp(0.35 * np.clip(latent_t, -1, 3)), 0.0)
        n_tickets_this_month = rng.poisson(np.clip(expected_tickets, 0, 3))
        neg_prob = 1 / (1 + np.exp(-(latent_t - 0.6)))  # sigmoid: higher latent -> more negative

        for idx in np.nonzero(n_tickets_this_month)[0]:
            for _ in range(int(n_tickets_this_month[idx])):
                if rng.random() < neg_prob[idx]:
                    category = rng.choice(
                        ["cancellation_intent", "bug_report", "billing"], p=[0.45, 0.35, 0.20]
                    )
                else:
                    category = rng.choice(
                        ["feature_request", "onboarding", "general_praise", "billing"],
                        p=[0.30, 0.25, 0.30, 0.15],
                    )
                template = rng.choice(_TICKET_TEMPLATES[category])
                text = template.format(
                    org=org_names[idx],
                    plan=plan_tier[idx],
                    feature=rng.choice(["reporting", "integrations", "workflows", "analytics"]),
                )
                if rng.random() < 0.45:
                    filler = rng.choice(_SHARED_FILLER)
                    text = f"{text} {filler}" if rng.random() < 0.5 else f"{filler} {text}"

                # ~6% of tickets get mis-tagged, same as a human agent
                # occasionally filing a ticket under the wrong category.
                recorded_category = category
                if rng.random() < 0.06:
                    recorded_category = rng.choice(
                        [c for c in TICKET_CATEGORIES if c != category]
                    )

                ticket_rows.append(
                    {
                        "ticket_id": ticket_id_counter,
                        "account_id": int(accounts["account_id"].iat[idx]),
                        "month_index": t,
                        "category": recorded_category,
                        "text": text,
                    }
                )
                ticket_id_counter += 1

        churn_month = np.where(will_churn_this_month & (churn_month < 0), t, churn_month)
        active = active & ~will_churn_this_month

    monthly_usage = pd.concat(usage_rows, ignore_index=True)
    tickets = pd.DataFrame(ticket_rows)
    return monthly_usage, tickets, effective_latent


def simulate_experiment(
    accounts: pd.DataFrame,
    monthly_usage: pd.DataFrame,
    effective_latent: np.ndarray,
    cfg: HarborConfig,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Simulates a retention-offer RCT at the observation cutoff month.

    Eligibility uses an *observable* proxy (recent engagement decline), not
    the hidden latent variable -- exactly like a real retention team would
    target accounts using dashboard signals, not ground truth. The true
    treatment effect is heterogeneous by engagement segment (config-driven),
    which is what the uplift model is expected to recover.
    """
    exp_cfg = cfg.section("experiment")
    data_cfg = cfg.section("data")
    cutoff = data_cfg["observation_cutoff_month"]
    horizon = data_cfg["churn_horizon_months"]
    n_months = data_cfg["n_months_history"]

    active_at_cutoff = (
        monthly_usage.loc[monthly_usage["month_index"] == cutoff - 1]
        .set_index("account_id")["is_active"]
    )
    active_ids = active_at_cutoff[active_at_cutoff].index.to_numpy()

    # observable recent-engagement proxy: trailing-3-month login trend
    trail = monthly_usage[
        (monthly_usage["month_index"] >= cutoff - 3) & (monthly_usage["month_index"] < cutoff)
    ]
    trail_login = trail.groupby("account_id")["login_count"].mean()
    early = monthly_usage[
        (monthly_usage["month_index"] >= cutoff - 6) & (monthly_usage["month_index"] < cutoff - 3)
    ]
    early_login = early.groupby("account_id")["login_count"].mean()
    decline_ratio = (trail_login - early_login) / (early_login + 1.0)
    decline_ratio = decline_ratio.reindex(active_ids).fillna(0.0)

    risk_threshold = decline_ratio.quantile(1 - exp_cfg["eligible_risk_percentile"])
    eligible_ids = decline_ratio[decline_ratio <= risk_threshold].index.to_numpy()

    is_treated = rng.random(len(eligible_ids)) < exp_cfg["treatment_fraction"]
    group = np.where(is_treated, "treatment", "control")

    # Split within the eligible (already at-risk) population, not against the
    # whole portfolio's median -- eligibility already selects the
    # most-declining accounts, so comparing to the population median would
    # put nearly everyone in "low_engagement" and collapse the very
    # heterogeneity this experiment exists to measure.
    eligible_decline = decline_ratio.loc[eligible_ids]
    engagement_segment = np.where(
        eligible_decline.to_numpy() <= eligible_decline.median(),
        "low_engagement",
        "high_engagement",
    )

    id_to_row = {aid: i for i, aid in enumerate(accounts["account_id"].to_numpy())}
    obs_months = [cutoff, cutoff + 1, min(cutoff + 2, n_months - 1)][:horizon]

    baseline_p0 = np.zeros(len(eligible_ids))
    for j, aid in enumerate(eligible_ids):
        row = id_to_row[aid]
        hazards = 0.010 * np.exp(1.15 * np.clip(effective_latent[row, obs_months], -2, 4))
        hazards = np.clip(hazards, 0.0, 0.35)
        baseline_p0[j] = 1 - np.prod(1 - hazards)

    true_effect = np.where(
        engagement_segment == "low_engagement",
        exp_cfg["true_effect_low_engagement"],
        exp_cfg["true_effect_high_engagement"],
    )
    p_used = np.where(is_treated, np.clip(baseline_p0 + true_effect, 0.0, 1.0), baseline_p0)
    observed_churn_90d = rng.random(len(eligible_ids)) < p_used

    return pd.DataFrame(
        {
            "account_id": eligible_ids,
            "assigned_month": cutoff,
            "group": group,
            "engagement_segment": engagement_segment,
            "baseline_risk_proxy": np.round(decline_ratio.loc[eligible_ids].to_numpy(), 4),
            "true_p0": np.round(baseline_p0, 4),
            "true_effect": true_effect,
            "observed_churn_90d": observed_churn_90d,
        }
    )


def generate_all(cfg: HarborConfig) -> dict[str, pd.DataFrame]:
    rng = _rng(cfg.random_seed)
    accounts = generate_accounts(cfg, rng)
    monthly_usage, tickets, effective_latent = simulate_lifecycle(accounts, cfg, rng)
    experiment = simulate_experiment(accounts, monthly_usage, effective_latent, cfg, rng)

    latent_df = pd.DataFrame(
        effective_latent, columns=[f"m{t}" for t in range(effective_latent.shape[1])]
    )
    latent_df.insert(0, "account_id", accounts["account_id"].to_numpy())

    return {
        "accounts": accounts,
        "monthly_usage": monthly_usage,
        "support_tickets": tickets,
        "experiment_assignments": experiment,
        "_ground_truth_latent": latent_df,  # debug-only, not a feature source
    }


def write_all(tables: dict[str, pd.DataFrame], cfg: HarborConfig) -> None:
    raw_dir = cfg.path(key="raw_dir")
    raw_dir.mkdir(parents=True, exist_ok=True)
    data_cfg = cfg.section("data")

    tables["accounts"].to_csv(raw_dir / data_cfg["accounts_file"], index=False)
    tables["monthly_usage"].to_csv(raw_dir / data_cfg["usage_file"], index=False)
    tables["support_tickets"].to_csv(raw_dir / data_cfg["tickets_file"], index=False)
    tables["experiment_assignments"].to_csv(raw_dir / data_cfg["experiment_file"], index=False)
    tables["_ground_truth_latent"].to_parquet(
        raw_dir / "_ground_truth_latent_debug.parquet", index=False
    )
    logger.info("Wrote synthetic data to %s", raw_dir)
