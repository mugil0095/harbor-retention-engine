"""Statistical analysis for the retention-offer experiment: guardrails first
(sample ratio mismatch), then the actual hypothesis test, with confidence
intervals, effect sizes, a power check, and multiple-testing correction
across the overall test plus per-segment tests.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize


@dataclass
class SrmCheckResult:
    n_treatment: int
    n_control: int
    expected_ratio: float
    chi2_stat: float
    p_value: float
    passed: bool


def check_sample_ratio_mismatch(
    n_treatment: int, n_control: int, expected_ratio: float = 0.5, alpha: float = 0.001
) -> SrmCheckResult:
    """SRM check: a guardrail on the experiment infrastructure itself, not
    the treatment effect. A significant imbalance at a strict alpha (0.001,
    the conventional SRM threshold) means the randomization is broken and
    the effect estimate below should not be trusted regardless of its p-value.
    """
    n_total = n_treatment + n_control
    expected_treatment = n_total * expected_ratio
    expected_control = n_total * (1 - expected_ratio)
    chi2_stat, p_value = stats.chisquare(
        f_obs=[n_treatment, n_control], f_exp=[expected_treatment, expected_control]
    )
    return SrmCheckResult(
        n_treatment=n_treatment,
        n_control=n_control,
        expected_ratio=expected_ratio,
        chi2_stat=float(chi2_stat),
        p_value=float(p_value),
        passed=p_value >= alpha,
    )


@dataclass
class TwoProportionTestResult:
    p_treatment: float
    p_control: float
    absolute_effect: float  # p_treatment - p_control (negative = churn reduced)
    relative_effect: float
    ci_low: float
    ci_high: float
    z_stat: float
    p_value: float
    cohens_h: float
    n_treatment: int
    n_control: int


def two_proportion_test(
    successes_treatment: int, n_treatment: int, successes_control: int, n_control: int, alpha: float = 0.05
) -> TwoProportionTestResult:
    p1 = successes_treatment / n_treatment
    p2 = successes_control / n_control
    p_pool = (successes_treatment + successes_control) / (n_treatment + n_control)
    se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / n_treatment + 1 / n_control))
    z_stat = (p1 - p2) / se_pool if se_pool > 0 else 0.0
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    se_diff = np.sqrt(p1 * (1 - p1) / n_treatment + p2 * (1 - p2) / n_control)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    diff = p1 - p2
    ci_low, ci_high = diff - z_crit * se_diff, diff + z_crit * se_diff

    return TwoProportionTestResult(
        p_treatment=p1,
        p_control=p2,
        absolute_effect=diff,
        relative_effect=(p1 - p2) / p2 if p2 > 0 else float("nan"),
        ci_low=ci_low,
        ci_high=ci_high,
        z_stat=float(z_stat),
        p_value=float(p_value),
        cohens_h=float(proportion_effectsize(p1, p2)),
        n_treatment=n_treatment,
        n_control=n_control,
    )


def required_sample_size_per_arm(
    baseline_rate: float, minimum_detectable_effect: float, alpha: float = 0.05, power: float = 0.8
) -> int:
    """Minimum per-arm sample size to detect `minimum_detectable_effect` (an
    absolute change in the churn rate) at the given alpha/power, via a
    normal-approximation two-proportion power calculation."""
    p2 = baseline_rate
    p1 = baseline_rate + minimum_detectable_effect
    effect_size = proportion_effectsize(p1, p2)
    analysis = NormalIndPower()
    n = analysis.solve_power(effect_size=abs(effect_size), alpha=alpha, power=power, ratio=1.0)
    return int(np.ceil(n))


def apply_multiple_testing_correction(p_values: list[float], method: str = "holm") -> np.ndarray:
    """Corrects for testing the overall effect plus per-segment effects on
    the same experiment. Holm's method controls the family-wise error rate
    with less power loss than Bonferroni."""
    _, p_adjusted, _, _ = multipletests(p_values, method=method)
    return p_adjusted


def cuped_adjust(outcome: np.ndarray, covariate: np.ndarray) -> tuple[np.ndarray, float]:
    """CUPED (Controlled-experiment Using Pre-Experiment Data; Deng et al.
    2013): adjusts the outcome using a pre-experiment covariate correlated
    with it, to reduce variance without changing the expected treatment
    effect. `theta` is the OLS slope of outcome on covariate -- the value
    that minimizes Var(outcome - theta * covariate).

    Unbiasedness depends on `theta` being computed from the *pooled* sample
    (both arms together, as done here), not separately per arm: since
    `theta` doesn't depend on treatment assignment, subtracting
    `theta * (covariate - mean(covariate))` shifts every unit by a
    quantity with pooled mean exactly zero, so it cannot systematically
    favor either arm -- it only removes the part of the outcome's
    variance that the covariate already explains, which is where the
    power gain comes from.
    """
    theta = float(np.cov(outcome, covariate, ddof=1)[0, 1] / np.var(covariate, ddof=1))
    adjusted = outcome - theta * (covariate - covariate.mean())
    return adjusted, theta


@dataclass
class CupedResult:
    theta: float
    variance_raw: float
    variance_adjusted: float
    variance_reduction_pct: float
    treatment_mean_adjusted: float
    control_mean_adjusted: float
    absolute_effect: float
    ci_low: float
    ci_high: float
    t_stat: float
    p_value: float
    n_treatment: int
    n_control: int


def cuped_treatment_effect(
    outcome: np.ndarray, covariate: np.ndarray, treatment: np.ndarray, alpha: float = 0.05
) -> CupedResult:
    """Re-runs the treatment-effect test on the CUPED-adjusted outcome. The
    adjusted outcome is continuous (even when the raw outcome is binary
    churn, 0/1), so this uses Welch's t-test rather than the binomial
    two-proportion test above -- the standard way CUPED is applied to
    binary outcomes in practice (the adjustment is a linear projection, it
    doesn't require the outcome to be continuous to begin with).
    """
    adjusted, theta = cuped_adjust(outcome, covariate)
    variance_raw = float(np.var(outcome, ddof=1))
    variance_adjusted = float(np.var(adjusted, ddof=1))
    variance_reduction_pct = (
        (variance_raw - variance_adjusted) / variance_raw * 100 if variance_raw > 0 else float("nan")
    )

    treated = treatment == 1
    treat_vals, control_vals = adjusted[treated], adjusted[~treated]
    t_stat, p_value = stats.ttest_ind(treat_vals, control_vals, equal_var=False)

    mean_diff = float(treat_vals.mean() - control_vals.mean())
    se_diff = float(np.sqrt(treat_vals.var(ddof=1) / len(treat_vals) + control_vals.var(ddof=1) / len(control_vals)))
    z_crit = stats.norm.ppf(1 - alpha / 2)

    return CupedResult(
        theta=theta,
        variance_raw=variance_raw,
        variance_adjusted=variance_adjusted,
        variance_reduction_pct=variance_reduction_pct,
        treatment_mean_adjusted=float(treat_vals.mean()),
        control_mean_adjusted=float(control_vals.mean()),
        absolute_effect=mean_diff,
        ci_low=mean_diff - z_crit * se_diff,
        ci_high=mean_diff + z_crit * se_diff,
        t_stat=float(t_stat),
        p_value=float(p_value),
        n_treatment=int(treated.sum()),
        n_control=int((~treated).sum()),
    )
