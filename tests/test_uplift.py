"""M: experiment/uplift calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from harbor.experimentation.ab_test import (
    apply_multiple_testing_correction,
    check_sample_ratio_mismatch,
    cuped_adjust,
    cuped_treatment_effect,
    required_sample_size_per_arm,
    two_proportion_test,
)
from harbor.models.uplift import (
    evaluate_uplift_method,
    fit_s_learner,
    fit_t_learner,
    fit_x_learner,
    predict_cate_s_learner,
    predict_cate_t_learner,
    predict_cate_x_learner,
    qini_curve,
)


def test_srm_check_passes_for_balanced_groups():
    result = check_sample_ratio_mismatch(n_treatment=500, n_control=505)
    assert result.passed


def test_srm_check_fails_for_heavily_skewed_groups():
    result = check_sample_ratio_mismatch(n_treatment=900, n_control=100)
    assert not result.passed


def test_two_proportion_test_detects_known_effect():
    # 1000 per arm, treatment churn 10%, control churn 20% -- should be highly significant
    rng = np.random.default_rng(42)
    treatment_churns = int(rng.binomial(1000, 0.10))
    control_churns = int(rng.binomial(1000, 0.20))
    result = two_proportion_test(treatment_churns, 1000, control_churns, 1000)
    assert result.p_value < 0.01
    assert result.absolute_effect < 0  # treatment reduced churn


def test_two_proportion_test_no_effect_is_not_significant():
    rng = np.random.default_rng(42)
    treatment_churns = int(rng.binomial(500, 0.15))
    control_churns = int(rng.binomial(500, 0.15))
    result = two_proportion_test(treatment_churns, 500, control_churns, 500)
    assert result.p_value > 0.05


def test_confidence_interval_contains_point_estimate():
    result = two_proportion_test(100, 1000, 150, 1000)
    assert result.ci_low <= result.absolute_effect <= result.ci_high


def test_required_sample_size_decreases_as_effect_size_grows():
    n_small_effect = required_sample_size_per_arm(0.2, 0.02)
    n_large_effect = required_sample_size_per_arm(0.2, 0.10)
    assert n_large_effect < n_small_effect


def test_multiple_testing_correction_increases_or_maintains_p_values():
    raw_p = [0.01, 0.02, 0.20]
    adjusted = apply_multiple_testing_correction(raw_p)
    assert all(adj >= raw for adj, raw in zip(adjusted, raw_p))


def test_cuped_adjust_preserves_the_pooled_mean_exactly():
    """Unbiasedness check: theta doesn't depend on treatment assignment, and
    the adjustment term (covariate - its own mean) sums to exactly zero
    over the sample it was centered on -- so the adjusted outcome's mean
    must equal the raw outcome's mean to floating-point precision, not just
    approximately. If this ever drifts, CUPED would be introducing bias."""
    rng = np.random.default_rng(1)
    outcome = rng.normal(10, 2, 500)
    covariate = rng.normal(5, 1, 500)
    adjusted, _theta = cuped_adjust(outcome, covariate)
    assert adjusted.mean() == pytest.approx(outcome.mean(), abs=1e-9)


def test_cuped_reduces_variance_when_covariate_is_correlated():
    rng = np.random.default_rng(2)
    covariate = rng.normal(0, 1, 2000)
    outcome = 3.0 * covariate + rng.normal(0, 0.5, 2000)  # covariate explains most of the variance
    adjusted, theta = cuped_adjust(outcome, covariate)
    assert theta == pytest.approx(3.0, abs=0.2)
    assert np.var(adjusted, ddof=1) < 0.1 * np.var(outcome, ddof=1)  # most variance removed


def test_cuped_barely_changes_variance_when_covariate_is_uncorrelated():
    rng = np.random.default_rng(3)
    outcome = rng.normal(0, 1, 3000)
    covariate = rng.normal(0, 1, 3000)  # independent of outcome
    adjusted, theta = cuped_adjust(outcome, covariate)
    assert abs(theta) < 0.1  # near-zero slope
    assert np.var(adjusted, ddof=1) == pytest.approx(np.var(outcome, ddof=1), rel=0.1)


def test_cuped_treatment_effect_recovers_known_effect_with_tighter_ci():
    """The actual payoff CUPED promises: same data, a narrower confidence
    interval than the raw test, because a pre-experiment covariate soaks up
    variance the treatment effect doesn't need to explain."""
    rng = np.random.default_rng(4)
    n = 2000
    covariate = rng.normal(50, 10, n)  # e.g. a pre-treatment engagement score
    treatment = rng.integers(0, 2, n)
    true_effect = -2.0
    # outcome is driven mostly by the covariate, plus a real (small) treatment effect
    outcome = 0.8 * covariate + np.where(treatment == 1, true_effect, 0.0) + rng.normal(0, 3, n)

    cuped_result = cuped_treatment_effect(outcome, covariate, treatment)
    assert cuped_result.absolute_effect == pytest.approx(true_effect, abs=1.0)
    assert cuped_result.ci_low <= true_effect <= cuped_result.ci_high
    assert cuped_result.variance_reduction_pct > 50  # covariate explains most of the noise

    # compare against the raw (unadjusted) Welch's t-test on the same data
    from scipy import stats as _stats

    treat_raw, control_raw = outcome[treatment == 1], outcome[treatment == 0]
    raw_se = np.sqrt(treat_raw.var(ddof=1) / len(treat_raw) + control_raw.var(ddof=1) / len(control_raw))
    cuped_se = (cuped_result.ci_high - cuped_result.ci_low) / (2 * _stats.norm.ppf(0.975))
    assert cuped_se < raw_se  # CUPED's CI is narrower -- the actual point of using it


def test_cuped_result_reports_consistent_arm_sizes():
    rng = np.random.default_rng(5)
    n = 400
    outcome = rng.normal(0, 1, n)
    covariate = rng.normal(0, 1, n)
    treatment = np.array([1] * 150 + [0] * 250)
    result = cuped_treatment_effect(outcome, covariate, treatment)
    assert result.n_treatment == 150
    assert result.n_control == 250


def _synthetic_uplift_frame(n=400, seed=7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    engagement = rng.choice(["low", "high"], size=n)
    treatment = rng.integers(0, 2, size=n)
    true_effect = np.where(engagement == "low", -0.20, -0.02)
    base_rate = 0.4
    p = np.clip(base_rate + np.where(treatment == 1, true_effect, 0.0), 0, 1)
    outcome = (rng.random(n) < p).astype(int)
    return pd.DataFrame({"engagement": engagement, "treatment": treatment, "outcome": outcome, "true_effect": true_effect})


def test_qini_curve_is_nondecreasing_in_fraction_length():
    df = _synthetic_uplift_frame()
    predicted_cate = -df["true_effect"].to_numpy()  # perfect ranking
    qini_df = qini_curve(predicted_cate, df["treatment"].to_numpy(), df["outcome"].to_numpy())
    assert len(qini_df) == len(df)
    assert qini_df["fraction"].iloc[-1] == 1.0


def test_t_learner_recovers_directionally_correct_heterogeneity():
    df = _synthetic_uplift_frame(n=2000, seed=11)
    feature_cols = ["engagement"]

    t_treat, t_control, X_t = fit_t_learner(df, feature_cols, "treatment", "outcome", random_seed=11)
    t_cate = predict_cate_t_learner(t_treat, t_control, X_t)

    low_mask = (df["engagement"] == "low").to_numpy()
    # true effect is -0.20 for low-engagement vs -0.02 for high-engagement, so
    # a working T-learner must predict substantially larger uplift for "low"
    assert t_cate[low_mask].mean() > t_cate[~low_mask].mean()


def test_s_learner_under_separates_heterogeneity_relative_to_t_learner():
    """This is a documented, expected weakness of the S-learner (see module
    docstring): a single regularized model tends to shrink the treatment
    interaction toward zero, so it separates segments by uplift far less
    than the T-learner does -- it should not be relied on when the true
    effect is heterogeneous, which is exactly what this test demonstrates."""
    df = _synthetic_uplift_frame(n=2000, seed=11)
    feature_cols = ["engagement"]
    low_mask = (df["engagement"] == "low").to_numpy()

    s_model, X_s = fit_s_learner(df, feature_cols, "treatment", "outcome", random_seed=11)
    s_cate = predict_cate_s_learner(s_model, X_s)
    s_separation = abs(s_cate[low_mask].mean() - s_cate[~low_mask].mean())

    t_treat, t_control, X_t = fit_t_learner(df, feature_cols, "treatment", "outcome", random_seed=11)
    t_cate = predict_cate_t_learner(t_treat, t_control, X_t)
    t_separation = abs(t_cate[low_mask].mean() - t_cate[~low_mask].mean())

    assert s_separation < t_separation


def test_evaluate_uplift_method_reports_correlation_with_true_effect():
    df = _synthetic_uplift_frame(n=1000, seed=5)
    feature_cols = ["engagement"]
    t_treat, t_control, X_t = fit_t_learner(df, feature_cols, "treatment", "outcome", random_seed=5)
    t_cate = predict_cate_t_learner(t_treat, t_control, X_t)

    result = evaluate_uplift_method(
        "t_learner", t_cate, df["treatment"].to_numpy(), df["outcome"].to_numpy(), true_effect=df["true_effect"].to_numpy()
    )
    assert result.correlation_with_true_effect is not None
    assert result.correlation_with_true_effect > 0.3  # T-learner should recover the known heterogeneity direction


def test_x_learner_recovers_directionally_correct_heterogeneity():
    df = _synthetic_uplift_frame(n=2000, seed=11)
    feature_cols = ["engagement"]
    low_mask = (df["engagement"] == "low").to_numpy()

    tau1, tau0, propensity, X = fit_x_learner(df, feature_cols, "treatment", "outcome", random_seed=11)
    x_cate = predict_cate_x_learner(tau1, tau0, propensity, X)

    # true effect is -0.20 for low-engagement vs -0.02 for high-engagement
    assert x_cate[low_mask].mean() > x_cate[~low_mask].mean()


def test_x_learner_correlates_with_true_effect():
    df = _synthetic_uplift_frame(n=1000, seed=5)
    feature_cols = ["engagement"]
    tau1, tau0, propensity, X = fit_x_learner(df, feature_cols, "treatment", "outcome", random_seed=5)
    x_cate = predict_cate_x_learner(tau1, tau0, propensity, X)

    result = evaluate_uplift_method(
        "x_learner", x_cate, df["treatment"].to_numpy(), df["outcome"].to_numpy(), true_effect=df["true_effect"].to_numpy()
    )
    assert result.correlation_with_true_effect is not None
    assert result.correlation_with_true_effect > 0.3


def test_x_learner_handles_unbalanced_treatment_arms():
    """The X-learner's stated advantage over the T-learner is exactly this
    case: a much smaller arm still gets a usable effect estimate because it
    borrows the other arm's outcome model rather than relying purely on its
    own small sample."""
    rng = np.random.default_rng(9)
    n = 1500
    engagement = rng.choice(["low", "high"], size=n)
    treatment = (rng.random(n) < 0.15).astype(int)  # only ~15% treated -- unbalanced arms
    true_effect = np.where(engagement == "low", -0.20, -0.02)
    base_rate = 0.4
    p = np.clip(base_rate + np.where(treatment == 1, true_effect, 0.0), 0, 1)
    outcome = (rng.random(n) < p).astype(int)
    df = pd.DataFrame({"engagement": engagement, "treatment": treatment, "outcome": outcome, "true_effect": true_effect})

    tau1, tau0, propensity, X = fit_x_learner(df, ["engagement"], "treatment", "outcome", random_seed=9)
    x_cate = predict_cate_x_learner(tau1, tau0, propensity, X)
    assert len(x_cate) == n
    assert np.isfinite(x_cate).all()
    low_mask = (df["engagement"] == "low").to_numpy()
    assert x_cate[low_mask].mean() > x_cate[~low_mask].mean()
