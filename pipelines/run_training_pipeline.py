#!/usr/bin/env python
"""End-to-end training pipeline: raw data -> validated tables -> NLP model ->
feature store -> churn models (baseline/classical/advanced) -> forecasting
backtests -> experiment analysis -> uplift models -> persisted artifacts and
a machine-readable training_summary.json that docs/EVALUATION_REPORT.md is
generated from (never hand-written numbers).

Usage:
    python pipelines/run_training_pipeline.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from harbor.config import load_config
from harbor.data.schema import (
    ACCOUNTS,
    EXPERIMENT_ASSIGNMENTS,
    MONTHLY_USAGE,
    SUPPORT_TICKETS,
)
from harbor.data.validate import assert_valid, validate_table
from harbor.decision.decision_engine import UPLIFT_FEATURE_COLUMNS
from harbor.experimentation.ab_test import (
    apply_multiple_testing_correction,
    check_sample_ratio_mismatch,
    cuped_treatment_effect,
    required_sample_size_per_arm,
    two_proportion_test,
)
from harbor.features.build_features import (
    CATEGORICAL_FEATURE_COLUMNS,
    NUMERIC_FEATURE_COLUMNS,
    build_feature_store,
)
from harbor.models.churn_model import (
    LABEL_COLUMN,
    STATIC_FEATURE_COLUMNS,
    calibration_curve_points,
    evaluate_model,
    expected_calibration_error,
    segment_performance,
    top_feature_importances,
    train_advanced_model,
    train_baseline_model,
    train_calibrated_advanced_model,
    train_classical_model,
    tune_advanced_model,
)
from harbor.models.explainability import compute_shap_importance
from harbor.models.forecasting import (
    attach_plan_tier,
    build_monthly_series,
    rolling_origin_backtest,
)
from harbor.models.uplift import (
    evaluate_uplift_method,
    fit_s_learner,
    fit_t_learner,
    fit_x_learner,
    predict_cate_s_learner,
    predict_cate_t_learner,
    predict_cate_x_learner,
)
from harbor.nlp.ticket_analysis import compare_nlp_approaches
from harbor.utils.experiment_tracker import RunTracker
from harbor.utils.logging_config import configure_logging

logger = logging.getLogger(__name__)


def load_raw_tables(cfg) -> dict[str, pd.DataFrame]:
    raw_dir = cfg.path(key="raw_dir")
    data_cfg = cfg.section("data")
    accounts = pd.read_csv(raw_dir / data_cfg["accounts_file"])
    usage = pd.read_csv(raw_dir / data_cfg["usage_file"])
    tickets = pd.read_csv(raw_dir / data_cfg["tickets_file"])
    experiment = pd.read_csv(raw_dir / data_cfg["experiment_file"])
    return {"accounts": accounts, "monthly_usage": usage, "support_tickets": tickets, "experiment": experiment}


def main() -> None:
    configure_logging()
    cfg = load_config()
    models_dir = cfg.models_dir
    models_dir.mkdir(parents=True, exist_ok=True)
    processed_dir = cfg.path(key="processed_dir")
    processed_dir.mkdir(parents=True, exist_ok=True)

    tables = load_raw_tables(cfg)
    reference = {"accounts": tables["accounts"]}
    assert_valid(validate_table(tables["accounts"], ACCOUNTS))
    assert_valid(validate_table(tables["monthly_usage"], MONTHLY_USAGE, reference))
    assert_valid(validate_table(tables["support_tickets"], SUPPORT_TICKETS, reference))
    assert_valid(validate_table(tables["experiment"], EXPERIMENT_ASSIGNMENTS, reference))
    logger.info("Raw tables validated.")

    summary: dict = {}

    # ---------------------------------------------------------------- NLP --
    logger.info("Comparing NLP baseline vs. TF-IDF classifier...")
    nlp_result, ticket_classifier = compare_nlp_approaches(tables["support_tickets"], random_seed=cfg.random_seed)
    summary["nlp"] = nlp_result.__dict__
    logger.info(
        "NLP: baseline F1=%.3f AUC=%.3f | classifier F1=%.3f AUC=%.3f",
        nlp_result.baseline_f1, nlp_result.baseline_auc, nlp_result.classifier_f1, nlp_result.classifier_auc,
    )

    # ------------------------------------------------------- Feature store --
    data_cfg = cfg.section("data")
    logger.info("Building feature store across snapshots %s ...", data_cfg["snapshot_months"])
    feature_store = build_feature_store(
        tables["accounts"],
        tables["monthly_usage"],
        tables["support_tickets"],
        ticket_classifier,
        data_cfg["snapshot_months"],
        data_cfg["churn_horizon_months"],
    )
    feature_store.to_parquet(processed_dir / data_cfg["feature_store_file"], index=False)
    logger.info("Feature store: %d rows across %d snapshots", len(feature_store), feature_store["snapshot_month"].nunique())

    test_month = data_cfg["test_snapshot_month"]
    train_df = feature_store[feature_store["snapshot_month"] != test_month]
    test_df = feature_store[feature_store["snapshot_month"] == test_month]
    logger.info("Train rows=%d (snapshots != %d), test rows=%d (snapshot == %d)", len(train_df), test_month, len(test_df), test_month)
    summary["label_base_rate_train"] = float(train_df[LABEL_COLUMN].mean())
    summary["label_base_rate_test"] = float(test_df[LABEL_COLUMN].mean())

    # ------------------------------------------------------- Churn models --
    full_feature_cols = NUMERIC_FEATURE_COLUMNS + CATEGORICAL_FEATURE_COLUMNS

    logger.info("Training baseline churn model (static attributes only)...")
    baseline_pipe = train_baseline_model(train_df, random_seed=cfg.random_seed)
    baseline_metrics = evaluate_model(baseline_pipe, test_df, STATIC_FEATURE_COLUMNS)
    summary["churn_baseline"] = baseline_metrics
    logger.info("Baseline: %s", baseline_metrics)

    logger.info("Training strong classical churn model (logistic regression, full features)...")
    classical_pipe = train_classical_model(train_df, NUMERIC_FEATURE_COLUMNS, CATEGORICAL_FEATURE_COLUMNS, random_seed=cfg.random_seed)
    classical_metrics = evaluate_model(classical_pipe, test_df, full_feature_cols)
    summary["churn_classical"] = classical_metrics
    logger.info("Classical: %s", classical_metrics)

    logger.info("Tuning advanced churn model (gradient boosting) via rolling-origin CV...")
    tune_result = tune_advanced_model(
        feature_store, NUMERIC_FEATURE_COLUMNS, CATEGORICAL_FEATURE_COLUMNS,
        data_cfg["snapshot_months"], test_month, random_seed=cfg.random_seed,
    )
    logger.info("Best params: %s (CV mean PR-AUC=%.3f)", tune_result.best_params, tune_result.best_score)
    advanced_pipe = train_advanced_model(
        train_df, NUMERIC_FEATURE_COLUMNS, CATEGORICAL_FEATURE_COLUMNS, tune_result.best_params, random_seed=cfg.random_seed,
    )
    advanced_metrics_uncalibrated = evaluate_model(advanced_pipe, test_df, full_feature_cols)
    summary["churn_advanced_best_params"] = tune_result.best_params
    summary["churn_advanced_cv_all_results"] = tune_result.all_results
    logger.info("Advanced (uncalibrated): %s", advanced_metrics_uncalibrated)

    # Interpretability (permutation importance, SHAP) is computed on the raw
    # tree pipeline regardless of whether calibration is shipped: isotonic
    # calibration is a monotonic rescaling of the output, it doesn't change
    # which features drive the ranking, and SHAP's TreeExplainer needs
    # direct access to the tree ensemble (not the calibrator wrapper).
    seg_plan = segment_performance(advanced_pipe, test_df, full_feature_cols, "plan_tier")
    seg_industry = segment_performance(advanced_pipe, test_df, full_feature_cols, "industry")
    summary["churn_advanced_segment_by_plan"] = seg_plan.to_dict(orient="records")
    summary["churn_advanced_segment_by_industry"] = seg_industry.to_dict(orient="records")

    importances = top_feature_importances(advanced_pipe, test_df, full_feature_cols, random_seed=cfg.random_seed)
    summary["churn_advanced_top_features"] = importances.to_dict(orient="records")

    logger.info("Computing SHAP values for the advanced model...")
    shap_result = compute_shap_importance(
        advanced_pipe, test_df, full_feature_cols, categorical_cols=CATEGORICAL_FEATURE_COLUMNS, random_seed=cfg.random_seed
    )
    summary["churn_advanced_shap_top_features"] = shap_result.to_dict(orient="records")
    permutation_ranking = [row["feature"] for row in summary["churn_advanced_top_features"]]
    shap_ranking = shap_result["feature"].tolist()
    top5_overlap = len(set(permutation_ranking[:5]) & set(shap_ranking[:5]))
    summary["shap_vs_permutation_top5_overlap"] = top5_overlap
    logger.info("SHAP vs. permutation importance: %d/5 features agree in the top 5.", top5_overlap)

    logger.info("Checking calibration of the advanced model...")
    y_test = test_df[LABEL_COLUMN].to_numpy()
    proba_uncalibrated = advanced_pipe.predict_proba(test_df[full_feature_cols])[:, 1]
    ece_uncalibrated = expected_calibration_error(y_test, proba_uncalibrated)
    predicted_uncal, observed_uncal = calibration_curve_points(y_test, proba_uncalibrated)

    calibrated_pipe = train_calibrated_advanced_model(
        train_df, NUMERIC_FEATURE_COLUMNS, CATEGORICAL_FEATURE_COLUMNS, tune_result.best_params, random_seed=cfg.random_seed,
    )
    proba_calibrated = calibrated_pipe.predict_proba(test_df[full_feature_cols])[:, 1]
    ece_calibrated = expected_calibration_error(y_test, proba_calibrated)
    predicted_cal, observed_cal = calibration_curve_points(y_test, proba_calibrated)

    calibration_improves = ece_calibrated < ece_uncalibrated
    final_churn_pipe = calibrated_pipe if calibration_improves else advanced_pipe
    # Headline "churn_advanced" metrics reflect whichever pipe is actually
    # shipped as churn_advanced_model.joblib -- reporting the uncalibrated
    # model's metrics while shipping the calibrated one (or vice versa)
    # would make the evaluation report describe a model nobody can query.
    advanced_metrics = evaluate_model(final_churn_pipe, test_df, full_feature_cols)
    summary["churn_advanced"] = advanced_metrics

    summary["churn_advanced_calibration"] = {
        "uncalibrated": {
            "ece": ece_uncalibrated,
            "brier_score": advanced_metrics_uncalibrated["brier_score"],
            "predicted": predicted_uncal,
            "observed": observed_uncal,
        },
        "isotonic_calibrated": {
            "ece": ece_calibrated,
            "brier_score": float(brier_score_loss(y_test, proba_calibrated)),
            "predicted": predicted_cal,
            "observed": observed_cal,
        },
        "calibration_recommended": calibration_improves,
    }
    logger.info(
        "Calibration: uncalibrated ECE=%.4f (Brier=%.4f) vs. isotonic-calibrated ECE=%.4f (Brier=%.4f) -- %s",
        ece_uncalibrated, advanced_metrics_uncalibrated["brier_score"], ece_calibrated,
        summary["churn_advanced_calibration"]["isotonic_calibrated"]["brier_score"],
        "shipping calibrated model" if calibration_improves else "shipping uncalibrated model",
    )

    for name, pipe in [("churn_baseline_model", baseline_pipe), ("churn_classical_model", classical_pipe), ("churn_advanced_model", final_churn_pipe)]:
        tracker = RunTracker(models_dir, name)
        tracker.log_params(random_seed=cfg.random_seed)
        tracker.save_artifact(pipe, f"{name}.joblib")
        tracker.finalize()

    # ------------------------------------------------------------- Forecast --
    logger.info("Backtesting forecasting methods...")
    forecast_cfg = cfg.section("model")["forecasting"]
    usage_with_plan = attach_plan_tier(tables["monthly_usage"], tables["accounts"])
    forecast_summary = {}
    for segment in [None, "starter", "growth", "enterprise"]:
        series = build_monthly_series(usage_with_plan, segment=segment)
        results = rolling_origin_backtest(series, forecast_cfg["horizon_months"], forecast_cfg["min_history_months"])
        key = segment or "portfolio"
        forecast_summary[key] = [r.__dict__ for r in results]
        logger.info("Forecast segment=%s: %s", key, forecast_summary[key])
    summary["forecasting"] = forecast_summary

    # -------------------------------------------------- Experiment / uplift --
    logger.info("Analyzing retention-offer experiment...")
    exp = tables["experiment"]
    n_treat = int((exp["group"] == "treatment").sum())
    n_control = int((exp["group"] == "control").sum())
    srm = check_sample_ratio_mismatch(n_treat, n_control)
    summary["experiment_srm"] = srm.__dict__

    overall_test = two_proportion_test(
        int(exp.loc[exp["group"] == "treatment", "observed_churn_90d"].sum()), n_treat,
        int(exp.loc[exp["group"] == "control", "observed_churn_90d"].sum()), n_control,
    )
    segment_tests = {}
    for seg in exp["engagement_segment"].unique():
        seg_df = exp[exp["engagement_segment"] == seg]
        nt = int((seg_df["group"] == "treatment").sum())
        nc = int((seg_df["group"] == "control").sum())
        st = int(seg_df.loc[seg_df["group"] == "treatment", "observed_churn_90d"].sum())
        sc = int(seg_df.loc[seg_df["group"] == "control", "observed_churn_90d"].sum())
        segment_tests[seg] = two_proportion_test(st, nt, sc, nc)

    all_p = [overall_test.p_value] + [t.p_value for t in segment_tests.values()]
    adjusted_p = apply_multiple_testing_correction(all_p)
    summary["ab_test_overall"] = {**overall_test.__dict__, "p_value_adjusted": float(adjusted_p[0])}
    summary["ab_test_by_segment"] = {
        seg: {**t.__dict__, "p_value_adjusted": float(adjusted_p[i + 1])}
        for i, (seg, t) in enumerate(segment_tests.items())
    }

    # CUPED: `baseline_risk_proxy` (the trailing-engagement-decline score
    # used for experiment eligibility) is known *before* treatment
    # assignment, which is exactly the pre-experiment covariate CUPED
    # needs -- it's already correlated with the outcome by construction
    # (it's literally what was used to target at-risk accounts), so this
    # should show a real variance reduction, not a token one.
    logger.info("Applying CUPED variance reduction using baseline_risk_proxy...")
    cuped_result = cuped_treatment_effect(
        outcome=exp["observed_churn_90d"].to_numpy().astype(float),
        covariate=exp["baseline_risk_proxy"].to_numpy(),
        treatment=(exp["group"] == "treatment").to_numpy().astype(int),
    )
    summary["ab_test_cuped"] = cuped_result.__dict__
    # Diagnostic only (uses the synthetic-data-only ground-truth column,
    # same as the uplift-model "correlation with true effect" checks
    # elsewhere) -- explains *why* the variance reduction is modest: the
    # covariate correlates reasonably with the true latent risk but far
    # less with the single noisy Bernoulli draw CUPED actually has to work
    # with, which is the realistic bottleneck for CUPED on rare binary
    # outcomes, not a flaw in the covariate choice.
    summary["ab_test_cuped"]["covariate_corr_with_realized_outcome"] = float(
        np.corrcoef(exp["baseline_risk_proxy"], exp["observed_churn_90d"])[0, 1]
    )
    summary["ab_test_cuped"]["covariate_corr_with_true_latent_risk"] = float(
        np.corrcoef(exp["baseline_risk_proxy"], exp["true_p0"])[0, 1]
    )
    raw_ci_width = overall_test.ci_high - overall_test.ci_low
    cuped_ci_width = cuped_result.ci_high - cuped_result.ci_low
    logger.info(
        "CUPED: variance reduction=%.1f%%, raw CI width=%.4f, CUPED CI width=%.4f (%.2f%% narrower)",
        cuped_result.variance_reduction_pct, raw_ci_width, cuped_ci_width,
        (1 - cuped_ci_width / raw_ci_width) * 100 if raw_ci_width > 0 else float("nan"),
    )

    # Power analysis for a follow-up experiment: given each segment's
    # *observed* effect size in this pilot, how large would a properly
    # powered follow-up need to be to confirm it at conventional
    # significance? This is what "run a bigger follow-up" should mean in
    # practice, not just a suggestion left unquantified.
    power_analysis = {}
    for seg, t in segment_tests.items():
        current_n_per_arm = int((exp[exp["engagement_segment"] == seg]["group"] == "treatment").sum())
        mde = abs(t.absolute_effect)
        if mde <= 0 or t.p_control <= 0 or t.p_control >= 1:
            continue
        required_n = required_sample_size_per_arm(t.p_control, mde, alpha=0.05, power=0.8)
        # An astronomically large required-n (millions) isn't a real target
        # to plan a follow-up around -- it means the observed effect is
        # small enough to be indistinguishable from noise at any realistic
        # sample size, i.e. the honest conclusion is "not worth pursuing,"
        # not "collect a few hundred thousand more rows."
        impractical = required_n > 50_000
        power_analysis[seg] = {
            "observed_control_rate": t.p_control,
            "observed_absolute_effect": t.absolute_effect,
            "current_n_per_arm": current_n_per_arm,
            "required_n_per_arm_for_80pct_power": required_n,
            "additional_n_per_arm_needed": max(0, required_n - current_n_per_arm),
            "practical_to_pursue": not impractical,
        }
    summary["ab_test_power_analysis"] = power_analysis
    for seg, p in power_analysis.items():
        if p["practical_to_pursue"]:
            logger.info(
                "Power analysis (%s): observed effect=%.3f, need ~%d per arm for 80%% power (have %d, %d more needed)",
                seg, p["observed_absolute_effect"], p["required_n_per_arm_for_80pct_power"],
                p["current_n_per_arm"], p["additional_n_per_arm_needed"],
            )
        else:
            logger.info(
                "Power analysis (%s): observed effect=%.3f is too small to be worth chasing -- required sample "
                "size (~%d/arm) is not a realistic follow-up target.",
                seg, p["observed_absolute_effect"], p["required_n_per_arm_for_80pct_power"],
            )

    logger.info("Fitting uplift models (S-learner baseline vs. T-learner vs. X-learner)...")
    exp_features = tables["accounts"].merge(exp, on="account_id")
    exp_features = exp_features.merge(
        feature_store[feature_store["snapshot_month"] == data_cfg["observation_cutoff_month"]][
            ["account_id", "tenure_months", "avg_login_trailing_3m", "avg_login_trailing_6m", "login_trend_3v6", "avg_ticket_risk_proba_trailing_3m"]
        ],
        on="account_id", how="left",
    ).fillna(0.0)
    exp_features["treatment"] = (exp_features["group"] == "treatment").astype(int)

    s_model, X_s = fit_s_learner(exp_features, UPLIFT_FEATURE_COLUMNS, "treatment", "observed_churn_90d", random_seed=cfg.random_seed)
    s_cate = predict_cate_s_learner(s_model, X_s)
    s_eval = evaluate_uplift_method(
        "s_learner", s_cate, exp_features["treatment"].to_numpy(), exp_features["observed_churn_90d"].to_numpy(),
        true_effect=exp_features["true_effect"].to_numpy(),
    )

    t_model_treat, t_model_control, X_t = fit_t_learner(exp_features, UPLIFT_FEATURE_COLUMNS, "treatment", "observed_churn_90d", random_seed=cfg.random_seed)
    t_cate = predict_cate_t_learner(t_model_treat, t_model_control, X_t)
    t_eval = evaluate_uplift_method(
        "t_learner", t_cate, exp_features["treatment"].to_numpy(), exp_features["observed_churn_90d"].to_numpy(),
        true_effect=exp_features["true_effect"].to_numpy(),
    )

    x_tau1, x_tau0, x_propensity, X_x = fit_x_learner(exp_features, UPLIFT_FEATURE_COLUMNS, "treatment", "observed_churn_90d", random_seed=cfg.random_seed)
    x_cate = predict_cate_x_learner(x_tau1, x_tau0, x_propensity, X_x)
    x_eval = evaluate_uplift_method(
        "x_learner", x_cate, exp_features["treatment"].to_numpy(), exp_features["observed_churn_90d"].to_numpy(),
        true_effect=exp_features["true_effect"].to_numpy(),
    )

    summary["uplift_s_learner"] = {k: v for k, v in s_eval.__dict__.items() if k != "qini_curve"}
    summary["uplift_t_learner"] = {k: v for k, v in t_eval.__dict__.items() if k != "qini_curve"}
    summary["uplift_x_learner"] = {k: v for k, v in x_eval.__dict__.items() if k != "qini_curve"}
    logger.info("S-learner AUUC=%.4f corr_with_truth=%s", s_eval.auuc, s_eval.correlation_with_true_effect)
    logger.info("T-learner AUUC=%.4f corr_with_truth=%s", t_eval.auuc, t_eval.correlation_with_true_effect)
    logger.info("X-learner AUUC=%.4f corr_with_truth=%s", x_eval.auuc, x_eval.correlation_with_true_effect)

    import joblib
    joblib.dump(s_model, models_dir / "uplift_s_learner.joblib")
    joblib.dump({"treat": t_model_treat, "control": t_model_control}, models_dir / "uplift_t_learner.joblib")
    joblib.dump({"tau1": x_tau1, "tau0": x_tau0, "propensity": x_propensity}, models_dir / "uplift_x_learner.joblib")
    joblib.dump(list(X_t.columns), models_dir / "uplift_feature_columns_encoded.joblib")

    with open(models_dir / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Wrote training_summary.json")


if __name__ == "__main__":
    main()
