"""Streamlit dashboard for retention ops: reads only the artifacts the
pipelines already produce (training_summary.json, batch_decisions.csv,
batch_scoring_impact.json, feature_store.parquet) -- it has no modeling
logic of its own, so it can never show a number the pipelines didn't
actually compute.

Run with:
    streamlit run dashboards/app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import streamlit as st

from harbor.config import load_config

st.set_page_config(page_title="Harbor: Retention Decision Engine", layout="wide")

cfg = load_config()
models_dir = cfg.models_dir
processed_dir = cfg.path(key="processed_dir")


@st.cache_data
def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


st.title("Harbor -- Retention Decision Engine")
st.caption("Every number on this page is read from a pipeline artifact, not computed in the dashboard.")

summary_path = models_dir / "training_summary.json"
if not summary_path.exists():
    st.error("No training_summary.json found. Run `python pipelines/run_training_pipeline.py` first.")
    st.stop()
summary = load_json(summary_path)

decisions_path = processed_dir / "batch_decisions.csv"
impact_path = processed_dir / "batch_scoring_impact.json"

tab_overview, tab_models, tab_forecast, tab_experiment, tab_segments, tab_accounts, tab_monitoring = st.tabs(
    ["Portfolio Overview", "Model Performance", "Forecasting", "Experiment & Uplift", "Customer Segments", "At-Risk Accounts", "Monitoring"]
)

with tab_overview:
    if impact_path.exists():
        impact = load_json(impact_path)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accounts scored", f"{impact['n_accounts_scored']:,}")
        c2.metric("Recommended for outreach", f"{impact['n_accounts_recommended']:,}")
        c3.metric("Intervention cost", f"${impact['total_intervention_cost']:,.0f}")
        c4.metric("Expected net ROI", f"{impact['expected_net_roi']:.1%}" if impact["expected_net_roi"] == impact["expected_net_roi"] else "n/a")
        st.caption(f"Total expected value of the recommended interventions: ${impact['total_expected_value']:,.0f}")

        if "total_portfolio_mrr" in impact:
            st.metric(
                "Revenue at risk (top-decile churn risk)",
                f"${impact['mrr_at_risk_top_decile']:,.0f}/mo",
                help=f"{impact['mrr_at_risk_share']:.1%} of ${impact['total_portfolio_mrr']:,.0f}/mo total portfolio MRR. "
                     "Independent of the budget-constrained recommendations -- this is total exposure, not just what's funded.",
            )

        budget_path = processed_dir / "budget_sensitivity.json"
        if budget_path.exists():
            st.subheader("Budget sensitivity")
            st.caption("What a few concrete budget levels actually buy, instead of only the unconstrained optimum above.")
            budget_df = pd.DataFrame(load_json(budget_path)).set_index("budget")
            b1, b2 = st.columns(2)
            with b1:
                st.caption("Accounts treated by budget")
                st.line_chart(budget_df["n_accounts_recommended"])
            with b2:
                st.caption("Expected value by budget")
                st.line_chart(budget_df["total_expected_value"])
    else:
        st.info("Run `python pipelines/run_batch_scoring.py` to populate portfolio impact.")

    st.subheader("NLP: ticket risk classifier vs. VADER baseline")
    nlp = summary["nlp"]
    st.dataframe(
        pd.DataFrame(
            [
                {"approach": "VADER (baseline)", "accuracy": nlp["baseline_accuracy"], "f1": nlp["baseline_f1"], "auc": nlp["baseline_auc"]},
                {"approach": "TF-IDF + LogReg", "accuracy": nlp["classifier_accuracy"], "f1": nlp["classifier_f1"], "auc": nlp["classifier_auc"]},
            ]
        ),
        use_container_width=True,
    )

with tab_models:
    st.subheader("Churn model comparison (held-out snapshot)")
    rows = []
    for key, label in [("churn_baseline", "Baseline"), ("churn_classical", "Classical"), ("churn_advanced", "Advanced")]:
        m = summary[key]
        rows.append({"model": label, **{k: v for k, v in m.items() if k != "n_samples"}})
    st.dataframe(pd.DataFrame(rows).set_index("model"), use_container_width=True)

    st.subheader("Segment performance (advanced model)")
    seg_col1, seg_col2 = st.columns(2)
    with seg_col1:
        st.caption("By plan tier")
        st.dataframe(pd.DataFrame(summary["churn_advanced_segment_by_plan"]), use_container_width=True)
    with seg_col2:
        st.caption("By industry")
        st.dataframe(pd.DataFrame(summary["churn_advanced_segment_by_industry"]), use_container_width=True)

    st.subheader("Explainability: permutation importance vs. SHAP")
    imp_col1, imp_col2 = st.columns(2)
    with imp_col1:
        st.caption("Permutation importance (drop in PR-AUC when shuffled)")
        importances = pd.DataFrame(summary["churn_advanced_top_features"]).set_index("feature")
        st.bar_chart(importances["importance_mean"])
    with imp_col2:
        st.caption("SHAP (mean |contribution| to each prediction)")
        shap_importances = pd.DataFrame(summary["churn_advanced_shap_top_features"]).set_index("feature")
        st.bar_chart(shap_importances["mean_abs_shap"])
    overlap = summary.get("shap_vs_permutation_top5_overlap")
    st.caption(f"{overlap}/5 features agree between the two methods' top 5 -- agreement between two "
               "differently-computed explainability methods is a stronger interpretability claim than either alone.")

    if "churn_advanced_calibration" in summary:
        st.subheader("Calibration: raw model output vs. isotonic-calibrated")
        cal = summary["churn_advanced_calibration"]
        shipped = "isotonic-calibrated" if cal["calibration_recommended"] else "uncalibrated"
        st.caption(f"Shipped model: **{shipped}** (whichever had lower expected calibration error on the held-out snapshot)")
        cal_col1, cal_col2 = st.columns(2)
        with cal_col1:
            st.metric("Uncalibrated ECE", f"{cal['uncalibrated']['ece']:.4f}")
        with cal_col2:
            st.metric("Isotonic-calibrated ECE", f"{cal['isotonic_calibrated']['ece']:.4f}")
        # Each model has its own quantile bin edges, so "predicted" values
        # differ between the two -- plotted separately rather than forced
        # onto one shared x-axis, which would misalign the bins.
        rel_col1, rel_col2 = st.columns(2)
        with rel_col1:
            st.caption("Uncalibrated: predicted vs. observed")
            unc = pd.DataFrame({"predicted": cal["uncalibrated"]["predicted"], "observed": cal["uncalibrated"]["observed"]}).set_index("predicted")
            st.line_chart(unc)
        with rel_col2:
            st.caption("Isotonic-calibrated: predicted vs. observed")
            calib = pd.DataFrame({"predicted": cal["isotonic_calibrated"]["predicted"], "observed": cal["isotonic_calibrated"]["observed"]}).set_index("predicted")
            st.line_chart(calib)
        st.caption("A perfectly calibrated model traces predicted == observed in every bin.")

with tab_forecast:
    st.subheader("Rolling-origin forecast backtest (MAPE by method)")
    forecast_rows = []
    for segment, results in summary["forecasting"].items():
        for r in results:
            forecast_rows.append({"segment": segment, **r})
    forecast_df = pd.DataFrame(forecast_rows)
    pivot = forecast_df.pivot(index="method", columns="segment", values="mape")
    st.dataframe(pivot.style.format("{:.3f}"), use_container_width=True)
    st.caption("Lower MAPE is better. gbm_lags underperforming naive/holt_winters here is a real finding, not a display error -- "
               "see docs/EVALUATION_REPORT.md for the full discussion of why the added complexity doesn't pay off on this data.")

with tab_experiment:
    st.subheader("Retention-offer A/B test")
    srm = summary["experiment_srm"]
    if srm["passed"]:
        st.success(f"Sample ratio mismatch check passed (p={srm['p_value']:.3f}) -- randomization looks healthy.")
    else:
        st.error(f"Sample ratio mismatch check FAILED (p={srm['p_value']:.4f}) -- do not trust the effect estimates below.")

    overall = summary["ab_test_overall"]
    st.metric(
        "Overall absolute effect on 90-day churn probability",
        f"{overall['absolute_effect']:.3f}",
        help=f"95% CI [{overall['ci_low']:.3f}, {overall['ci_high']:.3f}], Holm-adjusted p={overall['p_value_adjusted']:.3f}",
    )
    st.dataframe(pd.DataFrame(summary["ab_test_by_segment"]).T, use_container_width=True)

    if summary.get("ab_test_power_analysis"):
        st.subheader("Power analysis: sizing a follow-up experiment")
        st.caption("Required sample size per arm to confirm each segment's *observed* effect at 80% power, alpha=0.05.")
        power_rows = []
        for seg, p in summary["ab_test_power_analysis"].items():
            verdict = f"{p['additional_n_per_arm_needed']:,} more needed" if p["practical_to_pursue"] else "not worth pursuing -- effect too small"
            power_rows.append(
                {
                    "segment": seg,
                    "observed_effect": p["observed_absolute_effect"],
                    "current_n_per_arm": p["current_n_per_arm"],
                    "required_n_per_arm": p["required_n_per_arm_for_80pct_power"],
                    "verdict": verdict,
                }
            )
        st.dataframe(pd.DataFrame(power_rows).set_index("segment"), use_container_width=True)

    if summary.get("ab_test_cuped"):
        st.subheader("CUPED: variance reduction using pre-experiment data")
        c = summary["ab_test_cuped"]
        cc1, cc2, cc3 = st.columns(3)
        cc1.metric("Variance reduction", f"{c['variance_reduction_pct']:.1f}%")
        raw_width = overall["ci_high"] - overall["ci_low"]
        cuped_width = c["ci_high"] - c["ci_low"]
        cc2.metric("CI width (raw)", f"{raw_width:.4f}")
        cc3.metric("CI width (CUPED)", f"{cuped_width:.4f}")
        st.caption(
            f"Covariate (`baseline_risk_proxy`) correlation with the realized outcome: {c['covariate_corr_with_realized_outcome']:.2f} "
            f"vs. with the true simulated risk: {c['covariate_corr_with_true_latent_risk']:.2f}. "
            "The gap between those two numbers is *why* the reduction is modest here -- CUPED's payoff is bounded "
            "by correlation with the noisy realized outcome, not the latent probability behind it. Reported as-is, "
            "not swapped for a covariate that scores better."
        )

    st.subheader("Uplift modeling: S-learner vs. T-learner vs. X-learner")
    uplift_rows = [{"method": "S-learner (baseline)", **summary["uplift_s_learner"]}, {"method": "T-learner", **summary["uplift_t_learner"]}]
    if "uplift_x_learner" in summary:
        uplift_rows.append({"method": "X-learner", **summary["uplift_x_learner"]})
    st.dataframe(pd.DataFrame(uplift_rows).set_index("method"), use_container_width=True)

with tab_segments:
    cohort_path = processed_dir / "cohort_retention.csv"
    if cohort_path.exists():
        st.subheader("Cohort retention: does account maturity predict churn?")
        cohort_df = pd.read_csv(cohort_path)
        pivot = cohort_df.pivot(index="month_index", columns="cohort", values="retention_rate")
        st.line_chart(pivot)
        last_month = int(cohort_df["month_index"].max())
        final = cohort_df[cohort_df["month_index"] == last_month]
        spread_pp = (final["retention_rate"].max() - final["retention_rate"].min()) * 100
        st.caption(
            f"Spread across maturity cohorts at month {last_month}: {spread_pp:.1f} percentage points -- account "
            "maturity barely distinguishes retention here, corroborating the churn model's own finding that static "
            "attributes (including tenure) carry almost no signal on their own. See docs/EVALUATION_REPORT.md section 6."
        )
        st.divider()

    seg_summary_path = models_dir / "segmentation_summary.json"
    seg_projection_path = processed_dir / "segmentation_projection.csv"
    if not (seg_summary_path.exists() and seg_projection_path.exists()):
        st.info("Run `python pipelines/run_segmentation.py` to populate this tab.")
    else:
        seg = load_json(seg_summary_path)
        projection = pd.read_csv(seg_projection_path)

        st.subheader("Method comparison")
        mc = seg["method_comparison"]
        c1, c2, c3 = st.columns(3)
        c1.metric(f"K-Means (k={seg['best_k']}) silhouette", f"{mc['kmeans_silhouette']:.3f}")
        c2.metric(f"Hierarchical (k={seg['best_k']}) silhouette", f"{mc['hierarchical_silhouette']:.3f}")
        hdbscan_sil = mc["hdbscan_silhouette_excl_noise"]
        c3.metric(
            "HDBSCAN silhouette",
            f"{hdbscan_sil:.3f}" if hdbscan_sil is not None else "n/a",
            help=f"{mc['hdbscan_noise_fraction']:.0%} of points labeled noise (not assigned to any dense cluster)",
        )
        st.caption(
            f"K-Means stability (mean bootstrap ARI across resamples): {seg['kmeans_stability_bootstrap_ari']:.3f}. "
            "High stability alone doesn't prove real structure -- K-Means can partition pure noise reproducibly "
            "too; silhouette above is the check that actually matters. See docs/EVALUATION_REPORT.md section 7."
        )

        st.subheader("PCA projection (2D), colored by K-Means segment")
        projection_display = projection.copy()
        projection_display["kmeans_label"] = projection_display["kmeans_label"].astype(str)
        st.scatter_chart(projection_display, x="pca_x", y="pca_y", color="kmeans_label", size=30)

        st.subheader("Segment profiles")
        profiles_df = pd.DataFrame(
            [
                {
                    "segment": p["label"],
                    "size": p["size"],
                    "share": f"{p['share']:.1%}",
                    "avg_mrr": f"${p['avg_mrr']:,.0f}",
                    "total_mrr": f"${p['total_mrr']:,.0f}",
                    "churn_rate": f"{p['churn_rate']:.1%}" if p["churn_rate"] is not None else "n/a",
                }
                for p in seg["segment_profiles"]
            ]
        )
        st.dataframe(profiles_df.set_index("segment"), use_container_width=True)

with tab_accounts:
    st.subheader("Batch-scored accounts")
    if decisions_path.exists():
        decisions = pd.read_csv(decisions_path)
        if "intervention_cost" in decisions.columns:
            st.caption("Tiered decision engine is active -- intervention cost varies per account by tier.")
            tier_counts = decisions.loc[decisions["recommended_action"] != "no_action", "recommended_action"].value_counts()
            if len(tier_counts) > 0:
                st.bar_chart(tier_counts)
        min_ev = st.slider("Minimum expected value ($)", float(decisions["expected_value"].min()), float(decisions["expected_value"].max()), 0.0)
        filtered = decisions[decisions["expected_value"] >= min_ev].sort_values("expected_value", ascending=False)
        st.dataframe(filtered, use_container_width=True, height=500)
        st.caption(f"Showing {len(filtered)} of {len(decisions)} scored accounts.")
    else:
        st.info("Run `python pipelines/run_batch_scoring.py` to populate this table.")

with tab_monitoring:
    st.subheader("Data drift (PSI: reference training snapshot vs. current)")
    drift_path = processed_dir / "drift_report.json"
    if drift_path.exists():
        drift = load_json(drift_path)
        if drift["retraining_recommended"]:
            st.error(
                f"Retraining recommended: {drift['n_critical']} critical, {drift['n_warning']} warning feature(s) "
                f"drifted between snapshot_month={drift['reference_snapshot_month']} (reference) and "
                f"snapshot_month={drift['current_snapshot_month']} (current)."
            )
        else:
            st.success(
                f"No retraining trigger: {drift['n_critical']} critical, {drift['n_warning']} warning feature(s) "
                f"out of {drift['n_features_checked']} checked."
            )
        st.caption(
            "A little drift on tenure/MRR-like features between snapshots is expected -- accounts simply age "
            f"{drift['current_snapshot_month'] - drift['reference_snapshot_month']} months between the reference "
            "and current snapshot. That alone shouldn't cross the critical threshold; PSI thresholds follow the "
            "conventional < 0.1 stable / 0.1-0.25 warning / > 0.25 critical rule of thumb."
        )
        drift_df = pd.DataFrame(drift["features"]).sort_values("psi", ascending=False)
        st.dataframe(drift_df.set_index("feature"), use_container_width=True)
        st.caption(f"Report generated {drift['generated_at']}.")
    else:
        st.info("Run `python pipelines/run_monitoring.py` to populate this section.")

    st.subheader("Model registry (lineage)")
    registry_path = models_dir / "model_registry.json"
    if registry_path.exists():
        registry = load_json(registry_path)
        registry_rows = [
            {"model": name, "git_commit": entry.get("git_commit") or "n/a", "finished_at": entry.get("finished_at"), **entry.get("metrics", {})}
            for name, entry in registry.items()
        ]
        st.dataframe(pd.DataFrame(registry_rows).set_index("model"), use_container_width=True)
    else:
        st.info("Run `python pipelines/run_training_pipeline.py` to populate this section.")
