# Evaluation Report

_Generated 2026-09-09T07:00:12.992811+00:00 by `pipelines/run_evaluation.py` from `models_store/training_summary.json`. All figures below come from a single run on synthetic data (seed=42) and will vary slightly if data is regenerated with a different seed._

## 1. NLP: ticket risk classifier vs. VADER baseline

| Approach | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| VADER lexicon (baseline) | 0.556 | 0.388 | 0.752 |
| TF-IDF + Logistic Regression | 0.967 | 0.968 | 0.967 |

Evaluated on 3450 held-out tickets, task = detect billing/bug/cancellation-intent tickets from text alone.

## 2. Churn prediction: baseline -> classical -> advanced

Label base rate: train=0.056, test (snapshot held out)=0.058

| Model | ROC-AUC | PR-AUC | Brier | Best F1 (threshold) | Top-decile lift |
|---|---|---|---|---|---|
| Baseline (static attrs, logistic reg.) | 0.541 | 0.074 | 0.055 | 0.134 (t=0.063) | 1.449 |
| Strong classical (logistic reg., full features) | 0.756 | 0.196 | 0.052 | 0.286 (t=0.105) | 3.079 |
| Advanced (gradient boosting, tuned) | 0.746 | 0.197 | 0.052 | 0.308 (t=0.132) | 3.623 |

_F1 is reported at each model's own best threshold, not a fixed 0.5 cutoff -- at a ~5% base rate, every model's predicted probabilities stay well under 0.5, so F1@0.5 is trivially 0 for all three and would not distinguish them._

Best advanced-model hyperparameters (rolling-origin CV): `{'max_iter': 150, 'max_depth': 3, 'learning_rate': 0.03, 'l2_regularization': 0.0}`

### Segment-level performance (advanced model, held-out snapshot)

**By plan tier**

| plan_tier | ROC-AUC | PR-AUC | n |
|---|---|---|---|
| enterprise | 0.770 | 0.258 | 197 |
| growth | 0.716 | 0.274 | 648 |
| starter | 0.766 | 0.182 | 1038 |

**By industry**

| industry | ROC-AUC | PR-AUC | n |
|---|---|---|---|
| b2b_software | 0.650 | 0.115 | 297 |
| fintech | 0.817 | 0.375 | 332 |
| healthtech | 0.799 | 0.380 | 294 |
| logistics | 0.790 | 0.175 | 310 |
| media | 0.726 | 0.236 | 333 |
| retail | 0.711 | 0.222 | 317 |

### Top permutation-importance features (advanced model)

| feature | importance (mean drop in PR-AUC) |
|---|---|
| avg_feature_adoption_trailing_12m | 0.0674 |
| login_trend_3v6 | 0.0217 |
| login_trend_6v12 | 0.0175 |
| avg_login_trailing_3m | 0.0174 |
| tenure_months | 0.0140 |
| avg_feature_adoption_trailing_6m | 0.0131 |
| ticket_count_trailing_3m | 0.0078 |
| avg_api_calls_trailing_3m | 0.0069 |
| avg_ticket_risk_proba_trailing_3m | 0.0068 |
| avg_ticket_sentiment_trailing_3m | 0.0058 |

### Top SHAP features (advanced model) -- cross-checked against permutation importance

| feature | mean abs(SHAP) |
|---|---|
| avg_feature_adoption_trailing_12m | 0.3572 |
| avg_feature_adoption_trailing_6m | 0.1086 |
| avg_feature_adoption_trailing_3m | 0.0976 |
| avg_login_trailing_3m | 0.0835 |
| login_trend_3v6 | 0.0650 |
| tenure_months | 0.0557 |
| login_trend_6v12 | 0.0457 |
| avg_api_calls_trailing_3m | 0.0412 |
| avg_api_calls_trailing_6m | 0.0364 |
| avg_login_trailing_6m | 0.0300 |

3/5 features agree between the SHAP and permutation-importance top-5 lists. Two explainability methods measuring different things (marginal prediction contribution vs. performance drop under shuffling) landing on mostly the same features is a stronger interpretability claim than either method alone.

### Calibration: raw gradient-boosting output vs. isotonic-calibrated

| | ECE | Brier score |
|---|---|---|
| Uncalibrated | 0.0152 | 0.0516 |
| Isotonic-calibrated | 0.0164 | 0.0518 |

**Shipped as `churn_advanced_model.joblib`: the uncalibrated model** -- whichever achieved lower expected calibration error (ECE) on the held-out snapshot, applying the same "keep it only if it measurably helps" standard used everywhere else in this project.

Reliability points (predicted probability -> observed churn rate, by quantile bin), uncalibrated vs. calibrated:

| predicted (uncalibrated) | observed (uncalibrated) | predicted (calibrated) | observed (calibrated) |
|---|---|---|---|
| 0.017 | 0.016 | 0.018 | 0.020 |
| 0.027 | 0.027 | 0.028 | 0.026 |
| 0.033 | 0.016 | 0.034 | 0.015 |
| 0.038 | 0.021 | 0.037 | 0.018 |
| 0.043 | 0.021 | 0.041 | 0.024 |
| 0.051 | 0.048 | 0.049 | 0.041 |
| 0.065 | 0.043 | 0.063 | 0.090 |
| 0.077 | 0.090 | 0.076 | 0.059 |
| 0.096 | 0.090 | 0.097 | 0.091 |
| 0.161 | 0.212 | 0.153 | 0.201 |

_A perfectly calibrated model has predicted == observed in every row. Compare how close each column pair is to the diagonal, not just the summary ECE._

## 3. Forecasting: naive -> seasonal -> Holt-Winters -> GBM+lags (rolling-origin backtest)

**Segment: portfolio**

| Method | MAPE | RMSE | n backtest origins |
|---|---|---|---|
| naive | 0.040 | 19268.8 | 19 |
| seasonal_naive | 0.218 | 98754.8 | 19 |
| holt_winters | 0.035 | 18292.3 | 19 |
| gbm_lags | 0.080 | 40214.5 | 19 |

**Segment: starter**

| Method | MAPE | RMSE | n backtest origins |
|---|---|---|---|
| naive | 0.041 | 2944.4 | 19 |
| seasonal_naive | 0.219 | 14866.4 | 19 |
| holt_winters | 0.036 | 2770.6 | 19 |
| gbm_lags | 0.084 | 6412.9 | 19 |

**Segment: growth**

| Method | MAPE | RMSE | n backtest origins |
|---|---|---|---|
| naive | 0.042 | 7741.0 | 19 |
| seasonal_naive | 0.219 | 38545.3 | 19 |
| holt_winters | 0.039 | 7560.7 | 19 |
| gbm_lags | 0.084 | 16511.8 | 19 |

**Segment: enterprise**

| Method | MAPE | RMSE | n backtest origins |
|---|---|---|---|
| naive | 0.038 | 9045.4 | 19 |
| seasonal_naive | 0.216 | 45479.4 | 19 |
| holt_winters | 0.034 | 8665.5 | 19 |
| gbm_lags | 0.075 | 17511.9 | 19 |

## 4. Retention-offer experiment (A/B test)

Sample ratio mismatch check: treatment n=407, control n=401, chi2 p=0.8328 -> **PASSED**

**Overall effect on 90-day churn probability**

- Treatment churn rate: 0.054, Control churn rate: 0.082
- Absolute effect: -0.028 (95% CI [-0.063, 0.006])
- Cohen's h (effect size): -0.113
- p-value: 0.1110, Holm-adjusted: 0.2221

**By engagement segment**

| segment | treatment rate | control rate | absolute effect | 95% CI | p (adj.) |
|---|---|---|---|---|---|
| low_engagement | 0.045 | 0.102 | -0.056 | [-0.107, -0.006] | 0.0912 |
| high_engagement | 0.062 | 0.062 | 0.001 | [-0.046, 0.048] | 0.9780 |

**Power analysis for a follow-up experiment** (given each segment's *observed* effect size in this pilot, how large a follow-up would need to be to confirm it at 80% power, alpha=0.05):

| segment | observed effect | current n/arm | required n/arm | verdict |
|---|---|---|---|---|
| low_engagement | -0.056 | 198 | 551 | 353 more needed |
| high_engagement | 0.001 | 209 | 2,075,913 | **not worth pursuing -- effect too small to detect at any realistic sample size** |

_This is what "run a bigger follow-up" should mean concretely -- a target sample size, not just a suggestion. Required n is sized off this pilot's own observed effect, which is itself noisy at this sample size; treat it as a starting estimate for the follow-up's design, not a guarantee. A segment flagged "not worth pursuing" isn't a data or calculation error -- it means the observed effect is so close to zero that no realistic experiment could confirm it, which is itself a useful answer._

**CUPED variance reduction** (using `baseline_risk_proxy`, known *before* treatment assignment, as the pre-experiment covariate):

| | Absolute effect | 95% CI | CI width |
|---|---|---|---|
| Raw (two-proportion test) | -0.028 | [-0.063, 0.006] | 0.069 |
| CUPED-adjusted | -0.028 | [-0.063, 0.007] | 0.069 |

Variance reduction: **1.1%** (CI narrowed by 0.43%, same 808 accounts -- no new data collected).

_This is a real but modest reduction, reported as-is rather than swapped for a covariate that happens to look better. The reason it's modest: `baseline_risk_proxy` correlates reasonably with the *true* simulated churn probability (r=-0.34, only knowable because this is synthetic data) but far less with the single noisy Bernoulli draw that is the actual observed outcome (r=-0.10). CUPED's payoff is bounded by correlation with the *realized* metric, not the latent probability behind it -- on a rare binary outcome from a modest sample, that bound is low almost regardless of covariate choice. CUPED earns its keep on continuous, aggregated metrics (revenue, session count) far more reliably than on infrequent binary events like this one -- a real limitation of the technique here, not an implementation shortfall._

## 5. Uplift modeling: S-learner (baseline) vs. T-learner vs. X-learner

| Method | AUUC | Mean predicted uplift | Correlation with true simulated effect | MAE vs. true effect |
|---|---|---|---|---|
| S-learner (baseline) | 8.0799 | 0.0287 | 0.275 | 0.0498 |
| T-learner | 11.0361 | 0.0309 | 0.110 | 0.0567 |
| X-learner | 10.0714 | 0.0309 | 0.162 | 0.0560 |

_Correlation/MAE against the true simulated effect is only measurable because this is synthetic data with a known data-generating process -- it validates that the modeling methodology can recover a known causal effect, not that real-world uplift estimates are this accurate._

## 6. Cohort retention: does account maturity predict churn?

Accounts bucketed into 4 quartiles by how mature they already were at the start of the observation window (a maturity/vintage cohort, not a calendar-signup-month cohort -- every account here already existed before the window with a different starting age). Retention tracked forward through month 29:

| Cohort | Starting size | Retained by month 29 |
|---|---|---|
| Q1 (newest) | 799 | 59.7% |
| Q2 | 746 | 60.3% |
| Q3 | 729 | 59.8% |
| Q4 (oldest) | 726 | 58.0% |

**Spread across cohorts: 2.3 percentage points** -- account maturity barely distinguishes retention here. This corroborates, via a completely independent method, the churn model's own finding in section 2: the *baseline* model (static attributes including tenure) was barely better than random (ROC-AUC 0.54), while behavioral features (usage decline, ticket sentiment) carried the real signal. Two different analyses landing on the same conclusion is a stronger claim than either alone -- and a real answer a retention team would want before assuming "newer customers are riskier" without checking.

## 7. Behavioral segmentation: K-Means vs. hierarchical vs. HDBSCAN

Segmented 1883 active accounts on the latest snapshot using 8 standardized behavioral features (`current_mrr, tenure_months, avg_login_trailing_3m, avg_login_trailing_12m, login_trend_3v6, avg_feature_adoption_trailing_3m, ticket_count_trailing_3m, avg_ticket_sentiment_trailing_3m`).

K selection via silhouette score over k=2..8: best k=2 (silhouette=0.327).

| Method | Silhouette | Notes |
|---|---|---|
| K-Means (k=2) | 0.327 | |
| Hierarchical (k=2, Ward) | 0.346 | |
| HDBSCAN (density-based) | 0.212 | 55% of points labeled noise |

K-Means stability (mean bootstrap ARI across resamples): **0.982**

PCA 2D projection explains 50.1% of variance (components: ['0.327', '0.173']).

_HDBSCAN did find clusters at one swept threshold (min_cluster_size=19), but its silhouette (0.212) is markedly weaker than K-Means's (0.327) and 55% of accounts are unassigned noise. This corroborates K-Means/hierarchical's 2-segment split as the more defensible finding, rather than treating three methods' disagreement as inconclusive -- the weaker method's own confidence (silhouette) is what resolves the disagreement._

### Segment profiles (K-Means, business value)

| segment | size | share | avg MRR | total MRR | churn rate |
|---|---|---|---|---|---|
| 0 | 311 | 16.5% | $624 | $194077 | 1.9% |
| 1 | 1572 | 83.5% | $121 | $189913 | 6.6% |

_Stability (highly reproducible partitions) is not by itself evidence of real cluster structure -- K-Means can partition pure noise very reproducibly too (see `tests/test_segmentation.py::test_stability_alone_cannot_detect_meaningless_clusters_silhouette_can`). The silhouette scores above are what actually indicate whether these segments reflect real structure, not just the stability number._

## 8. Decision engine: portfolio impact of latest batch scoring run

- **Accounts scored**: 1,883
- **Accounts recommended for outreach**: 69
- **Total intervention cost**: $3,105
- **Total expected value**: $670
- **Expected net ROI**: 21.6%
- **Revenue at risk**: $41,918/mo in the top-decile churn-risk accounts -- 10.9% of $383,990/mo total portfolio MRR

_Revenue at risk is independent of the budget-constrained recommendations above -- it's what's exposed if the highest-risk 10% of accounts churn, regardless of how many the intervention budget actually covers. Comparing this to the MRR forecast in section 3 is a natural next step: is the portfolio-level forecast already showing this decline, or is the risk concentrated enough that it wouldn't show up in an aggregate forecast yet?_

### Budget sensitivity

A retention team rarely has an unlimited budget -- this shows what a few concrete budget levels actually buy, instead of only the unconstrained optimum above.

| Budget | Accounts treated | Total expected value | Net ROI |
|---|---|---|---|
| $310 | 6 | $255 | 94.6% |
| $776 | 17 | $436 | 57.0% |
| $1552 | 34 | $591 | 38.6% |
| $2329 | 51 | $652 | 28.4% |
| $3105 | 69 | $670 | 21.6% |
