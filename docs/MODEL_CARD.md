# Model Card

Numbers below are from the run documented in `docs/EVALUATION_REPORT.md`
(synthetic data, seed=42). Regenerating data with a different seed will
shift them somewhat; the relative ordering and qualitative findings are
stable across seeds we spot-checked.

## Churn model

**Purpose**: estimate the probability an active account churns within the
next 3 months, from behavioral, billing, and support-ticket features.

**Three versions, in order of complexity:**

| | Features | ROC-AUC | Top-decile lift | Notes |
|---|---|---|---|---|
| Baseline | plan/industry/size/region/tenure only | 0.54 | 1.4x | barely better than random -- static attributes carry almost no churn signal |
| Classical | + behavioral trends + NLP ticket features (logistic regression) | 0.76 | 3.1x | most of the achievable lift comes from here |
| Advanced | same features, gradient boosting (tuned) | 0.75 | 3.6x | **does not beat classical on ROC-AUC**, but wins on top-decile lift and best-achievable F1 |

**Recommendation**: ship the advanced model if the business use case is
"rank accounts to prioritize outreach" (top-decile lift is what matters).
If the use case needs a globally well-calibrated, easily-explained
probability, the classical logistic regression is a defensible choice and
loses almost nothing. This is a "don't use complexity for its own sake"
finding, not a hedge -- both are reported so the choice is checkable.

**Known limitation**: the churn base rate in any 3-month window is ~5-6%.
F1 is reported at each model's own best threshold, not a fixed 0.5 cutoff
(at 0.5, every model here predicts negative for everyone).

**Segment performance** varies meaningfully by industry (PR-AUC ranges from
0.12 for `b2b_software` to 0.38 for `fintech`/`healthtech` in the sampled
run) -- a single portfolio-wide threshold will over-flag some industries and
under-flag others. See `docs/EVALUATION_REPORT.md` for the full table.

**Explainability**: permutation importance and SHAP were computed
independently and cross-checked against each other (3/5 agreement in the
top-5 features in the sampled run, `avg_feature_adoption_trailing_12m`
dominant in both). Partial disagreement is expected -- the two methods
measure different things (performance drop under shuffling vs. marginal
prediction contribution) -- but substantial overlap is what makes either
ranking trustworthy rather than an artifact of one method's assumptions.

**Calibration**: isotonic recalibration was tried and *rejected* -- the raw
gradient-boosting output was already better calibrated (ECE 0.015 vs. 0.016
after calibration) on the held-out snapshot in the sampled run. The
uncalibrated model is what's actually shipped as `churn_advanced_model.joblib`.
This is reported as a genuine methodology check, not a step skipped for
convenience: calibration is computed and compared every training run, and
only kept when it measurably helps.

## NLP: ticket-risk classifier

**Purpose**: convert raw support-ticket text into a feature indicating how
"at-risk" a ticket sounds (billing / bug / cancellation-intent vs. neutral).

| | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| VADER lexicon (baseline, no training) | ~0.56 | ~0.39 | ~0.75 |
| TF-IDF + logistic regression (trained) | ~0.97 | ~0.97 | ~0.97 |

**Important caveat**: this text is templated synthetic data with ~6% label
noise and shared filler sentences injected specifically to avoid a
misleadingly perfect score. Real support-ticket text is far noisier
(typos, mixed topics per ticket, sarcasm, multilingual content); expect
meaningfully lower real-world performance. The comparison's *direction*
(a trained classifier beats a lexicon baseline) generalizes; the absolute
numbers do not.

## Behavioral segmentation

**Purpose**: group the active customer base by behavior (not just plan
tier) to see whether distinct, actionable segments exist.

Three clustering families were compared on the same standardized
behavioral features (current MRR, tenure, trailing engagement, login
trend, ticket volume, ticket sentiment):

| Method | Silhouette | Notes |
|---|---|---|
| K-Means (k=2, chosen by silhouette sweep k=2..8) | ~0.33 | moderate, not strong separation |
| Hierarchical (Ward, same k) | ~0.35 | agrees closely with K-Means |
| HDBSCAN (density-based) | n/a | found **no dense clusters** (100% noise) |

**Finding, reported honestly**: the customer base does not split into
sharply distinct density clusters -- HDBSCAN found nothing at a reasonable
density threshold, and K-Means/hierarchical only achieve moderate
silhouette scores (well under the ~0.5+ that would indicate strong,
unambiguous separation). What *does* exist is a real, business-relevant,
highly stable 2-way split: a smaller high-value/low-churn segment (~17% of
accounts, ~$620 avg MRR, ~2% churn) and a much larger mass-market/
higher-churn segment (~83%, ~$120 avg MRR, ~7% churn).

**A methodological point worth calling out explicitly**: this segmentation
was validated with *two* different checks that measure different things --
bootstrap stability (do the same clusters reappear on a resampled subset?)
and silhouette score (are the clusters actually well-separated?). They can
disagree: K-Means forced onto pure, structureless noise still produces a
highly *stable* partition (it reproducibly slices a unimodal blob into the
same wedges every time), so stability alone is not evidence of real
structure. Silhouette score is what actually distinguishes the case above
from real clusters -- see
`tests/test_segmentation.py::test_stability_alone_cannot_detect_meaningless_clusters_silhouette_can`
for a runnable demonstration. Both checks are always reported together for
exactly this reason.

**Recommendation**: use the 2-segment split for prioritization (it's real,
stable, and maps to a clear business distinction), but don't oversell it as
a rich taxonomy -- it's one meaningful cut, not five or six distinct
customer personas.

## Forecasting

**Purpose**: forecast portfolio and per-plan-tier MRR 3 months ahead.

| Method | Typical MAPE (portfolio) |
|---|---|
| Naive (repeat last value) | ~0.040 |
| Seasonal naive (12-month lag) | ~0.22 (worse -- see below) |
| Holt-Winters (additive trend+seasonal) | ~0.035 (best) |
| Gradient boosting + lag features | ~0.080 (worse than naive) |

**Finding, reported honestly**: seasonal naive and GBM-with-lags both
underperform the simple naive baseline here. Seasonal naive fails because
MRR trend dominates over any seasonal component in this simulation; GBM
underperforms because 30 months of history leaves too few independent
rolling-origin windows to fit lag-12 features reliably. **Recommendation:
Holt-Winters for production**, with naive as the fallback when history is
short. This is the "establish baselines first, don't add complexity that
doesn't pay for itself" principle applied to forecasting.

## Experimentation & uplift

**Purpose**: estimate whether a retention offer causally reduces churn, and
for whom.

- Sample-ratio-mismatch guardrail: passes (randomization looks healthy).
- **Overall effect is not statistically significant** after Holm correction
  (adjusted p ~ 0.22) -- do not ship a company-wide retention-offer rollout
  on this evidence alone.
- The `low_engagement` segment shows a larger, closer-to-significant effect
  (raw p ~ 0.09 after adjustment) -- suggestive, not confirmed. A follow-up
  experiment focused on that segment, with a larger sample, is the
  appropriate next step (see `required_sample_size_per_arm` in
  `ab_test.py` for the power calculation to size it).
- **CUPED** (pre-experiment covariate adjustment) was applied using
  `baseline_risk_proxy`, and delivered a real but modest ~1% variance
  reduction -- reported honestly rather than swapped for a covariate that
  happens to score better. The reason: CUPED's payoff is bounded by the
  covariate's correlation with the *realized* outcome, and a single binary
  churn draw is inherently noisy even when the covariate correlates well
  with the *true* underlying probability (see `docs/EVALUATION_REPORT.md`
  section 4 for the full breakdown). CUPED is far more effective on
  continuous, aggregated metrics than on rare binary events like this one --
  a real limitation of the technique on this outcome, not a bug.
- **Uplift models**: three learners were compared, not just one. S-learner
  shows the highest correlation with the (only knowable-because-synthetic)
  true effect (0.28), T-learner the highest AUUC/ranking ability (11.0), and
  X-learner lands in between on both (0.16 correlation, AUUC 10.1) --
  consistent with how X-learner is designed to behave (it blends the
  S-learner's and T-learner's estimates via propensity weighting, so
  landing between them is expected, not a sign either implementation is
  wrong). No single learner dominates every axis -- reported as-is rather
  than picking a "winner." T-learner is used in the batch-scoring pipeline
  since ranking ability (AUUC) is what the decision engine actually needs
  to prioritize outreach.

**This is intentionally not a success story dressed up as one.** A
retention team reading this would correctly conclude: promising signal in
one segment, not yet strong enough to act on portfolio-wide, worth a
better-powered follow-up experiment before committing budget.

## Decision engine

**Purpose**: turn the above into a per-account recommendation.
`expected_value = predicted_uplift x monthly_margin x months_credited - cost`.
In the sampled batch-scoring run: 69 of 1,883 scored accounts had positive
expected value, total intervention cost ~$3,105, total expected value
~$670 (a modest ~22% expected ROI) -- consistent with the experiment's
weak-but-not-zero effect above. This number should be read as "what the
model believes given the (statistically inconclusive) experiment," not as
a validated ROI claim.

## Intended use / out of scope

- Intended: as a prioritization and decision-support tool for a retention
  team, with a human reviewing recommendations before acting on them.
- Not intended: fully automated intervention without human review, or as
  evidence for a statistically confirmed causal claim (the underlying
  experiment is underpowered for that).
