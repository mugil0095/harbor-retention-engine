# Progress Log

A dated, human-readable record of what happened in each work session on
this project. Unlike `CLAUDE.md` (a snapshot of *current* state), this is a
chronological history — append to it, never rewrite past entries.

---

## 2026-08-28 — Initial build

- Selected the concept (retention decision engine) from 3 candidates, over
  two alternatives (marketplace pricing intelligence, support-ticket triage).
- Built the full pipeline from scratch: synthetic data generation, feature
  engineering (leakage-safe, time-aware), churn models (baseline → classical
  → gradient boosting), NLP ticket classifier, forecasting, A/B test
  analysis, S-/T-learner uplift models, decision engine, FastAPI service,
  Streamlit dashboard.
- Wrote the initial test suite, ran it, found and fixed 4 real bugs in the
  process (a missing `tenure_months` column, an experiment-segmentation bug
  that had collapsed 100% of participants into one segment, a meaningless
  F1@0.5 metric, forecasters silently degrading to naive with too little
  history).
- Wrote full documentation set (README, ARCHITECTURE, DATA_DICTIONARY,
  MODEL_CARD, API, DEPLOYMENT, LIMITATIONS, generated EVALUATION_REPORT).

## 2026-08-29 — Closing gaps, self-review

- User asked "did you complete the entire project?" — audited honestly
  against the original spec and found real gaps: no segmentation/clustering
  module, no EDA notebook, only S-/T-learner (no X-learner), CI never
  actually executed, nothing committed to git.
- Built the segmentation module (K-Means, hierarchical, HDBSCAN — each
  swept across parameters, compared on silhouette + bootstrap stability).
- Wrote and executed a real EDA notebook (`notebooks/01_eda.ipynb`).
- Added calibration checking (isotonic vs. raw — isotonic was tried and
  **rejected**, honestly reported) and SHAP explainability (cross-checked
  against permutation importance).
- Added the X-learner uplift model (lands between S- and T-learner on both
  metrics, as theory predicts — a good sanity check).
- Added power analysis for a follow-up experiment and a budget-sensitivity
  sweep for the decision engine.
- User asked for a genuine architecture/design review (not interview
  quizzing). Found and fixed 5 real issues: `models_store` hardcoded in 5
  files despite the project's own "nothing hardcoded" principle; a dead
  config key (`uplift.learner`) that did nothing; HDBSCAN given only one
  untested parameter guess while K-Means got a fair sweep; segmentation and
  forecasting were dead-end reports never feeding the actual decision
  output; the "real-time" API endpoint required callers to hand-supply ~20
  features, which no real caller would have.
- Ran an actual coverage report (not assumed) — found `experiment_tracker.py`
  at 0% coverage and `holt_winters_forecast`'s real fit path completely
  untested. Fixed both with real tests.
- **Found and fixed a real deployment bug**: the Docker images used
  `pip install .` (non-editable), which breaks `harbor.config.REPO_ROOT`
  and would crash both containers on startup. Reproduced by building and
  installing an actual wheel, fixed with `-e .`, pinned as a regression
  test (`tests/test_packaging.py`).
- Final state: **150/150 tests passing, 97% line coverage, ruff clean.**

## 2026-08-30 — Communication/process corrections

- Published an architecture diagram as a Claude Artifact without being
  asked to — user corrected this: visuals should stay local and
  in-conversation only, not published to a hosted link by default. Deleted
  the artifact, saved the preference to memory (`feedback_no_artifact_publishing.md`).
- Rebuilt the diagram as a local file (`docs/diagrams/architecture-blueprint.html`)
  and delivered it via `SendUserFile` instead.
- Created `CLAUDE.md` and memory files (`user_context.md`,
  `feedback_no_artifact_publishing.md`) so project state and collaboration
  preferences survive conversation compaction.
- Created this progress log for the same reason, at the user's request.

## 2026-09-01 — CUPED variance reduction

- User asked to continue with "more technical improvements" — offered a
  choice (GitHub push / more technical work / interview prep materials);
  user chose more technical work.
- Implemented CUPED (Controlled-experiment Using Pre-Experiment Data) in
  `harbor.experimentation.ab_test`: `cuped_adjust` and
  `cuped_treatment_effect`, with 5 new tests (unbiasedness, variance
  reduction under correlation, no reduction under independence, recovery
  of a known effect with a tighter CI, arm-size consistency).
- Wired into the training pipeline using `baseline_risk_proxy` (known
  before treatment assignment) as the covariate. Real result: only a
  **~1.1% variance reduction** — investigated *why* rather than swapping
  in a better-looking covariate: the covariate correlates with the *true*
  simulated risk at r=-0.34 but with the *realized* binary outcome at only
  r=-0.10, and CUPED's payoff is bounded by the latter. Reported honestly
  in `docs/EVALUATION_REPORT.md`, `docs/MODEL_CARD.md`, and the dashboard,
  including the diagnostic that explains the gap.
- Test count: 150 → 155, still 97% coverage, ruff clean.

### Same day — cohort retention curve

- Built the cohort-retention "layer cake" chart: new `harbor.analytics`
  module, bucketing accounts into maturity/vintage quartiles (by tenure at
  the start of the observation window) and tracking retention decay
  forward. New pipeline `run_cohort_analysis.py`, 7 tests (including a
  strong invariant check: retention within a cohort can only decrease,
  never increase).
- Real result: only a **2.3 percentage-point spread** across cohorts —
  account maturity barely predicts churn here. Cross-checked this against
  the earlier finding that the baseline churn model (tenure + static
  attributes only) was barely better than random (AUC 0.54) — two
  independent methods landing on the same conclusion, reported explicitly
  as corroboration in `docs/EVALUATION_REPORT.md` section 6.
- Wired into the dashboard's Customer Segments tab, verified visually
  (the line chart shows the classic layer-cake shape with minimal fan-out,
  matching the honest 2.3pp finding).
- Found and fixed one more leftover instance of the `models_store`
  hardcoding bug (missed in `dashboards/app.py` during the earlier
  architecture review) while touching that file.
- Test count: 155 → 162, 97% coverage maintained, ruff clean.
- Both additions (CUPED + cohort retention) fully wired end-to-end:
  pipeline → tests → real numbers → EVALUATION_REPORT.md → MODEL_CARD.md
  (CUPED only) → dashboard.

---

## Template for new entries

```
## YYYY-MM-DD — <short theme>

- What was asked.
- What was built/fixed/decided.
- Any new bugs found, and how they were caught.
- Current test count / coverage if it changed.
- Anything still pending or awaiting the user's decision.
```
