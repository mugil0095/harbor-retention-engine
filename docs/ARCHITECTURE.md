# Architecture

## Data flow

```
configs/config.yaml (seed, snapshot months, cost/margin assumptions)
        |
        v
pipelines/run_data_generation.py
        |  generates accounts.csv, monthly_usage.csv, support_tickets.csv,
        |  experiment_assignments.csv from one internally-consistent DGP
        v
   data/raw/*.csv  -----> harbor.data.validate (schema + FK + range checks, fails loudly)
        |
        v
pipelines/run_training_pipeline.py
        |
        |-- harbor.nlp.ticket_analysis
        |     fits TF-IDF+LogReg ticket-risk classifier (vs. VADER baseline)
        |
        |-- harbor.features.build_features
        |     builds one row per (account, snapshot_month), features strictly
        |     from month_index < snapshot_month, label from
        |     [snapshot_month, snapshot_month + horizon)
        |     -> data/processed/feature_store.parquet
        |
        |-- harbor.models.churn_model + harbor.models.explainability
        |     baseline (static attrs) -> classical (logistic, full features)
        |     -> advanced (gradient boosting, rolling-origin-CV-tuned)
        |     trained on all snapshots except the held-out test_snapshot_month;
        |     permutation importance cross-checked against SHAP; isotonic
        |     calibration tried and shipped only if it improves ECE
        |
        |-- harbor.models.forecasting
        |     naive -> seasonal_naive -> Holt-Winters -> GBM+lags,
        |     rolling-origin backtested per plan-tier segment
        |
        |-- harbor.experimentation.ab_test
        |     SRM guardrail, two-proportion test, CI, Holm correction across
        |     overall + per-segment tests, power calc for a follow-up test
        |
        |-- harbor.models.uplift
        |     S-learner (baseline) vs. T-learner vs. X-learner on the RCT sample
        |
        v
   models_store/*.joblib + training_summary.json
        |
        v
pipelines/run_segmentation.py  (reads feature_store.parquet)
        |  harbor.segmentation.clustering: K-Means / hierarchical / HDBSCAN,
        |  each swept across its own parameter (k for K-Means, min_cluster_size
        |  for HDBSCAN -- neither gets a single untested guess), silhouette
        |  comparison, bootstrap stability, PCA projection, per-segment
        |  business-value profiling
        v
   models_store/segmentation_summary.json + data/processed/segmentation_projection.csv
        |
        v
pipelines/run_batch_scoring.py
        |  applies advanced churn model + T-learner to the latest snapshot,
        |  prices each account via harbor.decision.decision_engine, computes
        |  revenue-at-risk (top-decile churn-risk MRR), and enriches each
        |  recommendation with its behavioral segment from the step above --
        |  segmentation and forecasting are not dead-end reports, they feed
        |  the same decision output the rest of the system produces
        v
   data/processed/batch_decisions.csv, budget_sensitivity.json
        |
pipelines/run_cohort_analysis.py  (reads accounts.csv + monthly_usage.csv directly;
        |                          independent of the modeling pipeline above)
        v
   data/processed/cohort_retention.csv
        |
        +--> pipelines/run_evaluation.py --> docs/EVALUATION_REPORT.md
        +--> dashboards/app.py (Streamlit, reads artifacts only)
        +--> harbor.serving.api (FastAPI, reads artifacts only)
```

No component recomputes another component's output -- the dashboard and API
are pure consumers of what the pipelines already wrote to disk. This means
adding a new metric means changing the pipeline that computes it once, not
every downstream consumer.

## Batch vs. real-time inference

Three inference paths, deliberately different, not for variety:

- **`GET /accounts/{id}/churn-score`** is real-time, by ID: features are
  looked up from the feature store the training pipeline already wrote,
  and scored synchronously. This is the realistic shape of "score this
  account right now" -- a real caller (a CS console, an account page) has
  an account ID, not a hand-computed feature vector.
- **`POST /score/churn`** is real-time, by raw features: for a caller that
  already has its own feature vector (a different system, a what-if
  scenario). Kept as a separate endpoint rather than folded into the one
  above because the two fail differently -- 422 on a malformed vector vs.
  404 on an unknown account -- and conflating them would hide which
  failure a caller is actually hitting.
- **`GET /accounts/{id}/decision`** is a lookup against a nightly batch
  table. The full decision requires the uplift model plus portfolio-level
  budget allocation, is not needed at sub-second freshness, and is cheaper
  to compute once for the whole portfolio than per-request.

A production system would still run batch scoring on a schedule (e.g.
nightly cron / Airflow DAG) rather than the manual `python pipelines/run_*.py`
sequence used here, and would materialize the feature store from a real
streaming/warehouse pipeline instead of `build_feature_store` reading flat
files. That orchestration layer -- Airflow, a message queue, a managed
feature store -- was deliberately not built; see [LIMITATIONS.md](LIMITATIONS.md).

## Time-aware validation

Every model here is validated with rolling-origin / snapshot-based splits,
never a random row split:

- **Churn model**: multiple observation snapshots (`data.snapshot_months`
  in config) are stacked into one feature store; training uses all
  snapshots except the last, which is held out entirely. Hyperparameter
  tuning uses successive rolling-origin folds within the training snapshots
  (`rolling_origin_splits` in `churn_model.py`), never the held-out test
  snapshot.
- **Forecasting**: `rolling_origin_backtest` walks the origin forward one
  month at a time, training only on data strictly before each origin.
- **Feature engineering**: `_assert_no_leakage` in `build_features.py` is a
  runtime assertion, not just a convention -- it raises if any row used to
  build features for snapshot `S` has `month_index >= S`.

## Monitoring & retraining triggers

`harbor.monitoring.drift` implements Population Stability Index (PSI) drift
detection, the standard for churn/credit-risk model monitoring: bin a
reference distribution (the training feature distribution), compare a new
sample's bin proportions, summarize as one number. `retraining_recommended`
triggers on any single feature with PSI >= 0.25 (critical) or three-plus
features simultaneously >= 0.1 (warning) -- the conventional thresholds.
This is implemented and tested but not wired into a scheduler in this
project (see Limitations).

## What's deliberately not built

- **Airflow/Prefect orchestration** -- four sequential scripts are enough to
  demonstrate pipeline structure without adding infrastructure that a single
  developer running this locally gets no benefit from.
- **A real feature store** (Feast, Tecton) -- `feature_store.parquet` plays
  that role at the scale this project operates at.
- **MLflow tracking server** -- `harbor.utils.experiment_tracker.RunTracker`
  writes the same params/metrics/artifact triad to local JSON + joblib
  files; swapping in real MLflow calls would not change any calling code.
- **Prophet / deep learning forecasters** -- evaluated and rejected: with 30
  months of monthly history, a heavier model has more parameters than
  signal to fit against. See `docs/EVALUATION_REPORT.md` section 3 for why
  even GBM+lags underperforms Holt-Winters here.
