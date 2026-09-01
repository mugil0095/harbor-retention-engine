# Harbor -- a Customer-Retention Decision Engine

Harbor answers one question for a subscription SaaS business: **for a given
at-risk account, should we intervene, and is it worth it?**

It's not a churn classifier. Churn probability alone doesn't tell a retention
team what to do -- you also need to know whether an intervention would
actually change the outcome (a causal question, not a predictive one), and
whether the expected benefit is worth the cost. Harbor combines:

1. **Behavioral segmentation** -- K-Means vs. hierarchical vs. HDBSCAN
   (density-based), compared on silhouette score, with bootstrap stability
   checked separately from cluster quality.
2. **Churn prediction** -- baseline -> classical -> gradient-boosted, evaluated
   with a genuinely time-aware split, cross-checked with SHAP vs. permutation
   importance, and calibration-tested (isotonic recalibration is tried and
   only kept if it actually improves on the raw output).
3. **NLP on support tickets** -- a TF-IDF classifier vs. a lexicon baseline,
   used as a churn-risk feature.
4. **A retention-offer A/B test + uplift modeling (S-learner vs. T-learner
   vs. X-learner)**, with CUPED variance reduction -- estimating *who the
   offer actually helps*, not just who is at risk.
5. **MRR forecasting** -- naive -> seasonal -> Holt-Winters -> gradient
   boosting with lags, backtested with rolling-origin validation.
6. **Cohort retention analysis** -- an independent check (by account
   maturity, not behavior) that corroborates the churn model's own finding
   that static attributes alone carry almost no signal.

...into one decision layer that prices `churn_risk x causal_uplift x cost`
into a recommended action per account, served through a FastAPI scoring
service and a Streamlit ops dashboard.

**All data is synthetic and generated locally** (see [Reproducing the data](#reproducing-the-data)) --
no external downloads, no API keys, no cost to run.

## Why this exists

Most churn-model portfolios stop at "predict who will churn." That's the
easy 80%. The hard, high-value part -- and the part that maps directly onto
what a product/applied data scientist is actually asked to do -- is turning
a risk score into a decision: is this worth intervening on, and how do we
know the intervention works rather than just correlating with accounts that
were going to stay anyway? That's what the experiment + uplift + decision
layer here is for.

## Quickstart

```bash
python -m venv .venv
source .venv/Scripts/activate   # or .venv\Scripts\activate on Windows cmd
pip install -e ".[dev]"

python pipelines/run_data_generation.py     # generates data/raw/*.csv
python pipelines/run_training_pipeline.py   # trains all models, writes models_store/
python pipelines/run_segmentation.py        # clusters the customer base, writes segmentation_summary.json
python pipelines/run_batch_scoring.py       # scores the portfolio, enriched with behavioral segment
python pipelines/run_cohort_analysis.py     # cohort retention curves, writes cohort_retention.csv
python pipelines/run_evaluation.py          # renders docs/EVALUATION_REPORT.md from the runs above

pytest                                      # 162 tests
streamlit run dashboards/app.py             # ops dashboard on :8501
uvicorn harbor.serving.api:app --reload     # scoring API on :8000
```

Or with Docker:

```bash
cd deployment
docker compose up --build
```

(Docker images bake in whatever is currently in `models_store/` and
`data/processed/` at build time -- run the pipelines above first.)

## Reproducing the data

There is no real dataset here on purpose: no public source combines
subscription billing, product usage, support-ticket text, and a retention
experiment for the same accounts. `pipelines/run_data_generation.py`
simulates all four from one internally-consistent data-generating process
(a latent "account health" variable drives usage decline, ticket sentiment,
and churn hazard together -- see the docstring in
[`generate_synthetic_data.py`](src/harbor/data/generate_synthetic_data.py)).
Same `random_seed` in [`configs/config.yaml`](configs/config.yaml) always
reproduces byte-identical data.

Because the data-generating process is known, this project can also check
whether the uplift models actually recover the *true* simulated treatment
effect -- see section 5 of [`docs/EVALUATION_REPORT.md`](docs/EVALUATION_REPORT.md).
That check is a methodology validation specific to synthetic data, not a
claim about real-world uplift accuracy.

## Documentation

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) -- system design, data flow, batch-vs-real-time decisions
- [`docs/DATA_DICTIONARY.md`](docs/DATA_DICTIONARY.md) -- every table/column, target variables, known data-quality issues
- [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) -- what each model does, its limitations, intended use
- [`docs/EVALUATION_REPORT.md`](docs/EVALUATION_REPORT.md) -- **generated from actual pipeline output**, not hand-written
- [`docs/API.md`](docs/API.md) -- endpoint reference
- [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) -- local, Docker, and cloud deployment notes
- [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) -- what this project does not claim to solve

## Repository layout

```
src/harbor/
  data/            synthetic data generation + schema + validation
  features/        leakage-safe feature engineering (time-aware snapshots)
  nlp/             ticket text: VADER baseline vs. TF-IDF classifier
  models/          churn_model.py, forecasting.py, uplift.py
  segmentation/    K-Means / hierarchical / HDBSCAN clustering, stability, PCA projection
  analytics/       cohort retention curves (maturity/vintage cohorts, independent of the modeling pipeline)
  experimentation/ A/B test statistics (SRM check, two-proportion test, power, multiple testing, CUPED)
  decision/        churn risk x uplift x cost -> recommended action
  serving/         FastAPI app
  monitoring/      PSI-based drift detection
  utils/           config, logging, experiment tracking, artifact guards
pipelines/         run_data_generation / run_training_pipeline / run_segmentation / run_batch_scoring / run_cohort_analysis / run_evaluation
dashboards/        Streamlit ops dashboard
notebooks/         01_eda.ipynb -- exploratory data analysis, executed with outputs saved
tests/             162 pytest tests -- data validation, leakage, models, segmentation, calibration, SHAP, API, drift, e2e
deployment/        Dockerfile(s), docker-compose.yml
docs/              architecture, data dictionary, model card, evaluation report, limitations
configs/config.yaml  every tunable parameter -- nothing is hard-coded in source
```

## Status

Implemented, tested (162/162 passing, 97% line coverage), and run end-to-end
on synthetic data. Results in `docs/EVALUATION_REPORT.md` are from an actual
pipeline run, not illustrative numbers.
