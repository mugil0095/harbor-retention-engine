# Limitations

Being explicit about what this project does not do, on the theory that a
reviewer will find these gaps anyway and it's better to have already named
them.

## Data

- **All data is synthetic.** The qualitative findings (behavioral features
  beat static attributes, a trained NLP classifier beats a lexicon
  baseline, Holt-Winters beats naive) are the kind of thing that generalizes
  to real SaaS data; the absolute numbers (0.97 F1, 0.76 AUC) do not, and
  should not be quoted as if they were measured on a real product.
- The synthetic data-generating process is smoother and more separable than
  real customer behavior -- there is no real-world noise like data-entry
  errors, mid-contract plan changes with retroactive billing adjustments,
  multi-product accounts, or genuinely adversarial/ambiguous support text.

## Modeling

- The advanced (gradient boosting) churn model does not clearly beat the
  classical (logistic regression) model on every metric. This is reported
  honestly rather than hidden, but it does mean the "advanced model" story
  here is "we checked, and the complexity mostly doesn't pay off" rather
  than "the fancy model wins" -- which is the more useful and more common
  real-world outcome, but worth being upfront about.
- The retention-offer experiment's overall effect is not statistically
  significant after multiple-testing correction. The decision-engine numbers
  downstream of it (expected value, ROI) inherit that uncertainty and
  should be read as "what the model currently believes," not a validated
  business outcome.
- Uplift modeling uses S-, T-, and X-learners with logistic/ridge base
  learners, not causal forests or doubly-robust estimators (e.g. via
  `econml`/`causalml`). Those would likely do better with more data, but
  add a heavy dependency and more moving parts than an ~800-row RCT sample
  can support -- not implemented, listed under Future Improvements instead.
- The tiered decision engine (`configs/config.yaml`'s `decision.intervention_tiers`,
  off by default via `use_tiered_decision_engine: false`) prices two extra
  intervention options (a cheap email nudge, a pricier white-glove call) by
  applying a hand-picked `effect_multiplier` to the *one* RCT-validated
  offer's uplift estimate. Only `standard_offer` (multiplier 1.0) reflects
  an actual experiment -- the other multipliers are an illustrative modeling
  assumption, not a measured finding, since no RCT arm for those
  interventions was ever run. That is exactly why the flag defaults off.
- The tiered engine assigns each account its own best positive-EV tier
  independently, then funds top-down by EV within budget -- a greedy
  heuristic, not a true cross-account knapsack. A real optimizer could gain
  a bit more total value by downgrading one account to a cheaper tier to
  free budget for two higher-EV accounts elsewhere; that's real
  combinatorial optimization and disproportionate complexity for the
  marginal gain here, so it's listed under Future Improvements instead.

## Engineering

- **No orchestrator** (Airflow/Prefect/Dagster). Four scripts run in
  sequence. Appropriate for a single-person project's scale; would need
  real orchestration (retries, backfills, DAG-level monitoring) at team
  scale.
- **No real feature store.** `GET /accounts/{id}/churn-score` looks features
  up from `feature_store.parquet` -- a flat file frozen at the last
  training run, not a live, continuously-updated store. A production system
  needs point-in-time correct feature retrieval (the features for account X
  as of the moment of the request, not as of the last batch job), which is
  a genuinely hard problem (Feast/Tecton exist specifically to solve it)
  that this project doesn't attempt.
- **Model artifacts are baked into Docker images** rather than pulled from
  an artifact store at container start. Fine for a single environment;
  would need to change for multi-environment (staging/prod) promotion.
- **No authentication/authorization on the API.** Not appropriate to ship
  without it, but out of scope for demonstrating the modeling/decision
  pipeline this project is about.
- **Monitoring is implemented but not scheduled.** `harbor.monitoring.drift`
  computes PSI and a retraining recommendation correctly (and is tested),
  but nothing calls it on a schedule against live traffic -- there is no
  live traffic, since this isn't deployed against a real product.
- No `--config` CLI flag on the pipeline scripts; they load
  `configs/config.yaml` directly, though every function underneath accepts
  an explicit config object, so wiring a flag through would be a small
  change if multiple environments were needed.

## Statistical

- The A/B test's statistical power was not designed in advance -- the
  experiment sample size in the synthetic data was chosen for demonstration
  purposes, not calculated from a target minimum detectable effect. A real
  experiment should size its sample from `required_sample_size_per_arm`
  *before* running, not evaluate power after the fact.
- Segment-level A/B test results (by `engagement_segment`) are exploratory
  in this run, not pre-registered -- in a real experiment, segment cuts
  decided after seeing results would need to be flagged as such or
  corrected more conservatively than the Holm procedure used here, which
  assumes the segments were specified in advance.
