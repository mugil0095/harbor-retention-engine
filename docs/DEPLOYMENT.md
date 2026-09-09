# Deployment

## Local development

```bash
pip install -e ".[dev]"
python pipelines/run_data_generation.py
python pipelines/run_training_pipeline.py
python pipelines/run_batch_scoring.py
uvicorn harbor.serving.api:app --reload            # API on :8000
streamlit run dashboards/app.py                    # dashboard on :8501
```

## Docker (local, no cloud account needed)

```bash
cd deployment
docker compose up --build
```

This builds two images (`deployment/Dockerfile` for the API,
`deployment/Dockerfile.dashboard` for the dashboard) and starts both.
Model artifacts and processed data are copied into the image at build time
-- run the training/batch-scoring pipelines on the host first so
`models_store/` and `data/processed/` are populated before `docker compose
build`.

This is a deliberately simple deployment story for a single-person
project. It does **not** attempt to be a production MLOps setup; see below
for what that would add.

**Verification note**: `docker build`/`docker compose up` are not runnable in
the sandboxed environment this project was developed in (no Docker there),
but `.github/workflows/ci.yml` now builds and runs both images for real on
every push/PR, using GitHub's `ubuntu-latest` runners (which do have Docker):
it reuses the data/model artifacts the pipeline smoke-test step already
generated as the build context, runs `docker compose up --build -d --wait`,
then curls both `/health` (API) and `/_stcore/health` (dashboard) against
the live containers before tearing them down. This is the first real,
CI-verified Docker build+run this project has had -- see the workflow file
for the exact steps, and check the Actions tab for the latest result rather
than assuming it's green.

Before that CI step existed, the Dockerfiles were only reviewed by tracing
through what they actually do, which caught a real bug: `pip install .`
(non-editable) breaks `harbor.config.REPO_ROOT` because a non-editable
install physically copies package files out of `src/` into site-packages,
which would have made the API and dashboard containers fail immediately on
startup with `FileNotFoundError: Config file not found`. Reproduced and
fixed (Dockerfiles now use `pip install -e .`); see
`tests/test_packaging.py`, which builds a real wheel and installs it
non-editably to pin the failure mode as a regression test.

## Cloud deployment sketch (not implemented, described for completeness)

A minimal, low-cost path on any major cloud:

1. Push the two images to a container registry.
2. Run the API as a single always-on container (e.g. Cloud Run / Fargate /
   App Runner, scale-to-zero where available) -- this workload is light
   enough that a serverless container tier is appropriate and inexpensive.
3. Run `run_training_pipeline.py` and `run_batch_scoring.py` as a scheduled
   job (e.g. a nightly Cloud Run Job / ECS scheduled task) writing artifacts
   to object storage (S3/GCS) instead of the local filesystem; the API
   container pulls the latest artifact from there at startup instead of
   having it baked into the image.
4. Put the dashboard behind the same or a separate scale-to-zero container;
   it has no write path, so it's safe to run multiple replicas.

Total expected cost for this workload at low traffic: low tens of dollars a
month on any of the above, dominated by the always-on API container rather
than compute for training (training runs in under 5 minutes on a laptop).

## Configuration

Everything environment-specific lives in `configs/config.yaml` --
account/month counts, snapshot months, cost/margin assumptions, drift
thresholds. No source file hard-codes a path, seed, or business constant.
To point at a different config file, pass its path to `load_config()`
(all pipeline entry points accept this, though the CLI scripts currently
default to the repo's config -- see Limitations for what a `--config` flag
would add).

## Secrets

There are none to manage in this project -- no external APIs, no database
credentials, no cloud SDK calls. If this were extended to pull from a real
warehouse or object store, credentials would be read from environment
variables (never committed), following the same pattern FastAPI/uvicorn
already uses for its own runtime config.
