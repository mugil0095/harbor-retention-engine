# CLAUDE.md — working memory for this repo

This file is read automatically at the start of future sessions in this
directory. It exists so context survives conversation compaction/clearing —
keep it updated as state changes, don't let it go stale.

## What this project is

**Harbor** — a customer-retention decision engine for a subscription SaaS
business, built entirely by Claude as a **data science portfolio project**
for the user (vexorin8@gmail.com) to use in job applications. The user did
not write any of the code themselves — they need to be able to explain and
defend the project's design decisions in interviews, so favor clear,
first-principles explanations over jargon when discussing it with them.

One-line pitch: combines churn prediction, NLP, causal/uplift modeling,
behavioral segmentation, and MRR forecasting into one decision engine that
prices `churn_risk x causal_uplift x cost -> recommended action` per
account — not five disconnected notebooks.

**Full detail lives in the project's own docs, not here** — don't duplicate
it, read it when needed:
- [README.md](README.md) — quickstart, repo layout, what's implemented
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — system design, data flow
- [docs/MODEL_CARD.md](docs/MODEL_CARD.md) — what each model does, honest findings
- [docs/EVALUATION_REPORT.md](docs/EVALUATION_REPORT.md) — generated from real pipeline runs, not hand-written
- [docs/LIMITATIONS.md](docs/LIMITATIONS.md) — what's deliberately not built
- [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) — local/Docker/cloud notes, including a real packaging bug found and fixed
- [docs/PROGRESS_LOG.md](docs/PROGRESS_LOG.md) — **dated, chronological history of every session.** Read this for "what happened when." Append a new dated entry at the end of every future session (template at the bottom of that file) — don't just update this file's snapshot and lose the timeline.

## Current status (as of last session)

- Fully implemented, tested, and run end-to-end on synthetic data.
- **162/162 tests passing, 97% line coverage.** `ruff check` clean.
- Every number in `docs/EVALUATION_REPORT.md` comes from an actual pipeline
  run (`run_data_generation.py` -> `run_training_pipeline.py` ->
  `run_segmentation.py` -> `run_batch_scoring.py` -> `run_cohort_analysis.py` -> `run_evaluation.py`,
  in that order — segmentation must run before batch scoring so behavioral
  segment membership can be joined into the decisions table).
- A real bug was found and fixed via careful review (not assumed working):
  the Docker images used a non-editable `pip install .`, which breaks
  `harbor.config.REPO_ROOT` resolution and would have crashed both
  containers on startup. Fixed (`-e .` in both Dockerfiles), reproduced and
  pinned as a regression test in `tests/test_packaging.py`.

## Repository

**Pushed to GitHub**: https://github.com/mugil0095/harbor-retention-engine
(`main` branch). The user pushes it themselves from a local terminal (not
via Claude); latest verified push on 2026-09-02, commit `e92da32`. Verified
from this side each time via `git ls-remote origin main` matching the
local commit — this actually happened, not just claimed.

Before committing anything new: `.gitignore` was tightened the same day to
exclude `data/raw/*`, `data/processed/*`, `models_store/*` (all
reproducible), and `.coverage` — check `git status` looks sane (no
`.venv`, no `*.joblib`/`*.parquet`, no `.coverage`) before staging.

## Explicitly NOT done — don't assume otherwise

- **CI passed for the first time on 2026-09-02** (run `33605551341`, after
  fixing a real failure — see PROGRESS_LOG). Verified via `gh run watch`,
  not assumed. If you touch `pipelines/*.py` again on Windows, remember new
  files there need the executable bit set explicitly
  (`git update-index --chmod=+x`) or ruff's EXE001 will fail on Linux CI
  while passing silently on Windows.
- **Docker has never actually been built.** No Docker available in this
  environment. The packaging bug above was found via manual wheel
  build/install reproduction (see `tests/test_packaging.py`), not via
  `docker build`. If Docker ever becomes available, actually build it
  before claiming it works.

## How to run things

```bash
pip install -e ".[dev]"
python pipelines/run_data_generation.py
python pipelines/run_training_pipeline.py
python pipelines/run_segmentation.py
python pipelines/run_batch_scoring.py
python pipelines/run_cohort_analysis.py
python pipelines/run_evaluation.py
pytest                                      # 162 tests, ~2 min (test_packaging.py builds a real wheel, adds ~15s)
ruff check src tests pipelines dashboards
streamlit run dashboards/app.py             # :8501
uvicorn harbor.serving.api:app --reload     # :8000
```

## User collaboration preferences (apply broadly, not just here)

- **Do not publish visuals as Claude Artifacts unless explicitly asked to
  "publish" or "share."** Default: save the file locally and deliver via
  `SendUserFile` with `display: "render"` so it shows inline in the
  conversation without creating a hosted claude.ai link. (Full detail:
  `memory/feedback_no_artifact_publishing.md` in the memory system.)
- The user wants to genuinely understand this project (not just receive
  a finished artifact) — they've asked for plain-language walkthroughs and
  an architecture visualization. Expect more of these requests; answer them
  thoroughly and in first-principles terms, don't assume prior ML/DS
  background beyond what's evident in the conversation.
- Style: the user's own writing is terse/informal with non-native-English
  phrasing — match their directness in responses, but keep explanations
  fully spelled out (don't mirror the terseness into unclear answers).

## Working style established across this project (worth continuing)

- Every claim gets verified by actually running the code, not asserted —
  this project's credibility rests on that (e.g. the calibration section
  honestly reports that isotonic recalibration was tried and *rejected*
  because it didn't help; the A/B test's overall effect is reported as
  *not* statistically significant rather than dressed up).
- When adding anything new, follow the existing pattern: implement -> write
  real tests -> run the actual pipeline -> update the relevant doc(s) with
  real numbers -> re-run the full test suite before considering it done.
- Diagrams/visuals for this project so far: `docs/diagrams/architecture-blueprint.html`
  (local file, IBM Plex Sans/Mono, blueprint/chart-inspired palette — fits
  the "Harbor" nautical-chart branding). Reuse that visual language if more
  diagrams are requested, rather than starting a new style from scratch.
