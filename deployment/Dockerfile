# Serves the scoring API. Model artifacts and batch-scored decisions are
# baked into the image for simplicity (this is a single-person portfolio
# project, not a fleet); a real deployment would instead mount a volume or
# pull artifacts from an object store / model registry at container start,
# so a new model version doesn't require a full image rebuild.
FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml .
COPY src/ src/
# -e (editable), not a regular install: harbor.config.REPO_ROOT is computed
# as Path(__file__).resolve().parents[2], which only lands on /app if
# config.py stays at /app/src/harbor/config.py. A non-editable install
# copies package files into site-packages instead, silently pointing
# REPO_ROOT at the Python installation directory and breaking every
# config-derived path (configs/, models_store/, data/) -- verified by
# building a wheel and installing it non-editably; load_config() raised
# FileNotFoundError immediately. Do not "simplify" this to `pip install .`.
RUN pip install --no-cache-dir -e .

COPY configs/ configs/
COPY models_store/ models_store/
COPY data/processed/ data/processed/

EXPOSE 8000
ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "harbor.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
