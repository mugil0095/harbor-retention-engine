"""FastAPI service layer.

Three inference paths, deliberately different, matching how a real
retention system would be built:

- `GET /accounts/{id}/churn-score` -- **real-time, by ID**. This is what a
  real caller (a CS console, an account page) actually does: give me this
  account's current churn risk, right now. Features are looked up from the
  feature store already produced by the training pipeline, not supplied by
  the caller -- no real caller has ~20 trailing-window features sitting
  around to hand over.
- `POST /score/churn` -- **real-time, by raw features**. For a caller that
  already computed its own feature vector (a different system, a what-if
  scenario, a feature not yet materialized into the feature store). Kept
  distinct from the endpoint above rather than merged, since the two have
  different failure modes: this one 422s on a malformed feature vector, the
  ID-based one 404s on an unknown account.
- `GET /accounts/{id}/decision` -- **batch** lookup. The full decision
  (churn risk x causal uplift x cost -> recommended action) is expensive to
  recompute per request (it requires the uplift model and portfolio-level
  budget allocation) and doesn't need sub-second freshness, so it is
  computed nightly by `pipelines/run_batch_scoring.py` and served here as a
  lookup against the latest batch output.

Model artifacts are loaded once at startup and exposed with a version
string (git commit + filename) so responses are traceable to the run that
produced them.
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from harbor.config import load_config
from harbor.features.build_features import (
    CATEGORICAL_FEATURE_COLUMNS,
    NUMERIC_FEATURE_COLUMNS,
)
from harbor.serving.schemas import (
    AccountFeatures,
    ChurnScoreResponse,
    DecisionResponse,
    DriftReportResponse,
    HealthResponse,
)
from harbor.utils.logging_config import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

_state: dict = {
    "churn_model": None,
    "model_version": "unloaded",
    "batch_decisions": None,
    "feature_store": None,
    "drift_report": None,
    "model_registry": None,
}


def _churn_model_path() -> Path:
    cfg = load_config()
    return cfg.models_dir / "churn_advanced_model.joblib"


def _batch_decisions_path() -> Path:
    cfg = load_config()
    return cfg.path(key="processed_dir") / "batch_decisions.csv"


def _feature_store_path() -> Path:
    cfg = load_config()
    return cfg.path(key="processed_dir") / cfg.section("data")["feature_store_file"]


def _drift_report_path() -> Path:
    cfg = load_config()
    return cfg.path(key="processed_dir") / "drift_report.json"


def _model_registry_path() -> Path:
    cfg = load_config()
    return cfg.models_dir / "model_registry.json"


def load_artifacts() -> None:
    registry_path = _model_registry_path()
    if registry_path.exists():
        with open(registry_path, "r", encoding="utf-8") as f:
            _state["model_registry"] = json.load(f)
        logger.info("Loaded model registry (%d tracked models) from %s", len(_state["model_registry"]), registry_path)
    else:
        logger.warning("No model registry found at %s; run the training pipeline first.", registry_path)

    model_path = _churn_model_path()
    if model_path.exists():
        _state["churn_model"] = joblib.load(model_path)
        registry_entry = (_state["model_registry"] or {}).get("churn_advanced_model")
        version_tag = (
            registry_entry["git_commit"]
            if registry_entry and registry_entry.get("git_commit")
            else int(model_path.stat().st_mtime)
        )
        _state["model_version"] = f"{model_path.name}@{version_tag}"
        logger.info("Loaded churn model from %s", model_path)
    else:
        logger.warning("No churn model artifact found at %s; run the training pipeline first.", model_path)

    drift_path = _drift_report_path()
    if drift_path.exists():
        with open(drift_path, "r", encoding="utf-8") as f:
            _state["drift_report"] = json.load(f)
        logger.info("Loaded drift report from %s", drift_path)
    else:
        logger.warning("No drift report found at %s; run pipelines/run_monitoring.py first.", drift_path)

    decisions_path = _batch_decisions_path()
    if decisions_path.exists():
        _state["batch_decisions"] = pd.read_csv(decisions_path).set_index("account_id")
        logger.info("Loaded %d batch decisions from %s", len(_state["batch_decisions"]), decisions_path)
    else:
        logger.warning("No batch decisions found at %s; run the batch scoring pipeline first.", decisions_path)

    feature_store_path = _feature_store_path()
    if feature_store_path.exists():
        cfg = load_config()
        latest_snapshot = cfg.section("data")["test_snapshot_month"]
        store = pd.read_parquet(feature_store_path)
        _state["feature_store"] = store[store["snapshot_month"] == latest_snapshot].set_index("account_id")
        logger.info(
            "Loaded feature store (%d accounts, latest snapshot_month=%d) from %s",
            len(_state["feature_store"]), latest_snapshot, feature_store_path,
        )
    else:
        logger.warning("No feature store found at %s; run the training pipeline first.", feature_store_path)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    load_artifacts()
    yield


app = FastAPI(title="Harbor Retention Decision Engine", version="0.1.0", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        churn_model_loaded=_state["churn_model"] is not None,
        batch_decisions_loaded=_state["batch_decisions"] is not None,
        feature_store_loaded=_state["feature_store"] is not None,
        drift_monitoring_loaded=_state["drift_report"] is not None,
        model_registry_loaded=_state["model_registry"] is not None,
    )


@app.get("/monitoring/drift", response_model=DriftReportResponse)
def get_drift_report() -> DriftReportResponse:
    """Latest data-drift report, produced by pipelines/run_monitoring.py."""
    if _state["drift_report"] is None:
        raise HTTPException(status_code=503, detail="No drift report loaded. Run pipelines/run_monitoring.py first.")
    return DriftReportResponse(**_state["drift_report"])


@app.get("/models/registry")
def get_model_registry() -> dict:
    """Lineage for every tracked model: git commit, training timestamp, and
    metrics logged at training time. Not a version store -- reflects
    whatever is currently on disk in models_store/, one entry per model."""
    if _state["model_registry"] is None:
        raise HTTPException(status_code=503, detail="No model registry loaded. Run the training pipeline first.")
    return _state["model_registry"]


@app.get("/accounts/{account_id}/churn-score", response_model=ChurnScoreResponse)
def get_churn_score(account_id: int) -> ChurnScoreResponse:
    """Real-time score for one account, looked up by ID -- the realistic
    shape of a real-time request (a caller has an account ID, not a
    pre-computed feature vector)."""
    if _state["churn_model"] is None:
        raise HTTPException(status_code=503, detail="Churn model not loaded. Run the training pipeline first.")
    if _state["feature_store"] is None:
        raise HTTPException(status_code=503, detail="Feature store not loaded. Run the training pipeline first.")
    if account_id not in _state["feature_store"].index:
        raise HTTPException(status_code=404, detail=f"No feature-store row on file for account_id={account_id}")

    feature_cols = NUMERIC_FEATURE_COLUMNS + CATEGORICAL_FEATURE_COLUMNS
    row = _state["feature_store"].loc[[account_id], feature_cols]
    proba = float(_state["churn_model"].predict_proba(row)[:, 1][0])
    return ChurnScoreResponse(account_id=account_id, churn_probability=proba, model_version=_state["model_version"])


@app.post("/score/churn", response_model=ChurnScoreResponse)
def score_churn(features: AccountFeatures) -> ChurnScoreResponse:
    if _state["churn_model"] is None:
        raise HTTPException(status_code=503, detail="Churn model not loaded. Run the training pipeline first.")

    row = pd.DataFrame([features.model_dump()])
    feature_cols = NUMERIC_FEATURE_COLUMNS + CATEGORICAL_FEATURE_COLUMNS
    missing = [c for c in feature_cols if c not in row.columns]
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing feature columns: {missing}")

    proba = float(_state["churn_model"].predict_proba(row[feature_cols])[:, 1][0])
    return ChurnScoreResponse(
        account_id=features.account_id, churn_probability=proba, model_version=_state["model_version"]
    )


@app.get("/accounts/{account_id}/decision", response_model=DecisionResponse)
def get_decision(account_id: int) -> DecisionResponse:
    if _state["batch_decisions"] is None:
        raise HTTPException(status_code=503, detail="Batch decisions not loaded. Run the batch scoring pipeline first.")
    if account_id not in _state["batch_decisions"].index:
        raise HTTPException(status_code=404, detail=f"No decision on file for account_id={account_id}")

    row = _state["batch_decisions"].loc[account_id]
    return DecisionResponse(
        account_id=account_id,
        churn_probability=float(row["churn_probability"]),
        predicted_uplift=float(row["predicted_uplift"]),
        expected_value=float(row["expected_value"]),
        recommended_action=str(row["recommended_action"]),
        scored_at=str(row["scored_at"]),
        model_version=str(row["model_version"]),
    )
