"""Lightweight, file-based experiment/model tracker.

This intentionally avoids adding an MLflow server dependency: every training
run writes a timestamped JSON record (params, metrics, git commit if
available) plus a joblib artifact to `models_store/`. That is sufficient for
a single-person, reproducible portfolio project, and the JSON schema is
compatible with being replaced by real MLflow calls later (same
params/metrics/artifact triad) without changing calling code.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        # git not installed, not a repo, or the call timed out -- none of
        # that should block a training run from finishing.
        return None


class RunTracker:
    """Records one training run: params in, metrics out, artifact saved."""

    def __init__(self, store_dir: Path, run_name: str) -> None:
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.run_name = run_name
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.params: dict[str, Any] = {}
        self.metrics: dict[str, Any] = {}

    def log_params(self, **params: Any) -> None:
        self.params.update(params)

    def log_metrics(self, **metrics: Any) -> None:
        self.metrics.update(metrics)

    def save_artifact(self, obj: Any, filename: str) -> Path:
        artifact_path = self.store_dir / filename
        joblib.dump(obj, artifact_path)
        return artifact_path

    def finalize(self) -> Path:
        record = {
            "run_name": self.run_name,
            "started_at": self.started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "git_commit": _git_commit(),
            "params": self.params,
            "metrics": self.metrics,
        }
        record_path = self.store_dir / f"{self.run_name}_run.json"
        with open(record_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, default=str)
        return record_path


def build_registry_summary(store_dir: Path) -> dict[str, dict]:
    """Aggregates every `*_run.json` written by RunTracker.finalize() in
    `store_dir` into one lineage manifest: run_name -> {git_commit,
    finished_at, metrics}. This is deliberately not a version *store* --
    every run still overwrites its artifact file in place -- just a single
    place to see what's currently live and how it was validated, matching
    this project's "no MLflow server, file-based, upgradeable later" tracking
    philosophy (see the module docstring)."""
    store_dir = Path(store_dir)
    registry: dict[str, dict] = {}
    for record_path in sorted(store_dir.glob("*_run.json")):
        with open(record_path, "r", encoding="utf-8") as f:
            record = json.load(f)
        run_name = record.get("run_name", record_path.stem.removesuffix("_run"))
        registry[run_name] = {
            "git_commit": record.get("git_commit"),
            "finished_at": record.get("finished_at"),
            "metrics": record.get("metrics", {}),
        }
    return registry
