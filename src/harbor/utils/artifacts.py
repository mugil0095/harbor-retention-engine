"""Guards for loading required model/data artifacts."""

from __future__ import annotations

from pathlib import Path


def require_files_exist(paths: list[Path], context: str) -> None:
    """Fails loudly and specifically when required artifacts are missing,
    instead of letting joblib/pandas raise a generic, harder-to-diagnose
    error deep inside a loading call."""
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"{context}: missing required artifact(s): {missing}. "
            "Run pipelines/run_training_pipeline.py first."
        )
