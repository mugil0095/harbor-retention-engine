"""RunTracker: the lightweight MLflow-alternative used by
run_training_pipeline.py to save every model artifact and its run metadata.
Had zero test coverage despite being on the path of every training run --
a bug here (e.g. an artifact silently not saved, or a run record with the
wrong schema) would have broken the pipeline without any test catching it.
"""

from __future__ import annotations

import json
import subprocess

import joblib
import numpy as np
import pytest

from harbor.utils.experiment_tracker import RunTracker, _git_commit


def test_run_tracker_creates_store_dir_if_missing(tmp_path):
    store_dir = tmp_path / "does_not_exist_yet"
    assert not store_dir.exists()
    RunTracker(store_dir, "test_run")
    assert store_dir.exists()


def test_log_params_accumulates_across_multiple_calls(tmp_path):
    tracker = RunTracker(tmp_path, "test_run")
    tracker.log_params(a=1)
    tracker.log_params(b=2)
    assert tracker.params == {"a": 1, "b": 2}


def test_log_params_overwrites_same_key(tmp_path):
    tracker = RunTracker(tmp_path, "test_run")
    tracker.log_params(a=1)
    tracker.log_params(a=2)
    assert tracker.params == {"a": 2}


def test_log_metrics_accumulates_across_multiple_calls(tmp_path):
    tracker = RunTracker(tmp_path, "test_run")
    tracker.log_metrics(roc_auc=0.75)
    tracker.log_metrics(pr_auc=0.20)
    assert tracker.metrics == {"roc_auc": 0.75, "pr_auc": 0.20}


def test_save_artifact_round_trips_via_joblib(tmp_path):
    tracker = RunTracker(tmp_path, "test_run")
    original = {"weights": [1, 2, 3], "note": "a fitted model would go here"}
    artifact_path = tracker.save_artifact(original, "model.joblib")
    assert artifact_path == tmp_path / "model.joblib"
    assert artifact_path.exists()
    assert joblib.load(artifact_path) == original


def test_finalize_writes_valid_json_with_expected_schema(tmp_path):
    tracker = RunTracker(tmp_path, "churn_advanced_model")
    tracker.log_params(random_seed=42)
    tracker.log_metrics(roc_auc=0.75, pr_auc=0.20)
    record_path = tracker.finalize()

    assert record_path == tmp_path / "churn_advanced_model_run.json"
    with open(record_path, encoding="utf-8") as f:
        record = json.load(f)

    assert record["run_name"] == "churn_advanced_model"
    assert record["params"] == {"random_seed": 42}
    assert record["metrics"] == {"roc_auc": 0.75, "pr_auc": 0.20}
    assert "started_at" in record and "finished_at" in record
    assert record["started_at"] <= record["finished_at"]  # ISO-8601 strings sort chronologically


def test_finalize_serializes_numpy_metrics_without_raising(tmp_path):
    """Real training-pipeline metrics can include numpy scalars (e.g. from
    sklearn). np.float64 is a subclass of Python float, so it serializes as
    a native JSON number; np.int64 is NOT a subclass of Python int, so it
    falls through to `default=str` and comes back as a string -- a real,
    slightly surprising asymmetry worth a test rather than an assumption.
    Either way, finalize() must not raise on numpy input.
    """
    tracker = RunTracker(tmp_path, "test_run")
    tracker.log_metrics(roc_auc=np.float64(0.756), n_samples=np.int64(1883))
    record_path = tracker.finalize()  # must not raise
    with open(record_path, encoding="utf-8") as f:
        record = json.load(f)
    assert record["metrics"]["roc_auc"] == pytest.approx(0.756)
    assert isinstance(record["metrics"]["roc_auc"], float)
    assert record["metrics"]["n_samples"] == "1883"  # stringified by default=str
    assert isinstance(record["metrics"]["n_samples"], str)


def test_finalize_includes_git_commit_when_run_inside_a_git_repo(tmp_path):
    tracker = RunTracker(tmp_path, "test_run")
    record_path = tracker.finalize()
    with open(record_path, encoding="utf-8") as f:
        record = json.load(f)
    # this test suite runs inside the project's own git repo, so a real
    # short commit hash (or None if truly detached/unborn) is expected --
    # either is valid, but the key must always be present
    assert "git_commit" in record


def test_git_commit_returns_none_when_git_executable_is_missing(monkeypatch):
    def fake_run(*args, **kwargs):
        raise FileNotFoundError("git not found")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert _git_commit() is None


def test_git_commit_returns_none_on_timeout(monkeypatch):
    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="git", timeout=5)

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert _git_commit() is None


def test_git_commit_returns_stripped_stdout_on_success(monkeypatch):
    class FakeResult:
        stdout = "abc1234\n"

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: FakeResult())
    assert _git_commit() == "abc1234"


def test_git_commit_returns_none_for_empty_stdout(monkeypatch):
    """An empty result (e.g. a repo with no commits yet) should be treated
    the same as "no commit available", not an empty string."""

    class FakeResult:
        stdout = ""

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: FakeResult())
    assert _git_commit() is None


def test_two_trackers_in_the_same_store_dir_do_not_clobber_each_others_records(tmp_path):
    """The real training pipeline creates three RunTrackers (baseline,
    classical, advanced) writing into the same models_store/ directory --
    their records must stay independent, keyed by run_name."""
    for run_name in ["model_a", "model_b"]:
        tracker = RunTracker(tmp_path, run_name)
        tracker.log_metrics(marker=run_name)
        tracker.finalize()

    with open(tmp_path / "model_a_run.json", encoding="utf-8") as f:
        record_a = json.load(f)
    with open(tmp_path / "model_b_run.json", encoding="utf-8") as f:
        record_b = json.load(f)

    assert record_a["metrics"]["marker"] == "model_a"
    assert record_b["metrics"]["marker"] == "model_b"
