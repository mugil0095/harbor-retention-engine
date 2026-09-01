"""Reproduces, as a real regression test, a bug found by manual review
rather than by any other test in this suite: `deployment/Dockerfile` used
to run `pip install .` (non-editable). A non-editable install copies
package files out of `src/` into site-packages, which breaks
`harbor.config.REPO_ROOT` (computed as `Path(__file__).resolve().parents[2]`,
correct only when this file stays at `<repo>/src/harbor/config.py`) --
`load_config()` then raises `FileNotFoundError` immediately, since it can
never find `configs/config.yaml`.

This was invisible to every other test because the whole suite runs against
this project's own editable dev install (`pip install -e ".[dev]"`), and
Docker was never actually built in this environment (no Docker available)
to catch it that way either. The fix is `deployment/Dockerfile` using
`pip install -e .` instead -- this test exists so that fix can't silently
regress back to a plain `pip install .` without a test failing.

Slower than the rest of the suite (builds and installs a real wheel), but
this is exactly the class of bug worth the extra runtime: it would have
broken the API and dashboard containers immediately on startup.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kwargs)


@pytest.fixture(scope="module")
def built_wheel(tmp_path_factory) -> Path:
    dist_dir = tmp_path_factory.mktemp("dist")
    _run([sys.executable, "-m", "build", "--wheel", "--outdir", str(dist_dir), str(REPO_ROOT)])
    wheels = list(dist_dir.glob("*.whl"))
    assert len(wheels) == 1, f"expected exactly one wheel, found {wheels}"
    return wheels[0]


@pytest.mark.slow
def test_non_editable_install_breaks_repo_root(built_wheel, tmp_path):
    """The failure mode this project's Dockerfile used to have. If this
    test ever starts failing because REPO_ROOT correctly resolves, that's
    good news (config.py got more robust) -- update the assertion, don't
    delete the test, since it means the underlying fragility was fixed at
    the source rather than worked around at the deployment layer."""
    install_dir = tmp_path / "installed"
    _run([
        sys.executable, "-m", "pip", "install", "--quiet", "--no-deps",
        "--target", str(install_dir), str(built_wheel),
    ])

    result = subprocess.run(
        [sys.executable, "-c", "import harbor.config as c; print(c.REPO_ROOT)"],
        cwd=str(install_dir),
        env={"PYTHONPATH": str(install_dir), "SYSTEMROOT": _systemroot()},
        capture_output=True, text=True, check=False,
    )
    resolved_repo_root = Path(result.stdout.strip())

    # the whole point: under a non-editable install, REPO_ROOT is NOT the
    # real project directory, so nothing config-derived can be found there
    assert resolved_repo_root != REPO_ROOT
    assert not (resolved_repo_root / "configs" / "config.yaml").exists()


@pytest.mark.slow
def test_non_editable_install_makes_load_config_fail_loudly(built_wheel, tmp_path):
    """Confirms the failure is a clear, immediate FileNotFoundError (as
    documented) rather than some other, more confusing failure mode."""
    install_dir = tmp_path / "installed"
    _run([
        sys.executable, "-m", "pip", "install", "--quiet", "--no-deps",
        "--target", str(install_dir), str(built_wheel),
    ])

    result = subprocess.run(
        [sys.executable, "-c", "import harbor.config as c; c.load_config()"],
        cwd=str(install_dir),
        env={"PYTHONPATH": str(install_dir), "SYSTEMROOT": _systemroot()},
        capture_output=True, text=True, check=False,
    )
    assert result.returncode != 0
    assert "FileNotFoundError" in result.stderr
    assert "Config file not found" in result.stderr


def _systemroot() -> str:
    """Windows' Python needs SYSTEMROOT on the child process's env for
    basic DLL loading (socket, etc.) -- passing a minimal `env=` dict
    without it makes the subprocess fail for unrelated reasons."""
    import os

    return os.environ.get("SYSTEMROOT", r"C:\Windows")
