"""HarborConfig: path resolution. Untested until now -- config.py is small
but every pipeline depends on it resolving paths correctly, and it was the
site of a real bug (models_store was hard-coded in five places instead of
going through this module)."""

from __future__ import annotations

from pathlib import Path

from harbor.config import REPO_ROOT, HarborConfig, load_config
from tests.conftest import make_test_config


def test_load_config_reads_real_config_file():
    cfg = load_config()
    assert cfg.random_seed == 42
    assert cfg.section("data")["n_accounts"] > 0


def test_decision_intervention_tiers_have_required_keys():
    """Guards against configs/config.yaml drifting out of sync with
    InterventionTier's fields -- build_tiered_recommendations() unpacks each
    tier dict as InterventionTier(**tier), which fails with a confusing
    TypeError (not a clear config error) if a key is renamed or dropped."""
    cfg = load_config()
    tiers = cfg.section("decision")["intervention_tiers"]
    assert len(tiers) > 0
    for tier in tiers:
        assert set(tier.keys()) == {"name", "cost_per_account", "effect_multiplier"}


def test_repo_root_is_pinned_to_the_actual_project_directory():
    """REPO_ROOT is computed from `Path(__file__).resolve().parents[2]`,
    which is correct only when this file lives at <repo>/src/harbor/config.py
    (true under `pip install -e .`, false under a regular install -- see
    the fragility comment above REPO_ROOT's definition, and
    test_packaging.py for a real reproduction of the non-editable-install
    failure this pins against). This test at least catches the module ever
    silently resolving to the wrong place under the normal editable-install
    dev/CI setup every other test in this suite already depends on.
    """
    assert (REPO_ROOT / "pyproject.toml").exists()
    assert (REPO_ROOT / "configs" / "config.yaml").exists()
    assert (REPO_ROOT / "src" / "harbor" / "config.py").exists()


def test_models_dir_resolves_relative_to_repo_root():
    cfg = load_config()
    assert cfg.models_dir == REPO_ROOT / "models_store"


def test_models_dir_respects_absolute_override(tmp_path):
    cfg = HarborConfig(raw={"models_dir": str(tmp_path / "custom_models")})
    assert cfg.models_dir == tmp_path / "custom_models"


def test_models_dir_falls_back_to_default_when_unset():
    cfg = HarborConfig(raw={})
    assert cfg.models_dir == REPO_ROOT / "models_store"


def test_path_resolves_relative_to_repo_root():
    cfg = make_test_config()
    resolved = cfg.path(key="raw_dir")
    assert resolved == REPO_ROOT / "data" / "raw"
    assert isinstance(resolved, Path)


def test_path_respects_absolute_override(tmp_path):
    # A POSIX-style path like "/absolute/path" is deliberately NOT used here:
    # on Windows it has no drive letter, so pathlib's `.is_absolute()` is
    # False for it and `path()` would silently join it under REPO_ROOT's
    # drive instead of using it as an override -- a real platform quirk in
    # `path()`, harmless in practice since every config value in this
    # project is a relative path, but worth a test documenting it rather
    # than asserting behavior that doesn't hold across platforms.
    absolute_override = tmp_path / "custom_raw"
    cfg = HarborConfig(raw={"data": {"raw_dir": str(absolute_override)}})
    resolved = cfg.path(key="raw_dir")
    assert resolved == absolute_override
    assert resolved.is_absolute()
