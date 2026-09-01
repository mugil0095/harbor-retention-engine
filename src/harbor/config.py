"""Typed access to configs/config.yaml. No behavior is hard-coded in modules;
everything that could plausibly change between environments or experiments
lives in the YAML file and is loaded through this module."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
# This only resolves to the actual project root under an EDITABLE install
# (`pip install -e .`): editable installs keep this file at
# <repo>/src/harbor/config.py, so parents[2] is <repo>. A regular
# (non-editable) install copies package files into site-packages instead,
# so parents[2] lands somewhere inside the Python installation -- silently
# breaking every config-derived path (configs/, models_store/, data/).
# See deployment/Dockerfile's `-e` flag, added specifically because of this.
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "config.yaml"


@dataclass(frozen=True)
class HarborConfig:
    raw: dict[str, Any]

    @property
    def random_seed(self) -> int:
        return int(self.raw["random_seed"])

    def section(self, name: str) -> dict[str, Any]:
        return self.raw[name]

    def path(self, *, key: str, section: str = "data") -> Path:
        """Resolve a relative path from the config against the repo root."""
        rel = self.raw[section][key]
        p = Path(rel)
        return p if p.is_absolute() else REPO_ROOT / p

    @property
    def models_dir(self) -> Path:
        """Where trained model artifacts are read/written. A top-level key
        (not under `data`) since it holds pipeline outputs, not inputs --
        every pipeline script reads this instead of hard-coding
        `Path("models_store")`, which is the one path that used to bypass
        the config entirely."""
        rel = self.raw.get("models_dir", "models_store")
        p = Path(rel)
        return p if p.is_absolute() else REPO_ROOT / p


def load_config(path: str | Path | None = None) -> HarborConfig:
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return HarborConfig(raw=raw)
