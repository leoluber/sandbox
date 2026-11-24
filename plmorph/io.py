"""Saving and loading experiment results. See SPEC §10."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def save_run(run_dir: str | Path, config: dict, arrays: dict[str, np.ndarray]) -> Path:  # INTERFACE
    """Write config.json and data.npz into run_dir (created if needed). Returns the path."""
    raise NotImplementedError


def load_run(run_dir: str | Path) -> tuple[dict, dict[str, np.ndarray]]:  # INTERFACE
    """Return (config, arrays)."""
    raise NotImplementedError
