"""Saving and loading experiment results. See SPEC §10."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def save_run(run_dir: str | Path, config: dict, arrays: dict[str, np.ndarray]) -> Path:  # INTERFACE
    """Write a JSON config and NumPy arrays into ``run_dir``.

    Parameters
    ----------
    run_dir : str or pathlib.Path
        Output directory for the run.
    config : dict
        JSON-serializable run metadata.
    arrays : dict[str, numpy.ndarray]
        Named arrays; object arrays are disallowed so runs load without pickle.

    Returns
    -------
    pathlib.Path
        The created run directory.
    """
    out = Path(run_dir)
    out.mkdir(parents=True, exist_ok=True)
    normalized: dict[str, np.ndarray] = {}
    for name, array in arrays.items():
        value = np.asarray(array)
        if value.dtype.hasobject:
            raise TypeError(f"array {name!r} has object dtype; use a non-object array")
        normalized[name] = value
    (out / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    np.savez_compressed(out / "data.npz", **normalized)
    return out


def load_run(run_dir: str | Path) -> tuple[dict, dict[str, np.ndarray]]:  # INTERFACE
    """Load ``config.json`` and ``data.npz`` from a run directory.

    Parameters
    ----------
    run_dir : str or pathlib.Path
        Directory containing the saved run.

    Returns
    -------
    tuple[dict, dict[str, numpy.ndarray]]
        Configuration and named arrays.
    """
    run = Path(run_dir)
    config: dict[str, Any] = json.loads((run / "config.json").read_text(encoding="utf-8"))
    with np.load(run / "data.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    return config, arrays
