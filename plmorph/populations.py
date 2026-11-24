"""Population models on the morphology grid. See SPEC §5."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .grid import Grid

Kind = Literal["isotropic", "npl"]
REL_SIGMA_BOUNDS: tuple[float, float] = (0.05, 0.20)


@dataclass
class Population:  # INTERFACE
    """One morphology population.

    Attributes
    ----------
    kind : "isotropic" or "npl" (thickness on z, integer ML)
    mean : (3,) mean edge lengths in unit cells
    rel_sigma : (3,) sigma_i / mu_i, each in REL_SIGMA_BOUNDS (sigma_z ignored for npl)
    weight : mixture weight A_k (mixture weights sum to 1)
    corr : (3, 3) correlation matrix, or None for identity (phase-1 default)
    """

    kind: Kind
    mean: np.ndarray
    rel_sigma: np.ndarray
    weight: float = 1.0
    corr: np.ndarray | None = None
    label: str = ""

    def validate(self, grid: Grid) -> None:
        """Raise ValueError if bounds, kind rules or grid range are violated."""
        raise NotImplementedError


def density(pop: Population, grid: Grid) -> tuple[np.ndarray, float]:  # INTERFACE
    """Discretized multivariate normal on the integer lattice.

    Returns
    -------
    P : (nx, ny, nz) array, sums to 1 over the grid
    mass_outside : float, fraction of the continuous mass outside the grid (reported)
    """
    raise NotImplementedError


def mixture_density(pops: list[Population], grid: Grid) -> np.ndarray:  # INTERFACE
    """P = sum_k A_k P_k, shape (nx, ny, nz), sums to 1."""
    raise NotImplementedError


def to_params(pops: list[Population], grid: Grid) -> np.ndarray:  # INTERFACE
    """Map populations to one unconstrained real vector (scaled sigmoids, softmax)."""
    raise NotImplementedError


def from_params(theta: np.ndarray, kinds: list[Kind], grid: Grid) -> list[Population]:  # INTERFACE
    """Inverse of ``to_params``. Every output satisfies all bounds by construction."""
    raise NotImplementedError


def random_population(rng: np.random.Generator, kind: Kind, grid: Grid) -> Population:  # INTERFACE
    """Draw a random valid population (used for D3)."""
    raise NotImplementedError


def _pop(kind: Kind, mean, rel_sigma, label: str) -> Population:
    return Population(kind, np.asarray(mean, float), np.asarray(rel_sigma, float), label=label)


# D1 library: first proposal (PLAN.md §6). Thickness on z for NPLs.
LIBRARY: dict[str, Population] = {
    "cube_6": _pop("isotropic", [6, 6, 6], [0.10] * 3, "cube 6 uc"),
    "cube_10": _pop("isotropic", [10, 10, 10], [0.10] * 3, "cube 10 uc"),
    "cube_15": _pop("isotropic", [15, 15, 15], [0.10] * 3, "cube 15 uc"),
    "cube_25": _pop("isotropic", [25, 25, 25], [0.10] * 3, "cube 25 uc"),
    "npl_2ML_L30": _pop("npl", [30, 30, 2], [0.10, 0.10, 0.0], "2 ML, L 30 uc"),
    "npl_3ML_L20": _pop("npl", [20, 20, 3], [0.10, 0.10, 0.0], "3 ML, L 20 uc"),
    "npl_4ML_L40": _pop("npl", [40, 40, 4], [0.10, 0.10, 0.0], "4 ML, L 40 uc"),
    "npl_3ML_rect": _pop("npl", [15, 30, 3], [0.10, 0.10, 0.0], "3 ML, 15x30 uc"),
    "edge_narrow": _pop("isotropic", [10, 10, 10], [0.05] * 3, "cube 10 uc, 5 %"),
    "edge_broad": _pop("isotropic", [10, 10, 10], [0.20] * 3, "cube 10 uc, 20 %"),
    "edge_large": _pop("isotropic", [45, 45, 45], [0.05] * 3, "cube 45 uc (grid edge)"),
}
