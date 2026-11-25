"""Population models on the morphology grid. See README §3."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy.special import expit, ndtr
from scipy.stats import multivariate_normal

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
        if self.kind not in ("isotropic", "npl"):
            raise ValueError(f"unknown population kind {self.kind!r}")
        mean = np.asarray(self.mean, dtype=np.float64)
        rel_sigma = np.asarray(self.rel_sigma, dtype=np.float64)
        if mean.shape != (3,) or rel_sigma.shape != (3,):
            raise ValueError("mean and rel_sigma must each have shape (3,)")
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(rel_sigma)):
            raise ValueError("mean and rel_sigma must be finite")
        if np.any((mean < grid.n_min) | (mean > grid.n_max)):
            raise ValueError("population means must lie within the grid")
        if self.kind == "isotropic" and not np.allclose(mean, mean[0], rtol=0, atol=1e-12):
            raise ValueError("isotropic populations must share one mean on all axes")
        if self.kind == "npl" and not float(mean[2]).is_integer():
            raise ValueError("NPL thickness mean must be an integer ML on z")
        active_sigma = rel_sigma if self.kind == "isotropic" else rel_sigma[:2]
        if np.any((active_sigma < REL_SIGMA_BOUNDS[0]) |
                  (active_sigma > REL_SIGMA_BOUNDS[1])):
            raise ValueError(f"active relative sigmas must lie in {REL_SIGMA_BOUNDS}")
        if not np.isfinite(self.weight) or not 0.0 <= self.weight <= 1.0:
            raise ValueError("population weight must be finite and lie in [0, 1]")
        if self.corr is not None:
            corr = np.asarray(self.corr, dtype=np.float64)
            if corr.shape != (3, 3) or not np.all(np.isfinite(corr)):
                raise ValueError("corr must be a finite 3x3 matrix")
            if not np.allclose(corr, corr.T, rtol=0, atol=1e-12):
                raise ValueError("corr must be symmetric")
            if not np.allclose(np.diag(corr), 1.0, rtol=0, atol=1e-12):
                raise ValueError("corr must have a unit diagonal")
            if np.linalg.eigvalsh(corr).min() <= 0:
                raise ValueError("corr must be positive definite")


def _corr_for(pop: Population, ndim: int) -> np.ndarray:
    """Return the active correlation block."""
    if pop.corr is None:
        return np.eye(ndim, dtype=np.float64)
    return np.asarray(pop.corr, dtype=np.float64)[:ndim, :ndim]


def _normal_pdf_lattice(
    coordinates: tuple[np.ndarray, ...], mean: np.ndarray, sigma: np.ndarray,
    corr: np.ndarray,
) -> np.ndarray:
    """Evaluate a multivariate normal density at lattice points."""
    covariance = sigma[:, None] * corr * sigma[None, :]
    chol = np.linalg.cholesky(covariance)
    points = np.stack([c.ravel() for c in coordinates])
    standardized = np.linalg.solve(chol, points - mean[:, None])
    quadratic = np.einsum("ij,ij->j", standardized, standardized)
    log_density = -0.5 * quadratic - np.log(np.diag(chol)).sum()
    log_density -= 0.5 * len(mean) * np.log(2.0 * np.pi)
    density = np.exp(log_density - np.max(log_density)).reshape(coordinates[0].shape)
    total = density.sum(dtype=np.float64)
    if not np.isfinite(total) or total <= 0:
        raise ValueError("population density cannot be normalized on this grid")
    return density / total


def _mass_inside_grid(mean: np.ndarray, sigma: np.ndarray, corr: np.ndarray,
                      grid: Grid) -> float:
    """Continuous Gaussian mass inside the grid's half-cell boundaries."""
    lower = np.full(mean.shape, grid.n_min - 0.5, dtype=np.float64)
    upper = np.full(mean.shape, grid.n_max + 0.5, dtype=np.float64)
    if np.allclose(corr, np.eye(mean.size), rtol=0, atol=1e-14):
        mass = np.prod(ndtr((upper - mean) / sigma) - ndtr((lower - mean) / sigma))
    else:
        covariance = sigma[:, None] * corr * sigma[None, :]
        mass = multivariate_normal.cdf(
            upper, mean=mean, cov=covariance, lower_limit=lower,
            maxpts=1_000_000, abseps=1e-8, releps=1e-8,
            rng=np.random.default_rng(0),
        )
    return float(np.clip(mass, 0.0, 1.0))


def density(pop: Population, grid: Grid) -> tuple[np.ndarray, float]:  # INTERFACE
    """Discretized multivariate normal on the integer lattice.

    Returns
    -------
    P : (nx, ny, nz) array, sums to 1 over the grid
    mass_outside : float, fraction of the continuous mass outside the grid (reported)
    """
    pop.validate(grid)
    axis = grid.axis.astype(np.float64)
    if pop.kind == "isotropic":
        coordinates = tuple(np.meshgrid(axis, axis, axis, indexing="ij"))
        mean = np.asarray(pop.mean, dtype=np.float64)
        sigma = mean * np.asarray(pop.rel_sigma, dtype=np.float64)
        corr = _corr_for(pop, 3)
        probability = _normal_pdf_lattice(coordinates, mean, sigma, corr)
    else:
        xx, yy = np.meshgrid(axis, axis, indexing="ij")
        mean = np.asarray(pop.mean[:2], dtype=np.float64)
        sigma = mean * np.asarray(pop.rel_sigma[:2], dtype=np.float64)
        corr = _corr_for(pop, 2)
        lateral_probability = _normal_pdf_lattice((xx, yy), mean, sigma, corr)
        probability = np.zeros(grid.shape, dtype=np.float64)
        z_index = int(pop.mean[2]) - grid.n_min
        probability[:, :, z_index] = lateral_probability
    mass_inside = _mass_inside_grid(mean, sigma, corr, grid)
    return probability, float(1.0 - mass_inside)


def mixture_density(pops: list[Population], grid: Grid) -> np.ndarray:  # INTERFACE
    """P = sum_k A_k P_k, shape (nx, ny, nz), sums to 1."""
    if not 1 <= len(pops) <= 3:
        raise ValueError("a mixture must contain between one and three populations")
    for pop in pops:
        pop.validate(grid)
    weights = np.asarray([pop.weight for pop in pops], dtype=np.float64)
    if len(pops) == 1:
        return density(pops[0], grid)[0]
    if not np.isclose(weights.sum(), 1.0, rtol=0, atol=1e-10):
        raise ValueError("mixture weights must sum to one")
    result = np.zeros(grid.shape, dtype=np.float64)
    for pop in pops:
        probability, _ = density(pop, grid)
        result += pop.weight * probability
    return result


def _to_logit(value: float, lower: float, upper: float) -> float:
    fraction = np.clip((value - lower) / (upper - lower), 1e-12, 1.0 - 1e-12)
    return float(np.log(fraction / (1.0 - fraction)))


def _from_logit(value: float, lower: float, upper: float) -> float:
    return float(lower + (upper - lower) * expit(value))


def to_params(pops: list[Population], grid: Grid) -> np.ndarray:  # INTERFACE
    """Map populations to one unconstrained real vector (scaled sigmoids, softmax)."""
    if not 1 <= len(pops) <= 3:
        raise ValueError("parameterization supports one to three populations")
    for pop in pops:
        pop.validate(grid)
    weights = np.asarray([pop.weight for pop in pops], dtype=np.float64)
    if np.any(weights <= 0) or not np.isclose(weights.sum(), 1.0, atol=1e-10):
        raise ValueError("parameterization requires positive weights summing to one")
    values: list[float] = []
    for pop in pops:
        if pop.kind == "isotropic":
            values.append(_to_logit(float(pop.mean[0]), grid.n_min, grid.n_max))
            values.extend(_to_logit(float(s), *REL_SIGMA_BOUNDS) for s in pop.rel_sigma)
        else:
            values.extend(_to_logit(float(m), grid.n_min, grid.n_max) for m in pop.mean)
            values.extend(_to_logit(float(s), *REL_SIGMA_BOUNDS) for s in pop.rel_sigma[:2])
    values.extend(np.log(weights).tolist())
    return np.asarray(values, dtype=np.float64)


def from_params(theta: np.ndarray, kinds: list[Kind], grid: Grid) -> list[Population]:  # INTERFACE
    """Inverse of ``to_params``. Every output satisfies all bounds by construction."""
    if not 1 <= len(kinds) <= 3:
        raise ValueError("parameterization supports one to three populations")
    theta = np.asarray(theta, dtype=np.float64)
    expected = sum(4 if kind == "isotropic" else 5 for kind in kinds) + len(kinds)
    if theta.shape != (expected,) or not np.all(np.isfinite(theta)):
        raise ValueError(f"theta must be a finite vector of shape ({expected},)")
    idx = 0
    populations: list[Population] = []
    for kind in kinds:
        if kind == "isotropic":
            mean_value = _from_logit(theta[idx], grid.n_min, grid.n_max)
            idx += 1
            sigmas = [_from_logit(v, *REL_SIGMA_BOUNDS) for v in theta[idx:idx + 3]]
            idx += 3
            populations.append(Population(kind, np.full(3, mean_value),
                                          np.asarray(sigmas)))
        elif kind == "npl":
            mean_xy = [_from_logit(v, grid.n_min, grid.n_max) for v in theta[idx:idx + 2]]
            mean_z_continuous = _from_logit(theta[idx + 2], grid.n_min, grid.n_max)
            mean_z = int(np.clip(np.rint(mean_z_continuous), grid.n_min, grid.n_max))
            idx += 3
            sigmas = [_from_logit(v, *REL_SIGMA_BOUNDS) for v in theta[idx:idx + 2]]
            idx += 2
            populations.append(Population(kind, np.array([*mean_xy, mean_z]),
                                          np.array([*sigmas, 0.0])))
        else:
            raise ValueError(f"unknown population kind {kind!r}")
    weight_logits = theta[idx:]
    shifted = weight_logits - np.max(weight_logits)
    weights = np.exp(shifted)
    weights /= weights.sum()
    for pop, weight in zip(populations, weights):
        pop.weight = float(weight)
    return populations


def random_population(rng: np.random.Generator, kind: Kind, grid: Grid) -> Population:  # INTERFACE
    """Draw a random valid population (used for D3)."""
    if kind not in ("isotropic", "npl"):
        raise ValueError(f"unknown population kind {kind!r}")
    active_mean = rng.uniform(grid.n_min, grid.n_max)
    rel_sigma = rng.uniform(*REL_SIGMA_BOUNDS, size=3)
    if kind == "isotropic":
        mean = np.full(3, active_mean, dtype=np.float64)
    else:
        mean = np.array([
            rng.uniform(grid.n_min, grid.n_max),
            rng.uniform(grid.n_min, grid.n_max),
            rng.integers(grid.n_min, grid.n_max + 1),
        ], dtype=np.float64)
        rel_sigma[2] = 0.0
    return Population(kind, mean, rel_sigma)


def _pop(kind: Kind, mean, rel_sigma, label: str) -> Population:
    return Population(kind, np.asarray(mean, float), np.asarray(rel_sigma, float), label=label)


# D1 library. Thickness on z for NPLs.
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
