"""Forward operator: morphology distribution -> transition-energy distribution. See SPEC §4.

Phase 1: sparse binning operator T (no optical kernel).
Later: K = T * GaLa, with the same interface.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .grid import EnergyGrid, Grid
from .maps import BrightnessMap, EnergyMap
from .populations import Population, density, mixture_density


@dataclass
class BinningOperator:  # INTERFACE
    """Sparse push-forward of grid weights onto energy bins (linear interpolation).

    For each grid point g (flattened, G = nx*ny*nz):
      lower[g]   : int, index of the lower neighbouring bin
      w_upper[g] : float in [0, 1], weight given to bin lower[g] + 1
      valid[g]   : bool, False if E_map(g) is outside the energy grid
    """

    grid_shape: tuple[int, int, int]
    n_bins: int
    lower: np.ndarray
    w_upper: np.ndarray
    valid: np.ndarray
    map_name: str = ""
    map_version: str = ""

    @classmethod
    def build(cls, grid: Grid, egrid: EnergyGrid, e_map: EnergyMap) -> "BinningOperator":
        x, y, z = grid.mesh()
        energy = np.asarray(e_map(x, y, z), dtype=np.float64)
        if energy.shape != grid.shape or not np.all(np.isfinite(energy)):
            raise ValueError("energy map must return finite values with the grid shape")
        position = (energy.ravel() - egrid.centers[0]) / egrid.de
        valid = (position >= 0.0) & (position <= egrid.n_bins - 1)
        lower = np.floor(position).astype(np.int64)
        # The final centre is represented by the final pair with upper weight 1.
        lower = np.clip(lower, 0, egrid.n_bins - 2)
        w_upper = np.clip(position - lower, 0.0, 1.0).astype(np.float64)
        w_upper[~valid] = 0.0
        return cls(
            grid_shape=grid.shape,
            n_bins=egrid.n_bins,
            lower=lower,
            w_upper=w_upper,
            valid=valid,
            map_name=getattr(e_map, "name", type(e_map).__name__),
            map_version=getattr(e_map, "version", ""),
        )

    @property
    def n_out_of_range(self) -> int:
        return int((~self.valid).sum())

    def apply(self, W: np.ndarray) -> np.ndarray:
        """(nx, ny, nz) -> (n_bins,). Unnormalized push-forward T W."""
        weights = np.asarray(W, dtype=np.float64)
        if weights.shape != self.grid_shape:
            raise ValueError(f"W must have shape {self.grid_shape}")
        valid = self.valid
        flat_weights = weights.ravel()[valid]
        lower = self.lower[valid]
        upper_weight = self.w_upper[valid]
        result = np.bincount(
            lower, weights=flat_weights * (1.0 - upper_weight), minlength=self.n_bins
        )
        result += np.bincount(
            lower + 1, weights=flat_weights * upper_weight, minlength=self.n_bins
        )
        return result.astype(np.float64, copy=False)

    def adjoint(self, r: np.ndarray) -> np.ndarray:
        """(n_bins,) -> (nx, ny, nz). T^T r; must satisfy <T W, r> = <W, T^T r>."""
        residual = np.asarray(r, dtype=np.float64)
        if residual.shape != (self.n_bins,):
            raise ValueError(f"r must have shape ({self.n_bins},)")
        result = np.zeros(int(np.prod(self.grid_shape)), dtype=np.float64)
        valid = self.valid
        lower = self.lower[valid]
        upper_weight = self.w_upper[valid]
        result[valid] = (1.0 - upper_weight) * residual[lower] + upper_weight * residual[lower + 1]
        return result.reshape(self.grid_shape)


@dataclass
class ForwardResult:
    p: np.ndarray             # (n_bins,) total density, sum(p) * de = 1
    components: np.ndarray    # (K, n_bins) per-population contributions (sum over K = p)
    W: np.ndarray             # (nx, ny, nz) brightness-weighted distribution
    n_out_of_range: int


def forward(
    pops: list[Population],
    grid: Grid,
    egrid: EnergyGrid,
    T: BinningOperator,
    b_map: BrightnessMap,
) -> ForwardResult:  # INTERFACE
    """P -> W = P * B -> p(E) = normalize(T W). Also returns per-population components."""
    if T.grid_shape != grid.shape or T.n_bins != egrid.n_bins:
        raise ValueError("binning operator does not match the supplied grids")
    if not pops:
        raise ValueError("at least one population is required")
    probability = mixture_density(pops, grid)
    x, y, z = grid.mesh()
    brightness = np.asarray(b_map(x, y, z), dtype=np.float64)
    if brightness.shape != grid.shape or not np.all(np.isfinite(brightness)):
        raise ValueError("brightness map must return finite values with the grid shape")
    if np.any(brightness < 0):
        raise ValueError("brightness map must be nonnegative")

    weighted_probability = probability * brightness
    raw_total = T.apply(weighted_probability)
    norm = raw_total.sum(dtype=np.float64) * egrid.de
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("forward spectrum has no positive in-range brightness")
    components = np.empty((len(pops), egrid.n_bins), dtype=np.float64)
    for index, pop in enumerate(pops):
        pop_probability, _ = density(pop, grid)
        components[index] = T.apply(pop.weight * pop_probability * brightness) / norm
    p = raw_total / norm
    return ForwardResult(
        p=p,
        components=components,
        W=weighted_probability,
        n_out_of_range=T.n_out_of_range,
    )
