"""Forward operator: morphology distribution -> transition-energy distribution. See SPEC §4.

Phase 1: sparse binning operator T (no optical kernel).
Later: K = T * GaLa, with the same interface.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .grid import EnergyGrid, Grid
from .maps import BrightnessMap, EnergyMap
from .populations import Population


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
        raise NotImplementedError

    @property
    def n_out_of_range(self) -> int:
        return int((~self.valid).sum())

    def apply(self, W: np.ndarray) -> np.ndarray:
        """(nx, ny, nz) -> (n_bins,). Unnormalized push-forward T W."""
        raise NotImplementedError

    def adjoint(self, r: np.ndarray) -> np.ndarray:
        """(n_bins,) -> (nx, ny, nz). T^T r; must satisfy <T W, r> = <W, T^T r>."""
        raise NotImplementedError


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
    raise NotImplementedError
