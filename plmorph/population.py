"""Step 1: a population and its morphology distribution P(x, y, z).

A population is a normal distribution of edge lengths with no correlation between the axes.
So P is the outer product of three 1D distributions, one per axis:

    P[i, j, k] = px[i] * py[j] * pz[k]

Two kinds:
- "isotropic": one mean shared by x, y, z (sigma may differ per axis).
- "npl": lateral means on x, y; thickness z is a single integer ML (pz is 1 at that ML, 0 elsewhere).
"""

from dataclasses import dataclass
from typing import Literal
import numpy as np
from scipy.special import ndtr
from .grid import Grid


@dataclass
class Population:
    kind: Literal["isotropic", "npl"]
    mean: np.ndarray       # (3,) mean edge lengths in unit cells; for "npl", mean[2] = ML
    rel_sigma: np.ndarray  # (3,) sigma / mean per axis; rel_sigma[2] unused for "npl"
    weight: float = 1.0    # share of this population in a mixture
    label: str = ""

    def __post_init__(self):
        self.mean = np.asarray(self.mean, dtype=float)
        self.rel_sigma = np.asarray(self.rel_sigma, dtype=float)
        if self.kind == "isotropic" and not np.all(self.mean == self.mean[0]):
            raise ValueError("isotropic: x, y, z must share one mean")
        if self.kind == "npl" and not float(self.mean[2]).is_integer():
            raise ValueError("npl: thickness mean[2] must be an integer ML")

    def marginals(self, grid: Grid) -> list[np.ndarray]:
        """The three 1D distributions [px, py, pz] on grid.axis, each sums to 1."""
        n = grid.axis.astype(float)
        out = []
        for axis in range(3):
            if self.kind == "npl" and axis == 2:
                p = (n == self.mean[2]).astype(float)
            else:
                mu = self.mean[axis]
                sigma = mu * self.rel_sigma[axis]
                p = np.exp(-0.5 * ((n - mu) / sigma) ** 2)
            out.append(p / p.sum())
        return out

    def density(self, grid: Grid) -> np.ndarray:
        """P(x, y, z), shape grid.shape, sums to 1."""
        px, py, pz = self.marginals(grid)
        return np.einsum("i,j,k->ijk", px, py, pz)

    def mass_outside(self, grid: Grid) -> float:
        """Fraction of the continuous normal distribution that falls outside the grid.

        This mass is cut off (``density`` renormalizes to 1); reported, not used.
        """
        lo, hi = grid.n_min - 0.5, grid.n_max + 0.5
        axes = [0, 1] if self.kind == "npl" else [0, 1, 2]
        inside = 1.0
        for axis in axes:
            mu = self.mean[axis]
            sigma = mu * self.rel_sigma[axis]
            inside *= ndtr((hi - mu) / sigma) - ndtr((lo - mu) / sigma)
        return 1.0 - inside
