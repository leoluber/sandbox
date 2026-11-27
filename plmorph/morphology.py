"""Step 1: the morphology distribution P(x, y, z) the solver works on.

A ``Morphology`` is a weighted sum of K modes. Each mode is a discrete normal on the integer
lattice, with no correlation between the axes (diagonal covariance), so it is the outer product
of three 1D distributions:

    mode_k[i, j, l] = px[i] * py[j] * pz[l],   px(n) ∝ exp(-(n - mean_x)**2 / (2 sigma_x**2))

The normal is evaluated at the lattice points and renormalized (not integrated over unit cells).
A small sigma (e.g. 0.1 uc) puts a mode on a single ML; a mean between two integers with
sigma ~ 0.4 spreads it over both. There are no "kinds": cubes and plates are just different means
and sigmas. See solver.md in the project.
"""

from dataclasses import dataclass

import numpy as np

from .grid import Grid


@dataclass
class Morphology:
    means: np.ndarray           # (K, 3) mean edge lengths in unit cells
    sigmas: np.ndarray          # (K, 3) standard deviations in unit cells, > 0
    weights: np.ndarray = None  # (K,) mode weights; default equal; renormalized to sum 1

    def __post_init__(self):
        self.means = np.atleast_2d(np.asarray(self.means, dtype=float))
        self.sigmas = np.atleast_2d(np.asarray(self.sigmas, dtype=float))
        if self.weights is None:
            self.weights = np.ones(len(self.means))
        self.weights = np.asarray(self.weights, dtype=float) / np.sum(self.weights)

    @property
    def n_modes(self) -> int:
        return len(self.means)

    def mode(self, k: int, grid: Grid) -> np.ndarray:
        """Mode k on the grid, shape grid.shape, sums to 1 (weight not applied)."""
        n = grid.axis.astype(float)
        px, py, pz = (np.exp(-0.5 * ((n - m) / s) ** 2) for m, s in zip(self.means[k], self.sigmas[k]))
        return np.einsum("i,j,l->ijl", px / px.sum(), py / py.sum(), pz / pz.sum())

    def density(self, grid: Grid) -> np.ndarray:
        """P(x, y, z) = sum_k weight_k * mode_k, shape grid.shape, sums to 1."""
        return sum(w * self.mode(k, grid) for k, w in enumerate(self.weights))
