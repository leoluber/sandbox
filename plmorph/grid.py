"""Morphology and energy grids. See SPEC §1–2."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

UNIT_CELL_NM: float = 0.59  # CsPbBr3, 1 unit cell = 1 ML


@dataclass(frozen=True)
class Grid:  # INTERFACE
    """Cubic integer lattice of morphologies (x, y, z) in unit cells.

    Tensors on this grid have shape ``(n, n, n)`` with ``indexing="ij"``.
    """

    n_min: int = 2
    n_max: int = 50

    @property
    def axis(self) -> np.ndarray:
        """Integer edge lengths, shape (n,)."""
        return np.arange(self.n_min, self.n_max + 1)

    @property
    def shape(self) -> tuple[int, int, int]:
        n = self.axis.size
        return (n, n, n)

    def mesh(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return X, Y, Z, each of shape ``self.shape``, in unit cells."""
        a = self.axis
        return np.meshgrid(a, a, a, indexing="ij")


@dataclass(frozen=True)
class EnergyGrid:  # INTERFACE
    """Uniform energy bins (eV)."""

    e_min: float = 2.35
    e_max: float = 3.20
    de: float = 0.001

    @property
    def centers(self) -> np.ndarray:
        """Bin centres in eV, shape (n_bins,)."""
        n = int(round((self.e_max - self.e_min) / self.de)) + 1
        return self.e_min + self.de * np.arange(n)

    @property
    def n_bins(self) -> int:
        return self.centers.size
