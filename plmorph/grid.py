"""Step 0: the two grids everything lives on.

- ``Grid``: integer edge lengths (x, y, z) in unit cells. Arrays on it have shape (n, n, n).
- ``EnergyGrid``: energy bin centres in eV.
"""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Grid:
    """Morphology grid: every edge runs from ``n_min`` to ``n_max`` unit cells."""

    n_min: int = 2
    n_max: int = 50

    @property
    def axis(self) -> np.ndarray:
        """Edge lengths in unit cells, shape (n,)."""
        return np.arange(self.n_min, self.n_max + 1)

    @property
    def shape(self) -> tuple[int, int, int]:
        n = self.axis.size
        return (n, n, n)

    def mesh(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """X, Y, Z in unit cells, each of shape (n, n, n)."""
        return np.meshgrid(self.axis, self.axis, self.axis, indexing="ij")


@dataclass(frozen=True)
class EnergyGrid:
    """Uniform energy bins from ``e_min`` to ``e_max`` (eV), spacing ``de``."""

    e_min: float = 2.0
    e_max: float = 3.0
    de: float = 0.001

    @property
    def centers(self) -> np.ndarray:
        """Bin centres in eV, shape (n_bins,)."""
        n = int(round((self.e_max - self.e_min) / self.de)) + 1
        return self.e_min + self.de * np.arange(n)

    @property
    def n_bins(self) -> int:
        return self.centers.size
