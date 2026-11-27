"""Step 3: binning. Puts the weight of every grid point onto the energy grid.

Each grid point has an energy E from the energy map. It falls between two neighbouring
bin centres, ``lo`` and ``lo + 1``, and its weight is split between them by linear
interpolation: a share ``frac`` goes to ``lo + 1`` and ``1 - frac`` to ``lo``.
Grid points outside the energy window are dropped (``outside``).
"""

import numpy as np

from .grid import EnergyGrid


class Binning:
    def __init__(self, E: np.ndarray, egrid: EnergyGrid):
        """E: transition energies on the morphology grid (eV), shape (n, n, n)."""
        position = (E.ravel() - egrid.centers[0]) / egrid.de  # in units of bins
        self.lo = np.floor(position).astype(int)
        self.frac = position - self.lo
        self.outside = (self.lo < 0) | (self.lo >= egrid.n_bins - 1)
        self.n_bins = egrid.n_bins

    def apply(self, W: np.ndarray) -> np.ndarray:
        """Weights on the morphology grid, shape (n, n, n) -> weights per energy bin, (n_bins,)."""
        keep = ~self.outside
        w, lo, frac = W.ravel()[keep], self.lo[keep], self.frac[keep]
        to_lower = np.bincount(lo, weights=w * (1 - frac), minlength=self.n_bins)
        to_upper = np.bincount(lo + 1, weights=w * frac, minlength=self.n_bins)
        return to_lower + to_upper
