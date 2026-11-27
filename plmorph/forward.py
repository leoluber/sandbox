"""Forward pass: populations -> transition-energy distribution p(E).

    for each population:  P = pop.density(grid)          step 1
                          component = weight * bin(P)    steps 2-3 (binning built from E(x, y, z))
    p = sum of components, normalized so that sum(p) * de = 1

No lineshape (GaLa) and no brightness yet: p(E) is the pure inhomogeneous distribution.
"""

from dataclasses import dataclass

import numpy as np

from .binning import Binning
from .grid import EnergyGrid, Grid
from .population import Population


@dataclass
class ForwardResult:
    p: np.ndarray           # (n_bins,) total, sum(p) * de = 1
    components: np.ndarray  # (K, n_bins) one row per population, rows sum to p


def forward(pops: list[Population], grid: Grid, egrid: EnergyGrid, binning: Binning) -> ForwardResult:
    components = np.array([pop.weight * binning.apply(pop.density(grid)) for pop in pops])
    components /= components.sum() * egrid.de
    return ForwardResult(p=components.sum(axis=0), components=components)
