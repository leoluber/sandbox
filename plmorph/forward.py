"""Forward pass: morphology -> transition-energy distribution p(E).

    for each mode k:  component_k = weight_k * bin(mode_k)    (binning built once from E(x, y, z))
    p = sum of components, normalized so that sum(p) * de = 1

No lineshape (GaLa) and no brightness yet: p(E) is the pure inhomogeneous distribution.
"""

from dataclasses import dataclass

import numpy as np

from .binning import Binning
from .grid import EnergyGrid, Grid
from .morphology import Morphology


@dataclass
class ForwardResult:
    p: np.ndarray           # (n_bins,) total, sum(p) * de = 1
    components: np.ndarray  # (K, n_bins) one row per mode, rows sum to p


def forward(morph: Morphology, grid: Grid, egrid: EnergyGrid, binning: Binning) -> ForwardResult:
    components = np.array([w * binning.apply(morph.mode(k, grid)) for k, w in enumerate(morph.weights)])
    components /= components.sum() * egrid.de
    return ForwardResult(p=components.sum(axis=0), components=components)
