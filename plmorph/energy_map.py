"""Step 2: the energy map E(x, y, z).

E_map v1 (CsPbBr3 PL), from the xyzPL confinement analysis (24 Nov 2026).
Sort the edges so that d1 <= d2 <= d3 (in nm). Then

    E = E0 + A / (d1 + d0)**p  +  b * (1/d2**2 + 1/d3**2)
             thinnest edge         two long edges (weak)

A, d0 and p are fitted jointly to the plates (672 / 488 / 306 / 223 meV for 1-4 ML)
and the own cube trend 33 meV * (13.4 nm / L)**1.45. b is the plate lateral fit
(b left free in the joint fit gives 0.48). Plates come out at 705 / 440 / 308 / 232 meV;
2 ML is the largest miss (-10 %). Emission mapping: the Stokes shift is folded in.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EnergyMap:
    A: float = 1.329            # eV nm^p
    d0: float = 0.955           # nm
    p: float = 1.457
    b: float = 0.5              # eV nm^2 per long edge
    E0: float = 2.375           # eV, unconfined level (fixed)
    unit_cell_nm: float = 0.59  # CsPbBr3, 1 unit cell = 1 ML

    def __call__(self, x, y, z) -> np.ndarray:
        """Transition energy in eV. x, y, z: edge lengths in unit cells (any matching shapes)."""
        edges = np.sort(np.stack(np.broadcast_arrays(x, y, z)), axis=0) * self.unit_cell_nm
        d1, d2, d3 = edges
        return self.E0 + self.A / (d1 + self.d0) ** self.p + self.b * (1 / d2**2 + 1 / d3**2)
