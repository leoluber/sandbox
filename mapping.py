"""Symmetric Kane two-band mapping for cuboid CsPbBr3 nanocrystals."""
import numpy as np

NAME = "M2 Kane two-band with effective length and weak long axes"
E0 = 2.375  # eV, fixed bulk transition energy
PARAMS = {
    "C": (4.65, 0.01, 20.0),
    "delta": (0.93, 0.0, 5.0),
    "w": (0.74, 0.0, 1.0),
}

def E(x, y, z, p):
    d1, d2, d3 = np.sort(np.array([x, y, z], dtype=float) * 0.59, axis=0)
    S = 1 / (d1 + p["delta"])**2 + p["w"] * (
        1 / (d2 + p["delta"])**2 + 1 / (d3 + p["delta"])**2
    )
    return np.sqrt(E0**2 + E0 * p["C"] * S)
