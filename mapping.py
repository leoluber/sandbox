"""Candidate E(x, y, z) mapping - the ONLY file the agent edits.
x, y, z: edge lengths in unit cells (1 unit cell = 0.59 nm); numpy arrays, non-integer allowed.
Returns the transition energy in eV. Free parameters in PARAMS are fitted by eval.py."""
import numpy as np

NAME = "v0 baseline: lateral a_eff = sqrt(xy) + separate thickness term z (E0 fixed)"

E0 = 2.375  # eV, unconfined level (fixed, validated experimentally)

PARAMS = {  # name: (initial, lower, upper)  or  name: initial
    "a_lat": (0.6271, 0.0, 10.0),
    "p_lat": (1.7425, 0.3, 3.0),
    "a_z": (0.5949, 0.0, 10.0),
    "p_z": (1.1228, 0.3, 3.0),
}


def E(x, y, z, p):
    nm = 0.59
    a_eff = np.sqrt(x * y) * nm
    return E0 + p["a_lat"] * a_eff ** -p["p_lat"] + p["a_z"] * (z * nm) ** -p["p_z"]
