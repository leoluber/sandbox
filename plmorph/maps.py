"""Physics maps on the morphology grid: transition energy and brightness. See SPEC §3.

Maps are swappable: callers only use the ``EnergyMap`` / ``BrightnessMap`` protocols
and the registries below.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np

from .grid import UNIT_CELL_NM

E0_EV: float = 2.375  # unconfined level, fixed (DECISIONS.md)


class EnergyMap(Protocol):  # INTERFACE
    name: str
    version: str

    def __call__(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Transition energy in eV, elementwise, same shape as the inputs.

        Inputs are edge lengths in unit cells. Must be symmetric under any
        permutation of (x, y, z), and E >= E0_EV.
        """
        ...


class BrightnessMap(Protocol):  # INTERFACE
    name: str
    version: str

    def __call__(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Relative brightness B >= 0, elementwise."""
        ...


def sorted_edges(x, y, z) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (d1, d2, d3) with d1 <= d2 <= d3, elementwise. Same shapes as the inputs."""
    s = np.sort(np.stack(np.broadcast_arrays(x, y, z)), axis=0)
    return s[0], s[1], s[2]


@dataclass(frozen=True)
class ToyConfinementMap:
    """Placeholder map: dE = f(d1) + g(d2, d3), with lengths in nm.

    f(d) = A / (d + d0)**p         strongly confined thinnest edge
    g    = c * (1/d2**2 + 1/d3**2) weak lateral (centre-of-mass-like) term

    Rough default parameters: p = 1 with A, d0 matched to the 1 ML and 4 ML plate
    values (672 and 223 meV); c = 0.5 eV nm^2 per edge (xyzPL_confinement_mapping.md).
    NOT a calibrated mapping. Replace with the morphPL-sandbox mapping (M1).
    """

    A: float = 0.591   # eV nm^p
    d0: float = 0.289  # nm
    p: float = 1.0
    c: float = 0.5     # eV nm^2
    name: str = "toy"
    version: str = "0.1"

    def __call__(self, x, y, z) -> np.ndarray:
        d1, d2, d3 = (d * UNIT_CELL_NM for d in sorted_edges(x, y, z))
        dE = self.A / (d1 + self.d0) ** self.p + self.c * (1.0 / d2**2 + 1.0 / d3**2)
        return E0_EV + dE


@dataclass(frozen=True)
class UnitBrightness:
    name: str = "unit"
    version: str = "1"

    def __call__(self, x, y, z) -> np.ndarray:
        return np.ones(np.broadcast(x, y, z).shape)


_ENERGY_MAPS: dict[str, Callable[[], EnergyMap]] = {"toy": ToyConfinementMap}
_BRIGHTNESS_MAPS: dict[str, Callable[[], BrightnessMap]] = {"unit": UnitBrightness}


def get_energy_map(name: str = "toy") -> EnergyMap:  # INTERFACE
    return _ENERGY_MAPS[name]()


def get_brightness_map(name: str = "unit") -> BrightnessMap:  # INTERFACE
    return _BRIGHTNESS_MAPS[name]()


def register_energy_map(name: str, factory: Callable[[], EnergyMap]) -> None:
    _ENERGY_MAPS[name] = factory


def register_brightness_map(name: str, factory: Callable[[], BrightnessMap]) -> None:
    _BRIGHTNESS_MAPS[name] = factory
