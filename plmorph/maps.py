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
    """Toy map (test fixture only; use "v1" for physics): dE = f(d1) + g(d2, d3), with lengths in nm.

    f(d) = A / (d + d0)**p         strongly confined thinnest edge
    g    = c * (1/d2**2 + 1/d3**2) weak lateral (centre-of-mass-like) term

    Rough default parameters: p = 1 with A, d0 matched to the 1 ML and 4 ML plate
    values (672 and 223 meV); c = 0.5 eV nm^2 per edge (xyzPL_confinement_mapping.md).
    NOT calibrated. Kept as a simple second map for interface and swap tests.
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
class ConfinementMapV1:
    """E_map v1 (CsPbBr3 PL), from the xyzPL confinement analysis (24 Nov 2026).

    dE = f(d1) + g(d2, d3), with sorted edges d1 <= d2 <= d3 in nm:
      f(d) = A / (d + d0)**p        thinnest edge, bends at 1-2 ML
      g    = b * (1/d2**2 + 1/d3**2) weak lateral term (exciton centre-of-mass confinement)

    b = 0.5 eV nm^2 per edge is fixed (plate lateral fit). A, d0 and p are fitted jointly to:
      - plate dE at large lateral size: 672, 488, 306, 223 meV for 1-4 ML;
      - own cube trend: dE = 33 meV * (13.4 nm / L)**1.45 for L = 4-20 nm.
    With b left free, the same joint fit gives b = 0.48, which independently confirms the plate value.

    Fit quality: cubes within ~1 % of the own trend; plates 705 / 440 / 308 / 232 meV
    (+5 %, -10 %, +1 %, +4 %). 2 ML is the largest miss (the doc flags 1-2 ML as uncertain).
    Cube/plate ratio at equal thinnest edge is 2.6 / 2.0 / 1.8 / 1.6 for 2-5 ML
    (doc: 2.8 / 2.45 / 2.2 / 1.9 with extrapolated cubes): the same trend, slightly low.
    Emission (PL) mapping: the Stokes shift is folded in. Not valid for wires (excluded in phase 1).
    """

    A: float = 1.329   # eV nm^p
    d0: float = 0.955  # nm
    p: float = 1.457
    b: float = 0.5     # eV nm^2 per lateral edge
    name: str = "v1"
    version: str = "1.0 (2026-11-24)"

    def __call__(self, x, y, z) -> np.ndarray:
        d1, d2, d3 = (d * UNIT_CELL_NM for d in sorted_edges(x, y, z))
        dE = self.A / (d1 + self.d0) ** self.p + self.b * (1.0 / d2**2 + 1.0 / d3**2)
        return E0_EV + dE


@dataclass(frozen=True)
class UnitBrightness:
    name: str = "unit"
    version: str = "1"

    def __call__(self, x, y, z) -> np.ndarray:
        return np.ones(np.broadcast(x, y, z).shape)


_ENERGY_MAPS: dict[str, Callable[[], EnergyMap]] = {"v1": ConfinementMapV1, "toy": ToyConfinementMap}
_BRIGHTNESS_MAPS: dict[str, Callable[[], BrightnessMap]] = {"unit": UnitBrightness}


def get_energy_map(name: str = "v1") -> EnergyMap:  # INTERFACE
    return _ENERGY_MAPS[name]()


def get_brightness_map(name: str = "unit") -> BrightnessMap:  # INTERFACE
    return _BRIGHTNESS_MAPS[name]()


def register_energy_map(name: str, factory: Callable[[], EnergyMap]) -> None:
    _ENERGY_MAPS[name] = factory


def register_brightness_map(name: str, factory: Callable[[], BrightnessMap]) -> None:
    _BRIGHTNESS_MAPS[name] = factory
