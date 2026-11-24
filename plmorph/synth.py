"""Synthetic measurements with known ground truth. See SPEC §6."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .grid import EnergyGrid
from .populations import Population


@dataclass
class Measurement:  # INTERFACE
    p: np.ndarray                   # (n_bins,) measured density, sum(p) * de = 1
    egrid: EnergyGrid
    truth: list[Population]         # ground-truth populations
    meta: dict = field(default_factory=dict)  # corruption settings, seed, map name/version


@dataclass
class Corruption:
    n_particles: int | None = None  # multinomial sampling if set
    noise_sigma: float = 0.0        # additive Gaussian noise, relative to max(p)
    energy_shift_ev: float = 0.0
    seed: int = 0


def make_measurement(
    p_true: np.ndarray,
    truth: list[Population],
    egrid: EnergyGrid,
    corruption: Corruption | None = None,
) -> Measurement:  # INTERFACE
    """Apply the corruptions to a clean p(E) and return a Measurement.

    Map mismatch is done by the caller: generate p_true with a perturbed map, invert with the nominal one.
    """
    raise NotImplementedError
