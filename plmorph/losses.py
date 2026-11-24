"""Loss functions between model and data energy distributions. See SPEC §7.

Each loss: ``loss(p_model, p_data, egrid) -> float`` and
``grad(p_model, p_data, egrid) -> (n_bins,)`` (derivative with respect to p_model).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from .grid import EnergyGrid


class Loss(Protocol):  # INTERFACE
    name: str

    def __call__(self, p_model: np.ndarray, p_data: np.ndarray, egrid: EnergyGrid) -> float: ...

    def grad(self, p_model: np.ndarray, p_data: np.ndarray, egrid: EnergyGrid) -> np.ndarray: ...


class Chi2:
    """Sum of squared residuals, optionally weighted by per-bin sigma."""
    name = "chi2"

    def __init__(self, sigma: np.ndarray | float = 1.0):
        self.sigma = sigma

    def __call__(self, p_model, p_data, egrid):
        raise NotImplementedError

    def grad(self, p_model, p_data, egrid):
        raise NotImplementedError


class KL:
    """KL(p_data || p_model) with epsilon smoothing for empty bins."""
    name = "kl"

    def __init__(self, eps: float = 1e-12):
        self.eps = eps

    def __call__(self, p_model, p_data, egrid):
        raise NotImplementedError

    def grad(self, p_model, p_data, egrid):
        raise NotImplementedError


class Wasserstein1:
    """W1 in 1D: integral of |CDF_model - CDF_data| dE (eV)."""
    name = "wasserstein1"

    def __call__(self, p_model, p_data, egrid):
        raise NotImplementedError

    def grad(self, p_model, p_data, egrid):
        raise NotImplementedError


class Cramer:
    """Integral of (CDF_model - CDF_data)^2 dE."""
    name = "cramer"

    def __call__(self, p_model, p_data, egrid):
        raise NotImplementedError

    def grad(self, p_model, p_data, egrid):
        raise NotImplementedError


class Moments:
    """Squared differences of mean, standard deviation and skewness of E (weighted)."""
    name = "moments"

    def __init__(self, weights: tuple[float, float, float] = (1.0, 1.0, 0.1)):
        self.weights = weights

    def __call__(self, p_model, p_data, egrid):
        raise NotImplementedError

    def grad(self, p_model, p_data, egrid):
        raise NotImplementedError


@dataclass
class CompositeLoss:  # INTERFACE
    """Weighted sum of losses: {loss: weight}."""
    terms: list[tuple[Loss, float]] = field(default_factory=list)
    name: str = "composite"

    def __call__(self, p_model, p_data, egrid) -> float:
        return float(sum(w * L(p_model, p_data, egrid) for L, w in self.terms))

    def grad(self, p_model, p_data, egrid) -> np.ndarray:
        return sum(w * L.grad(p_model, p_data, egrid) for L, w in self.terms)


LOSSES = {"chi2": Chi2, "kl": KL, "wasserstein1": Wasserstein1, "cramer": Cramer, "moments": Moments}
