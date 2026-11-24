"""Evaluation metrics: compare inferred and true populations."""

from __future__ import annotations

import numpy as np

from .grid import Grid
from .populations import Population


def match_populations(true: list[Population], est: list[Population]) -> list[tuple[int, int]]:
    """Resolve label switching: optimal assignment (Hungarian) on parameter distance."""
    raise NotImplementedError


def parameter_errors(true: list[Population], est: list[Population]) -> dict[str, np.ndarray]:
    """Relative errors of mean, rel_sigma and weight per matched population."""
    raise NotImplementedError


def marginal_wasserstein(true: list[Population], est: list[Population], grid: Grid) -> np.ndarray:
    """1D W1 distance between the true and estimated mixture marginals, per axis. Shape (3,), unit cells."""
    raise NotImplementedError


def k_correct(true: list[Population], est: list[Population]) -> bool:
    return len(true) == len(est)
