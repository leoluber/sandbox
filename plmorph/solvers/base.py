"""Common solver interface. See SPEC §8.

Planned modules: gradient.py (Adam, L-BFGS), global_opt.py (CMA-ES, differential evolution,
basin hopping), linear.py (regularized NNLS on the grid), bayes.py (Laplace, HMC/NUTS),
sbi.py (later).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np

from ..grid import EnergyGrid, Grid
from ..losses import Loss
from ..maps import BrightnessMap, EnergyMap
from ..populations import Kind, Population
from ..synth import Measurement


@dataclass
class Problem:  # INTERFACE
    grid: Grid
    egrid: EnergyGrid
    e_map: EnergyMap
    b_map: BrightnessMap
    measurement: Measurement
    kinds: list[Kind]  # one entry per population; len(kinds) = K


@dataclass
class SolverResult:  # INTERFACE
    populations: list[Population]
    loss: float
    loss_history: np.ndarray                 # (n_iter,)
    param_history: np.ndarray                # (n_iter, n_params), unconstrained theta
    runtime_s: float
    n_forward_evals: int
    meta: dict = field(default_factory=dict)


class Solver(Protocol):  # INTERFACE
    name: str

    def solve(
        self,
        problem: Problem,
        loss: Loss,
        init: list[Population] | None,
        rng: np.random.Generator,
    ) -> SolverResult: ...


SOLVERS: dict[str, Callable[..., Solver]] = {}


def register_solver(name: str, factory: Callable[..., Solver]) -> None:
    SOLVERS[name] = factory


def get_solver(name: str, **kwargs) -> Solver:
    return SOLVERS[name](**kwargs)
