"""Solver families. Each registers itself in ``base.SOLVERS`` by name."""

from .base import SOLVERS, Problem, Solver, SolverResult, get_solver, register_solver

__all__ = ["SOLVERS", "Problem", "Solver", "SolverResult", "get_solver", "register_solver"]
