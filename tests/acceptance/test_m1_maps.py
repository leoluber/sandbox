"""M1 acceptance tests (LOCKED). Run against every registered energy map."""

import itertools

import numpy as np
import pytest

from plmorph.grid import EnergyGrid, Grid
from plmorph.maps import E0_EV, _ENERGY_MAPS, get_energy_map

MAPS = sorted(_ENERGY_MAPS)


@pytest.mark.parametrize("name", MAPS)
def test_m1_permutation_symmetry(name):
    e_map = get_energy_map(name)
    rng = np.random.default_rng(0)
    pts = rng.integers(2, 51, size=(200, 3))
    ref = e_map(pts[:, 0], pts[:, 1], pts[:, 2])
    for perm in itertools.permutations(range(3)):
        q = pts[:, perm]
        np.testing.assert_allclose(e_map(q[:, 0], q[:, 1], q[:, 2]), ref, rtol=0, atol=1e-12)


@pytest.mark.parametrize("name", MAPS)
def test_m1_above_unconfined_level_and_monotone(name):
    e_map = get_energy_map(name)
    X, Y, Z = Grid().mesh()
    E = e_map(X, Y, Z)
    assert np.all(E >= E0_EV)
    # increasing any one edge never increases the energy
    assert np.all(np.diff(E, axis=0) <= 1e-12)
    assert np.all(np.diff(E, axis=2) <= 1e-12)


@pytest.mark.parametrize("name", MAPS)
def test_m1_shapes_preserved(name):
    e_map = get_energy_map(name)
    X, Y, Z = Grid().mesh()
    assert e_map(X, Y, Z).shape == X.shape


@pytest.mark.parametrize("name", MAPS)
def test_m1_energy_grid_covers_map(name):
    e_map = get_energy_map(name)
    X, Y, Z = Grid().mesh()
    E = e_map(X, Y, Z)
    eg = EnergyGrid()
    frac_out = np.mean((E < eg.e_min) | (E > eg.e_max))
    assert frac_out < 0.01, f"{frac_out:.1%} of grid outside the energy grid"
