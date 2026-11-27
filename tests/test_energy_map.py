import itertools

import numpy as np

from plmorph.energy_map import EnergyMap
from plmorph.grid import EnergyGrid, Grid

E_MAP = EnergyMap()


def test_symmetric_under_axis_swaps():
    pts = np.random.default_rng(0).integers(2, 51, size=(200, 3))
    ref = E_MAP(*pts.T)
    for perm in itertools.permutations(range(3)):
        np.testing.assert_allclose(E_MAP(*pts[:, perm].T), ref, atol=1e-12)


def test_above_E0_and_decreasing_with_size():
    E = E_MAP(*Grid().mesh())
    assert np.all(E >= E_MAP.E0)
    assert np.all(np.diff(E, axis=0) <= 1e-12)
    assert np.all(np.diff(E, axis=2) <= 1e-12)


def test_plate_values():
    # large lateral size: 705 / 440 / 308 / 232 meV for 1-4 ML (README)
    dE = E_MAP([1, 2, 3, 4], 1e6, 1e6) - E_MAP.E0
    np.testing.assert_allclose(dE * 1000, [705, 440, 308, 232], atol=1.5)


def test_energy_grid_covers_map():
    E = E_MAP(*Grid().mesh())
    eg = EnergyGrid()
    assert np.mean((E < eg.e_min) | (E > eg.e_max)) < 0.01
