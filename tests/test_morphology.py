import numpy as np
import pytest

from plmorph.examples import EXAMPLES
from plmorph.grid import Grid
from plmorph.morphology import Morphology

GRID = Grid(n_max=70)


@pytest.mark.parametrize("key", sorted(EXAMPLES))
def test_density_normalized(key):
    P = EXAMPLES[key].density(GRID)
    assert P.shape == GRID.shape
    assert np.all(P >= 0)
    assert abs(P.sum() - 1) < 1e-10


def test_small_sigma_is_single_ml():
    z_mass = EXAMPLES["3 ML, 15 nm"].density(GRID).sum(axis=(0, 1))
    assert z_mass.max() > 1 - 1e-12
    assert GRID.axis[np.argmax(z_mass)] == 3


def test_mode_between_two_ml():
    # lattice-point evaluation: mean 3.4, sigma 0.4 -> 65 / 35 on 3 / 4 ML
    z_mass = EXAMPLES["3/4 ML, 20 nm"].density(GRID).sum(axis=(0, 1))
    np.testing.assert_allclose(z_mass[GRID.axis == 3], 0.650, atol=0.002)
    np.testing.assert_allclose(z_mass[GRID.axis == 4], 0.348, atol=0.002)


def test_weights_and_sum_of_modes():
    m = Morphology(means=[[10, 10, 10], [30, 30, 3]], sigmas=[[1, 1, 1], [3, 3, 0.1]], weights=[3, 1])
    np.testing.assert_allclose(m.weights, [0.75, 0.25])
    np.testing.assert_allclose(m.density(GRID), 0.75 * m.mode(0, GRID) + 0.25 * m.mode(1, GRID))
