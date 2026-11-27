import numpy as np
import pytest

from plmorph.examples import EXAMPLES
from plmorph.grid import Grid

GRID = Grid()


@pytest.mark.parametrize("key", sorted(EXAMPLES))
def test_density_normalized(key):
    pop = EXAMPLES[key]
    P = pop.density(GRID)
    assert P.shape == GRID.shape
    assert np.all(P >= 0)
    assert abs(P.sum() - 1) < 1e-10
    assert 0 <= pop.mass_outside(GRID) < 1


def test_npl_is_single_ml():
    P = EXAMPLES["npl_3ML_L20"].density(GRID)
    z_mass = P.sum(axis=(0, 1))
    assert z_mass.max() > 1 - 1e-12
    assert GRID.axis[np.argmax(z_mass)] == 3


def test_isotropic_marginals_match():
    px, py, pz = EXAMPLES["cube_10"].marginals(GRID)
    np.testing.assert_allclose(px, py, atol=1e-12)
    np.testing.assert_allclose(px, pz, atol=1e-12)


def test_mass_outside_grid_edge():
    assert EXAMPLES["cube_10"].mass_outside(GRID) < 1e-12
    assert EXAMPLES["edge_large"].mass_outside(GRID) > 0.01  # 45 uc, 5 %: +2 sigma beyond 50
