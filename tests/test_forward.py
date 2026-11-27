import numpy as np
import pytest

from plmorph.binning import Binning
from plmorph.energy_map import EnergyMap
from plmorph.examples import EXAMPLES
from plmorph.forward import forward
from plmorph.grid import EnergyGrid, Grid
from plmorph.morphology import Morphology

GRID, EGRID = Grid(), EnergyGrid()


@pytest.fixture(scope="module")
def E():
    return EnergyMap()(*GRID.mesh())


@pytest.fixture(scope="module")
def binning(E):
    return Binning(E, EGRID)


def test_binning_conserves_mass(binning):
    P = EXAMPLES["cube 10 nm"].density(GRID)
    inside = ~binning.outside.reshape(GRID.shape)
    assert abs(binning.apply(P).sum() - P[inside].sum()) < 1e-10


def test_binning_matches_loop(binning, E):
    W = np.random.default_rng(2).random(GRID.shape)
    c = EGRID.centers
    ref = np.zeros(EGRID.n_bins)
    for e, w in zip(E.ravel(), W.ravel()):
        j = int(np.floor((e - c[0]) / EGRID.de))
        if 0 <= j < EGRID.n_bins - 1:
            t = (e - c[j]) / EGRID.de
            ref[j] += w * (1 - t)
            ref[j + 1] += w * t
    np.testing.assert_allclose(binning.apply(W), ref, rtol=1e-9, atol=1e-12)


def test_forward_components_sum_to_p(binning):
    morph = Morphology(means=[[17, 17, 17], [34, 34, 3]], sigmas=[[1.7, 1.7, 1.7], [3.4, 3.4, 0.1]],
                       weights=[0.7, 0.3])
    res = forward(morph, GRID, EGRID, binning)
    assert res.components.shape == (2, EGRID.n_bins)
    np.testing.assert_allclose(res.components.sum(axis=0), res.p, atol=1e-12)
    assert abs(res.p.sum() * EGRID.de - 1) < 1e-10
