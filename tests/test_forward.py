"""M2 acceptance tests (LOCKED; formerly M2): populations and forward operator."""

import numpy as np
import pytest

from plmorph.forward import BinningOperator, forward
from plmorph.grid import EnergyGrid, Grid
from plmorph.maps import get_brightness_map, get_energy_map
from plmorph.populations import (
    LIBRARY,
    REL_SIGMA_BOUNDS,
    density,
    from_params,
    mixture_density,
    to_params,
)

GRID, EGRID = Grid(), EnergyGrid()


@pytest.fixture(scope="module")
def T():
    return BinningOperator.build(GRID, EGRID, get_energy_map("toy"))


@pytest.mark.parametrize("key", sorted(LIBRARY))
def test_m2_density_normalized_and_valid(key):
    pop = LIBRARY[key]
    pop.validate(GRID)
    P, mass_outside = density(pop, GRID)
    assert P.shape == GRID.shape
    assert np.all(P >= 0)
    assert abs(P.sum() - 1) < 1e-10
    assert 0 <= mass_outside < 1


def test_m2_npl_is_single_ml():
    P, _ = density(LIBRARY["npl_3ML_L20"], GRID)
    z_mass = P.sum(axis=(0, 1))
    assert z_mass.max() > 1 - 1e-12  # all mass in one z slice
    assert GRID.axis[np.argmax(z_mass)] == 3


def test_m2_isotropic_marginals_match():
    P, _ = density(LIBRARY["cube_10"], GRID)
    mx, my, mz = P.sum(axis=(1, 2)), P.sum(axis=(0, 2)), P.sum(axis=(0, 1))
    np.testing.assert_allclose(mx, my, atol=1e-12)
    np.testing.assert_allclose(mx, mz, atol=1e-12)


def test_m2_params_roundtrip_and_bounds():
    pops = [LIBRARY["cube_10"], LIBRARY["npl_3ML_L20"]]
    pops[0].weight, pops[1].weight = 0.6, 0.4
    kinds = [p.kind for p in pops]
    back = from_params(to_params(pops, GRID), kinds, GRID)
    np.testing.assert_allclose(back[0].mean, pops[0].mean, rtol=1e-8)
    np.testing.assert_allclose(back[1].weight, 0.4, rtol=1e-8)
    rng = np.random.default_rng(1)
    for _ in range(50):
        theta = rng.normal(scale=10, size=to_params(pops, GRID).shape)
        for p in from_params(theta, kinds, GRID):
            p.validate(GRID)
            rs = p.rel_sigma[:2] if p.kind == "npl" else p.rel_sigma
            assert np.all(rs >= REL_SIGMA_BOUNDS[0]) and np.all(rs <= REL_SIGMA_BOUNDS[1])


def test_m2_mass_conservation(T):
    P = mixture_density([LIBRARY["cube_10"]], GRID)
    valid = T.valid.reshape(GRID.shape)
    assert abs(T.apply(P).sum() - P[valid].sum()) < 1e-10


def test_m2_apply_matches_direct_summation(T):
    rng = np.random.default_rng(2)
    W = rng.random(GRID.shape)
    p = T.apply(W)
    X, Y, Z = GRID.mesh()
    E = get_energy_map("toy")(X, Y, Z).ravel()
    w, c = W.ravel(), EGRID.centers
    p_ref = np.zeros(EGRID.n_bins)
    for Eg, wg in zip(E, w):
        j = int(np.floor((Eg - c[0]) / EGRID.de))
        if 0 <= j < EGRID.n_bins - 1:
            t = (Eg - c[j]) / EGRID.de
            p_ref[j] += wg * (1 - t)
            p_ref[j + 1] += wg * t
    np.testing.assert_allclose(p, p_ref, rtol=1e-9, atol=1e-12)


def test_m2_adjoint_identity(T):
    rng = np.random.default_rng(3)
    W, r = rng.random(GRID.shape), rng.normal(size=EGRID.n_bins)
    lhs = T.apply(W) @ r
    rhs = np.sum(W * T.adjoint(r))
    assert abs(lhs - rhs) < 1e-9 * max(1.0, abs(lhs))


def test_m2_forward_components_sum(T):
    pops = [LIBRARY["cube_10"], LIBRARY["npl_3ML_L20"]]
    pops[0].weight, pops[1].weight = 0.7, 0.3
    res = forward(pops, GRID, EGRID, T, get_brightness_map("unit"))
    assert res.components.shape == (2, EGRID.n_bins)
    np.testing.assert_allclose(res.components.sum(0), res.p, atol=1e-12)
    assert abs(res.p.sum() * EGRID.de - 1) < 1e-10
