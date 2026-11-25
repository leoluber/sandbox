# PLmorph sandbox

Phase 1 of "the exciton as a ruler": invert a transition-energy distribution **p(E)** into
morphology distribution(s) on synthetic data. This file is the single spec. It is changed only by
Leo or Claude; the history is in git.

## 1. Layout

```
plmorph/grid.py         morphology grid (2..50 uc per axis) and energy grid
plmorph/maps.py         energy maps (v1 default, toy for swap tests) and brightness map
plmorph/populations.py  Population, density, mixture, to_params/from_params, LIBRARY
plmorph/forward.py      sparse binning operator T (apply, adjoint) and forward()
tests/                  locked tests (maps, forward)
```

Plotting, run outputs and new modules are added per milestone, by name, in this file.

## 2. Scope and decisions

Confirmed:
- Synthetic data only. The pipeline stops at p(E); no GaLa kernel yet (later K = T ∗ GaLa, same interface).
- Populations: isotropic or nanoplatelets (NPL) only; no rods or wires. K ≤ 3.
- σ/μ in [5 %, 20 %] per dimension; every edge 2–50 unit cells (1 uc = 1 ML = 0.59 nm, CsPbBr₃).
- E0 = 2.375 eV, fixed. The energy map is symmetric under permutations of (x, y, z): ΔE = f(d₁) + g(d₂, d₃).
- Absorption is not an observable.

Provisional:
- Stack: NumPy/SciPy; JAX (CPU) vs PyTorch for autodiff is open.
- Isotropic = one shared mean on x, y, z (σ may differ per axis).
- NPL = one integer ML on z (σ_z ignored), 2D normal over (x, y).
- Diagonal covariance only (corr = identity).
- Energy grid 2.35–3.20 eV, 1 meV bins.

## 3. Model and interfaces

- **Units:** lengths in unit cells (nm only inside maps and plots); energies in eV; float64.
- **Grids:** `Grid` axes 2..50, tensors shape (49, 49, 49), `ij` indexing. `EnergyGrid` 851 bin centres.
- **E_map v1** (`ConfinementMapV1`): ΔE = A/(d₁+d₀)^p + b(1/d₂² + 1/d₃²), sorted edges in nm;
  A = 1.329, d₀ = 0.955 nm, p = 1.457, b = 0.5 eV·nm² (fixed). Joint fit to plates 1–4 ML
  (672 / 488 / 306 / 223 meV) and the cube trend 33 meV × (13.4 nm / L)^1.45; b free gives 0.48.
  Largest miss 2 ML (−10 %). Emission mapping, Stokes shift included.
- **Brightness:** B = 1 (`unit`).
- **Populations:** `Population(kind, mean (3,), rel_sigma (3,), weight, corr=None)`. Density = normal
  evaluated on the integer lattice, normalised to 1 on the grid; mass outside the grid is returned.
  Mixture P = Σ A_k P_k. `to_params`/`from_params` map to an unconstrained vector (scaled sigmoids, softmax).
- **Forward:** W = P ⊙ B → T (sparse: per grid point the lower bin and the upper-bin weight; linear
  interpolation) → p, normalised Σ p·dE = 1. `apply(W)`, `adjoint(r)`, ⟨TW, r⟩ = ⟨W, Tᵀr⟩.
  Grid points outside the energy window are counted (`n_out_of_range`). Never build T dense.

## 4. Milestones

- **M2 (done, needs numbers):** forward pass on `LIBRARY`.
- **M3:** invert the library populations from clean and noisy p(E).
- **M4:** random populations, reconstruction statistics.
- **M5:** losses, solvers, hyperparameters.
- **M6:** 2- and 3-population systems.
- Later: uncertainties (Laplace, HMC), resolvability sweep, map mismatch (~17 % in ΔE).

Figures (designed and plotted by Leo/Claude, not by the agent): F1 ruler map, F2 corner plots,
F3 tensor pipeline (key method figure), F4 residual back-projection, F5 trajectory, F6 recovery,
F7 resolvability, F8 posterior vs prior width, F9 solver comparison, F10 loss landscapes.

## 5. Running

```powershell
conda run -p "C:\Users\leolu\miniconda3\envs\research" python -m pytest
```
