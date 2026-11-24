# TASKS — milestones

Work in order. Each milestone: implement → acceptance tests pass → figures in gallery → RESULTS entry.
Acceptance tests for M3 and later are written together with the human before those milestones start.

## M0 — Figure style and gallery
- Implement `plmorph/viz/style.py` (finish the stubs) and `plmorph/io.py`.
- Make a demo figure with `figures/scripts/fig_template.py`.
- **Tests:** `tests/acceptance/test_m0_style.py`
- **Figures:** `demo` (can be deleted later)

## M1 — Grid and maps
- Finish `grid.py` and `maps.py`. The toy map must be symmetric under permutations and satisfy E ≥ E0.
- `E_map` v1 is implemented (`ConfinementMapV1`); the toy map is kept for swap tests. F1 shows v1 with the plate values and the own cube trend overlaid.
- **Tests:** `tests/acceptance/test_m1_maps.py`
- **Figures:** **F1** ruler: ΔE over (lateral size, thickness) on log axes with iso-energy contours; the (x, y) plane at fixed z.

## M2 — D1: population library and forward pass
- Implement `populations.py` (density, mixture, parameterization, `LIBRARY`) and `forward.py` (sparse T, apply, adjoint).
- Run the forward pass for every library population and save the results.
- **Tests:** `tests/acceptance/test_m2_forward.py`
- **Figures:** **F2** corner plots of the library populations; **F3** tensor pipeline P → ⊙B → W → T → p(E).

## M3 — D2: inversion of library populations
- Implement `synth.py`, `losses.py` (at least `chi2` and `wasserstein1`) and one gradient-based solver (L-BFGS with restarts).
- Invert every library population from a clean p(E) and from a noisy p(E).
- **Tests:** to be written (recovery tolerances)
- **Figures:** **F4** back-projection of the residual; **F5** optimization trajectory.

## M4 — D3: random populations
- `random_population(rng, kind)`; reconstruction statistics over ≥ 200 random truths per class.
- **Figures:** **F6** recovered vs true parameters; example fits.

## M5 — D4: losses, solvers, hyperparameters
- All losses in SPEC §7; solvers: Adam, L-BFGS, CMA-ES, differential evolution, NNLS on the grid with regularization.
- Hyperparameter optimization of loss weights and solver settings on a validation set of random truths.
- **Figures:** **F9** solver comparison; **F10** loss landscapes.

## M6 — D5: 2- and 3-population systems
- Mixtures; choice of K (BIC or evidence); handling of label switching.
- **Figures:** **F5** and **F6** for mixtures.

## M7 — Uncertainties and identifiability
- Laplace approximation, then HMC/NUTS. Posterior width vs prior width per dimension.
- **Figures:** **F8**

## M8 — Resolvability and robustness
- Sweep over separation × amplitude ratio × noise; map mismatch of about 17 % in ΔE.
- **Figures:** **F7**
