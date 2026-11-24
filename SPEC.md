# SPEC — phase 1 (synthetic, no optical kernel)

*Source of truth for interfaces. Changes only by agreement and with an entry in `DECISIONS.md`.*

## 1. Units and constants
- Lengths are in **unit cells** (CsPbBr₃: 1 unit cell = 1 ML = `UNIT_CELL_NM = 0.59` nm). Convert to nm only inside maps and plots.
- Energies are in **eV**. Unconfined level `E0_EV = 2.375` (fixed, never fitted).

## 2. Grids (`plmorph/grid.py`)
- `Grid`: integer axes `n_min..n_max` (default 2..50) for x, y and z; tensors have shape `(nx, ny, nz)` and index order `ij`.
- `EnergyGrid`: bin centres from `e_min` to `e_max` with width `de` (default 2.35–3.20 eV, 1 meV).

## 3. Maps (`plmorph/maps.py`)
- `EnergyMap(x, y, z) -> E` (eV), elementwise on arrays of any shape. It **must be symmetric under any permutation of (x, y, z)** and satisfy E ≥ E0.
- `BrightnessMap(x, y, z) -> B ≥ 0`. Default: `unit_brightness` (B = 1).
- Maps carry `name` and `version`. Get them through `get_energy_map(name)` / `get_brightness_map(name)`.
- v1 form: `ΔE = f(d₁) + g(d₂, d₃)` with sorted edges d₁ ≤ d₂ ≤ d₃. The `toy` map is a placeholder until the `morphPL-sandbox` mapping is copied into `legacy/` and wrapped.

## 4. Forward operator (`plmorph/forward.py`)
1. `W = P ⊙ B`, shape `(nx, ny, nz)`.
2. Push-forward with the binning operator T: each grid point g deposits `W_g` into the two energy bins next to `E_map(g)`, with linear-interpolation weights.
   `p_e = Σ_g W_g T[g, e]`.
3. Normalize: `Σ_e p_e · de = 1`.

T is stored **sparse** (per grid point: the lower bin index and the upper-bin weight). Dense (grid × energy) is about 10⁸ entries and is not allowed.
- `apply(W) -> p`, shape `(n_bins,)`
- `adjoint(r) -> ∂/∂W`, shape `(nx, ny, nz)`
- Must satisfy `<T W, r> = <W, Tᵀ r>` to machine precision.

Grid points whose energy lies outside the energy grid are counted and reported (`n_out_of_range`), not silently dropped.

**Later (GaLa layer):** T → K = T ∗ GaLa(σ₀, w_L). Same interface.

## 5. Populations (`plmorph/populations.py`)
- `Population(kind, mean, rel_sigma, weight, corr)`:
  - `kind ∈ {"isotropic", "npl"}`;
  - `mean`: shape (3,), in unit cells, within the grid range;
  - `rel_sigma = σᵢ/μᵢ`: shape (3,), each in **[0.05, 0.20]**;
  - `corr`: 3×3 correlation matrix, or `None` (identity; phase-1 default).
- Density: a multivariate normal evaluated at the integer lattice points and normalized to sum 1 on the grid. Mass outside the grid is reported.
- **Isotropic** (provisional): all `mean` components share one value μ, `rel_sigma` may differ per axis.
- **NPL** (provisional): z is a single integer ML (one-hot slice, σ_z ignored); a 2D normal distribution over (x, y). Convention: thickness on z.
- No rods or wires.
- Mixture: `P = Σ_k A_k P_k`, with Σ A_k = 1, K ≤ 3.
- **Parameterization for solvers:** `to_params` / `from_params` map a population to and from an unconstrained real vector. The bounds are enforced by construction: rel_sigma via a scaled sigmoid onto [0.05, 0.20], means via a scaled sigmoid onto [n_min, n_max], weights via softmax.

## 6. Synthetic measurements (`plmorph/synth.py`)
`Measurement(p, egrid, truth, meta)`. Corruptions, all optional and seeded:
- finite number of particles (multinomial sampling, `n_particles`);
- additive Gaussian noise (`noise_sigma`, relative to the peak);
- energy shift (`energy_shift_ev`);
- map mismatch (generate with a perturbed map, invert with the nominal one).

## 7. Losses (`plmorph/losses.py`)
Signature: `loss(p_model, p_data, egrid) -> float`. Each loss also provides a gradient with respect to `p_model` (analytic or via autodiff).
Implementations: `chi2`, `kl`, `wasserstein1` (CDF L1), `cramer` (CDF L2), `moments`. `CompositeLoss(terms={name: weight})` combines them.

## 8. Solvers (`plmorph/solvers/`)
- `Problem`: grid, egrid, e_map, b_map, measurement, n_populations, kinds.
- `Solver.solve(problem, loss, init, rng) -> SolverResult`.
- `SolverResult`: populations, final loss, loss history, parameter history (for trajectory figures), runtime in s, number of forward evaluations, metadata.
- Solver families live in separate modules and are registered by name.

## 9. Figure style (`plmorph/viz/style.py`)
- Widths: single 89 mm, 1.5 column 120 mm, double 183 mm. Body text 7 pt, panel labels 8 pt bold lowercase.
- Colours:
  - `POP_COLORS` (fixed per population index);
  - `CMAP_DENSITY` (sequential);
  - `CMAP_DIVERGING` (centred at 0);
  - `CMAP_ENERGY`.
  - Truth: dark dashed line. Fit: solid coloured line or fill.
- `savefig(fig, name)` writes `figures/out/<name>.pdf` and `.png` (300 dpi).

## 10. Results I/O (`plmorph/io.py`)
`save_run(dir, config: dict, arrays: dict)` writes `config.json` and `data.npz`. `load_run(dir)` returns both.
