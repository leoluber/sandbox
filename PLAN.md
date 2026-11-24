# PLmorph sandbox — plan and main ideas

*Draft for discussion, 24 Nov 2026. Phase 1: inverse problem on synthetic data only.*

## 1. Goal of this sandbox

Build and understand the inverse solver **PL spectrum → morphology distribution(s)** on synthetic data, where the ground truth is known exactly. Every step should be *visible*: the sandbox produces a small set of clean, paper-ready figures that show what the tensors look like and what the solver is doing. The same figures later become the method and SI figures of the paper.

Out of scope for phase 1: real data, calibrating the maps, TCSPC, forecasting.

## 2. Working principles

- **Swappable physics.** `E_map`, `wL_map` and `B_map` are plug-in functions on the morphology grid. The current mapping comes from the `morphPL-sandbox` work (see `xyzPL_confinement_mapping.md`) and is replaced by one line when the database-calibrated version exists.
- **Figures are deliverables, not side effects.** Each milestone is done only when its figures exist and look right.
- **Figures are regenerated from saved data.** Computation writes `.npz`/`.json` results; separate plotting scripts read them. Restyling a figure never reruns the solver.
- **Locked acceptance tests** (written by us, not by the agent) decide when a milestone is done. Codex may add its own unit tests.
- **Small, reviewable reports.** After each milestone Codex appends a short entry to `RESULTS.md` (≤30 lines) and updates the figure gallery.

## 3. Model in one paragraph

Morphology grid g = (x, y, z) in lattice planes (CsPbBr₃ first, 1 ML = 0.59 nm). Populations are discretized multivariate normals, `P = Σ_k A_k P_k(μ_k, Σ_k)`. Brightness weighting `W = P ⊙ B`. Optical kernel `K[g, E] = GaLa(E − E_map(g); σ₀, w_L(g))`, precomputed once per map version. Model spectrum `S = s · einsum('xyz,xyze->e', W, K)`. Loss = noise-weighted residual + TEM hyperprior + w_L constraint. Gradients flow back through the same contraction (`einsum('e,xyze->xyz', r, K)` is the residual back-projected onto morphology space).

Physics notes carried over from the mapping work:
- E0 = 2.375 eV fixed (unconfined level, not fitted).
- Confinement looks like `ΔE = f(d₁) + g(d₂, d₃)` with sorted edges d₁ ≤ d₂ ≤ d₃: one strongly confined thin edge plus weak long edges. So the map must be symmetric under axis swaps, which the v0 mapping is not.
- Realistic error floor of the mapping: ~17 % in ΔE for cubes. Synthetic tests should include a "map mismatch" case at about this level, not only perfect maps.

## 4. Milestones (each with its figures)

| # | Milestone | Done when | Figures |
|---|---|---|---|
| M0 | Figure style + gallery | style module, one demo figure, gallery builds | — |
| M1 | Grid + maps interface | maps swappable, axis-swap symmetry test passes | F1 |
| M2 | Populations + forward model | einsum matches direct summation | F2, F3 |
| M3 | Synthetic generator | spectra with noise, known truth saved | F3 (with truth) |
| M4 | Single-population MAP inversion | recovers μ, Σ within tolerance | F4, F5 |
| M5 | Multi-population + K selection | correct K on test set | F5, F6 |
| M6 | Uncertainties (Laplace, then HMC) | calibrated credible regions | F6, F8 |
| M7 | Resolvability sweep | phase diagram computed | F7 |
| M8 | Robustness: map mismatch, σ₀ error, noise | degradation quantified | F7 variants |

## 5. Figure system

### Style (one module, `plmorph/figstyle.py`)
- Matplotlib only, one `rcParams` setup used everywhere.
- **Sizes in journal widths:** single column 89 mm, 1.5 column 120 mm, double 183 mm (Nature family; ACS is similar). Fonts 7 pt body, 8 pt panel labels (bold lowercase **a**, **b**, …), sans-serif (Arial or Helvetica if installed, otherwise DejaVu Sans).
- Thin lines (0.6–0.8 pt), no top/right spines, no gridlines unless they carry meaning, ticks outward.
- **Fixed colour semantics across all figures:**
  - populations: a fixed qualitative palette (population 1, 2, 3 keep their colour everywhere);
  - densities and probabilities: one perceptually uniform sequential map (e.g. `cividis` or `viridis`);
  - signed quantities (residuals, gradients): one diverging map centred on zero (e.g. `RdBu_r`);
  - energy: one dedicated sequential map, so anything coloured by energy is recognisable at a glance;
  - truth vs fit: truth as a thin dark line or dashed outline, fit as a filled or solid coloured line.
- Exports: vector PDF (for the paper) plus PNG at 300 dpi (for the gallery).
- Energy axes in eV, with optional wavelength on a top axis.

### Gallery
`figures/GALLERY.md` lists every current figure with a one-line caption and the commit or date it was made. You open one file and see the current state of the project.

## 6. Figure catalogue (first ideas)

**F1 — The exciton ruler (map).** ΔE as a heat map over (lateral size L, thickness z) on log axes, with iso-energy contours. Inset or second panel: the same over (x, y) at fixed z, showing the iso-energy curves (the lateral degeneracy). Later, the calibration data points sit on top.

**F2 — Morphology space.** A corner plot of the 3D distribution: 1D marginals on the diagonal, 2D projections (x–y, x–z, y–z) below, each population in its colour. The lattice quantization is visible as discrete pixels or dots. This is the standard way to show a 3D distribution cleanly.

**F3 — The tensor pipeline (key method figure).** One row of panels, left to right:
- P (marginal projection),
- B (brightness map),
- W = P ⊙ B,
- K (the kernel as an image: grid points sorted by E_map on one axis, energy on the other, so it looks like a smeared diagonal band),
- S(E): the total spectrum with each population's contribution shaded underneath.

Small "⊙" and "×K" symbols between panels make it read like an equation.

**F4 — Backpropagation made visible.** The residual r(E) and its back-projection ∂L/∂W onto morphology space (diverging map), shown next to the current and true population. It shows why the gradient pushes the population where it does.

**F5 — Optimization trajectory.** Loss vs iteration; parameter traces (μ, σ per population); population ellipses moving through morphology space from initialization to the optimum, with the truth marked. A small animation (GIF or MP4) of the same for intuition and talks.

**F6 — Recovery.** True vs inferred distributions (marginals with credible bands), and the measured vs fitted spectrum with residuals below.

**F7 — Resolvability phase diagram.** Recovery error (or the probability of finding the correct K) as a function of population separation (in meV or ML) and amplitude ratio, with one panel per noise level. Contours mark the "resolvable" boundary.

**F8 — What PL can and cannot see.** Posterior width per dimension vs lateral size, compared with the prior width. This shows where the data dominates and where the TEM prior does (the Bohr-radius range). Shown together with the iso-energy degeneracy from F1.

## 7. Proposed folder layout

```
PLmorph-sandbox/
├── PLAN.md            # this file (discussion, then frozen into SPEC.md)
├── AGENTS.md          # rules for Codex
├── SPEC.md            # model + interfaces (written after we agree on PLAN)
├── TASKS.md           # milestones M0–M8 with acceptance criteria
├── RESULTS.md         # Codex log, one short entry per milestone
├── DECISIONS.md       # dated one-liners
├── plmorph/
│   ├── grid.py  maps.py  lineshape.py  populations.py
│   ├── forward.py  inverse.py  synth.py  metrics.py
│   └── figstyle.py
├── figures/
│   ├── scripts/       # one script per figure, reads results/
│   ├── out/           # pdf + png
│   └── GALLERY.md
├── results/           # saved npz/json from runs
├── tests/acceptance/  # locked
└── tests/unit/        # Codex's own
```

## 8. Open points to decide

1. **Stack:** JAX or PyTorch? This is a Windows machine: JAX runs on Windows CPU only (GPU needs WSL2), while PyTorch has native CUDA on Windows. For grids of about 10⁶–10⁷ kernel entries, CPU is probably enough in phase 1. Which do you prefer?
2. **Material and grid for phase 1:** CsPbBr₃ only? Grid ranges, e.g. x, y up to about 40 lattice planes (≈24 nm) and z from 1 to about 30?
3. **Shape classes in phase 1:** start with NPLs (integer z plus a 2D lateral distribution) and cubes, and add rods later?
4. **GaLa implementation:** will you drop your existing code into `legacy/` so it gets wrapped rather than rewritten?
5. **Mapping:** use the current best form from `morphPL-sandbox` (f(d₁) + g(d₂, d₃)) as `E_map` v1?
6. **Target journal style** for the figures (Nature family vs ACS) — this mostly sets widths and fonts.
7. **Animations:** are GIF/MP4 for F5 wanted from the start, or later?
