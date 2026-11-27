# PLmorph sandbox

"The exciton as a ruler": turn a morphology distribution into a transition-energy distribution
p(E), and later invert it. Synthetic data only. This file is the spec; it is changed only by Leo
or Claude.

## Pipeline

```
Morphology ──► P(x,y,z) ──► E(x,y,z) ──► Binning ──► p(E)
 step 1         density      step 2       step 3     forward()
```

| File | Contents | Does |
|---|---|---|
| `plmorph/grid.py` | `Grid`, `EnergyGrid` | edge lengths 2–50 uc per axis (default); energy bins 2.0–3.0 eV, 1 meV |
| `plmorph/morphology.py` | `Morphology` | K modes (means, sigmas, weights) → P on the grid |
| `plmorph/energy_map.py` | `EnergyMap` | E(x, y, z) in eV, E_map v1 |
| `plmorph/binning.py` | `Binning` | grid weights → energy bins (linear interpolation between two bins) |
| `plmorph/forward.py` | `forward()` | morphology → p(E), plus one component per mode |
| `plmorph/examples.py` | `EXAMPLES` | 18 single modes from reported CsPbBr₃ sizes |
| `proof_of_concept.py` | script | examples → p(E), PL, peak/FWHM, local identifiability → `out/proof_of_concept.png` |

```python
grid, egrid = Grid(), EnergyGrid()
binning = Binning(EnergyMap()(*grid.mesh()), egrid)   # built once
res = forward(EXAMPLES["cube 10 nm"], grid, egrid, binning)   # res.p, res.components
```

## Model and decisions

- **Units:** edge lengths in unit cells (1 uc = 1 ML = 0.59 nm, CsPbBr₃; nm only inside the
  energy map), energies in eV.
- **Morphology:** the one object the solver works on. A weighted sum of K modes; each mode is a
  discrete normal with diagonal covariance (means and sigmas in uc), evaluated at the lattice points
  and renormalized. No kinds: a plate is a mode with small σ on one axis (σ = 0.1 → single ML),
  a mean between two ML with σ ≈ 0.4 spreads one mode over both. Decisions: `solver.md` (project).
- **E_map v1:** sorted edges d₁ ≤ d₂ ≤ d₃ in nm, ΔE = A/(d₁+d₀)^p + b(1/d₂² + 1/d₃²);
  A = 1.329, d₀ = 0.955 nm, p = 1.457, b = 0.5 eV·nm², E0 = 2.375 eV (fixed).
  Plates 705 / 440 / 308 / 232 meV for 1–4 ML (data 672 / 488 / 306 / 223); largest miss 2 ML −10 %.
  Emission mapping, Stokes shift included.
- **Forward:** p = Σ_k weight_k · Binning(mode_k), normalized to Σ p·dE = 1. Grid points whose energy
  falls outside the window are dropped (`Binning.outside`).
- **Not in yet:** lineshape (σ₀ / GaLa), brightness B (= 1), the solver.
- Absorption is not an observable.

## Running

```powershell
conda run -p "C:\Users\leolu\miniconda3\envs\research" python -m pytest
```
