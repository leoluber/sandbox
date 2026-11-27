# PLmorph sandbox

"The exciton as a ruler": turn a morphology distribution into a transition-energy distribution
p(E), and later invert it. Synthetic data only. This file is the spec; it is changed only by Leo
or Claude.

## Pipeline

```
Population ──► P(x,y,z) ──► E(x,y,z) ──► Binning ──► p(E)
 step 1         density      step 2       step 3     forward()
```

| File | Contents | Does |
|---|---|---|
| `plmorph/grid.py` | `Grid`, `EnergyGrid` | edge lengths 2–50 uc per axis; energy bins 2.35–3.20 eV, 1 meV |
| `plmorph/population.py` | `Population` | parameters → P on the grid (outer product of three 1D normals) |
| `plmorph/energy_map.py` | `EnergyMap` | E(x, y, z) in eV, E_map v1 |
| `plmorph/binning.py` | `Binning` | grid weights → energy bins (linear interpolation between two bins) |
| `plmorph/forward.py` | `forward()` | populations → p(E), plus one component per population |
| `plmorph/examples.py` | `EXAMPLES` | 11 test populations |

```python
grid, egrid = Grid(), EnergyGrid()
binning = Binning(EnergyMap()(*grid.mesh()), egrid)   # built once
res = forward([EXAMPLES["cube_10"]], grid, egrid, binning)   # res.p, res.components
```

## Model and decisions

- **Units:** edge lengths in unit cells (1 uc = 1 ML = 0.59 nm, CsPbBr₃; nm only inside the
  energy map), energies in eV.
- **Population:** `kind` "isotropic" (one mean shared by x, y, z) or "npl" (single integer ML on
  z, normal over x, y). Normal per axis, no correlation between axes, σ = rel_sigma × mean.
  P is renormalized on the grid; the cut-off mass is reported by `mass_outside`.
- **E_map v1:** sorted edges d₁ ≤ d₂ ≤ d₃ in nm, ΔE = A/(d₁+d₀)^p + b(1/d₂² + 1/d₃²);
  A = 1.329, d₀ = 0.955 nm, p = 1.457, b = 0.5 eV·nm², E0 = 2.375 eV (fixed).
  Plates 705 / 440 / 308 / 232 meV for 1–4 ML (data 672 / 488 / 306 / 223); largest miss 2 ML −10 %.
  Emission mapping, Stokes shift included.
- **Forward:** p = Σ_k weight_k · Binning(P_k), normalized to Σ p·dE = 1. Grid points whose energy
  falls outside the window are dropped (`Binning.outside`).
- **Not in yet:** lineshape (σ₀ / GaLa), brightness B (= 1), the solver.
- Absorption is not an observable.

## Running

```powershell
conda run -p "C:\Users\leolu\miniconda3\envs\research" python -m pytest
```
