# Decisions

Format: `date — decision (status)`. Status: **confirmed** or *provisional* (open for change; confirm before relying on it).

## Confirmed
- 2026-11-24 — Phase 1 is synthetic only; the pipeline stops at the transition-energy distribution p(E). No GaLa kernel yet (Leo). **confirmed**
- 2026-11-24 — Population classes: isotropic and nanoplatelets only; no rods or wires (Leo). **confirmed**
- 2026-11-24 — rel_sigma = σ/μ bounded to [5 %, 20 %] per dimension, adjustable (Leo). **confirmed**
- 2026-11-24 — Every dimension is limited to 2–50 unit cells (Leo). **confirmed**
- 2026-11-24 — E0 = 2.375 eV, fixed, never fitted. **confirmed**
- 2026-11-24 — The energy map must be symmetric under permutations of (x, y, z); form f(d₁) + g(d₂, d₃). **confirmed**
- 2026-11-24 — Absorption is not used as an observable (not quantitative). **confirmed**
- 2026-11-24 — Visualization is a deliverable of every milestone; figures are regenerated from saved results. **confirmed**

## Provisional
- 2026-11-24 — Core numerics in NumPy/SciPy; JAX (CPU) added for autodiff solvers and HMC. *provisional: confirm stack (JAX vs PyTorch)*
- 2026-11-24 — Isotropic = one shared mean μ for x, y, z. *provisional*
- 2026-11-24 — NPL = one integer ML in z (σ_z ignored); 2D normal over (x, y). *provisional*
- 2026-11-24 — Only diagonal covariances in phase 1 (corr = identity). *provisional*
- 2026-11-24 — Energy grid 2.35–3.20 eV, 1 meV bins. *provisional*
- 2026-11-24 — E_map v1 = f(d₁) + g(d₂, d₃) with A = 1.329, d₀ = 0.955 nm, p = 1.457, b = 0.5 eV·nm² (fixed). Joint fit to plates 1–4 ML and the own cube trend (xyzPL analysis); free b gives 0.48, which confirms 0.5. Largest miss: 2 ML (−10 %). *provisional until the database calibration*
- 2026-11-24 — Figure style: Nature-family widths (89/120/183 mm), 7 pt. *provisional*
