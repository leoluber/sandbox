# Results log

One entry per milestone, newest at the bottom, at most 30 lines per entry.

## M0 - Figure style and gallery (2026-11-24)
Status: done
Acceptance tests: 3/3
Key numbers:
- Journal widths: 89, 120, and 183 mm; body text: 7 pt.
- `savefig` produced PDF and PNG outputs for the demo.
- Run I/O round-trip passed for JSON config and NumPy arrays.
Figures: demo - `figures/out/demo.pdf`, `figures/out/demo.png`
Open problems: none
Interface proposals: none

## M1 - Grid and maps (2026-11-24)
Status: done
Acceptance tests: 8/8
Key numbers:
- Default morphology grid: 2-50 unit cells per axis; default energy grid: 2.35-3.20 eV at 1 meV spacing.
- Both registered maps passed symmetry, E >= E0, monotonicity, shape, and energy coverage checks.
- F1 overlays plate values and the cube trend; the revised figure adds a calibration check.
Figures: F1 - `figures/out/F1.pdf`, `figures/out/F1.png`
Open problems: none
Interface proposals: none

## M2 - Population library and forward pass (2026-11-24)
Status: done
Acceptance tests: 18/18
Key numbers:
- Forward spectra saved for all 11 library populations on a 49 x 49 x 49 grid and 851 energy bins.
- Sparse T stores `lower`, `w_upper`, and `valid` vectors (117,649 grid entries each); no dense grid-by-energy array is built.
- The saved npl_3ML_L20 pipeline run has normalized p(E); F3 displays selected sparse T rows.
Figures: F2 - `figures/out/F2.pdf`, `figures/out/F2.png`; F3 - `figures/out/F3.pdf`, `figures/out/F3.png`
Open problems: none
Interface proposals: none
Suggested commit: M2: population library and sparse forward pass
