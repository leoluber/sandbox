# Task: find a minimal effective mapping E(x, y, z)

Find the simplest closed-form function that maps the edge lengths of a CsPbBr3 cuboid nanocrystal (x, y, z in unit cells, 1 unit cell = 1 monolayer = 0.59 nm) to its PL transition energy in eV. It must fit the data and obey the rules below. This is a preliminary mapping, so prefer simple and physically sensible over a perfect fit.

## Python Environment

The project uses the Conda environment:

C:\Users\leolu\miniconda3\envs\research

IMPORTANT:
Always execute Python through `conda run -p` to ensure
the environment and its DLL dependencies are correctly loaded.

Example:

conda run -p "C:\Users\leolu\miniconda3\envs\research" python script.py

For package installation:

conda install -p "C:\Users\leolu\miniconda3\envs\research" PACKAGE

Never launch the environment's python.exe directly.
Never use system Python or the VS Code Python runner.

## Files

| File | Role | May you edit it? |
|---|---|---|
| `mapping.py` | the candidate: `NAME`, `PARAMS`, `E(x, y, z, p)` | **yes, this is the only file you edit** |
| `eval.py` | fits `PARAMS`, checks the rules, prints a report, appends to `results.tsv` | no |
| `confinement_data.csv` | data (`;`-separated, lengths in nm, notes column `no`) | no |
| `results.tsv` | log of every run, written by `eval.py` | no (read it) |

## Loop

1. Edit `mapping.py`: give the candidate a new, descriptive `NAME` and set `PARAMS` as `name: (initial, lower, upper)`.
2. Run `python eval.py`.
3. Read the report (fit, leave-one-out, AICc, rules, residuals per row), then go back to step 1.

## What counts as better

1. **Rules must pass** (`RULES: all pass`). A failing candidate is invalid, however well it fits.
2. **LOO-RMSE is the main score** (leave-one-out prediction error). The plain RMSE measures the fit, not how well the model predicts.
3. **Minimal:** add a parameter only if it lowers LOO-RMSE by about 2 meV or more *and* lowers AICc. Aim for 6 parameters or fewer. With similar scores, keep the simpler model.
4. Look at the `held` rows (excluded from the fit) as a sanity check. Don't tune the model to them.

## Rules (enforced in `eval.py`)

- **Symmetric:** E is unchanged when x, y and z are swapped, because a cuboid is the same object whatever its axis labels. Thin dimensions should therefore dominate, e.g. by sorting the edges or summing per-axis terms.
- **Monotonic:** E never increases as any edge gets longer.
- **Bulk level fixed:** E0 = 2.375 eV is the unconfined level. It has been tested extensively, so keep it as a constant and never make it a fitted parameter. E(2000, 2000, 2000) must lie within ±2 meV of it.
- **Finite** for all edges of 1 unit cell or more.
- **No memorising:** no lookup tables, interpolators, per-sample terms or special cases keyed to particular data rows.

## Data rules (`TAG_RULES` in `eval.py`, matched in the `no` notes column)

- Current data (cleaned 24 Nov 2026): cubes (3–100 nm) and 2–4 ML plates only. The 1 ML plate, wires, extrapolations and other materials have been removed. Column names are case-insensitive. `sigma_x_y` is optional and unused.
- Rows containing `issue`: held out. They aren't fitted, but their residuals are reported. (Tags `extrapolation`, `fake`, `PbS` and `Moreels` are still handled if they reappear.)
- All other rows (own data, `foreign` literature data, `agu`, `leoeval`): fitted with weight 1.
- Geometry: plates have z = thickness (n ML × 0.59 nm) and x, y lateral; cubes have x = y = z.

## What the data already show (ΔE = E − 2.375 eV, log-log)

1. **Cubes:** ΔE ∝ L^−1.7 over 3–20 nm. Own cubes give −1.45 and the literature rows −1.77 (−1.94 with the two points Li 12.1 nm and Rainò 14 nm). The cube exponent trades off strongly against E0 (+5 meV in E0 steepens it by about 0.1–0.15), which is why E0 is fixed.
2. **Noise floor:** the data deliberately span many syntheses and treatments. Cross-protocol scatter is about 17 % for cubes (roughly 25 meV at 5 nm, 8 meV at 10 nm) and about 10 meV for plates of one thickness. Don't chase errors below that. For the largest cubes (ΔE < 40 meV), judge errors in meV, not percent.
3. **Plates:** ΔE at large lateral size is about 488, 306 and 223 meV for 2, 3 and 4 ML (log-log slope about −1.1).
4. **Long edges are weak:** for plates, ΔE ≈ ΔE∞(z) + b/L² with b ≈ 1.0–1.1 eV·nm² for both lateral edges together. Per edge, that is about half of what the cubes imply.
5. **Not additive:** a plain sum of identical per-edge terms overestimates cubes relative to plates. The long edges must be weaker than the thinnest one, or the terms must combine sub-additively.

## Candidate models

Notation: edge lengths in nm, sorted so that d₁ ≤ d₂ ≤ d₃, and ΔE = E − E0 with E0 = 2.375 eV fixed. Useful constants: ħ²π²/m₀ = 0.752 eV·nm²; reduced exciton mass μ ≈ 0.13–0.18 m₀; total exciton mass M ≈ 4μ (if mₑ ≈ mₕ); exciton Bohr radius a_B ≈ 3.5 nm (diameter ≈ 7 nm); bulk exciton binding energy ≈ 33 meV. Fitted constants far from these values are a warning sign, not a success.

Quick tests on the current data (all pass the rules):

| Model | Params | LOO-RMSE | cube / plate LOO | AICc | Fitted values |
|---|---|---|---|---|---|
| M1 | 4 | 16.9 meV | 18.9 / 8.9 | −447.6 | d₀ → 0, p = 1.18, B = 0.86 |
| M2 | 3 | 17.2 meV | 17.5 / 16.6 | −446.7 | C = 4.65 (μ ≈ 0.16 m₀), δ = 0.93 nm, w = 0.74 |
| M2 with δ = 0 | 2 | 54.2 meV | 45 / 75 | −322.7 | an offset is needed |
| **M3** | **2** | **18.8 meV** | 20.3 / 13.6 | −436.6 | **μ = 0.127 m₀** (matches the measured value), δ = 1.18 nm |
| M4 | 4 | 15.6 meV | 17.3 / 9.1 | −455.6 | n = 1.42, p = 1.80, d₀ = 0.89 nm |

All of these are already at or near the noise floor. **The goal is now physical plausibility per parameter, not a lower error:** a 2-parameter model whose constants match the measured physics (M3) is worth more than a 4-parameter empirical one that is 3 meV better.

- **M1 – thin edge + weak long edges (empirical):** ΔE = A/(d₁ + d₀)^p + B·(d₂⁻² + d₃⁻²).
- **M2 – two-band (Kane) model, nonparabolic bands:** ΔE = √(E0² + E0·C·S) − E0, with S = 1/(d₁+δ)² + w·[1/(d₂+δ)² + 1/(d₃+δ)²]. C ≈ ħ²π²/μ; δ is an effective-length offset (ligand shell, wavefunction leaking out, sizing offset).
- **M3 – thin edge first, then the exciton moves as a whole:** the same two-band form with dᵢ → dᵢ + δ and confinement t = (ħ²π²/μ)/d₁² + (ħ²π²/m_L)·(1/d₂² + 1/d₃²), where m_L = μ + (M − μ)·a_B²/(a_B² + d₁²), M = 4μ and a_B = 3.5 nm fixed. In thin plates the long edges see the heavy exciton mass M; in cubes they approach μ. Free: μ and δ.
- **M4 – sub-additive combination:** εᵢ = A/(dᵢ + d₀)^p, then ΔE = (Σ εᵢⁿ)^(1/n), n ≥ 1.
- **M5 – M2 or M3 minus extra exciton binding in thin plates,** e.g. −E_b·(g − 1)·exp(−d₁/a_B). Only needed if thin-plate residuals stay systematic, which matters again once 1 ML data return.
- **M6 – weak-confinement exciton with a dead layer:** ΔE ≈ ħ²π²/(2M)·Σ 1/(dᵢ − η·a_B)², η ≈ 0.5–1. Only a check for large cubes; it diverges at small d.

Worth trying next: M3 with a_B or the M/μ ratio free (does a 3rd parameter pay off, and do the values stay physical?), and whether δ can shrink if the long-edge mass law changes.

## Sanity checks (report them; not scored)

The data no longer contain 1 ML plates, wires or sub-3 nm cubes, so the scores say nothing about those shapes. For the best 2–3 models, report predicted E for: a 1 ML plate (0.59 × 20 × 20 nm; measured ≈ 3.05 eV), a 3×3 ML wire 15 nm long, and cubes of 2 and 2.5 nm. Flag any model that predicts something unphysical there (E above about 3.5 eV for the 1 ML plate, or a wire below a 3 ML plate).

`eval.py` prints the error per family (cube / plate / wire). Report cube and plate errors separately.

## When done

Leave the best candidate in `mapping.py`. Reply with a short summary of the best model (formula, parameters, LOO-RMSE, AICc), the 2–3 runners-up from `results.tsv`, and the data rows that remain systematic outliers.

## Upgrading

To evaluate a new dataset with the same header: `python eval.py new_data.csv`. Change data roles in `TAG_RULES` and physics limits at the top of `eval.py`.
