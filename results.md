# Cuboid PL energy mapping: model comparison

Evaluated on `confinement_data.csv` with the unmodified `eval.py` harness, 55 fitted rows and 14 held `issue` rows. Every candidate below is symmetric in the three axes, and all listed candidates passed the evaluator's grid monotonicity, finiteness, and bulk-limit rules. Fit calls used the requested Conda environment. The starting mapping failed symmetry and is not considered valid.

## Selected candidate

Keep the three-parameter M2 Kane model in `mapping.py`. For sorted edge lengths in nm, `d1 <= d2 <= d3`, it is

```text
E = sqrt(E0^2 + E0*C*S)
S = 1/(d1 + delta)^2 + w * (1/(d2 + delta)^2 + 1/(d3 + delta)^2)
E0 = 2.375 eV
```

The fit gives `C = 4.648 eV nm^2`, `delta = 0.9297 nm`, and `w = 0.7431`. Interpreting `C ~= 0.752/mu` gives `mu ~= 0.162 m0`, consistent with the stated reduced-mass range. The offset is an effective length, and `w < 1` captures weaker confinement from the longer axes.

The model has RMSE 16.2 meV, LOO-RMSE 17.2 meV, AICc -446.7, and held-row RMSE 9.1 meV. Family LOO-RMSE is 17.5 meV for cubes and 16.6 meV for plates. There are no fitted wire rows.

## Models tested

| Candidate | Parameters / fitted values | LOO-RMSE | Cube / plate LOO | AICc | Assessment |
|---|---|---:|---:|---:|---|
| M2 Kane, weak long axes (selected) | 3: `C=4.648`, `delta=0.9297`, `w=0.7431` | 17.2 meV | 17.5 / 16.6 meV | -446.7 | Compact; effective mass is plausible. |
| M1 thin-axis power + long-axis inverse-square | 4: `A=0.5903`, `d0≈0`, `p=1.181`, `B=0.8626` | 16.9 meV | 18.9 / 8.9 meV | -447.6 | Only 0.3 meV better than M2; extra empirical parameter is not justified. |
| M4 sub-additive edge combination | 4: `A=1.824`, `d0=0.894`, `p=1.803`, `n=1.422` | 15.6 meV | 17.3 / 9.1 meV | -455.6 | Best simple empirical score, but only 1.6 meV better than M2 for one extra parameter; not enough under the stated ~2 meV rule. |
| M3 crossover, fitted mass ratio | 3: `mu=0.1703`, `delta=0.8543`, `M/mu=1.715` | 15.7 meV | 16.9 / 11.2 meV | -456.2 | Good fit, but the fitted mass ratio is substantially below the approximate physical value 4. |
| M3 crossover, fitted Bohr radius | 3: `mu=0.177`, `delta=0.7981`, `aB=1.199 nm` | 15.0 meV | not retained in short log | -460.6 | Lowest LOO score tested; radius is pinned near the 1 nm lower bound, far from the stated ~3.5 nm, so rejected on physical plausibility. |
| M2 Kane, equal axis weights | 2: `C=3.524`, `delta=0.6469` | 18.4 meV | 18.5 / 18.2 meV | -438.9 | Slightly simpler, but fails the useful weak-long-axis behavior and loses 1.2 meV versus selected M2. |
| M3 crossover, fixed `M/mu=4`, `aB=3.5 nm` | 2: `mu=0.1154`, `delta=1.289` | 21.5 meV | 23.0 / 16.2 meV | -421.7 | Physically constrained alternative, but poorer prediction on this dataset. |

The comparison follows the requested simplicity and physics criteria: M4's 1.6 meV gain over selected M2 does not clear the approximate 2 meV threshold. Freeing `aB` improves the score by 2.2 meV over M2 and lowers AICc, but produces an implausible boundary solution. The fitted M3 mass-ratio variant is practically tied with M4 but its mass ratio also conflicts with the stated estimate. M2 is therefore the best balance of prediction, complexity, and physical constants, not the numerical winner in isolation.

## Sanity predictions for selected M2

| Shape | Predicted energy |
|---|---:|
| 1 ML plate, `0.59 × 20 × 20 nm` | 3.234 eV |
| 3×3 ML wire, `1.77 × 1.77 × 15 nm` | 2.883 eV |
| 2 nm cube | 2.973 eV |
| 2.5 nm cube | 2.824 eV |

The 1 ML prediction is below the 3.5 eV flag threshold (though about 0.18 eV above the supplied ~3.05 eV measurement). The 3 ML plate at `1.77 × 20 × 20 nm` is predicted at 2.682 eV, below the wire prediction, so the wire sanity check passes.

## Remaining residual patterns

Residual is prediction minus observation. The largest selected-model residuals are mostly individual source offsets rather than a single geometry-wide failure:

- Underpredicted cubes: Cheng et al. at 3.9 nm (-38 meV); `agu leoeval` at 6.7 nm (-38 meV); Kline at 9.3 nm (-31 meV); and `agu 309 leoeval` at 9.9 nm (-30 meV).
- Overpredicted plates: Nina 3 ML at `5.3 × 5.3 × 1.77 nm` (+34 meV), and JP 65 2 ML at `4.7 × 4.7 × 2.36 nm` (+29 meV).
- Other notable cube deviations: the model overpredicts Akkerman sphere at 3.1 nm (+28 meV), Raino at 7 nm (+24 meV), Bodnarchuk at 5.5 nm (+21 meV), and Li at 12.1 nm (+19 meV); it underpredicts Liu at 4.9 nm (-23 meV) and Raino at 4.5 nm (-22 meV). The mixed signs point to source scatter rather than a smooth size-dependent bias.
- Largest held issue row: JP191 at `75 × 75 × 1.18 nm` is overpredicted by about 25 meV. Held rows are diagnostic only and were not fitted.

These large residuals are compatible with the stated cross-protocol scatter. They do not justify source-specific corrections or per-row terms. `results.tsv` contains the full per-run metrics; `mapping.py` contains the selected final candidate.
