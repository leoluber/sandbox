"""Single-mode example morphologies for CsPbBr3 (1 uc = 0.59 nm). Plate thickness on z.

Plates: own samples, lateral edges and lateral sigma from TEM (nm), PL peak measured.
Where no lateral sigma was measured, 13 % of the mean is assumed (median of the measured ones).
sigma_z = 0.1 uc puts a plate on a single ML. 3 ML rows are flagged "issue" in the confinement data.

Cubes / QDs (no own data here, reported sizes):
  - spheroidal QDs 3-13 nm, monodisperse: Akkerman et al., Science 2022
  - cubes 6.7 +- 0.7 to 15.2 +- 1.2 nm: Grisorio et al., Nanoscale Adv. 2021
"""

import numpy as np

from .morphology import Morphology

UC_NM = 0.59
ASSUMED_REL_SIGMA = 0.13


def plate(n_ml: int, a_nm: float, b_nm: float, sigma_nm: float | None = None) -> Morphology:
    """Single-ML plate with lateral edges a, b (nm) and lateral sigma (nm, same for both edges)."""
    if sigma_nm is None:
        sa, sb = ASSUMED_REL_SIGMA * a_nm, ASSUMED_REL_SIGMA * b_nm
    else:
        sa = sb = sigma_nm
    return Morphology(means=[[a_nm / UC_NM, b_nm / UC_NM, n_ml]],
                      sigmas=[[sa / UC_NM, sb / UC_NM, 0.1]])


def cube(edge_nm: float, sigma_nm: float) -> Morphology:
    return Morphology(means=[np.full(3, edge_nm / UC_NM)], sigmas=[np.full(3, sigma_nm / UC_NM)])


# name: (morphology, measured PL peak in eV or None)
EXAMPLES: dict[str, tuple[Morphology, float | None]] = {
    # cubes / QDs (literature sizes)
    "QD 4 nm":     (cube(4.0, 0.3), None),
    "QD 6 nm":     (cube(6.0, 0.35), None),
    "cube 6.7 nm": (cube(6.7, 0.7), None),
    "cube 10 nm":  (cube(10.0, 1.0), None),
    "cube 15 nm":  (cube(15.2, 1.2), None),
    # 3 ML plates (own samples)
    "jp142_48h":   (plate(3, 6.8, 7.5), 2.713),
    "jp201_1h":    (plate(3, 7.4, 8.2, 1.4), 2.713),
    "jp201_72h":   (plate(3, 8.1, 8.6, 1.4), 2.709),
    "jp82_24h":    (plate(3, 8.9, 10.5), 2.711),
    "jp137_5d":    (plate(3, 10.0, 9.0), 2.699),
    "jp169_96":    (plate(3, 10.0, 10.5), 2.703),
    "jp137_22d":   (plate(3, 12.0, 9.7), 2.699),
    "PL02P":       (plate(3, 16.0, 16.0, 3.9), 2.689),
    # 4 ML plates (own samples)
    "JP 65":       (plate(4, 4.7, 4.7), 2.650),
    "synth nina":  (plate(4, 6.0, 6.0, 0.7), 2.630),
    "synth joni":  (plate(4, 15.0, 8.8, 1.1), 2.610),
    "LL53":        (plate(4, 17.0, 11.7, 1.3), 2.603),
    "LL23":        (plate(4, 21.0, 10.5, 2.0), 2.603),
    "LL28":        (plate(4, 23.6, 13.2, 2.0), 2.601),
    # one mode over two thicknesses (illustration: 65 / 35 on 3 / 4 ML, lateral 10 nm)
    "3/4 ML, 10 nm": (Morphology(means=[[17, 17, 3.4]], sigmas=[[2.2, 2.2, 0.4]]), None),
}
