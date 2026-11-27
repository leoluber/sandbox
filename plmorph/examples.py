"""Single-mode example morphologies for CsPbBr3, based on reported sizes (1 uc = 0.59 nm).

Values in unit cells: (mean_x, mean_y, mean_z), (sigma_x, sigma_y, sigma_z). Plate thickness on z.
sigma_z = 0.1 puts a plate on a single ML. Sources (sizes and spreads, rounded):
  - spheroidal QDs 3-13 nm, monodisperse: Akkerman et al., Science 2022
  - cubes 4-15 nm: Protesescu et al., Nano Lett. 2015; 6.7 +- 0.7 to 15.2 +- 1.2 nm:
    Grisorio et al., Nanoscale Adv. 2021; typical spread 8-15 %, hot injection often > 15 %:
    Almeida et al., ACS Nano 2018
  - plates 2-6 ML: Bohn et al., Nano Lett. 2018; 3-5 ML: Akkerman et al., JACS 2016
  - cuboids with tunable aspect ratio: ACS Nano 2022, 16, 8318
Plate lateral sizes (10-30 nm, ~10 % spread) are typical values, not from one source.
"""

from .morphology import Morphology


def single(mean, sigma) -> Morphology:
    return Morphology(means=[mean], sigmas=[sigma])


EXAMPLES: dict[str, Morphology] = {
    # spheroidal QDs (narrow)
    "QD 4 nm":            single([7, 7, 7], [0.5, 0.5, 0.5]),
    "QD 6 nm":            single([10, 10, 10], [0.6, 0.6, 0.6]),
    "QD 9 nm":            single([15, 15, 15], [0.9, 0.9, 0.9]),
    # cubes
    "cube 6.7 nm":        single([11.4, 11.4, 11.4], [1.2, 1.2, 1.2]),
    "cube 10 nm":         single([17, 17, 17], [1.7, 1.7, 1.7]),
    "cube 15 nm":         single([25.8, 25.8, 25.8], [2.0, 2.0, 2.0]),
    "cube 10 nm, 18 %":   single([17, 17, 17], [3.0, 3.0, 3.0]),
    "cuboid 10x12x14 nm": single([17, 20, 24], [1.7, 2.0, 2.4]),
    # plates, single ML
    "2 ML, 15 nm":        single([25, 25, 2], [2.5, 2.5, 0.1]),
    "2 ML, 30 nm":        single([51, 51, 2], [5.1, 5.1, 0.1]),
    "3 ML, 15 nm":        single([25, 25, 3], [2.5, 2.5, 0.1]),
    "3 ML, 6 nm":         single([10, 10, 3], [1.0, 1.0, 0.1]),
    "3 ML, 10x25 nm":     single([17, 42, 3], [1.7, 4.2, 0.1]),
    "4 ML, 20 nm":        single([34, 34, 4], [3.4, 3.4, 0.1]),
    "5 ML, 20 nm":        single([34, 34, 5], [3.4, 3.4, 0.1]),
    "6 ML, 20 nm":        single([34, 34, 6], [3.4, 3.4, 0.1]),
    "8 ML, 15 nm":        single([25, 25, 8], [2.5, 2.5, 0.1]),
    # one mode over two thicknesses
    "3/4 ML, 20 nm":      single([34, 34, 3.4], [3.4, 3.4, 0.4]),
}
