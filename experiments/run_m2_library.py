"""Run the sparse forward pass for every M2 library population."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from plmorph.forward import BinningOperator, forward
from plmorph.grid import EnergyGrid, Grid
from plmorph.io import save_run
from plmorph.maps import get_brightness_map, get_energy_map
from plmorph.populations import LIBRARY, density


def main() -> None:
    """Save library morphologies, sparse bins, and all forward spectra."""
    grid, egrid = Grid(), EnergyGrid()
    energy_map = get_energy_map("v1")
    brightness_map = get_brightness_map("unit")
    operator = BinningOperator.build(grid, egrid, energy_map)
    names = list(LIBRARY)
    populations = [LIBRARY[name] for name in names]

    densities = np.empty((len(populations), *grid.shape), dtype=np.float64)
    spectra = np.empty((len(populations), egrid.n_bins), dtype=np.float64)
    masses_outside = np.empty(len(populations), dtype=np.float64)
    for index, population in enumerate(populations):
        densities[index], masses_outside[index] = density(population, grid)
        spectra[index] = forward(
            [population], grid, egrid, operator, brightness_map
        ).p

    x, y, z = grid.mesh()
    transition_energy = energy_map(x, y, z)
    brightness = brightness_map(x, y, z)
    selected_index = names.index("npl_3ML_L20")
    selected_probability = densities[selected_index]
    selected_weighted = selected_probability * brightness
    # Store only a handful of sparse rows to visualize T. The full operator
    # remains represented by its lower-bin and upper-weight vectors.
    valid_flat = np.flatnonzero(operator.valid)
    energies_flat = transition_energy.ravel()
    sorted_valid = valid_flat[np.argsort(energies_flat[valid_flat])]
    sample_positions = np.linspace(0, sorted_valid.size - 1, 8, dtype=int)
    sample_flat = sorted_valid[sample_positions]
    sample_xyz = np.column_stack(np.unravel_index(sample_flat, grid.shape)) + grid.n_min

    save_run(
        Path("results/m2/library_forward"),
        {
            "grid_n_min": grid.n_min,
            "grid_n_max": grid.n_max,
            "energy_e_min_ev": egrid.e_min,
            "energy_e_max_ev": egrid.e_max,
            "energy_de_ev": egrid.de,
            "energy_map": energy_map.name,
            "energy_map_version": energy_map.version,
            "brightness_map": brightness_map.name,
            "library_names": names,
            "sparse_operator_storage": ["lower", "w_upper", "valid"],
            "pipeline_population": "npl_3ML_L20",
        },
        {
            "names": np.asarray(names),
            "kinds": np.asarray([pop.kind for pop in populations]),
            "labels": np.asarray([pop.label for pop in populations]),
            "means_uc": np.stack([pop.mean for pop in populations]),
            "rel_sigma": np.stack([pop.rel_sigma for pop in populations]),
            "weights": np.asarray([pop.weight for pop in populations]),
            "mass_outside": masses_outside,
            "density": densities,
            "spectra": spectra,
            "energy_centers_ev": egrid.centers,
            "transition_energy_ev": transition_energy,
            "brightness": brightness,
            "pipeline_density": selected_probability,
            "pipeline_weighted_density": selected_weighted,
            "operator_lower": operator.lower,
            "operator_w_upper": operator.w_upper,
            "operator_valid": operator.valid,
            "sparse_sample_xyz_uc": sample_xyz,
            "sparse_sample_energy_ev": energies_flat[sample_flat],
            "sparse_sample_lower": operator.lower[sample_flat],
            "sparse_sample_w_upper": operator.w_upper[sample_flat],
        },
    )


if __name__ == "__main__":
    main()
