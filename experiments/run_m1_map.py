"""Save the deterministic M1 energy-map surfaces used by figure F1."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root; no install needed

import numpy as np

from plmorph.grid import UNIT_CELL_NM
from plmorph.io import save_run
from plmorph.maps import ConfinementMapV1, E0_EV


def main() -> None:
    """Evaluate the v1 ruler and save grids and calibration points."""
    map_v1 = ConfinementMapV1()
    lateral_nm = np.geomspace(2.5, 35.0, 180)
    thickness_nm = np.geomspace(UNIT_CELL_NM, 35.0, 180)
    l_mesh, z_mesh = np.meshgrid(lateral_nm, thickness_nm, indexing="xy")
    delta_e = map_v1(l_mesh / UNIT_CELL_NM, l_mesh / UNIT_CELL_NM,
                     z_mesh / UNIT_CELL_NM) - E0_EV

    xy_nm = np.geomspace(2.5, 35.0, 180)
    x_mesh, y_mesh = np.meshgrid(xy_nm, xy_nm, indexing="ij")
    fixed_z_ml = 3.0
    xy_delta_e = map_v1(x_mesh / UNIT_CELL_NM, y_mesh / UNIT_CELL_NM,
                        np.full_like(x_mesh, fixed_z_ml)) - E0_EV

    plate_ml = np.arange(1.0, 5.0)
    plate_delta_e = np.array([0.672, 0.488, 0.306, 0.223])
    cube_nm = np.linspace(4.0, 20.0, 100)
    cube_data_delta_e = 0.033 * (13.4 / cube_nm) ** 1.45
    cube_model_delta_e = map_v1(cube_nm / UNIT_CELL_NM,
                                cube_nm / UNIT_CELL_NM,
                                cube_nm / UNIT_CELL_NM) - E0_EV

    save_run(
        Path("results/m1/map_ruler"),
        {
            "map": map_v1.name,
            "version": map_v1.version,
            "unit_cell_nm": UNIT_CELL_NM,
            "e0_ev": E0_EV,
            "fixed_z_ml": fixed_z_ml,
            "plate_thickness_ml": plate_ml.tolist(),
            "plate_delta_e_ev": plate_delta_e.tolist(),
            "cube_trend": "0.033 * (13.4 / L_nm)**1.45 eV",
        },
        {
            "lateral_nm": lateral_nm,
            "thickness_nm": thickness_nm,
            "delta_e_ev": delta_e,
            "xy_nm": xy_nm,
            "xy_delta_e_ev": xy_delta_e,
            "fixed_z_ml": np.array(fixed_z_ml),
            "plate_ml": plate_ml,
            "plate_delta_e_ev": plate_delta_e,
            "cube_nm": cube_nm,
            "cube_data_delta_e_ev": cube_data_delta_e,
            "cube_model_delta_e_ev": cube_model_delta_e,
        },
    )


if __name__ == "__main__":
    main()
