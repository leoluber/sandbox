"""F1: v1 exciton ruler, read from the saved M1 map run."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plmorph.io import load_run
from plmorph.viz.style import CMAP_ENERGY, apply_style, figsize, panel_label, savefig

RUN_DIR = Path(__file__).resolve().parents[2] / "results" / "m1" / "map_ruler"


def main() -> None:
    """Plot lateral/thickness and fixed-thickness map sections."""
    apply_style()
    config, data = load_run(RUN_DIR)
    lateral = data["lateral_nm"]
    thickness = data["thickness_nm"]
    delta_e = data["delta_e_ev"] * 1000.0
    xy = data["xy_nm"]
    xy_delta_e = data["xy_delta_e_ev"] * 1000.0
    levels = np.linspace(0, 700, 15)

    fig, axes = plt.subplots(1, 2, figsize=figsize("double", 0.48), constrained_layout=True)
    ax = axes[0]
    filled = ax.contourf(lateral, thickness, delta_e, levels=levels, cmap=CMAP_ENERGY,
                         extend="max")
    contours = ax.contour(lateral, thickness, delta_e, levels=[25, 50, 100, 200, 400, 600],
                          colors="white", linewidths=0.55)
    ax.clabel(contours, fmt="%d meV", fontsize=6, inline=True)
    ax.set(xscale="log", yscale="log", xlabel="Lateral size L (nm)",
           ylabel="Thickness (nm)", title="v1 ruler")

    plate_ml = data["plate_ml"]
    plate_thickness_nm = plate_ml * config["unit_cell_nm"]
    # At the right edge, lateral confinement is negligible; the plate data are
    # shown at a common large lateral size to expose the thickness trend.
    ax.scatter(np.full_like(plate_ml, 30.0), plate_thickness_nm,
               c=data["plate_delta_e_ev"] * 1000, cmap=CMAP_ENERGY,
               vmin=0, vmax=700, edgecolors="black", linewidths=0.45,
               marker="o", s=18, label="Plate values (1–4 ML)")
    cube_nm = data["cube_nm"]
    ax.plot(cube_nm, cube_nm, color="white", linestyle="--", linewidth=1.0,
            label="Cube locus; marker colour = v1 energy")
    cube_points = np.linspace(0, cube_nm.size - 1, 10, dtype=int)
    ax.scatter(cube_nm[cube_points], cube_nm[cube_points],
               c=data["cube_data_delta_e_ev"][cube_points] * 1000,
               cmap=CMAP_ENERGY, vmin=0, vmax=700, edgecolors="black",
               linewidths=0.4, marker="D", s=15, label="Own cube trend")
    ax.legend(loc="upper right", fontsize=5.8, framealpha=0.85)
    fig.colorbar(filled, ax=ax, label="ΔE (meV)", fraction=0.046, pad=0.03)
    panel_label(ax, "a")

    ax = axes[1]
    xy_levels = np.linspace(300.0, 470.0, 16)
    image = ax.contourf(xy, xy, xy_delta_e.T, levels=xy_levels,
                        cmap=CMAP_ENERGY, extend="both")
    xy_contours = ax.contour(xy, xy, xy_delta_e.T, levels=8,
                             colors="white", linewidths=0.55)
    ax.clabel(xy_contours, fmt="%.0f", fontsize=6, inline=True)
    ax.set(xscale="log", yscale="log", xlabel="x (nm)", ylabel="y (nm)",
           title=f"Fixed thickness: {config['fixed_z_ml']:g} ML")
    fig.colorbar(image, ax=ax, label="ΔE (meV)", fraction=0.046, pad=0.03)
    panel_label(ax, "b")
    savefig(fig, "F1")


if __name__ == "__main__":
    main()
