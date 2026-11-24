"""F1: v1 exciton ruler, read from the saved M1 map run."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

from plmorph.io import load_run
from plmorph.viz.style import CMAP_ENERGY, apply_style, figsize, panel_label, savefig

RUN_DIR = Path(__file__).resolve().parents[2] / "results" / "m1" / "map_ruler"
PLAIN_LOG_TICKS = np.array([2, 3, 5, 10, 20, 30], dtype=float)


def _plain_log_ticks(ax, axis: str) -> None:
    """Show selected log ticks as plain numbers."""
    locator = FixedLocator(PLAIN_LOG_TICKS)
    formatter = FuncFormatter(lambda value, _: f"{value:g}")
    if axis == "x":
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        ax.xaxis.set_minor_formatter(NullFormatter())
    else:
        ax.yaxis.set_major_locator(locator)
        ax.yaxis.set_major_formatter(formatter)
        ax.yaxis.set_minor_formatter(NullFormatter())


def main() -> None:
    """Plot the ruler, fixed-thickness section, and calibration checks."""
    apply_style()
    config, data = load_run(RUN_DIR)
    lateral_nm = data["lateral_nm"]
    thickness_nm = data["thickness_nm"]
    delta_e_mev = data["delta_e_ev"] * 1000.0
    xy_nm = data["xy_nm"]
    xy_delta_e_mev = data["xy_delta_e_ev"].T * 1000.0
    cell_nm = float(config["unit_cell_nm"])

    fig = plt.figure(figsize=figsize("double", 1.08), constrained_layout=True)
    outer = fig.add_gridspec(2, 3, height_ratios=(1.0, 0.76),
                             width_ratios=(1.25, 0.62, 1.25))
    ax_a = fig.add_subplot(outer[0, 0])
    legend_ax = fig.add_subplot(outer[0, 1])
    ax_b = fig.add_subplot(outer[0, 2])
    bottom = outer[1, :].subgridspec(1, 2, wspace=0.28)
    ax_c_plate = fig.add_subplot(bottom[0, 0])
    ax_c_cube = fig.add_subplot(bottom[0, 1])

    # Thickness above L is redundant by permutation symmetry; shade the
    # excluded half and also mask it from the contour calculation.
    ax_a.fill_between(lateral_nm, lateral_nm, thickness_nm[-1],
                      color="0.88", zorder=0)
    valid = thickness_nm[:, None] <= lateral_nm[None, :]
    masked_energy = np.ma.masked_where(~valid, delta_e_mev)
    levels_a = np.arange(0.0, 701.0, 100.0)
    image_a = ax_a.contourf(lateral_nm, thickness_nm, masked_energy,
                            levels=levels_a, cmap=CMAP_ENERGY, extend="max")
    contours_a = ax_a.contour(lateral_nm, thickness_nm, masked_energy,
                              levels=np.arange(50.0, 701.0, 100.0),
                              colors="white", linewidths=0.55)
    ax_a.clabel(contours_a, fmt="%d", fontsize=5.5, inline=True)
    ax_a.plot(lateral_nm, lateral_nm, color="0.15", linestyle="--",
              linewidth=0.65, zorder=3)
    ax_a.set(xscale="log", yscale="log", xlim=(lateral_nm[0], lateral_nm[-1]),
             ylim=(thickness_nm[0], thickness_nm[-1]), xlabel="Lateral size L (nm)",
             ylabel="Thickness (nm)", title="v1 ruler")
    _plain_log_ticks(ax_a, "x")
    _plain_log_ticks(ax_a, "y")
    ax_a.set_xticks(PLAIN_LOG_TICKS, minor=False)
    ax_a.set_yticks(PLAIN_LOG_TICKS, minor=False)
    ml_axis = ax_a.secondary_yaxis(
        "right", functions=(lambda nm: nm / cell_nm, lambda ml: ml * cell_nm)
    )
    ml_axis.set_yscale("log")
    ml_ticks = np.array([1, 2, 3, 4, 6, 10, 20, 30, 50], dtype=float)
    ml_axis.yaxis.set_major_locator(FixedLocator(ml_ticks))
    ml_axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ml_axis.set_ylabel("Thickness (ML)")
    fig.colorbar(image_a, ax=ax_a, label="ΔE (meV)", ticks=levels_a,
                 fraction=0.046, pad=0.04)
    panel_label(ax_a, "a")

    image_b = ax_b.contourf(xy_nm, xy_nm, xy_delta_e_mev,
                            levels=np.arange(300.0, 461.0, 20.0),
                            cmap=CMAP_ENERGY, extend="both")
    contours_b = ax_b.contour(xy_nm, xy_nm, xy_delta_e_mev,
                              levels=np.arange(300.0, 461.0, 20.0),
                              colors="white", linewidths=0.55)
    ax_b.clabel(contours_b, fmt="%d", fontsize=5.5, inline=True)
    ax_b.set(xscale="log", yscale="log", xlabel="x (nm)", ylabel="y (nm)",
             title=f"Fixed thickness: {config['fixed_z_ml']:g} ML")
    _plain_log_ticks(ax_b, "x")
    _plain_log_ticks(ax_b, "y")
    fig.colorbar(image_b, ax=ax_b, label="ΔE (meV)",
                 ticks=np.arange(300.0, 461.0, 20.0), fraction=0.046, pad=0.04)
    panel_label(ax_b, "b")

    handles = [
        Line2D([], [], marker="o", linestyle="none", markerfacecolor="black",
               markeredgecolor="black", markersize=4, label="Plate data"),
        Line2D([], [], color="#1b7a8c", linewidth=1.0, label="v1 at L = 30 nm"),
        Line2D([], [], color="0.15", linestyle="--", linewidth=1.0,
               label="Own cube trend"),
        Line2D([], [], color="#d1603d", linewidth=1.0, label="v1 cube"),
    ]
    legend_ax.axis("off")
    legend_ax.legend(handles=handles, loc="center", frameon=False,
                     fontsize=5.5, handlelength=1.5, labelspacing=0.8)

    plate_thickness_ml = data["plate_ml"]
    plate_data_mev = data["plate_delta_e_ev"] * 1000.0
    # Interpolate the saved v1 surface at exactly L = 30 nm, staying within
    # the saved arrays and keeping all map evaluation out of this plot script.
    at_30_nm = np.array([
        np.interp(30.0, lateral_nm, row) for row in delta_e_mev
    ])
    thickness_ml = thickness_nm / cell_nm
    ax_c_plate.plot(thickness_ml, at_30_nm, color="#1b7a8c", linewidth=1.0,
                    label="v1 at L = 30 nm")
    ax_c_plate.scatter(plate_thickness_ml, plate_data_mev, c="black", s=14,
                       zorder=3, label="Plate data")
    ax_c_plate.set(xscale="log", yscale="log", xlim=(0.8, 5.0),
                   xlabel="Plate thickness (ML)", ylabel="ΔE (meV)",
                   title="Plate calibration")
    ax_c_plate.set_ylim(0.8 * min(plate_data_mev.min(), at_30_nm.min()),
                        1.2 * max(plate_data_mev.max(), at_30_nm.max()))
    _plain_log_ticks(ax_c_plate, "x")
    ax_c_plate.xaxis.set_major_locator(FixedLocator([1, 2, 3, 4]))
    ax_c_plate.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    panel_label(ax_c_plate, "c")

    cube_nm = data["cube_nm"]
    ax_c_cube.plot(cube_nm, data["cube_data_delta_e_ev"] * 1000.0,
                   color="0.15", linestyle="--", linewidth=1.0,
                   label="Own cube trend")
    ax_c_cube.plot(cube_nm, data["cube_model_delta_e_ev"] * 1000.0,
                   color="#d1603d", linewidth=1.0, label="v1 cube")
    ax_c_cube.set(xscale="log", yscale="log", xlabel="Cube edge length (nm)",
                  ylabel="ΔE (meV)", title="Cube calibration")
    _plain_log_ticks(ax_c_cube, "x")
    ax_c_cube.xaxis.set_major_locator(FixedLocator([5, 10, 20]))
    ax_c_cube.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax_c_cube.yaxis.set_major_locator(FixedLocator([20, 30, 50, 100, 200]))
    ax_c_cube.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    panel_label(ax_c_cube, "d")

    savefig(fig, "F1")


if __name__ == "__main__":
    main()
