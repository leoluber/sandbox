"""F3: visualize the morphology-to-energy tensor pipeline from saved data."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import numpy as np

from plmorph.io import load_run
from plmorph.viz.style import CMAP_DENSITY, POP_COLORS, apply_style, figsize, panel_label, savefig

RUN_DIR = Path(__file__).resolve().parents[2] / "results" / "m2" / "library_forward"


def main() -> None:
    """Draw P, B, W, selected sparse entries of T, and p(E)."""
    apply_style()
    config, data = load_run(RUN_DIR)
    n_min = int(config["grid_n_min"])
    n_max = int(config["grid_n_max"])
    probability = data["pipeline_density"]
    brightness = data["brightness"]
    weighted = data["pipeline_weighted_density"]
    xy_probability = probability.sum(axis=2)
    xy_brightness = brightness[:, :, 3 - n_min]
    xy_weighted = weighted.sum(axis=2)
    extent = (n_min - 0.5, n_max + 0.5, n_min - 0.5, n_max + 0.5)

    fig, axes = plt.subplots(1, 5, figsize=figsize("double", 0.58),
                             constrained_layout=True)
    fig.suptitle("P  →  (.)B  →  W  →  sparse T  →  p(E)", fontsize=8)
    for ax, values, title in zip(
        axes[:3], (xy_probability, xy_brightness, xy_weighted),
        ("P(x, y)", "B(x, y | z = 3)", "W(x, y)"),
    ):
        image = ax.imshow(values.T, origin="lower", extent=extent, aspect="auto",
                          cmap=CMAP_DENSITY,
                          vmin=0 if title.startswith("B(") else None)
        ax.set(title=title, xlabel="x (uc)", ylabel="y (uc)")
        ax.set_xlim(n_min, n_max)
        ax.set_ylim(n_min, n_max)
    axes[1].text(0.5, 0.5, "unit brightness", transform=axes[1].transAxes,
                 ha="center", va="center", fontsize=6, color="white")
    panel_label(axes[0], "a")
    panel_label(axes[1], "b")
    panel_label(axes[2], "c")

    centers = data["energy_centers_ev"]
    sample_lower = data["sparse_sample_lower"]
    sample_upper = data["sparse_sample_w_upper"]
    sample_xyz = data["sparse_sample_xyz_uc"]
    sparse_ax = axes[3]
    rows = np.arange(sample_lower.size)
    lower_weights = 1.0 - sample_upper
    sparse_ax.scatter(centers[sample_lower], rows,
                      s=16 + 36 * lower_weights, color=POP_COLORS[0],
                      edgecolors="black", linewidths=0.3, label="lower bin")
    sparse_ax.scatter(centers[sample_lower + 1], rows,
                      s=16 + 36 * sample_upper, color=POP_COLORS[1],
                      edgecolors="black", linewidths=0.3, label="upper bin")
    sparse_ax.set(title="Two entries per T row", xlabel="Energy (eV)",
                  ylabel="Sample morphology (uc)")
    sparse_ax.set_yticks(rows, [f"{x},{y},{z}" for x, y, z in sample_xyz])
    sparse_ax.tick_params(axis="y", labelsize=4.5)
    sparse_ax.legend(fontsize=4.5, loc="best", handletextpad=0.2,
                     borderpad=0.2, labelspacing=0.2)
    panel_label(sparse_ax, "d")

    spectrum_index = list(data["names"]).index("npl_3ML_L20")
    axes[4].plot(centers, data["spectra"][spectrum_index],
                 color=POP_COLORS[0], linewidth=0.9)
    axes[4].set(title="p(E)", xlabel="Transition energy (eV)", ylabel="Density (1/eV)")
    panel_label(axes[4], "e")
    savefig(fig, "F3")


if __name__ == "__main__":
    main()
