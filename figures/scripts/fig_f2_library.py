"""F2: corner projections for every saved M2 library population."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import numpy as np

from plmorph.io import load_run
from plmorph.viz.style import CMAP_DENSITY, POP_COLORS, apply_style, figsize, savefig

RUN_DIR = Path(__file__).resolve().parents[2] / "results" / "m2" / "library_forward"


def _pair_image(ax, values: np.ndarray, n_min: int, n_max: int,
                xlabel: str, ylabel: str) -> None:
    extent = (n_min - 0.5, n_max + 0.5, n_min - 0.5, n_max + 0.5)
    ax.imshow(values.T, origin="lower", extent=extent, aspect="auto",
              cmap=CMAP_DENSITY, interpolation="nearest")
    ax.set(xlim=(n_min - 0.5, n_max + 0.5), ylim=(n_min - 0.5, n_max + 0.5),
           xlabel=xlabel, ylabel=ylabel)


def main() -> None:
    """Create a small corner plot for each library member."""
    apply_style()
    config, data = load_run(RUN_DIR)
    n_min, n_max = int(config["grid_n_min"]), int(config["grid_n_max"])
    axis = np.arange(n_min, n_max + 1)
    names = data["names"]
    densities = data["density"]
    fig = plt.figure(figsize=figsize("double", 1.28), constrained_layout=True)
    outer = fig.add_gridspec(4, 3, wspace=0.08, hspace=0.13)

    for index, (name, probability) in enumerate(zip(names, densities)):
        row, col = divmod(index, 3)
        corner = outer[row, col].subgridspec(3, 3, wspace=0.08, hspace=0.08)
        axes = np.empty((3, 3), dtype=object)
        for r in range(3):
            for c in range(3):
                axes[r, c] = fig.add_subplot(corner[r, c])
                if c > r:
                    axes[r, c].set_axis_off()

        marginals = (
            probability.sum(axis=(1, 2)),
            probability.sum(axis=(0, 2)),
            probability.sum(axis=(0, 1)),
        )
        color = POP_COLORS[index % len(POP_COLORS)]
        for dimension in range(3):
            ax = axes[dimension, dimension]
            marginal = marginals[dimension]
            ax.plot(axis, marginal, color=color, linewidth=0.75)
            ax.fill_between(axis, marginal, color=color, alpha=0.16, linewidth=0)
            ax.set_xlim(n_min, n_max)
            ax.set_yticks([])
            ax.tick_params(labelsize=4.5, pad=1)
            if dimension < 2:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel("z (uc)", fontsize=5, labelpad=1)
            if dimension == 0:
                ax.set_title(str(name), loc="left", fontsize=5.5, pad=1)
                ax.set_ylabel("p(x)", fontsize=5, labelpad=1)
            elif dimension == 1:
                ax.set_ylabel("p(y)", fontsize=5, labelpad=1)
            else:
                ax.set_ylabel("p(z)", fontsize=5, labelpad=1)

        _pair_image(axes[1, 0], probability.sum(axis=2), n_min, n_max, "x (uc)", "y (uc)")
        _pair_image(axes[2, 0], probability.sum(axis=1), n_min, n_max, "x (uc)", "z (uc)")
        _pair_image(axes[2, 1], probability.sum(axis=0), n_min, n_max, "y (uc)", "z (uc)")
        for pair_ax in (axes[1, 0], axes[2, 0], axes[2, 1]):
            pair_ax.tick_params(labelsize=4, pad=0.5)

    for index in range(len(names), 12):
        row, col = divmod(index, 3)
        blank = fig.add_subplot(outer[row, col])
        blank.set_axis_off()

    savefig(fig, "F2")


if __name__ == "__main__":
    main()
