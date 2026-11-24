"""Template for figure scripts. Copy to fig_<id>_<name>.py.

Rules: read data from results/ only (never run solvers here), use plmorph.viz.style,
save via savefig(), then add a line to figures/GALLERY.md.
"""

import matplotlib.pyplot as plt
import numpy as np

from plmorph.viz import POP_COLORS, TRUTH_STYLE, apply_style, figsize, panel_label, savefig

# from plmorph.io import load_run
# config, data = load_run("results/m2/library_forward")


def main() -> None:
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=figsize("single", 0.45), constrained_layout=True)

    E = np.linspace(2.4, 2.9, 400)
    truth = np.exp(-0.5 * ((E - 2.62) / 0.02) ** 2)
    fit = np.exp(-0.5 * ((E - 2.625) / 0.021) ** 2)
    axes[0].plot(E, truth, **TRUTH_STYLE, label="truth")
    axes[0].fill_between(E, fit, color=POP_COLORS[0], alpha=0.35, lw=0, label="fit")
    axes[0].set(xlabel="Transition energy (eV)", ylabel="p(E)")
    axes[0].legend()
    panel_label(axes[0], "a")

    axes[1].imshow(np.random.default_rng(0).random((20, 20)), origin="lower")
    axes[1].set(xlabel="x (unit cells)", ylabel="y (unit cells)")
    panel_label(axes[1], "b")

    savefig(fig, "demo")


if __name__ == "__main__":
    main()
