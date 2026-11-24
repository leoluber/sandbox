"""Paper-ready figure style. See SPEC §9. Every figure uses this module."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

MM = 1 / 25.4
WIDTHS_MM = {"single": 89.0, "onehalf": 120.0, "double": 183.0}
OUT_DIR = Path(__file__).resolve().parents[2] / "figures" / "out"

# Fixed colour semantics (population index -> colour, the same in every figure).
POP_COLORS = ["#1b7a8c", "#d1603d", "#6b5bb5"]
CMAP_DENSITY = "cividis"
CMAP_DIVERGING = "RdBu_r"
CMAP_ENERGY = "plasma"
TRUTH_STYLE = dict(color="0.15", linestyle=(0, (3, 2)), linewidth=0.8)


def apply_style() -> None:  # INTERFACE
    """Set the global rcParams for all figures."""
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 7,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5,
        "legend.frameon": False,
        "axes.linewidth": 0.6,
        "lines.linewidth": 0.8,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "image.cmap": CMAP_DENSITY,
    })


def figsize(width: str = "single", aspect: float = 0.75) -> tuple[float, float]:  # INTERFACE
    """Figure size in inches for a journal width ('single' | 'onehalf' | 'double'); height = aspect * width."""
    w = WIDTHS_MM[width] * MM
    return (w, w * aspect)


def panel_label(ax, label: str, x: float = -0.18, y: float = 1.04) -> None:  # INTERFACE
    """Bold lowercase panel label ('a', 'b', ...) at the top left of an axis."""
    ax.text(x, y, label, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="left")


def savefig(fig, name: str, out_dir: Path | None = None) -> tuple[Path, Path]:  # INTERFACE
    """Save as figures/out/<name>.pdf and .png. Returns both paths."""
    out = Path(out_dir) if out_dir else OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    pdf, png = out / f"{name}.pdf", out / f"{name}.png"
    fig.savefig(pdf)
    fig.savefig(png)
    plt.close(fig)
    return pdf, png
