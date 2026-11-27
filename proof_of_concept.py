"""Proof of concept: push single-mode morphologies through the forward pass and look at the spectra.

For each example in plmorph/examples.py:
  forward pass -> p(E) -> convolve with a Gaussian sigma0 = 14.7 meV (stand-in for GaLa) -> PL(E)

Figure:
  left   PL spectra on one shared energy axis, one panel per group, colour = lateral size.
         Triangles: measured PL peak of the same sample.
  right  PL peak and PL FWHM vs lateral size for 3 and 4 ML: samples (model vs measured)
         and a sweep over square plates (13 % lateral sigma).

Run:  python proof_of_concept.py      -> out/proof_of_concept.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from plmorph.binning import Binning
from plmorph.energy_map import EnergyMap
from plmorph.examples import EXAMPLES, UC_NM, plate
from plmorph.forward import forward
from plmorph.grid import EnergyGrid, Grid
from plmorph.morphology import Morphology

GRID = Grid(n_max=70)
EGRID = EnergyGrid()
SIGMA0 = 0.0147  # eV
BINNING = Binning(EnergyMap()(*GRID.mesh()), EGRID)
E = EGRID.centers


def pl_spectrum(morph: Morphology) -> np.ndarray:
    """Forward pass, then Gaussian broadening with SIGMA0. Unit area."""
    p = forward(morph, GRID, EGRID, BINNING).p
    x = np.arange(-5 * SIGMA0, 5 * SIGMA0 + EGRID.de / 2, EGRID.de)
    kernel = np.exp(-0.5 * (x / SIGMA0) ** 2)
    return np.convolve(p, kernel / kernel.sum(), mode="same")


def peak_and_fwhm(y: np.ndarray) -> tuple[float, float]:
    """Peak energy and FWHM in eV (outermost half-maximum crossings, linearly interpolated)."""
    i = int(np.argmax(y))
    above = np.where(y >= y[i] / 2)[0]
    lo, hi = above[0], above[-1]
    e_lo = np.interp(y[i] / 2, [y[lo - 1], y[lo]], [E[lo - 1], E[lo]])
    e_hi = np.interp(y[i] / 2, [y[hi + 1], y[hi]], [E[hi + 1], E[hi]])
    return E[i], e_hi - e_lo


def lateral_nm(morph: Morphology) -> float:
    """Geometric mean of the two lateral edges (x, y) in nm."""
    return float(np.sqrt(morph.means[0, 0] * morph.means[0, 1]) * UC_NM)


# ---- compute ---------------------------------------------------------------------------------
results = {}
for name, (morph, measured) in EXAMPLES.items():
    pl = pl_spectrum(morph)
    peak, fwhm = peak_and_fwhm(pl)
    results[name] = dict(morph=morph, pl=pl, peak=peak, fwhm=fwhm, measured=measured,
                         size=lateral_nm(morph))
    m = f"measured {measured:.3f}, model - measured {(peak - measured) * 1e3:+4.0f} meV" if measured else ""
    print(f"{name:14s} lateral {results[name]['size']:5.1f} nm  peak {peak:.3f} eV  "
          f"FWHM {fwhm * 1e3:4.1f} meV  {m}")

sweep_sizes = np.linspace(4.5, 25, 30)
sweep = {n: np.array([peak_and_fwhm(pl_spectrum(plate(n, L, L, 0.13 * L))) for L in sweep_sizes])
         for n in (3, 4)}

# ---- figure ----------------------------------------------------------------------------------
INK, GREY = "#222222", "#9a9a9a"
TEAL = LinearSegmentedColormap.from_list("teal", ["#9fd8d2", "#0b5c55"])
BLUE = LinearSegmentedColormap.from_list("blue", ["#a9c8f5", "#0b3a8c"])
ML_COLOR = {3: "#1f5fbf", 4: "#c2410c"}
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": GREY, "xtick.color": GREY, "ytick.color": GREY,
                     "axes.labelcolor": INK, "legend.frameon": False, "legend.fontsize": 7})


def thickness(name: str) -> float:
    return results[name]["morph"].means[0, 2]


groups = [("Cubes / QDs (literature sizes)", ["QD 4 nm", "QD 6 nm", "cube 6.7 nm", "cube 10 nm", "cube 15 nm"], TEAL),
          ("3 ML plates (own samples)", [n for n in results if thickness(n) == 3 and "/" not in n], BLUE),
          ("4 ML plates (own samples)", [n for n in results if thickness(n) == 4], BLUE),
          ("One mode over 3 and 4 ML (65 / 35)", ["3/4 ML, 10 nm"], BLUE)]

fig = plt.figure(figsize=(14, 11))
gs = fig.add_gridspec(4, 2, width_ratios=[1.5, 1], hspace=0.45, wspace=0.25)
for row, (title, names, cmap) in enumerate(groups):
    ax = fig.add_subplot(gs[row, 0])
    sizes = np.array([results[n]["size"] for n in names])
    shade = (sizes - sizes.min()) / np.ptp(sizes) if len(sizes) > 1 else [0.8]
    for n, c in zip(names, shade):
        r, color = results[n], cmap(c)
        ax.plot(E, r["pl"] / r["pl"].max(), color=color, lw=1.4, label=f"{n}  ({r['size']:.1f} nm)")
        if r["measured"]:
            ax.plot(r["measured"], 1.06, marker="v", ms=6, color=color, mec="white", mew=0.8,
                    clip_on=False)
    ax.set_xlim(2.35, 2.80)
    ax.set_ylim(0, 1.12)
    ax.set_yticks([])
    ax.set_title(title, loc="left", fontsize=9, color=INK)
    ax.legend(loc="upper right" if row == 0 else "upper left", ncol=2 if len(names) > 4 else 1)
    if row == len(groups) - 1:
        ax.set_xlabel("E (eV)")

for i, (key, ylabel, title) in enumerate([("peak", "PL peak (eV)", "Peak vs lateral size"),
                                          ("fwhm", "PL FWHM (meV)", "Width vs lateral size")]):
    ax = fig.add_subplot(gs[2 * i:2 * i + 2, 1])
    scale = 1e3 if key == "fwhm" else 1
    for n_ml, color in ML_COLOR.items():
        ax.plot(sweep_sizes, sweep[n_ml][:, i] * scale, color=color, lw=1.2,
                label=f"{n_ml} ML model, square plates, 13 % σ")
        rs = [r for n, r in results.items() if thickness(n) == n_ml and "/" not in n]
        ax.scatter([r["size"] for r in rs], [r[key] * scale for r in rs], s=40, facecolor="white",
                   edgecolor=color, lw=1.5, zorder=3, label=f"{n_ml} ML samples, model")
        if key == "peak":
            ax.scatter([r["size"] for r in rs], [r["measured"] for r in rs], s=40, color=color,
                       edgecolor="white", lw=1, zorder=4, label=f"{n_ml} ML samples, measured")
    if key == "fwhm":
        ax.axhline(2.355 * SIGMA0 * 1e3, color=GREY, lw=0.8, ls="--")
        ax.text(25, 2.355 * SIGMA0 * 1e3, "σ0 alone", va="bottom", ha="right", fontsize=7, color=GREY)
    ax.set_xlabel("lateral size √(a·b) (nm)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=9, color=INK)
    ax.legend(loc="upper right")

fig.suptitle("Single modes through the forward pass: E_map v1, Gaussian σ0 = 14.7 meV. "
             "Spectra scaled to peak 1; ▼ = measured PL peak.", x=0.125, y=0.94, ha="left",
             fontsize=10, color=INK)
Path("out").mkdir(exist_ok=True)
fig.savefig("out/proof_of_concept.png", dpi=150, bbox_inches="tight")
print("saved out/proof_of_concept.png")
