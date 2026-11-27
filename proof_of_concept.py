"""Proof of concept: push single-mode morphologies through the forward pass and judge how
identifiable they would be in the backward pass.

For each example in plmorph/examples.py:
  1. forward pass  -> p(E), the inhomogeneous transition-energy distribution
  2. broaden p(E) with a Gaussian sigma0 = 14.7 meV -> PL(E)   (temporary stand-in for GaLa)
  3. peak and FWHM of PL(E)
  4. local identifiability: how well could PL pin down the 6 mode parameters
     (mean_x, mean_y, mean_z, sigma_x, sigma_y, sigma_z)?

Step 4 in plain words: change each parameter a little and see how much the spectrum changes
(Jacobian J, one column per parameter). With noise of 1 % of the peak (SNR 100) per 1 meV bin,
the singular values s of J / noise give the best possible precision (Cramer-Rao bound) along
the 6 independent parameter directions: 1 / s, in unit cells. A direction counts as
identifiable if 1 / s < 0.5 uc. This is a lower bound for a perfect model, PL only, one mode,
spectra normalized to unit area (the absolute intensity carries no information).

Run:  python proof_of_concept.py      -> out/proof_of_concept.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plmorph.binning import Binning
from plmorph.energy_map import EnergyMap
from plmorph.examples import EXAMPLES
from plmorph.forward import forward
from plmorph.grid import EnergyGrid, Grid
from plmorph.morphology import Morphology

GRID = Grid(n_max=70)     # larger than the default so the 30 nm plates fit
EGRID = EnergyGrid()
SIGMA0 = 0.0147           # eV, homogeneous broadening
SNR = 100                 # peak / noise per 1 meV bin
TOLERANCE_UC = 0.5        # a direction is identifiable if pinned down better than this
STEP_UC = 0.02            # finite-difference step for the Jacobian
PARAMS = ["mean_x", "mean_y", "mean_z", "sigma_x", "sigma_y", "sigma_z"]

BINNING = Binning(EnergyMap()(*GRID.mesh()), EGRID)
E = EGRID.centers


def broaden(p: np.ndarray) -> np.ndarray:
    """Convolve p(E) with a normalized Gaussian of width SIGMA0 (keeps the area)."""
    x = np.arange(-5 * SIGMA0, 5 * SIGMA0 + EGRID.de / 2, EGRID.de)
    kernel = np.exp(-0.5 * (x / SIGMA0) ** 2)
    return np.convolve(p, kernel / kernel.sum(), mode="same")


def spectra(mean, sigma) -> tuple[np.ndarray, np.ndarray]:
    """One mode -> (p(E), PL(E)), both with unit area."""
    p = forward(Morphology(means=[mean], sigmas=[sigma]), GRID, EGRID, BINNING).p
    return p, broaden(p)


def peak_and_fwhm(y: np.ndarray) -> tuple[float, float]:
    """Peak energy (eV) and FWHM (eV), half-maximum crossings linearly interpolated."""
    i = int(np.argmax(y))
    half = y[i] / 2
    lo = i - np.argmax(y[i::-1] < half)        # first bin below half, going left
    hi = i + np.argmax(y[i:] < half)           # first bin below half, going right
    e_lo = np.interp(half, [y[lo], y[lo + 1]], [E[lo], E[lo + 1]])
    e_hi = np.interp(half, [y[hi], y[hi - 1]], [E[hi], E[hi - 1]])
    return E[i], e_hi - e_lo


def precision_per_direction(mean, sigma) -> np.ndarray:
    """Best possible precision (uc) along the 6 independent parameter directions, sorted."""
    theta = np.concatenate([mean, sigma])
    _, pl = spectra(mean, sigma)
    noise = pl.max() / SNR
    J = np.empty((EGRID.n_bins, 6))
    for i in range(6):
        up, down = theta.copy(), theta.copy()
        up[i] += STEP_UC
        down[i] -= STEP_UC
        J[:, i] = (spectra(up[:3], up[3:])[1] - spectra(down[:3], down[3:])[1]) / (2 * STEP_UC)
    s = np.linalg.svd(J / noise, compute_uv=False)
    with np.errstate(divide="ignore"):
        return np.sort(1 / s)


# ---- compute ---------------------------------------------------------------------------------
rows = []
for name, morph in EXAMPLES.items():
    mean, sigma = morph.means[0], morph.sigmas[0]
    p, pl = spectra(mean, sigma)
    peak, fwhm = peak_and_fwhm(pl)
    precision = precision_per_direction(mean, sigma)
    rows.append(dict(name=name, p=p, pl=pl, peak=peak, fwhm=fwhm, precision=precision,
                     n_ok=int(np.sum(precision < TOLERANCE_UC))))
    print(f"{name:20s} peak {peak:.3f} eV  FWHM {fwhm * 1e3:5.1f} meV  "
          f"identifiable {rows[-1]['n_ok']}/6  precision (uc) "
          + " ".join(f"{v:8.2g}" for v in precision))

# ---- figure ----------------------------------------------------------------------------------
INK, GREY, LIGHT = "#222222", "#9a9a9a", "#d9d9d9"
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": GREY, "xtick.color": GREY, "ytick.color": GREY,
                     "axes.labelcolor": INK})
n_cols = 6
n_rows = int(np.ceil(len(rows) / n_cols))
fig = plt.figure(figsize=(15, 3 * n_rows + 4.5))
gs = fig.add_gridspec(n_rows + 1, n_cols, height_ratios=[1] * n_rows + [1.6], hspace=0.6, wspace=0.3,
                      top=0.94)
HALF_WINDOW = 0.12  # eV, same window width in every small panel so widths compare directly

for idx, r in enumerate(rows):
    ax = fig.add_subplot(gs[idx // n_cols, idx % n_cols])
    ax.fill_between(E, r["p"] / r["p"].max(), color=LIGHT, step="mid", lw=0, label="p(E)")
    ax.plot(E, r["pl"] / r["pl"].max(), color=INK, lw=1.2, label="PL(E)")
    ax.set_xlim(r["peak"] - HALF_WINDOW, r["peak"] + HALF_WINDOW)
    ax.set_ylim(0, 1.08)
    ax.set_yticks([])
    ax.set_title(f"{idx + 1}  {r['name']}", loc="left", fontsize=8.5, color=INK)
    ax.text(0.98, 0.95, f"{r['peak']:.3f} eV\n{r['fwhm'] * 1e3:.0f} meV\n{r['n_ok']}/6 ident.",
            transform=ax.transAxes, ha="right", va="top", fontsize=7, color=INK)
    if idx == 0:
        ax.legend(loc="upper left", frameon=False, fontsize=7)
    if idx // n_cols == n_rows - 1 or idx + n_cols >= len(rows):
        ax.set_xlabel("E (eV)")

# peak vs FWHM: modes close together here are hard to tell apart from PL alone
ax = fig.add_subplot(gs[n_rows, :3])
peaks = np.array([r["peak"] for r in rows])
fwhms = np.array([r["fwhm"] for r in rows]) * 1e3
ax.scatter(peaks, fwhms, s=40, color=INK, edgecolor="white", linewidth=1.5, zorder=3)
labels: dict[tuple, list[str]] = {}  # modes at (nearly) the same spot share one label
for i, (x, y) in enumerate(zip(peaks, fwhms)):
    labels.setdefault((round(x / 0.02), round(y / 1.0)), []).append(str(i + 1))
for i, (x, y) in enumerate(zip(peaks, fwhms)):
    group = labels[(round(x / 0.02), round(y / 1.0))]
    if group[0] == str(i + 1):
        ax.annotate(", ".join(group), (x, y), xytext=(4, 4), textcoords="offset points",
                    fontsize=7, color=INK)
ax.axhline(2.355 * SIGMA0 * 1e3, color=GREY, lw=0.8, ls="--")
ax.text(peaks.min(), 2.355 * SIGMA0 * 1e3 - 0.15, "FWHM of σ0 alone", va="top", ha="left",
        fontsize=7, color=GREY)
ax.set_xlabel("PL peak (eV)")
ax.set_xlim(peaks.min() - 0.03, peaks.max() + 0.05)
ax.set_ylim(2.355 * SIGMA0 * 1e3 - 1, fwhms.max() + 1)
ax.set_ylabel("PL FWHM (meV)")
ax.set_title("a  Peak and width of every mode", loc="left", fontsize=9, color=INK)

# precision per parameter direction: below the dashed line = identifiable from PL
ax = fig.add_subplot(gs[n_rows, 3:])
CAP = 1e3
for i, r in enumerate(rows):
    v = np.minimum(r["precision"], CAP)
    ax.scatter(np.full(6, i + 1), v, s=22, color=INK, edgecolor="white", linewidth=1, zorder=3)
ax.axhline(TOLERANCE_UC, color=GREY, lw=0.8, ls="--")
ax.text(len(rows) + 0.4, TOLERANCE_UC, f"{TOLERANCE_UC} uc", va="bottom", ha="right",
        fontsize=7, color=GREY)
ax.set_yscale("log")
ax.set_ylim(1e-3, CAP * 1.5)
ax.set_xticks(range(1, len(rows) + 1))
ax.set_xlabel("mode")
ax.set_ylabel(f"best precision per direction (uc), SNR {SNR}")
ax.set_title(f"b  Six parameter directions per mode (values ≥ {CAP:.0f} = not constrained)",
             loc="left", fontsize=9, color=INK)

fig.suptitle("Single modes through the forward pass (E_map v1, Gaussian σ0 = 14.7 meV, "
             "lattice-point normals). p(E) and PL each scaled to peak 1; 18: peak and FWHM of "
             "the main (3 ML) peak", x=0.125, y=0.975, ha="left", fontsize=10, color=INK)
Path("out").mkdir(exist_ok=True)
fig.savefig("out/proof_of_concept.png", dpi=150, bbox_inches="tight")
print("saved out/proof_of_concept.png")
