"""Draw docs/architecture.png: modules, core classes/functions, pipeline and connections.

Regenerate after interface changes:  python docs/make_architecture.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent / "architecture.png"

C = {  # module colours: (fill, edge, text)
    "pop": ("#E1F5EE", "#0F6E56", "#085041"),
    "maps": ("#FAEEDA", "#BA7517", "#633806"),
    "fwd": ("#EEEDFE", "#534AB7", "#3C3489"),
    "loss": ("#FAECE7", "#D85A30", "#712B13"),
    "synth": ("#E6F1FB", "#185FA5", "#0C447C"),
    "solve": ("#EAF3DE", "#3B6D11", "#27500A"),
    "util": ("#F1EFE8", "#5F5E5A", "#2C2C2A"),
}

fig = plt.figure(figsize=(22, 13.5), dpi=150)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 200)
ax.set_ylim(0, 123)
ax.axis("off")
plt.rcParams["font.family"] = "DejaVu Sans"


def box(x, y, w, h, key, title, lines, fs=9.5, dashed=False):
    fill, edge, txt = C[key]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.6",
                                fc=fill, ec=edge, lw=1.3, ls="--" if dashed else "-"))
    ax.text(x + 1.6, y + h - 2.2, title, fontsize=12, fontweight="bold", color=txt, va="top")
    for i, ln in enumerate(lines):
        ax.text(x + 1.8, y + h - 6.6 - i * 3.0, ln, fontsize=fs, color=txt, va="top",
                family="DejaVu Sans Mono")


def chip(x, y, w, h, key, text, fs=10.5, dashed=False):
    fill, edge, txt = C[key]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2",
                                fc=fill, ec=edge, lw=1.1, ls="--" if dashed else "-"))
    ax.text(x + w / 2, y + h / 2, text, fontsize=fs, color=txt, ha="center", va="center")


def arrow(x1, y1, x2, y2, color="#444441", ls="-", lw=1.3, rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13,
                                 color=color, lw=lw, ls=ls, connectionstyle=f"arc3,rad={rad}"))


def label(x, y, s, fs=9.5, color="#5F5E5A", **kw):
    ax.text(x, y, s, fontsize=fs, color=color, ha="center", va="center", **kw)


# ---- title ------------------------------------------------------------------------------
ax.text(4, 119.5, "PLmorph sandbox — architecture (phase 1: no optical kernel)",
        fontsize=17, fontweight="bold", color="#2C2C2A", va="center")
ax.text(4, 115.5, "Modules (boxes) with their core classes and functions · tensor pipeline "
        "(middle; solid = forward, dashed = backpropagation) · solver loop and outputs (bottom)", fontsize=11, color="#5F5E5A", va="center")

# ---- row A: physics modules ---------------------------------------------------------------
yA, hA = 80, 31
box(3, yA, 36, hA, "pop", "populations.py", [
    "Population(kind, mean, rel_sigma,",
    "           weight, corr)",
    "density(pop, grid) -> P_k, mass_out",
    "mixture_density(pops, grid) -> P",
    "to_params / from_params  (θ <-> pops)",
    "random_population(rng, kind, grid)",
    "LIBRARY  (11 named populations)",
    "REL_SIGMA_BOUNDS = (0.05, 0.20)",
])
box(42, yA, 34, hA, "maps", "maps.py", [
    "EnergyMap(x, y, z) -> E   [protocol]",
    "BrightnessMap(x, y, z) -> B",
    "ToyConfinementMap",
    "   ΔE = f(d1) + g(d2, d3), sorted",
    "UnitBrightness  (B = 1)",
    "get_energy_map / get_brightness_map",
    "register_*_map  (swap maps)",
    "E0_EV = 2.375",
])
box(79, yA, 38, hA, "fwd", "forward.py", [
    "BinningOperator  (sparse T)",
    "  .build(grid, egrid, e_map)",
    "  .apply(W) -> p        (G -> n_bins)",
    "  .adjoint(r) -> ∂W     (n_bins -> G)",
    "  .n_out_of_range",
    "forward(pops, grid, egrid, T, b_map)",
    "  -> ForwardResult(p, components,",
    "                   W, n_out_of_range)",
])
box(120, yA, 36, hA, "loss", "losses.py", [
    "Loss  [protocol]",
    "  __call__(p_model, p_data, egrid)",
    "  .grad(...) -> ∂L/∂p_model",
    "Chi2   KL   Wasserstein1",
    "Cramer   Moments",
    "CompositeLoss([(loss, weight), ...])",
    "LOSSES  registry",
])
box(159, yA, 38, hA, "synth", "synth.py", [
    "Measurement(p, egrid, truth, meta)",
    "Corruption(n_particles, noise_sigma,",
    "           energy_shift_ev, seed)",
    "make_measurement(p_true, truth,",
    "                 egrid, corruption)",
    "map mismatch: generate with perturbed",
    "  map, invert with nominal map",
])

# ---- tensor pipeline strip ----------------------------------------------------------------
yT, hT = 63, 8.5
chips = [  # x, w, key, text
    (3, 17, "pop", "θ  (unconstrained)"),
    (24, 19, "pop", "P  (nx, ny, nz)"),
    (47, 21, "maps", "W = P ⊙ B"),
    (72, 25, "fwd", "p_model = T · W   (n_bins)"),
    (101, 15, "fwd", "normalize"),
    (120, 36, "loss", "L = loss(p_model, p_data)"),
    (160, 37, "synth", "p_data  (Measurement.p)"),
]
for x, w, k, t in chips:
    chip(x, yT, w, hT, k, t)
mid = yT + hT / 2
for (x1, w1, *_), (x2, *_) in zip(chips[:4], chips[1:5]):
    arrow(x1 + w1 + 0.4, mid + 1.4, x2 - 0.4, mid + 1.4)
arrow(116.4, mid + 1.4, 119.6, mid + 1.4)
arrow(159.6, mid, 156.4, mid)
label(22, mid + 7.6, "from_params", fs=9)
label(45, mid + 7.6, "⊙ B(x,y,z)", fs=9)
label(70, mid + 7.6, "T from E_map", fs=9)
# backward (dashed) under the strip
yb = yT - 3.2
for xs, xe in [(132, 118), (99, 85), (70, 60), (45, 35), (22, 13)]:
    arrow(xs, yb, xe, yb, color="#993C1D", ls="--")
label(126, yb - 3.0, "r = ∂L/∂p", fs=9.5, color="#993C1D")
label(92, yb - 3.0, "Tᵀ r  (adjoint)", fs=9.5, color="#993C1D")
label(65, yb - 3.0, "∂L/∂W", fs=9.5, color="#993C1D")
label(40, yb - 3.0, "⊙ B → ∂L/∂P", fs=9.5, color="#993C1D")
label(17, yb - 3.0, "∂L/∂θ", fs=9.5, color="#993C1D")
chip(160, yb - 9.2, 37, 6, "fwd", "later: GaLa layer  K = T ∗ GaLa(σ0, w_L)", fs=9.5, dashed=True)

# vertical links from modules to the strip
for x in (21, 59, 98, 138, 178):
    arrow(x, yA - 0.3, x, yT + hT + 0.4, color="#888780", lw=1.0)

# ---- row B: solvers + utilities ------------------------------------------------------------
yB, hB = 22, 26
box(3, yB, 88, hB, "solve", "solvers/   base.py  +  one module per family", [
    "Problem(grid, egrid, e_map, b_map, measurement, kinds)",
    "Solver.solve(problem, loss, init, rng) -> SolverResult      [protocol]",
    "SolverResult(populations, loss, loss_history, param_history,",
    "             runtime_s, n_forward_evals, meta)",
    "SOLVERS registry · register_solver · get_solver",
    "gradient: Adam, L-BFGS   global: CMA-ES, DE, basin hopping",
    "linear: NNLS + regularization   bayes: Laplace, HMC   sbi: later",
], fs=9.5)
box(165, yB, 32, hB, "util", "grid.py", [
    "Grid(n_min=2, n_max=50)",
    "  .axis  .shape  .mesh()",
    "EnergyGrid(2.35, 3.20, 0.001)",
    "  .centers  .n_bins",
    "UNIT_CELL_NM = 0.59",
    "used by every module",
])
box(94, yB, 33, hB, "util", "metrics.py", [
    "match_populations  (Hungarian)",
    "parameter_errors",
    "marginal_wasserstein",
    "k_correct",
])
box(130, yB, 32, hB, "util", "io.py", [
    "save_run(dir, config, arrays)",
    "  -> config.json + data.npz",
    "load_run(dir)",
    "  -> (config, arrays)",
])

# solver loop arrows (strip <-> solver)
arrow(30, yB + hB + 0.4, 30, yT - 10.6, color="#3B6D11", lw=1.6)
arrow(14, yT - 10.6, 14, yB + hB + 0.4, color="#3B6D11", lw=1.6)
label(41.5, yB + hB + 4.2, "propose θ, evaluate L", fs=9.5, color="#27500A")
label(6.8, yB + hB + 2.0, "∂L/∂θ", fs=9.5, color="#27500A")
arrow(91.4, yB + 8, 93.6, yB + 8, color="#888780", lw=1.0)
arrow(127.4, yB + 8, 129.6, yB + 8, color="#888780", lw=1.0)


# ---- row C: outputs ---------------------------------------------------------------------
yC, hC = 5, 10
outs = [
    (3, 30, "solve", "experiments/m*_*.py"),
    (40, 36, "util", "results/<M>/<run>/  config.json · data.npz"),
    (83, 36, "util", "figures/scripts/fig_<id>.py\n+ viz.style (apply_style, savefig)"),
    (126, 32, "util", "figures/out/*.pdf · *.png"),
    (165, 32, "util", "figures/GALLERY.md"),
]
for x, w, k, t in outs:
    chip(x, yC, w, hC, k, t, fs=10)
for (x1, w1, *_), (x2, *_) in zip(outs[:-1], outs[1:]):
    arrow(x1 + w1 + 0.4, yC + hC / 2, x2 - 0.4, yC + hC / 2)
arrow(136, yB - 0.3, 72, yC + hC + 0.4, color="#888780", lw=1.0)
label(150, yB - 3.0, "io.save_run", fs=9.5)
label(58, yC + hC + 3.0, "plots read results only — never rerun solvers", fs=9.5)
label(20, yC + hC + 3.0, "tests/acceptance (locked) gate each milestone", fs=9.5, color="#3B6D11")

fig.savefig(OUT, dpi=150, facecolor="white")
print(f"wrote {OUT}")
