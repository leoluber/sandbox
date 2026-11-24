"""2D reduction of the CsPbBr3 confinement data: lateral L = sqrt(x*y), thickness z.
dE = E - 2.37 eV (unconfined level). All axes logarithmic."""
import csv, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

CSV = sys.argv[1] if len(sys.argv) > 1 else "confinement_data.csv"
E_BULK = 2.37
ML = 0.59

rows = []
for r in csv.DictReader(open(CSV, encoding="utf-8-sig"), delimiter=";"):
    n = r["no"].strip(); nl = n.lower()
    if any(t in nl for t in ("fake", "pbs", "moreels")):
        continue
    x, y, z, E = (float(r[k]) for k in ("x(nm)", "y(nm)", "z(nm)", "E(eV)"))
    L = np.sqrt(x * y)
    kind = "extrap" if "extrapolation" in nl else ("issue" if "issue" in nl else "ok")
    fam = "cube" if abs(np.log(z / L)) < 0.15 else ("plate" if z < L else "wire")
    rows.append(dict(L=L, z=z, dE=1000 * (E - E_BULK), kind=kind, fam=fam))
get = lambda key, **f: np.array([r[key] for r in rows if all(r[k] == v for k, v in f.items())])

# ---------------- style ----------------
BG, PANEL, INK, INK2, INK3, GRID = "#0d1017", "#141922", "#e8eaf0", "#9aa3b2", "#5c6575", "#252c38"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Inter", "Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size": 10, "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": GRID,
    "xtick.color": INK3, "ytick.color": INK3, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "figure.facecolor": BG, "axes.facecolor": PANEL, "savefig.facecolor": BG, "legend.frameon": False,
    "legend.labelcolor": INK2, "axes.spines.top": False, "axes.spines.right": False,
})
C_CUBE, C_PLATE, C_WIRE = "#4cc9f0", "#f4a261", "#c77dff"
MARK = dict(ok=dict(), issue=dict(facecolors="none"), extrap=dict(marker="x"))


def scatter(ax, xs, ys, kind, color, label=None, **kw):
    style = dict(s=34, linewidths=1.2, color=color, zorder=3, label=label)
    if kind == "issue":
        style.update(facecolors="none", edgecolors=color, label=None)
    if kind == "extrap":
        style.update(marker="x", label=None)
    ax.scatter(xs, ys, **style, **kw)


def loglog(ax):
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.grid(True, which="major", color=GRID, lw=0.6); ax.grid(True, which="minor", color=GRID, lw=0.3, alpha=0.5)
    ax.set_axisbelow(True)


fig, axs = plt.subplots(1, 3, figsize=(17, 6.2), gridspec_kw=dict(wspace=0.28))
fig.subplots_adjust(left=0.05, right=0.97, top=0.78, bottom=0.12)
fig.text(0.05, 0.945, "CsPbBr$_3$ confinement data in 2D", fontsize=19, weight="bold")
fig.text(0.05, 0.900, f"Lateral L = √(x·y), thickness z.   ΔE = E − {E_BULK} eV.   "
         "Filled = fitted data,  open = 'issue',  × = 'extrapolation'.   PbS / Moreels / fake rows removed.",
         fontsize=10.5, color=INK2)

# ---- A: coverage of the (L, z) plane ----
ax = axs[0]; loglog(ax)
norm = LogNorm(5, 1400)
for kind, mk in (("ok", "o"), ("issue", "o"), ("extrap", "X")):
    sel = [r for r in rows if r["kind"] == kind]
    Ls, zs, ds = (np.array([r[k] for r in sel]) for k in ("L", "z", "dE"))
    if kind == "issue":
        ax.scatter(Ls, zs, s=46, facecolors="none", edgecolors=plt.cm.magma(norm(ds)), lw=1.3, zorder=3)
    else:
        sc = ax.scatter(Ls, zs, c=ds, cmap="magma", norm=norm, s=46, marker=mk, edgecolors=BG, lw=0.6, zorder=3)
lim = (0.45, 600)
ax.plot(lim, lim, color=INK3, lw=0.8, ls=(0, (4, 3))); ax.text(30, 20, "cubes (z = L)", color=INK3, rotation=45, fontsize=9)
for n in range(1, 5):
    ax.axhline(n * ML, color=GRID, lw=0.8, zorder=1)
    ax.text(lim[1] * 0.95, n * ML * 1.04, f"{n} ML", ha="right", va="bottom", fontsize=8, color=INK3)
ax.text(1.9, 40, "wires", color=INK3, fontsize=9); ax.text(60, 0.75, "plates", color=INK3, fontsize=9)
ax.set(xlim=lim, ylim=(0.45, 150), xlabel="lateral L (nm)", ylabel="thickness z (nm)")
cb = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.05); cb.set_label("ΔE (meV)", color=INK2)
cb.outline.set_visible(False)
ax.set_title("A · Where the data sit", loc="left", fontsize=12.5, weight="bold", pad=10)

# ---- B: dE vs smallest dimension ----
ax = axs[1]; loglog(ax)
Lc, dc = get("L", fam="cube", kind="ok"), get("dE", fam="cube", kind="ok")
m = (Lc >= 3) & (Lc <= 20)
p, c = np.polyfit(np.log(Lc[m]), np.log(dc[m]), 1)
xx = np.geomspace(1.1, 110, 50)
ax.plot(xx, np.exp(c) * xx ** p, color=C_CUBE, lw=1.4, alpha=0.8)
ax.text(30, np.exp(c) * 30 ** p * 1.25, f"slope {p:.2f}", color=C_CUBE, fontsize=9.5, rotation=-28)
for k in ("ok", "issue", "extrap"):
    if len(get("L", fam="cube", kind=k)):
        scatter(ax, get("L", fam="cube", kind=k), get("dE", fam="cube", kind=k), k, C_CUBE, "cubes (vs L)")
for k in ("ok", "issue", "extrap"):
    if len(get("z", fam="plate", kind=k)):
        scatter(ax, get("z", fam="plate", kind=k), get("dE", fam="plate", kind=k), k, C_PLATE, "plates (vs z)")
for k in ("ok", "issue"):
    scatter(ax, get("L", fam="wire", kind=k), get("dE", fam="wire", kind=k), k, C_WIRE, "wires (vs thickness)")
# plate asymptotes (dE at large L, from dE = a + b/L^2 per thickness, fitted rows only)
zs, ainf = [], []
for n in (1, 2, 3, 4):
    Lp, dp = (np.array([r[k] for r in rows if r["fam"] == "plate" and r["kind"] == "ok"
                        and abs(r["z"] - n * ML) < 1e-3]) for k in ("L", "dE"))
    a = np.linalg.lstsq(np.c_[np.ones_like(Lp), Lp ** -2.0], dp, rcond=None)[0][0] if len(Lp) > 1 else dp[0]
    zs.append(n * ML); ainf.append(a)
ax.plot(zs, ainf, color=C_PLATE, lw=1.4, alpha=0.8, marker="D", ms=5, mec=BG)
for i in range(3):
    s = np.log(ainf[i + 1] / ainf[i]) / np.log(zs[i + 1] / zs[i])
    xm, ym = np.sqrt(zs[i] * zs[i + 1]), np.sqrt(ainf[i] * ainf[i + 1])
    ax.text(xm * (1.12 if i < 2 else 0.9), ym * (1.08 if i < 2 else 0.68), f"{s:.2f}", color=C_PLATE, fontsize=9, ha="left" if i < 2 else "right")
ax.set(xlim=(0.45, 150), ylim=(4, 1600), xlabel="smallest dimension (nm)", ylabel="ΔE (meV)")
ax.legend(loc="lower left", fontsize=9, handletextpad=0.3)
ax.set_title("B · Confinement vs the thinnest edge", loc="left", fontsize=12.5, weight="bold", pad=10)
ax.text(0.98, 0.97, "orange line: large-L plate limit,\nlocal log-log slopes between ML steps",
        transform=ax.transAxes, ha="right", va="top", fontsize=8.5, color=INK3)

# ---- C: dependence on the long dimensions ----
ax = axs[2]; loglog(ax)
cols = {1: "#ffd166", 2: "#f4a261", 3: "#e76f51", 4: "#d1495b"}
for n in (1, 2, 3, 4):
    for k in ("ok", "issue", "extrap"):
        sel = [r for r in rows if r["fam"] == "plate" and r["kind"] == k and abs(r["z"] - n * ML) < 1e-3]
        if sel:
            scatter(ax, [r["L"] for r in sel], [r["dE"] for r in sel], k, cols[n], f"{n} ML plates vs L")
for k in ("ok", "issue"):
    scatter(ax, get("z", fam="wire", kind=k), get("dE", fam="wire", kind=k), k, C_WIRE, "3×3 ML wires vs length")
ax.plot(xx, np.exp(c) * xx ** p, color=C_CUBE, lw=1.2, alpha=0.6)
ax.text(5, np.exp(c) * 5 ** p * 0.62, "cube trend", color=C_CUBE, fontsize=9, rotation=-30)
ax.set(xlim=(3, 600), ylim=(4, 1600), xlabel="long dimension(s) (nm)", ylabel="ΔE (meV)")
h, l = ax.get_legend_handles_labels(); u = dict(zip(l, h))
ax.legend(u.values(), u.keys(), loc="lower left", fontsize=9, handletextpad=0.3)
ax.set_title("C · Confinement vs the long edges", loc="left", fontsize=12.5, weight="bold", pad=10)

out = sys.argv[2] if len(sys.argv) > 2 else None
plt.savefig(out, dpi=120) if out else plt.show()
