"""Fixed evaluation harness: fits mapping.py to the data, checks the rules, logs the result.
Usage:  python eval.py [data.csv]        (default: confinement_data.csv)
Needs numpy and scipy. Appends one line per run to results.tsv."""
import csv, itertools, sys, time
import numpy as np
from scipy.optimize import least_squares
import mapping

LATTICE_NM = 0.59  # 1 unit cell = 1 monolayer

# ---- Data rules: the first tag found in the notes column ("no") sets the row's role. ----
# None = drop the row entirely, 0 = held out (not fitted, residual still reported), >0 = fit weight.
TAG_RULES = [
    ("fake", None),           # not real data
    ("pbs", None),            # other material
    ("moreels", None),        # other material
    ("extrapolation", 0),     # estimates, not measurements
    ("issue", 0),             # flagged as unreliable
]
DEFAULT_WEIGHT = 1.0

# ---- Physics rules that every mapping must pass. ----
REQUIRE_SYMMETRY = True       # a cuboid is the same object whatever its axis labels
E_BULK_EV = 2.375             # unconfined level, validated experimentally - not a free parameter
BULK_RANGE_EV = (E_BULK_EV - 0.002, E_BULK_EV + 0.002)  # E at 2000 unit cells per side
GRID = [1, 2, 3, 4, 6, 10, 20, 50, 100, 300]  # unit cells


def load(path):
    rows = []
    with open(path, encoding="utf-8-sig", errors="replace") as f:
        for r in csv.DictReader(f, delimiter=";"):
            r = {k.strip().lower(): v for k, v in r.items() if k}  # column names are case-insensitive
            note = (r.get("no") or "").strip()
            w = next((tw for tag, tw in TAG_RULES if tag in note.lower()), DEFAULT_WEIGHT)
            if w is None or not (r.get("e(ev)") or "").strip():
                continue
            rows.append([float(r["x(nm)"]), float(r["y(nm)"]), float(r["z(nm)"]),
                         float(r["e(ev)"]), w, note])
    d = np.array([r[:5] for r in rows], dtype=float)
    cells = d[:, :3] / LATTICE_NM
    return cells, d[:, 3], d[:, 4], [r[5] for r in rows]


def family(cells):
    """cube / plate / wire from the sorted edges (ratio threshold 1.35)."""
    a, b, c = np.sort(cells, axis=1).T
    return np.where(c / a < 1.35, "cube", np.where(b / a < 1.35, "wire", "plate"))


def check_rules(p):
    fails = []
    g = np.array(list(itertools.product(GRID, repeat=3)), dtype=float)
    e = np.asarray(mapping.E(g[:, 0], g[:, 1], g[:, 2], p), dtype=float)
    if not np.all(np.isfinite(e)):
        return ["non-finite energies on the grid"]
    if REQUIRE_SYMMETRY:
        dev = max(np.max(np.abs(e - mapping.E(*g[:, list(perm)].T, p))) for perm in
                  itertools.permutations(range(3)))
        if dev > 1e-4:
            fails.append(f"not symmetric under axis permutation (max dev {1000 * dev:.1f} meV)")
    n = len(GRID)
    cube = e.reshape(n, n, n)
    if any(np.any(np.diff(cube, axis=a) > 1e-6) for a in range(3)):
        fails.append("E increases with some edge length (must be non-increasing)")
    bulk = float(mapping.E(*np.full(3, 2000.0), p))
    if not BULK_RANGE_EV[0] <= bulk <= BULK_RANGE_EV[1]:
        fails.append(f"bulk limit {bulk:.3f} eV outside {BULK_RANGE_EV}")
    return fails


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "confinement_data.csv"
    X, Eobs, w, notes = load(path)
    names = list(mapping.PARAMS)
    spec = [v if isinstance(v, tuple) else (v, -np.inf, np.inf) for v in mapping.PARAMS.values()]
    x0, lo, hi = (np.array(c, dtype=float) for c in zip(*spec))
    P = lambda v: dict(zip(names, v))
    pred = lambda v, m: np.asarray(mapping.E(*X[m].T, P(v)), dtype=float)
    fit_m = w > 0

    def fit(m, start):
        if not names:
            return start
        res = lambda v: np.sqrt(w[m]) * (pred(v, m) - Eobs[m])
        return least_squares(res, start, bounds=(lo, hi), x_scale="jac").x

    best = fit(fit_m, x0)
    r = pred(best, np.ones(len(Eobs), bool)) - Eobs
    n, k = int(fit_m.sum()), len(names)
    rmse = 1000 * np.sqrt(np.average(r[fit_m] ** 2, weights=w[fit_m]))

    loo = []  # leave-one-out: refit without each row, predict it
    for i in np.flatnonzero(fit_m):
        m = fit_m.copy(); m[i] = False
        loo.append(pred(fit(m, best), np.arange(len(Eobs)) == i)[0] - Eobs[i])
    loo_rmse = 1000 * np.sqrt(np.mean(np.square(loo)))
    sse = np.sum(w[fit_m] * r[fit_m] ** 2)
    aicc = n * np.log(sse / n) + 2 * k + (2 * k * (k + 1) / (n - k - 1) if n - k - 1 > 0 else np.inf)
    held = ~fit_m
    held_rmse = 1000 * np.sqrt(np.mean(r[held] ** 2)) if held.any() else float("nan")
    fails = check_rules(P(best))

    print(f"\n{mapping.NAME}\n" + "-" * 72)
    print("params: " + ", ".join(f"{a}={b:.4g}" for a, b in zip(names, best)))
    print(f"fit rows {n}, params {k}   RMSE {rmse:.1f} meV   LOO-RMSE {loo_rmse:.1f} meV   "
          f"AICc {aicc:.1f}   held-out RMSE {held_rmse:.1f} meV")
    print("RULES: " + ("all pass" if not fails else "FAIL - " + "; ".join(fails)))
    fam, loo = family(X), np.array(loo)
    for f in ("cube", "plate", "wire"):
        mf, mh = fam[fit_m] == f, (fam == f) & held
        line = f"  {f:5}  fit n={mf.sum():2d}" + (f"  LOO-RMSE {1000 * np.sqrt(np.mean(loo[mf] ** 2)):5.1f} meV" if mf.any() else "")
        line += f"   held n={mh.sum():2d}" + (f"  RMSE {1000 * np.sqrt(np.mean(r[mh] ** 2)):5.1f} meV" if mh.any() else "")
        print(line)
    print("\nresid(meV)  role   x(nm)  y(nm)  z(nm)  E_obs  note   [sorted by |resid|]")
    for i in np.argsort(-np.abs(r)):
        nm = X[i] * LATTICE_NM
        print(f"{1000 * r[i]:+9.1f}  {'fit ' if fit_m[i] else 'held'}  {nm[0]:6.2f} {nm[1]:6.2f} "
              f"{nm[2]:6.2f}  {Eobs[i]:.3f}  {notes[i]}")

    with open("results.tsv", "a", encoding="utf-8") as f:
        if f.tell() == 0:
            f.write("time\tdata\tname\tk\tn\trmse_meV\tloo_meV\taicc\theld_meV\trules\tparams\n")
        f.write(f"{time.strftime('%Y-%m-%d %H:%M')}\t{path}\t{mapping.NAME}\t{k}\t{n}\t{rmse:.1f}\t"
                f"{loo_rmse:.1f}\t{aicc:.1f}\t{held_rmse:.1f}\t{'pass' if not fails else 'FAIL'}\t"
                + ", ".join(f"{a}={b:.4g}" for a, b in zip(names, best)) + "\n")


if __name__ == "__main__":
    main()
