# Rules for the coding agent

Read `SPEC.md`, `TASKS.md` and `DECISIONS.md` before starting any work. `PLAN.md` and `LeosNotes.md` give background.


## Python Environment
The project uses the Conda environment:

C:\Users\leolu\miniconda3\envs\research

IMPORTANT: Always execute Python through conda run -p to ensure the environment and its DLL dependencies are correctly loaded.

Example:

conda run -p "C:\Users\leolu\miniconda3\envs\research" python script.py

For package installation:

conda install -p "C:\Users\leolu\miniconda3\envs\research" PACKAGE

Never launch the environment's python.exe directly. Never use system Python or the VS Code Python runner.

## What you may and may not edit

| May edit | Must NOT edit |
|---|---|
| `plmorph/` (implementations) | `tests/acceptance/` (locked) |
| `experiments/` | `SPEC.md`, `PLAN.md`, `LeosNotes.md`, `DECISIONS.md` |
| `figures/scripts/`, `figures/out/` | `legacy/` (read-only reference) |
| `tests/unit/` | any signature or dataclass field marked `# INTERFACE` |
| `RESULTS.md`, `figures/GALLERY.md` | |

**Need an interface change?** Do not implement it. Write a short proposal (what, why, alternatives) under "Interface proposals" in your `RESULTS.md` entry and stop that line of work.

## How to work

1. Do **one milestone at a time, in order** (M0, M1, …). Do not start the next milestone until the current one is done.
2. A milestone is done when **all three** hold:
   - `pytest tests/acceptance -k <mN>` passes (for example `-k m2`);
   - the figures listed for that milestone in `TASKS.md` exist in `figures/out/` and are listed in `figures/GALLERY.md`;
   - a `RESULTS.md` entry is written using the template.
3. Never weaken, skip or `xfail` an acceptance test. If you think a test is wrong, say so in `RESULTS.md`.
4. **Do not run any git commands** (no add, commit, push, branch). The user reviews and commits. At the end of a milestone, suggest a one-line commit message at the bottom of your `RESULTS.md` entry.

## Code conventions

- Python ≥ 3.11, type hints everywhere, NumPy-style docstrings that state **array shapes and units** (unit cells, eV).
- Float64 throughout. Explicit `numpy.random.Generator` seeds, no global random state.
- No global mutable state. Configs are plain dataclasses or JSON.
- Keep modules general: a new solver, loss or map must plug in through the existing interfaces without editing callers.
- The binning operator is **sparse** (see SPEC §4). Never build a dense (grid × energy) array for the full grid.

## Results and figures

- Experiments write to `results/<milestone>/<run_id>/` with `config.json` plus `data.npz` (use `plmorph.io`).
- **Plotting scripts only read from `results/`.** They never run solvers.
- Every figure uses `plmorph.viz.style` (`apply_style()`, `figsize()`, `savefig()`), the fixed colour semantics in SPEC §9, and is saved as PDF plus PNG in `figures/out/`.
- One script per figure: `figures/scripts/fig_<id>_<name>.py`. Start from `fig_template.py`.

## RESULTS.md entry template (≤ 30 lines)

```
## MN — <title> (<date>)
Status: done | partial | blocked
Acceptance tests: <passed>/<total>
Key numbers: <3–6 bullet points>
Figures: <ids + paths>
Open problems: <bullets>
Interface proposals: <none | short proposal>
```

## Commands

PowerShell, always from the repo root, always via `conda run -p` (see "Python Environment"). Nothing needs installing: pytest finds `plmorph` through `pythonpath = ["."]` in `pyproject.toml`, and every script in `experiments/` and `figures/scripts/` starts with `sys.path.insert(0, <repo root>)` (see `figures/scripts/fig_template.py`). Do not run `pip install -e .`.

```powershell
conda run -p "C:\Users\leolu\miniconda3\envs\research" python -m pytest tests/acceptance -k m1   # acceptance tests for one milestone
conda run -p "C:\Users\leolu\miniconda3\envs\research" python -m pytest tests/unit               # your own tests
conda run -p "C:\Users\leolu\miniconda3\envs\research" python experiments/run_m1_map.py
conda run -p "C:\Users\leolu\miniconda3\envs\research" python figures/scripts/fig_f1_ruler.py
```
