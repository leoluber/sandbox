# Experiments

One script (and optional JSON config) per experiment, named by milestone, for example:
- `m2_library_forward.py`: forward pass for all `LIBRARY` populations (D1)
- `m3_library_inverse.py`: invert the library populations, clean and noisy (D2)
- `m4_random_recovery.py`: ≥ 200 random truths per class (D3)
- `m5_hparam_search.py`: losses × solvers × settings (D4)
- `m6_mixtures.py`: 2- and 3-population systems (D5)

Scripts start with `sys.path.insert(0, str(Path(__file__).resolve().parents[1]))` so `plmorph` imports without installing.

Each run writes `results/<milestone>/<run_id>/config.json` plus `data.npz` via `plmorph.io.save_run`. Seeds go in the config.
