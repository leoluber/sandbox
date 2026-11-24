"""M0 acceptance tests (LOCKED)."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plmorph.io import load_run, save_run
from plmorph.viz import apply_style, figsize, savefig


def test_m0_savefig_writes_pdf_and_png(tmp_path):
    apply_style()
    fig, ax = plt.subplots(figsize=figsize("single"))
    ax.plot([0, 1], [0, 1])
    pdf, png = savefig(fig, "t", out_dir=tmp_path)
    assert pdf.exists() and png.exists()


def test_m0_figsize_widths():
    assert abs(figsize("single")[0] - 89 / 25.4) < 1e-9
    assert abs(figsize("double")[0] - 183 / 25.4) < 1e-9


def test_m0_io_roundtrip(tmp_path):
    cfg = {"seed": 1, "name": "x"}
    arr = {"a": np.arange(5.0), "b": np.eye(3)}
    save_run(tmp_path / "run", cfg, arr)
    assert json.loads((tmp_path / "run" / "config.json").read_text()) == cfg
    cfg2, arr2 = load_run(tmp_path / "run")
    assert cfg2 == cfg
    np.testing.assert_array_equal(arr2["b"], arr["b"])
