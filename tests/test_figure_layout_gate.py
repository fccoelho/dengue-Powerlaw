"""The gate that keeps a committed figure and its caption the same vintage.

The bug these tests pin down
----------------------------
Figure 10 is a scatter the manuscript draws and the manifest does not hash: the
image is not an artefact of the database, so no step of ``verify_snapshot``
could tell whether the PNG beside a rewritten caption had been drawn by the code
the caption describes. A caption can therefore promise four panels over an image
with one, and every check still passes.

``generate_figures.fig10`` writes ``fig10_states.json`` beside the image with
the pixel shape of the file it saved. Comparing that record against the
committed PNG answers the question in one step.
"""
import json
import os

import numpy as np
import pytest

import verify_snapshot as vs


def _figdir(tmp_path):
    d = tmp_path / "Manuscript" / "figures"
    d.mkdir(parents=True)
    return d


def _png(path, height, width):
    """Write a genuine PNG of the given pixel shape."""
    image = pytest.importorskip("matplotlib.image")
    image.imsave(str(path), np.zeros((height, width, 3), dtype=float))


def test_png_shape_is_read_from_the_header(tmp_path):
    d = _figdir(tmp_path)
    _png(d / "fig10_pred_vs_obs.png", 834, 961)
    assert vs._png_shape(str(d / "fig10_pred_vs_obs.png")) == (961, 834)


def test_image_without_a_sidecar_is_refused(tmp_path):
    d = _figdir(tmp_path)
    _png(d / "fig10_pred_vs_obs.png", 679, 859)
    problems = vs.check_figure_layout(str(tmp_path))
    assert problems, "an image no current run produced must be refused"
    assert "fig10_states.json" in problems[0]
    assert "generate_figures.py 10" in problems[0]


def test_sidecar_from_a_different_layout_is_refused(tmp_path):
    d = _figdir(tmp_path)
    _png(d / "fig10_pred_vs_obs.png", 679, 859)
    (d / "fig10_states.json").write_text(json.dumps({"png_shape": [961, 834]}))
    problems = vs.check_figure_layout(str(tmp_path))
    assert problems
    assert "859x679" in problems[0] and "961x834" in problems[0]


def test_agreeing_sidecar_passes(tmp_path):
    d = _figdir(tmp_path)
    _png(d / "fig10_pred_vs_obs.png", 834, 961)
    (d / "fig10_states.json").write_text(json.dumps({"png_shape": [961, 834]}))
    assert vs.check_figure_layout(str(tmp_path)) == []


def test_absent_image_is_not_a_violation(tmp_path):
    _figdir(tmp_path)
    assert vs.check_figure_layout(str(tmp_path)) == []
