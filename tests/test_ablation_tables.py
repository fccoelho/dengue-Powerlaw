"""Tests for the ablation-to-LaTeX generator.

The defects these lock down are all ones that shipped: a contrast summary that
averaged over the 19-point quantile grid instead of over origins, a sign
convention that is easy to invert, and a table that would happily render numbers
from a superseded experiment.
"""

import json

import numpy as np
import pandas as pd
import pytest

import make_ablation_tables as mat


def _contrast_frame(n_tau=3, rel=(0.4, 0.2, -0.1, 0.3, -0.5)):
    """A minimal ablation frame: one contrast row per origin per tau."""
    rows = []
    for tau in np.linspace(0.05, 0.95, n_tau):
        for i, r in enumerate(rel):
            rows.append({
                "record_type": "contrast", "level": "state", "task": "size",
                "loss": "pinball", "tau": tau, "origin": 2021 + i,
                "contrast": "M3-M2", "rung": "M3",
                "value": r * 100.0, "rel_improvement": r,
                "ci_excludes_zero": abs(r) > 0.35,
                "improves": r > 0.35,
            })
    return pd.DataFrame(rows)


def test_contrast_summary_filters_by_tau():
    """Without the tau filter the median is over the grid, not over origins.

    That is not a cosmetic difference: with 19 quantiles the summary reports a
    median over 95 rows, which looks plausible and answers a different question.
    """
    df = _contrast_frame(n_tau=19, rel=(0.4, 0.2, -0.1, 0.3, -0.5))
    s = mat._contrast_summary(df, "size", 0.5)
    assert s["M3-M2"]["n_origins"] == 5
    assert s["M3-M2"]["median_rel"] == pytest.approx(0.2)


def test_exceedance_rows_without_tau_are_selectable():
    """Exceedance contrasts carry no tau; ``tau=None`` must select them."""
    df = _contrast_frame(n_tau=1).assign(tau=np.nan)
    s = mat._contrast_summary(df, "size", None)
    assert s["M3-M2"]["n_origins"] == 5


def test_positive_relative_improvement_means_the_later_rung_is_better():
    """The sign convention, pinned.

    run_ablation writes ``value = mean(L_ref - L_rung)``, so a positive value and
    a positive ``rel_improvement`` both mean the later rung has the lower loss.
    The manuscript states percentages on this scale, the figure plots it, and the
    caption was once written on the inverted one.
    """
    df = _contrast_frame(n_tau=1, rel=(0.5, 0.5, 0.5, 0.5, 0.5))
    s = mat._contrast_summary(df, "size", 0.05)["M3-M2"]
    assert s["median_rel"] > 0
    assert s["n_rel_positive"] == 5


def test_stale_source_is_not_used(tmp_path):
    """An artefact that predates the estimator change must not supply numbers."""
    df = _contrast_frame(n_tau=1)
    df.to_csv(tmp_path / "ablation_results.csv", index=False)
    (tmp_path / "ablation_results.meta.json").write_text(json.dumps({"model": "old"}))

    facts = mat.compute_facts(str(tmp_path))
    assert facts["sources"]["state_size"]["present"] is False
    assert "predates the current design" in facts["sources"]["state_size"]["problem"]
    assert not facts["size"]


def test_fresh_source_is_used(tmp_path):
    _contrast_frame(n_tau=1).to_csv(tmp_path / "ablation_results.csv", index=False)
    (tmp_path / "ablation_results.meta.json").write_text(json.dumps({
        "extrapolating_regressor": True, "classifier_model": "logistic"}))
    facts = mat.compute_facts(str(tmp_path))
    assert facts["sources"]["state_size"]["present"] is True


def test_exceed_table_marks_a_missing_block_instead_of_dropping_it():
    """A missing block is a visible row, never an absent one.

    Dropping the state block would silently delete the paper's only
    ``supported`` verdict; keeping stale numbers would be worse.
    """
    facts = {
        "exceed": {"state": None, "city": {
            "contrasts": {c: {"n_origins": 5, "median_diff": 0.002,
                              "n_diff_positive": 4, "n_same_direction_ci": 0,
                              "n_ci_excludes_zero": 0, "verdict": "not_supported"}
                          for c in mat.ADJACENT},
            "auc_median": {"M1": 0.58, "M2": 0.63, "M3": 0.65},
            "population_only_auc_median": 0.47, "n_units": 3976,
        }},
    }
    tex = mat.render_exceed_table(facts)
    assert "no committed artefact" in tex
    assert "run\\_ablation.py" in tex
    assert "Municipal (3{,}976 units)" in tex


def test_ablation_table_refuses_to_render_without_a_ladder():
    with pytest.raises(ValueError):
        mat.render_ablation_table({"size": {}})


def test_an_artefact_overwritten_by_another_task_is_not_used(tmp_path):
    """The defect behind "no size ladder in the artefacts".

    ``run_ablation.py`` used to default ``--output`` to ``ablation_results.csv``
    for every level and task, so a city exceedance run replaced the state size
    ladder in place. The columns still matched, so a column-only check called the
    file fresh: Table 3 lost its source while every check stayed green, and the
    failure surfaced as a bare "no size ladder" from the renderer.
    """
    df = _contrast_frame(n_tau=1).assign(task="exceedance", level="city",
                                         loss="brier", tau=np.nan)
    df.to_csv(tmp_path / "ablation_results.csv", index=False)
    (tmp_path / "ablation_results.meta.json").write_text(json.dumps({
        "extrapolating_regressor": True, "classifier_model": "logistic"}))

    facts = mat.compute_facts(str(tmp_path))
    src = facts["sources"]["state_size"]
    assert src["present"] is False
    assert "another run has written over it" in src["problem"]
    assert not facts["size"]
    with pytest.raises(ValueError, match="cannot render Table 3"):
        mat.render_ablation_table(facts)


def test_render_error_names_the_file_and_the_fix():
    """``no size ladder`` alone sent a reader hunting a bug in the renderer.

    The file was there all along, holding the wrong experiment; the message has
    to name it and give the command that rebuilds it.
    """
    with pytest.raises(ValueError) as ei:
        mat.render_ablation_table(
            {"size": {}, "sources": {"state_size": {"problem": "gone"}}})
    msg = str(ei.value)
    assert "ablation_results.csv" in msg
    assert "run_ablation.py --level state" in msg


def test_render_error_offers_the_restore_before_the_refit():
    """A refit is the expensive fix and it moves every hash; restore first.

    The refit command alone is also a dead end now: ``guard_output`` refuses to
    write over the artefact that caused the problem, so following that advice
    verbatim fails with a second, unrelated-looking error.
    """
    with pytest.raises(ValueError) as ei:
        mat.render_ablation_table(
            {"size": {}, "sources": {"state_size": {"problem": "gone"}}})
    msg = str(ei.value)
    assert "git checkout -- ablation_results.csv" in msg
    assert msg.index("git checkout") < msg.index("run_ablation.py")
    assert "--force" in msg
