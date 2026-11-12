"""Regression tests for the design gate that guards the figure module.

The bug these tests pin down
----------------------------
``verify_snapshot.check_design`` asked its questions of the ablation
``.meta.json`` files and of the ablation CSVs. None of those is what a figure
reads: ``generate_figures.py`` refits on its own for every panel that plots one
point per unit, so the artefacts could declare the reported estimator while the
panel beside them was drawn with one the manuscript had removed. Figure 13 kept
a random forest for two revisions that way -- every gate was answering a
question about a file the figure never opens.

These tests assert what the design requires:

* a figure module that constructs a tree ensemble is refused;
* the import that would supply one is refused too;
* a colour literal is not mistaken for a comment, and a comment is not
  mistaken for code;
* the repository's own figure module passes.
"""
import os
import textwrap

import verify_snapshot as vs


def _root_with(tmp_path, source):
    d = tmp_path / "Manuscript" / "figures"
    d.mkdir(parents=True)
    (d / "generate_figures.py").write_text(textwrap.dedent(source))
    return str(tmp_path)


def test_tree_ensemble_in_figure_module_is_refused(tmp_path):
    root = _root_with(tmp_path, """
        from sklearn.ensemble import RandomForestRegressor

        def panel():
            return RandomForestRegressor(n_estimators=150)
    """)
    problems = vs.check_figure_sources(root)
    assert problems, "a figure module fitting a random forest must be refused"
    assert "RandomForestRegressor" in problems[0]


def test_boosted_import_in_figure_module_is_refused(tmp_path):
    root = _root_with(tmp_path, """
        import lightgbm
    """)
    problems = vs.check_figure_sources(root)
    assert problems
    assert "import lightgbm" in problems[0]


def test_colour_literal_survives_the_comment_strip(tmp_path):
    root = _root_with(tmp_path, """
        def panel(ax):
            ax.plot([0], [0], color="#d62728")  # median APE
    """)
    assert vs.check_figure_sources(root) == []


def test_commented_mention_is_not_a_violation(tmp_path):
    root = _root_with(tmp_path, """
        def panel():
            # used to fit a RandomForestRegressor here; removed on purpose
            return None
    """)
    assert vs.check_figure_sources(root) == []


def test_missing_module_is_not_a_violation(tmp_path):
    assert vs.check_figure_sources(str(tmp_path)) == []


def test_repository_figure_module_passes():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    assert vs.check_figure_sources(here) == []
