"""The bibliography gate: every citation resolves, and no entry is dead weight.

A citation with no `.bib` entry does not break the LaTeX build -- BibTeX warns,
the page prints "[?]", and the manuscript ships. So the invariant is checked
here instead, in both directions. The parsing cases below are pinned because a
gate that miscounts is worse than no gate: a commented-out citation or an
optional argument that is read as a key invents a failure, and invented
failures are how a gate stops being read.
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "Manuscript", "verification"))

import verify_refs as vr  # noqa: E402


def _real() -> dict:
    """Run the gate over the committed manuscript and bibliography."""
    return vr.compare(
        vr.DEFAULT_TEX.read_text(encoding="utf-8"),
        vr.DEFAULT_BIB.read_text(encoding="utf-8"),
    )


def test_the_manuscript_resolves_every_citation():
    """No "[?]" can reach the PDF from the committed pair."""
    rep = _real()
    assert rep["missing"] == []
    assert rep["n_cited"] > 0
    assert rep["n_cited"] == rep["n_entries"]


def test_no_entry_is_dead_weight():
    """An uncited entry names either a typo's intended target or dead weight."""
    assert _real()["orphan"] == []


def test_commented_out_citation_is_ignored():
    tex = "real \\citep{a}.\n% \\citep{b} disabled\n"
    assert vr.citation_keys(tex) == {"a"}


def test_optional_arguments_are_not_mistaken_for_keys():
    tex = r"\citep[see][p.~3]{a,b} and \citep*{c}"
    assert vr.citation_keys(tex) == {"a", "b", "c"}


def test_nocite_star_is_not_a_key():
    assert vr.citation_keys(r"\nocite{*}") == set()


def test_keys_split_on_commas_and_newlines():
    tex = "\\citep{a, b,\n c}"
    assert vr.citation_keys(tex) == {"a", "b", "c"}


def test_directives_are_not_entries():
    bib = '@string{acm = "ACM"}\n@comment{note}\n@article{real,\n  title={x}\n}\n'
    assert vr.bib_keys(bib) == {"real"}


def test_a_key_with_no_entry_fails_the_run(tmp_path, capsys):
    tex = tmp_path / "m.tex"
    bib = tmp_path / "r.bib"
    tex.write_text(r"\citep{ghost}", encoding="utf-8")
    bib.write_text("@article{real,\n  title={x}\n}\n", encoding="utf-8")

    assert vr.main(["--tex", str(tex), "--bib", str(bib)]) == 1
    assert "MISSING" in capsys.readouterr().out


def test_an_uncited_entry_warns_until_strict(tmp_path):
    tex = tmp_path / "m.tex"
    bib = tmp_path / "r.bib"
    tex.write_text(r"\citep{real}", encoding="utf-8")
    bib.write_text(
        "@article{real,\n  title={x}\n}\n@article{spare,\n  title={y}\n}\n",
        encoding="utf-8",
    )

    assert vr.main(["--tex", str(tex), "--bib", str(bib)]) == 0
    assert vr.main(["--tex", str(tex), "--bib", str(bib), "--strict"]) == 1
