"""The survival block of the prose audit.

``make_tables.py --check`` is the only thing standing between the manuscript and
a hand-typed number, and until the survival section existed it audited the
database alone. These tests pin the three ways that block can go quiet: the
artefacts disappear, the prose drifts from them, or the skip is reported as a
pass.
"""

import os
import re

import pytest

import make_tables as mt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEX = os.path.join(ROOT, "Manuscript", "main.tex")


def test_survival_claims_are_derived_not_typed():
    """Every claim is a formatted string, and they cover the whole section."""
    claims = mt.survival_claims(root=ROOT)
    assert len(claims) >= 20
    assert all(isinstance(c["want"], str) and c["want"] for c in claims)
    # A literal brace means an f-string placeholder was not filled.
    assert not [c for c in claims if "{" in c["want"] or "}" in c["want"]]


def test_manuscript_carries_every_survival_number():
    """The invariant the audit exists to enforce, asserted directly here.

    Whitespace is collapsed first: the manuscript is reflowed between drafts, so
    a claim that only survives the original line breaks is a coincidence.
    """
    tex = re.sub(r"\s+", " ", open(TEX, encoding="utf-8").read())
    missing = [c["want"] for c in mt.survival_claims(root=ROOT)
               if c["want"] not in tex]
    assert not missing


def test_absent_artefacts_skip_the_block(tmp_path):
    """A reader who has not run the script gets no block, not a failure."""
    assert mt.survival_facts(root=str(tmp_path)) is None
    assert mt.survival_claims(root=str(tmp_path)) == []


def test_partial_artefacts_also_skip(tmp_path):
    """Half a directory is not enough: the facts would mix two runs."""
    d = tmp_path / mt.SURV_DIR
    d.mkdir(parents=True)
    for name in mt.SURV_FILES[:-1]:
        (d / name).write_text("a\n", encoding="utf-8")
    assert mt.survival_facts(root=str(tmp_path)) is None


def test_two_element_spelling():
    """Small counts are words in the prose; the audit must look for the word."""
    assert mt._spell(2) == "two"
    assert mt._spell(6) == "six"
    assert mt._spell(12) == "12"
