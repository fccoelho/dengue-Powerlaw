"""Prose is tracked by the snapshot but not policed by it.

Every test here exists because a hash over ``Manuscript/main.tex`` reported
every draft edit as drift, and the only repair a hash offers is to re-freeze --
step 10 of the README, run *before* the verification in step 11. The gate was
teaching the habit that defeats it.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

import make_tables
import provenance


def _fake_repo(tmp_path: Path) -> Path:
    """A minimal repository: the tracked artifacts plus a real, tiny database."""
    root = tmp_path / "repo"
    (root / "Manuscript").mkdir(parents=True)
    with sqlite3.connect(root / "powerlaw_results.db") as con:
        con.execute("CREATE TABLE powerlaw_fits (i INTEGER)")
        con.execute("INSERT INTO powerlaw_fits VALUES (1)")
    for rel in provenance.TRACKED_ARTIFACTS:
        if rel.endswith(".db"):
            continue
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(f"{rel}\n", encoding="utf-8")
    (root / "Manuscript" / "main.tex").write_text("original prose\n", encoding="utf-8")
    return root


def _freeze(root: Path) -> dict:
    """Freeze and return the manifest that was written."""
    provenance.write_manifest(root=str(root))
    return json.loads((root / provenance.MANIFEST_NAME).read_text())


def _manifest(root: Path) -> dict:
    return json.loads((root / provenance.MANIFEST_NAME).read_text())


def test_editing_prose_is_not_drift(tmp_path):
    root = _fake_repo(tmp_path)
    _freeze(root)
    (root / "Manuscript" / "main.tex").write_text("edited prose\n", encoding="utf-8")

    rep = provenance.compare(_manifest(root), root=str(root))

    assert rep["changed"] == []
    assert rep["prose"] == {"Manuscript/main.tex": "changed"}
    provenance.assert_snapshot(root=str(root))  # must not raise


def test_unchanged_prose_is_reported_ok(tmp_path):
    root = _fake_repo(tmp_path)
    _freeze(root)
    rep = provenance.compare(_manifest(root), root=str(root))
    assert rep["prose"] == {"Manuscript/main.tex": "ok"}


def test_missing_prose_still_raises(tmp_path):
    root = _fake_repo(tmp_path)
    _freeze(root)
    (root / "Manuscript" / "main.tex").unlink()

    with pytest.raises(provenance.SnapshotDrift) as err:
        provenance.assert_snapshot(root=str(root))
    assert "Manuscript/main.tex" in str(err.value)
    assert "not on disk" in str(err.value)


def test_version1_manifest_is_routed_by_path(tmp_path):
    """An old manifest lists the prose under ``artifacts``; that is not drift."""
    root = _fake_repo(tmp_path)
    man = _freeze(root)
    old = json.loads(json.dumps(man))
    old["manifest_version"] = 1
    old["artifacts"]["Manuscript/main.tex"] = old["prose"]["Manuscript/main.tex"]
    del old["prose"]
    (root / "Manuscript" / "main.tex").write_text("edited prose\n", encoding="utf-8")

    rep = provenance.compare(old, root=str(root))

    assert rep["changed"] == []
    assert rep["prose"] == {"Manuscript/main.tex": "changed"}


def test_write_records_prose_apart(tmp_path, monkeypatch):
    root = _fake_repo(tmp_path)
    monkeypatch.setattr(provenance, "claim_gate", lambda *a, **k: (True, "60 claims"))
    assert provenance.main(["--write", "--root", str(root)]) == 0

    man = _manifest(root)
    assert man["manifest_version"] == 2
    assert man["prose"] == {"Manuscript/main.tex": provenance.sha256_file(
        str(root / "Manuscript" / "main.tex"))}
    assert "Manuscript/main.tex" not in man["artifacts"]


def test_write_refuses_when_the_gate_is_red(tmp_path, monkeypatch):
    root = _fake_repo(tmp_path)
    _freeze(root)
    before = (root / provenance.MANIFEST_NAME).read_bytes()
    monkeypatch.setattr(
        provenance, "claim_gate",
        lambda *a, **k: (False, "1 claim(s) disagree with db: size window"))

    assert provenance.main(["--write", "--root", str(root)]) == 1
    assert (root / provenance.MANIFEST_NAME).read_bytes() == before


def test_force_freezes_over_a_red_gate(tmp_path, monkeypatch):
    root = _fake_repo(tmp_path)
    monkeypatch.setattr(
        provenance, "claim_gate",
        lambda *a, **k: (False, "1 claim(s) disagree with db: size window"))
    assert provenance.main(["--write", "--root", str(root), "--force"]) == 0
    assert (root / provenance.MANIFEST_NAME).exists()


def test_force_and_allow_missing_only_apply_to_write(tmp_path):
    root = _fake_repo(tmp_path)
    _freeze(root)
    assert provenance.main(["--check", "--force", "--root", str(root)]) == 2
    assert provenance.main(["--check", "--allow-missing", "--root", str(root)]) == 2


def test_check_reports_prose_and_exits_zero(tmp_path, capsys):
    root = _fake_repo(tmp_path)
    _freeze(root)
    (root / "Manuscript" / "main.tex").write_text("edited prose\n", encoding="utf-8")

    assert provenance.main(["--check", "--root", str(root)]) == 0
    out = capsys.readouterr().out
    assert "snapshot OK" in out
    assert "Manuscript/main.tex: edited since the freeze" in out


def test_gate_fails_closed_when_the_facts_cannot_be_derived(tmp_path):
    root = _fake_repo(tmp_path)
    (root / "powerlaw_results.db").unlink()
    passed, summary = provenance.claim_gate(root=str(root))
    assert passed is False
    assert "cannot derive the facts" in summary


def test_make_tables_check_sees_an_absent_manuscript(tmp_path):
    """`0 claims verified` is not `no claims violated`."""
    root = _fake_repo(tmp_path)
    assert make_tables.main(["--check", "--tex", str(root / "nope.tex")]) == 2


def test_make_tables_audit_reports_an_absent_file(tmp_path):
    facts = {"levels": {}, "national": {}, "state": {}, "exceedance_pct": {},
             "warnings": [], "total_fits": 0}
    audit = make_tables.audit_prose(str(tmp_path / "nope.tex"), facts)
    assert audit["error"]
    assert not audit["stale"] and not audit["missing"]  # the trap this guards
