"""Which artifacts belong to which data snapshot.

The repository holds several views of the same fit results: the SQLite
database (source of truth, written by ``fitpl.py``), the LaTeX tables and the
figures. Nothing in the code tied them together, so a later ``fitpl.py`` run
silently put the database ahead of the manuscript and of ``parquet_cache/``.

This module is that missing tie. A manifest records the SHA-256 of the
database and of every derived artifact, plus the database inventory. Any
artifact whose bytes no longer match the manifest was produced by a different
run and must be regenerated, not trusted.

Prose is the exception, and it is hashed apart (``PROSE_ARTIFACTS``). Editing
a sentence is not a stale artifact, and the only remedy a hash offers --
re-freezing -- is the command that would launder a real mismatch. So the
manuscript is reported, never policed, and its numbers are audited by
``make_tables.py --check`` instead. ``--write`` runs that audit before it
records anything: the manifest holds a digest, not a verdict, so the verdict
has to be taken at the moment of freezing.

Typical use::

    from provenance import assert_snapshot
    assert_snapshot()            # raises SnapshotDrift, never warns

    python provenance.py --write # record the current state as THE snapshot
    python provenance.py --check # verify, non-zero exit on drift
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
from datetime import datetime, timezone

MANIFEST_NAME = "snapshot_manifest.json"
#: 2 moved ``Manuscript/main.tex`` out of ``artifacts`` into ``prose``. A
#: version-1 manifest is still read, routed by path, so upgrading this module
#: does not by itself report drift.
MANIFEST_VERSION = 2

#: Manuscript audited by ``--write`` before it freezes anything.
GATE_TEX = os.path.join("Manuscript", "main.tex")

#: Tracked files that no script produces: the manuscript text itself.
#:
#: They are hashed and reported, never policed. "Regenerate, do not trust"
#: assumes a producer, and prose has none: editing a sentence is not a stale
#: artifact. Policing it made every draft change a DRIFT report whose only
#: available remedy was ``--write`` -- and step 10 of the README runs that
#: before step 11 verifies anything, so the habit it taught was free to hide a
#: genuine mismatch. The digest is still recorded, so the snapshot keeps saying
#: which text stood in front of which database at freeze time; it just stops
#: calling an edit a defect.
PROSE_ARTIFACTS = [
    "Manuscript/main.tex",
]

#: Artifacts the manuscript is built from, relative to the repo root. Every
#: entry is produced by a script, which is what makes regenerating it an
#: available response to drift.
TRACKED_ARTIFACTS = [
    "powerlaw_results.db",
    "Manuscript/generated/facts.json",
    "Manuscript/generated/tab_main.tex",
    "Manuscript/generated/tab_ablation.tex",
    "Manuscript/generated/tab_exceed.tex",
    "Manuscript/generated/ablation_facts.json",
    "ablation_results.csv",
    "ablation_results.meta.json",
    "ablation_city.csv",
    "ablation_city.meta.json",
    # Quoted by Table 4's state block and, until now, not hashed at all: the
    # manuscript could cite a run whose output nothing in the repository had to
    # produce. Tracking it makes its absence a named failure instead of prose
    # nobody can check.
    "ablation_state_exceedance.csv",
    "ablation_state_exceedance.meta.json",
    "classifier_results.parquet",
    "classifier_results_metrics.csv",
    "classifier_results_importance.csv",
    "classifier_results.meta.json",
    # The survival reading. Same argument as the state-exceedance table above:
    # the prose now quotes numbers that no single table in the manuscript
    # contains, so the files they come from have to belong to the snapshot. A
    # stale copy from an earlier panel is then a named failure rather than a
    # paragraph nobody can check.
    "analysis/survival/exceedance_within_season.csv",
    "analysis/survival/exceedance_transport.csv",
    "analysis/survival/duration_models.csv",
    "analysis/survival/duration_summary.json",
]


class SnapshotDrift(RuntimeError):
    """An artifact does not belong to the recorded snapshot."""


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    """Hex digest of a file, or ``"missing"`` when it does not exist."""
    if not os.path.exists(path):
        return "missing"
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def db_inventory(db_path: str = "powerlaw_results.db") -> dict:
    """Row count and schema digest for every table in the fits database."""
    if not os.path.exists(db_path):
        raise SnapshotDrift(f"fits database not found: {db_path}")
    out = {}
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
        tables = [
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
        ]
        for t in tables:
            n = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            cols = [r[1] for r in conn.execute(f"PRAGMA table_info({t})")]
            schema = hashlib.sha256(",".join(cols).encode()).hexdigest()[:16]
            out[t] = {"rows": n, "schema": schema, "columns": cols}
    return out


def build_manifest(db_path: str = "powerlaw_results.db", root: str = ".") -> dict:
    """Current state of the database and of every tracked artifact."""
    return {
        "manifest_version": MANIFEST_VERSION,
        "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "database": {
            "path": db_path,
            "sha256": sha256_file(os.path.join(root, db_path)),
            "inventory": db_inventory(os.path.join(root, db_path)),
        },
        "prose": {
            rel: sha256_file(os.path.join(root, rel)) for rel in PROSE_ARTIFACTS
        },
        "artifacts": {
            rel: sha256_file(os.path.join(root, rel))
            for rel in TRACKED_ARTIFACTS
            if rel not in PROSE_ARTIFACTS
        },
    }


def write_manifest(db_path: str = "powerlaw_results.db", root: str = ".") -> dict:
    """Record the current state as the snapshot. Overwrites any previous one."""
    man = build_manifest(db_path, root)
    path = os.path.join(root, MANIFEST_NAME)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(man, fh, indent=2, sort_keys=False)
        fh.write("\n")
    return man


def load_manifest(root: str = ".") -> dict:
    path = os.path.join(root, MANIFEST_NAME)
    if not os.path.exists(path):
        raise SnapshotDrift(
            f"{MANIFEST_NAME} not found; run `python provenance.py --write` first"
        )
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def compare(manifest: dict, root: str = ".") -> dict:
    """Classify every tracked artifact against the manifest.

    Returns ``{"db": ..., "changed": [...], "missing": [...],
    "untracked": [...], "ok": [...], "prose": {rel: state}}``. A changed
    *input* (the database) is fatal; a changed *output* is regenerable, but
    only if it is regenerated. A changed *prose* file is neither, so it is
    reported in ``prose`` and never makes the snapshot drift.
    """
    db_rel = manifest["database"]["path"]
    db_now = sha256_file(os.path.join(root, db_rel))
    rep = {"db": "ok" if db_now == manifest["database"]["sha256"] else "changed",
           "changed": [], "missing": [], "ok": [], "untracked": [], "prose": {}}
    if rep["db"] != "ok":
        rep["changed"].append(db_rel)
    # Route by path, not by the section a manifest happened to store it in: a
    # version-1 manifest still lists the prose under ``artifacts``, and reading
    # the code upgrade as drift would defeat the point of tolerating it.
    recorded_all = dict(manifest["artifacts"])
    recorded_all.update(manifest.get("prose") or {})
    for rel, recorded in recorded_all.items():
        now = sha256_file(os.path.join(root, rel))
        if rel in PROSE_ARTIFACTS:
            rep["prose"][rel] = ("missing" if now == "missing"
                                 else "ok" if now == recorded else "changed")
        elif now == "missing":
            rep["missing"].append(rel)
        elif recorded == "missing":
            rep["untracked"].append(rel)
        elif now != recorded:
            rep["changed"].append(rel)
        else:
            rep["ok"].append(rel)
    return rep


def assert_snapshot(db_path: str = "powerlaw_results.db", root: str = ".") -> dict:
    """Raise ``SnapshotDrift`` unless every artifact matches the manifest."""
    man = load_manifest(root)
    rep = compare(man, root)
    if rep["db"] != "ok":
        raise SnapshotDrift(
            f"{db_path} changed since the snapshot was frozen. The manuscript "
            f"numbers were derived from a different database: regenerate the "
            f"tables (`python make_tables.py`) or re-freeze deliberately "
            f"(`python provenance.py --write`)."
        )
    absent = [rel for rel, state in rep["prose"].items() if state == "missing"]
    if absent:
        raise SnapshotDrift(
            "prose recorded in the snapshot is not on disk: " + ", ".join(absent)
            + ". An edit is not drift, but a manuscript that disappeared is, and"
              " the digest is the only thing that notices."
        )
    if rep["changed"] or rep["missing"]:
        raise SnapshotDrift(
            "artifacts out of step with the database: "
            + ", ".join(rep["changed"] + rep["missing"])
        )
    return rep


def claim_gate(db_path: str = "powerlaw_results.db", tex_path: str = GATE_TEX,
               root: str = ".") -> tuple[bool, str]:
    """Re-derive the manuscript's numbers from the database: (passed, summary).

    Freezing is the only moment at which "this prose was checked against this
    database" can be recorded at all, because the manifest stores a digest and
    not a verdict. So the verdict is taken here, before the digest is written.

    Deliberately not a subprocess: the audit is two library calls, and a
    failure to import them has to fail *closed*. An `except: pass` around a
    missing module is how a gate stops being one.
    """
    try:
        import make_tables
    except Exception as e:  # noqa: BLE001 -- refuse to freeze unchecked
        return False, f"cannot import make_tables ({e}); nothing was audited"
    try:
        facts = make_tables.compute_facts(os.path.join(root, db_path))
    except Exception as e:  # noqa: BLE001 -- a gate that cannot run is not green
        return False, f"cannot derive the facts from {db_path}: {e}"
    audit = make_tables.audit_prose(os.path.join(root, tex_path), facts)
    if audit.get("error"):
        return False, audit["error"]
    bad = audit["stale"] + audit["missing"]
    if bad:
        named = ", ".join(entry["name"] for entry in bad)
        return False, f"{len(bad)} claim(s) disagree with {db_path}: {named}"
    return True, f"{len(audit['ok'])} claims verified against {db_path}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", default="powerlaw_results.db")
    ap.add_argument("--root", default=".")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", action="store_true", help="freeze current state")
    ap.add_argument("--allow-missing", action="store_true",
                    help="with --write: freeze even when a tracked artefact is "
                         "absent. Do not use for a manuscript build")
    ap.add_argument("--force", action="store_true",
                    help="with --write: freeze even when the manuscript fails "
                         "the claim audit. Do not use for a manuscript build")
    ap.add_argument("--tex", default=GATE_TEX,
                    help="manuscript audited by --write")
    g.add_argument("--check", action="store_true", help="verify against manifest")
    g.add_argument("--inventory", action="store_true", help="print table counts")
    a = ap.parse_args(argv)

    if (a.allow_missing or a.force) and not a.write:
        print("--allow-missing and --force only apply to --write", file=sys.stderr)
        return 2

    if a.inventory:
        inv = db_inventory(os.path.join(a.root, a.db))
        w = max(len(k) for k in inv)
        for t, d in inv.items():
            print(f"  {t:<{w}}  {d['rows']:>9,}")
        return 0

    if a.write:
        # Freezing a state that is missing a tracked artefact makes the gap
        # permanent: every later --check compares against a snapshot in which
        # the absent file is simply "missing", which is indistinguishable from
        # a file that was deleted afterwards.
        gone = [p for p in TRACKED_ARTIFACTS
                if not os.path.exists(os.path.join(a.root, p))]
        if gone and not a.allow_missing:
            print("refusing to freeze an incomplete snapshot:", file=sys.stderr)
            for p in gone:
                print(f"  absent: {p}", file=sys.stderr)
            print("Rebuild it (make_ablation_tables.py --sources names the "
                  "command), or pass --allow-missing.", file=sys.stderr)
            return 1
        passed, summary = claim_gate(a.db, a.tex, a.root)
        if not passed and not a.force:
            print("refusing to freeze: the manuscript is not in step with the "
                  "database", file=sys.stderr)
            print(f"  {summary}", file=sys.stderr)
            print("Run `python make_tables.py --check` for the full list and fix "
                  "the prose, or pass --force to record a snapshot you know is "
                  "inconsistent.", file=sys.stderr)
            return 1
        print(f"claim gate: {'OK' if passed else 'FAILED (forced)'} -- {summary}")
        man = write_manifest(a.db, a.root)
        print(f"frozen {MANIFEST_NAME}")
        print(f"  db sha256 {man['database']['sha256'][:16]}...")
        for t, d in man["database"]["inventory"].items():
            print(f"  {t:<32} {d['rows']:>9,}")
        return 0

    try:
        rep = assert_snapshot(a.db, a.root)
    except SnapshotDrift as e:
        print(f"DRIFT: {e}", file=sys.stderr)
        return 1
    print(f"snapshot OK ({len(rep['ok'])} artifacts match)")
    for rel, state in sorted(rep["prose"].items()):
        print(f"  prose {rel}: " + (
            "unchanged since the freeze" if state == "ok"
            else "edited since the freeze -- not drift; `python make_tables.py"
                 " --check` is what audits it"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
