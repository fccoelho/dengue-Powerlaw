"""Which artifacts belong to which data snapshot.

The repository holds several views of the same fit results: the SQLite
database (source of truth, written by ``fitpl.py``), the LaTeX tables and the
figures. Nothing in the code tied them together, so a later ``fitpl.py`` run
silently put the database ahead of the manuscript and of ``parquet_cache/``.

This module is that missing tie. A manifest records the SHA-256 of the
database and of every derived artifact, plus the database inventory. Any
artifact whose bytes no longer match the manifest was produced by a different
run and must be regenerated, not trusted.

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
MANIFEST_VERSION = 1

#: Artifacts the manuscript is built from, relative to the repo root.
TRACKED_ARTIFACTS = [
    "powerlaw_results.db",
    "Manuscript/main.tex",
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
        "artifacts": {
            rel: sha256_file(os.path.join(root, rel)) for rel in TRACKED_ARTIFACTS
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
    "untracked": [...], "ok": [...]}``. A changed *input* (the database) is
    fatal; a changed *output* is regenerable, but only if it is regenerated.
    """
    db_rel = manifest["database"]["path"]
    db_now = sha256_file(os.path.join(root, db_rel))
    rep = {"db": "ok" if db_now == manifest["database"]["sha256"] else "changed",
           "changed": [], "missing": [], "ok": [], "untracked": []}
    if rep["db"] != "ok":
        rep["changed"].append(db_rel)
    for rel, recorded in manifest["artifacts"].items():
        now = sha256_file(os.path.join(root, rel))
        if now == "missing":
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
    if rep["changed"] or rep["missing"]:
        raise SnapshotDrift(
            "artifacts out of step with the database: "
            + ", ".join(rep["changed"] + rep["missing"])
        )
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", default="powerlaw_results.db")
    ap.add_argument("--root", default=".")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", action="store_true", help="freeze current state")
    ap.add_argument("--allow-missing", action="store_true",
                    help="with --write: freeze even when a tracked artefact is "
                         "absent. Do not use for a manuscript build")
    g.add_argument("--check", action="store_true", help="verify against manifest")
    g.add_argument("--inventory", action="store_true", help="print table counts")
    a = ap.parse_args(argv)

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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
