"""One command that answers: does the manuscript match the database?

Three independent checks, because the repository drifted in three ways at once:

1. **Manifest** -- ``snapshot_manifest.json`` records the SHA-256 of the fits
   database and of every derived artifact. A mismatch means something was
   regenerated from a different run and must not be trusted together.
2. **Prose** -- ``make_tables.py --check`` re-derives every quantity the
   manuscript states and looks for stale literals. This is the check that was
   missing: Table 1 was hand-typed for months while Methods claimed no
   statistic was transcribed by hand.
3. **Orphan exports** -- ``parquet_cache/`` held Parquet exports of the same
   tables that no script read, so its row counts silently disagreed with the
   database (5,244 vs 5,247; 40,616 vs 41,635). Dead copies of the source of
   truth are how this class of bug survives. Any row-count disagreement is
   reported, and the directory is flagged for removal.

Exit code 0 only when all three pass. Suitable for a pre-commit hook or CI.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys

import make_tables
import provenance

#: Dead exports, if they reappear. Maps parquet stem -> database table.
ORPHAN_EXPORTS = {
    "powerlaw_fits": "powerlaw_fits",
    "episcanner_fits": "episcanner_fits",
    "episcanner_state_fits": "episcanner_state_fits",
    "city_stats": "city_stats",
}
ORPHAN_DIR = "parquet_cache"


def check_orphans(root: str = ".", db: str = "powerlaw_results.db") -> list[str]:
    """Compare any orphan Parquet export against the database it duplicates."""
    d = os.path.join(root, ORPHAN_DIR)
    if not os.path.isdir(d):
        return []
    problems = []
    con = sqlite3.connect(f"file:{os.path.join(root, db)}?mode=ro", uri=True)
    for stem, table in ORPHAN_EXPORTS.items():
        p = os.path.join(d, f"{stem}.parquet")
        if not os.path.exists(p):
            continue
        try:
            import pandas as pd
            n_parquet = len(pd.read_parquet(p))
        except Exception as e:  # unreadable is worse than stale
            problems.append(f"{ORPHAN_DIR}/{stem}.parquet unreadable: {e}")
            continue
        try:
            n_db = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        except sqlite3.Error as e:
            problems.append(f"{ORPHAN_DIR}/{stem}.parquet has no table {table}: {e}")
            continue
        if n_parquet != n_db:
            problems.append(
                f"{ORPHAN_DIR}/{stem}.parquet has {n_parquet:,} rows but "
                f"{table} has {n_db:,} -- a stale copy nothing reads; delete it"
            )
    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=".")
    ap.add_argument("--db", default="powerlaw_results.db")
    ap.add_argument("--tex", default=os.path.join("Manuscript", "main.tex"))
    ap.add_argument("--skip-manifest", action="store_true")
    a = ap.parse_args(argv)

    failed = False

    if not a.skip_manifest:
        try:
            rep = provenance.assert_snapshot(a.db, a.root)
            print(f"[1/3] manifest      OK ({len(rep['ok'])} artifacts)")
        except provenance.SnapshotDrift as e:
            print(f"[1/3] manifest      DRIFT: {e}", file=sys.stderr)
            failed = True
    else:
        print("[1/3] manifest      skipped")

    facts = make_tables.compute_facts(os.path.join(a.root, a.db))
    audit = make_tables.audit_prose(os.path.join(a.root, a.tex), facts)
    if audit["stale"] or audit["missing"]:
        failed = True
        print(f"[2/3] prose         STALE ({facts['total_fits']:,} fits in db)", file=sys.stderr)
        for s in audit["stale"]:
            print(f"        {s['name']:<26} has {s['found']} want {s['want']!r}", file=sys.stderr)
        for m in audit["missing"]:
            print(f"        {m['name']:<26} missing {m['want']!r}", file=sys.stderr)
    else:
        print(f"[2/3] prose         OK ({len(audit['ok'])} claims verified)")

    orphans = check_orphans(a.root, a.db)
    if orphans:
        failed = True
        print("[3/3] orphan exports  STALE", file=sys.stderr)
        for o in orphans:
            print(f"        {o}", file=sys.stderr)
    else:
        print("[3/3] orphan exports  OK (none)")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
