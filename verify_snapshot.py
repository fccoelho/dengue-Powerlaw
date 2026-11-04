"""One command that answers: does the manuscript match the database?

Three independent checks, because the repository drifted in three ways at once:

1. **Manifest** -- ``snapshot_manifest.json`` records the SHA-256 of the fits
   database and of every derived artifact. A mismatch means something was
   regenerated from a different run and must not be trusted together.
2. **Prose** -- ``make_tables.py --check`` re-derives every quantity the
   manuscript states and looks for stale literals. This is the check that was
   missing: Table 1 was hand-typed for months while Methods claimed no
   statistic was transcribed by hand.
3. **Design** -- the forecasting artefacts must declare the estimator the
   manuscript describes, and must not carry the duration target or the
   duration scaling descriptors that were removed with that aggregation level.
   A byte-identical artefact from a superseded design passes the manifest check
   and fails this one, which is the point: hashes cannot tell you that the
   experiment changed shape.
4. **Orphan exports** -- ``parquet_cache/`` held Parquet exports of the same
   tables that no script read, so its row counts silently disagreed with the
   database (5,244 vs 5,247; 40,616 vs 41,635). Dead copies of the source of
   truth are how this class of bug survives. Any row-count disagreement is
   reported, and the directory is flagged for removal.

Exit code 0 only when all four pass. Suitable for a pre-commit hook or CI.
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

#: Keys a forecasting artefact must carry, per file. Their absence means the
#: run predates the estimator change.
DESIGN_KEYS = {
    "ablation_results.meta.json": ("extrapolating_regressor",
                                   "extrapolating_classifier",
                                   "classifier_model"),
    "ablation_city.meta.json": ("extrapolating_regressor",
                                "extrapolating_classifier",
                                "classifier_model"),
    "classifier_results.meta.json": ("classifier_model",),
}

#: Columns whose presence means the artefact still encodes the removed
#: duration scaling rung.
FORBIDDEN_COLUMNS = ("prev_alpha_dur", "prev_xmin_dur")

#: Targets the ablation may report.
ALLOWED_TARGETS = {"total_cases"}


def check_design(root: str = ".") -> list[str]:
    """Refuse artefacts that belong to the superseded experimental design."""
    import json

    problems = []
    for name, keys in DESIGN_KEYS.items():
        path = os.path.join(root, name)
        if not os.path.exists(path):
            continue
        try:
            meta = json.loads(open(path, encoding="utf-8").read())
        except Exception as e:  # noqa: BLE001
            problems.append(f"{name} unreadable: {e}")
            continue
        missing = [k for k in keys if k not in meta]
        if missing:
            problems.append(
                f"{name} lacks {missing}: it was produced before the estimator "
                "change and must be regenerated (python run_ablation.py / "
                "run_classifier.py)"
            )
        ladder = meta.get("ladder") or {}
        stale = sorted({c for cols in ladder.values() for c in cols
                        if c in FORBIDDEN_COLUMNS})
        if stale:
            problems.append(
                f"{name} still carries the duration scaling descriptors {stale}"
            )

    for name in ("ablation_results.csv", "ablation_city.csv"):
        path = os.path.join(root, name)
        if not os.path.exists(path):
            continue
        try:
            import pandas as pd

            df = pd.read_csv(path)
        except Exception as e:  # noqa: BLE001
            problems.append(f"{name} unreadable: {e}")
            continue
        if "target" in df.columns:
            bad = sorted(set(df["target"].dropna()) - ALLOWED_TARGETS)
            if bad:
                problems.append(
                    f"{name} still reports the target(s) {bad}; duration and "
                    "peak timing are covariates now"
                )
    return problems


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
            print(f"[1/4] manifest      OK ({len(rep['ok'])} artifacts)")
        except provenance.SnapshotDrift as e:
            print(f"[1/4] manifest      DRIFT: {e}", file=sys.stderr)
            failed = True
    else:
        print("[1/4] manifest      skipped")

    facts = make_tables.compute_facts(os.path.join(a.root, a.db))
    audit = make_tables.audit_prose(os.path.join(a.root, a.tex), facts)
    if audit["stale"] or audit["missing"]:
        failed = True
        print(f"[2/4] prose         STALE ({facts['total_fits']:,} fits in db)", file=sys.stderr)
        for s in audit["stale"]:
            print(f"        {s['name']:<26} has {s['found']} want {s['want']!r}", file=sys.stderr)
        for m in audit["missing"]:
            print(f"        {m['name']:<26} missing {m['want']!r}", file=sys.stderr)
    else:
        print(f"[2/4] prose         OK ({len(audit['ok'])} claims verified)")

    design = check_design(a.root)
    if design:
        failed = True
        print("[3/4] design        STALE", file=sys.stderr)
        for d in design:
            print(f"        {d}", file=sys.stderr)
    else:
        print("[3/4] design        OK (current estimator, size-only targets)")

    orphans = check_orphans(a.root, a.db)
    if orphans:
        failed = True
        print("[4/4] orphan exports  STALE", file=sys.stderr)
        for o in orphans:
            print(f"        {o}", file=sys.stderr)
    else:
        print("[4/4] orphan exports  OK (none)")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
