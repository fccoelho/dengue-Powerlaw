"""One command that answers: does the manuscript match its artefacts?

Five independent checks, because the repository has drifted in five ways:

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
   experiment changed shape. The same question is asked of the figure module,
   because the artefacts cannot answer it: a figure that refits on its own can
   draw an estimator the artefacts never saw.
4. **Orphan exports** -- ``parquet_cache/`` held Parquet exports of the same
   tables that no script read, so its row counts silently disagreed with the
   database (5,244 vs 5,247; 40,616 vs 41,635). Dead copies of the source of
   truth are how this class of bug survives. Any row-count disagreement is
   reported, and the directory is flagged for removal.
5. **Ablation** -- the predictive results need the same treatment as the fits.
   Check 2 only covers the database; nothing watched ``ablation_results.csv``,
   and the result was a figure and its own caption reporting opposite signs for
   the same contrast on the same page. This check re-derives every ablation
   number the manuscript states, and it also fails when a *source artefact* is
   missing or predates the current design, because a number nobody can
   reproduce should not ship quietly.

Exit code 0 only when all five pass. Suitable for a pre-commit hook or CI.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import struct
import sys

import make_ablation_tables
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

#: Modules that draw the predictive figures.
FIGURE_SOURCES = (os.path.join("Manuscript", "figures", "generate_figures.py"),)

#: Figure index, image stem and the sidecar the drawing function writes.
#: The sidecar records the pixel shape of the file that run saved, which is the
#: only cheap evidence that the committed image and the code that claims to
#: draw it are the same vintage. A caption edited to describe a new layout over
#: an old image is otherwise invisible to every check here: the image is not in
#: the manifest, and no artefact knows how many panels it has.
FIGURE_SIDECARS = (
    ("10", "fig10_pred_vs_obs", "fig10_states.json"),
)

#: Constructors that cannot predict outside their training responses. They are
#: legitimate comparison arms inside ``run_ablation.py``, which is why that file
#: is not scanned; they have no business in a module that draws what the paper
#: reports.
SUPERSEDED_FIGURE_ESTIMATORS = (
    "RandomForestRegressor", "RandomForestClassifier",
    "GradientBoostingRegressor", "GradientBoostingClassifier",
    "HistGradientBoosting", "ExtraTreesRegressor", "ExtraTreesClassifier",
    "DecisionTreeRegressor", "LGBMRegressor", "LGBMClassifier",
    "import lightgbm", "import xgboost", "xgb.",
)


def check_figure_sources(root: str = ".") -> list[str]:
    """Refuse a figure module that fits an estimator the paper does not report.

    The artefact checks above cannot see this one. They read what the ablation
    wrote; a figure that plots a scatter of predictions refits on its own, so
    the artefacts can declare the current estimator while the panel beside them
    was drawn with an older one. That is how figure 13 kept a random forest for
    two revisions after the estimator was dropped from the text: every gate was
    answering a question about a file the figure never reads.
    """
    import re

    problems = []
    for rel in FIGURE_SOURCES:
        path = os.path.join(root, rel)
        if not os.path.exists(path):
            continue
        text = open(path, encoding="utf-8").read()
        # Drop comments only: colour literals such as "\"#d62728\"" must survive.
        code = re.sub(r"(?:^|\s+)#.*$", "", text, flags=re.MULTILINE)
        found = sorted({n for n in SUPERSEDED_FIGURE_ESTIMATORS if n in code})
        if found:
            problems.append(
                f"{rel} still fits {found}: a figure has to draw the estimator "
                "the ablation reports (run_ablation.fit_quantile_model / "
                "make_classifier), not one of its own"
            )
    return problems


def _png_shape(path: str) -> tuple[int, int]:
    """Width and height of a PNG, read from its IHDR chunk."""
    with open(path, "rb") as fh:
        head = fh.read(24)
    if head[:8] != b"\x89PNG\r\n\x1a\n" or head[12:16] != b"IHDR":
        raise ValueError("not a PNG")
    return struct.unpack(">II", head[16:24])


def check_figure_layout(root: str = ".") -> list[str]:
    """Refuse a committed figure that does not come from the current code.

    ``check_figure_sources`` reads the module; this reads the image the module
    wrote. The two are different questions, and only the second one catches a
    caption that was rewritten to describe panels nobody has drawn yet.
    """
    import json

    problems = []
    figdir = os.path.join(root, "Manuscript", "figures")
    for number, stem, sidecar in FIGURE_SIDECARS:
        png = os.path.join(figdir, f"{stem}.png")
        side = os.path.join(figdir, sidecar)
        if not os.path.exists(png):
            continue
        if not os.path.exists(side):
            problems.append(
                f"Manuscript/figures/{stem}.png has no {sidecar}: the image was "
                "not drawn by the current code. Regenerate it with "
                f"python Manuscript/figures/generate_figures.py {number}"
            )
            continue
        try:
            meta = json.loads(open(side, encoding="utf-8").read())
            recorded = tuple(int(v) for v in meta["png_shape"])
        except Exception as e:  # noqa: BLE001
            problems.append(f"Manuscript/figures/{sidecar} unreadable: {e}")
            continue
        try:
            actual = _png_shape(png)
        except Exception as e:  # noqa: BLE001
            problems.append(f"Manuscript/figures/{stem}.png unreadable: {e}")
            continue
        if actual != recorded:
            problems.append(
                f"Manuscript/figures/{stem}.png is {actual[0]}x{actual[1]} px "
                f"but {sidecar} records {recorded[0]}x{recorded[1]}: the "
                "committed image predates the current drawing code. Regenerate "
                f"it with python Manuscript/figures/generate_figures.py {number}"
            )
    return problems


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
    # Until now these two were defined and called only from the tests, so the
    # guard against a figure drawing a superseded estimator never ran in the
    # workflow that is supposed to enforce it.
    problems.extend(check_figure_sources(root))
    problems.extend(check_figure_layout(root))
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
    ap.add_argument("--allow-unverified", action="store_true",
                    help="do not fail on ablation numbers whose source "
                         "artefact is absent")
    a = ap.parse_args(argv)

    failed = False

    if not a.skip_manifest:
        try:
            rep = provenance.assert_snapshot(a.db, a.root)
            print(f"[1/5] manifest      OK ({len(rep['ok'])} artifacts)")
        except provenance.SnapshotDrift as e:
            print(f"[1/5] manifest      DRIFT: {e}", file=sys.stderr)
            failed = True
    else:
        print("[1/5] manifest      skipped")

    facts = make_tables.compute_facts(os.path.join(a.root, a.db))
    audit = make_tables.audit_prose(os.path.join(a.root, a.tex), facts)
    if audit["stale"] or audit["missing"]:
        failed = True
        print(f"[2/5] prose         STALE ({facts['total_fits']:,} fits in db)", file=sys.stderr)
        for s in audit["stale"]:
            print(f"        {s['name']:<26} has {s['found']} want {s['want']!r}", file=sys.stderr)
        for m in audit["missing"]:
            print(f"        {m['name']:<26} missing {m['want']!r}", file=sys.stderr)
    else:
        print(f"[2/5] prose         OK ({len(audit['ok'])} claims verified)")

    design = check_design(a.root)
    if design:
        failed = True
        print("[3/5] design        STALE", file=sys.stderr)
        for d in design:
            print(f"        {d}", file=sys.stderr)
    else:
        print("[3/5] design        OK (current estimator, size-only targets)")

    orphans = check_orphans(a.root, a.db)
    if orphans:
        failed = True
        print("[4/5] orphan exports  STALE", file=sys.stderr)
        for o in orphans:
            print(f"        {o}", file=sys.stderr)
    else:
        print("[4/5] orphan exports  OK (none)")

    abl = make_ablation_tables.audit_prose(
        os.path.join(a.root, a.tex),
        make_ablation_tables.compute_facts(a.root))
    bad = bool(abl["stale"] or abl["missing"]
               or (abl["unverified"] and not a.allow_unverified))
    if bad:
        failed = True
        label = "STALE" if (abl["stale"] or abl["missing"]) else "INCOMPLETE"
        print(f"[5/5] ablation      {label}", file=sys.stderr)
        for s in abl["stale"]:
            print(f"        {s['name']:<40} has {s['found']} want {s['want']!r}",
                  file=sys.stderr)
        for m in abl["missing"]:
            print(f"        {m['name']:<40} missing {m['want']!r}", file=sys.stderr)
        for u in abl["unverified"]:
            print(f"        {u['name']:<40} unverifiable: {u['source']} absent",
                  file=sys.stderr)
        print("        rerun: make_ablation_tables.py --check for the full list",
              file=sys.stderr)
    elif abl["unverified"]:
        print(f"[5/5] ablation      OK ({len(abl['ok'])} claims, "
              f"{len(abl['unverified'])} unverifiable, allowed)")
    else:
        print(f"[5/5] ablation      OK ({len(abl['ok'])} claims verified)")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
