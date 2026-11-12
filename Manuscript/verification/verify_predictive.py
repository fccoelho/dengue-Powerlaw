#!/usr/bin/env python3
"""Audit the predictive-layer claims of Manuscript/main.tex.

Usage:  python Manuscript/verification/verify_predictive.py [--strict]

Two-sided check
---------------
A gate that only compares numbers against artefacts drifts away from the
manuscript: it keeps re-deriving a value that a rewrite removed, reports DRIFT
for ever, and stops being read. Every entry below therefore has to pass twice:

  * the quoted value must still occur in ``main.tex``  (otherwise STALE), and
  * the committed artefact must still produce it        (otherwise DRIFT).

STALE and DRIFT both mean the gate and the manuscript have to be reconciled,
and they mean opposite things, so they are reported separately.

Sources
-------
Nothing is refitted and no panel is built. The four committed artefacts are
read directly:

    classifier_results_metrics.csv   municipal out-of-fold classifier, per origin
    ablation_city.csv                municipal exceedance ladder
    ablation_state_exceedance.csv    state exceedance ladder
    ablation_results.csv             state size ladder, 19 quantile levels

Counting rule
-------------
``ci_excludes_zero`` is direction-agnostic: an interval can exclude zero while
pointing *against* the rung. Every "satisfies the rule" count below is taken
jointly with ``improves``, and intervals pointing the other way are reported
separately. Counting the raw column inflates the pass count -- the municipal
M1-M0 contrast has five intervals excluding zero and four that favour the rung.

Claims the artefacts cannot support are listed under NOT AUDITABLE rather than
silently skipped; a quoted number with no artefact behind it is a defect of the
same kind as a wrong one.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TEX = ROOT / "Manuscript" / "main.tex"

#: Agreement is tested at half a unit in the last digit the manuscript prints.
#: One tolerance for everything would flag a correctly rounded 0.5825 quoted as
#: 0.58 as drift, which is the quickest way to make a gate be ignored.
TOL3 = 0.0006   # quoted to three decimals
TOL2 = 0.0051   # quoted to two decimals
TOL1 = 0.051    # quoted as a percentage with one decimal


def _read(name: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / name)


def _flat(text: str) -> str:
    """Collapse whitespace so a quoted phrase survives LaTeX reflow.

    ``main.tex`` is hard-wrapped by hand and by editors, so a literal that is
    one sentence in the manuscript can be two lines in the file. Searching the
    raw text makes the gate fail on a reflow, which is noise, not drift.
    """
    return re.sub(r"\s+", " ", text)


def _flags(d: pd.DataFrame):
    """Return (interval excludes zero, differential favours the later rung)."""
    excl = d["ci_excludes_zero"].astype("boolean").fillna(False).astype(bool)
    imp = d["improves"].astype("boolean").fillna(False).astype(bool)
    return excl, imp


def _contrast(d: pd.DataFrame, name: str) -> pd.DataFrame:
    return d[(d["record_type"] == "contrast") & (d["contrast"] == name)].copy()


def _passcount(s: pd.DataFrame):
    """(intervals excluding zero in favour, intervals excluding zero against)."""
    excl, imp = _flags(s)
    return int((excl & imp).sum()), int((excl & ~imp).sum())


def recompute() -> dict:
    clf = _read("classifier_results_metrics.csv")
    city = _read("ablation_city.csv")
    state = _read("ablation_state_exceedance.csv")
    size = _read("ablation_results.csv")

    v: dict = {}

    # -- municipal discrimination, per-unit threshold
    auc = clf.pivot_table(index="origin", columns="rung", values="auc")
    for rung in ("M0", "M1", "M2", "M3", "POP"):
        v[f"auc.{rung}"] = float(auc[rung].median())
    v["auc.POP.low"] = float(auc["POP"].min())
    v["auc.POP.high"] = float(auc["POP"].max())
    v["auc.m3_best"] = int((auc[["M1", "M2", "M3"]].idxmax(axis=1) == "M3").sum())
    v["auc.m3_gain"] = float((auc["M3"] - auc["M2"]).median())

    # -- exceedance, municipal
    c = _contrast(city, "M1-M0")
    imp, ag = _passcount(c)
    v["city.m1m0.median"] = float(c["value"].median())
    v["city.m1m0.pass"] = imp
    v["city.m1m0.against"] = ag
    v["city.m1m0.width"] = float((c["ci_high"] - c["ci_low"]).median())
    imp, ag = _passcount(_contrast(city, "M3-M2"))
    v["city.m3m2.pass"] = imp

    # -- exceedance, state
    s = _contrast(state, "M1-M0")
    imp, ag = _passcount(s)
    v["state.m1m0.median"] = float(s["value"].median())
    v["state.m1m0.positive"] = int((s["value"] > 0).sum())
    v["state.m1m0.max"] = float(s["value"].max())
    v["state.m1m0.pass"] = imp
    v["state.m1m0.against"] = ag
    v["state.m1m0.width"] = float((s["ci_high"] - s["ci_low"]).median())
    v["state.m2m1.median"] = float(_contrast(state, "M2-M1")["value"].median())
    sm = _contrast(state, "M3-M2")
    v["state.m3m2.median"] = float(sm["value"].median())
    v["state.m3m2.pass"] = _passcount(sm)[0]

    # -- size ladder, state, at the primary loss tau = 0.5
    skill = size[(size["record_type"] == "skill") & (size["tau"] == 0.5)]
    for rung in ("M1", "M2", "M3"):
        g = skill[skill["rung"] == rung]
        v[f"size.{rung.lower()}.skill"] = float(g["value"].median())
        if rung == "M1":
            v["size.m1.skill_low"] = float(g["value"].min())
            v["size.m1.skill_high"] = float(g["value"].max())
            v["size.m1.positive"] = int((g["value"] > 0).sum())
    v["size.m1m0.pass"] = _passcount(_contrast(size, "M1-M0").query("tau == 0.5"))[0]

    # -- size ladder, scaling block across the quantile grid
    grid = _contrast(size, "M3-M2")
    for tau in (0.5, 0.95):
        g = grid[np.isclose(grid["tau"], tau)]
        imp, ag = _passcount(g)
        tag = f"size.m3m2.tau{tau}"
        v[f"{tag}.median_rel"] = float(g["rel_improvement"].median() * 100)
        v[f"{tag}.positive"] = int((g["rel_improvement"] > 0).sum())
        v[f"{tag}.pass"] = imp
        v[f"{tag}.excl"] = imp + ag
    med = grid.groupby("tau")["rel_improvement"].median() * 100
    v["size.grid.median_low"] = float(med.min())
    v["size.grid.median_high"] = float(med.max())
    v["size.grid.n_tau"] = int(grid["tau"].nunique())
    favour = against = 0
    for _, g in grid.groupby("tau"):
        imp, ag = _passcount(g)
        if imp >= 3:
            favour += 1
        if ag >= 3:
            against += 1
    v["size.grid.quantiles_favouring"] = favour
    v["size.grid.quantiles_against"] = against
    return v


#: key, value quoted in main.tex, literal to find in the text (None = a count
#: that the text states in words), tolerance.
CLAIMS = [
    # municipal discrimination (section: classification of large seasons)
    ("auc.M0", 0.50, "0.50", TOL2),
    ("auc.M1", 0.58, "0.58", TOL2),
    ("auc.M2", 0.63, "0.63", TOL2),
    ("auc.M3", 0.65, "0.65", TOL2),
    ("auc.POP", 0.47, "0.47", TOL2),
    ("auc.POP.low", 0.454, "0.454", TOL3),
    ("auc.POP.high", 0.522, "0.522", TOL3),
    ("auc.m3_gain", 0.013, "0.013", TOL3),
    ("auc.m3_best", 3, None, 0),
    # municipal exceedance
    ("city.m1m0.median", 0.049, "0.049", TOL3),
    ("city.m1m0.pass", 4, None, 0),
    ("city.m1m0.width", 0.034, "0.034", TOL3),
    ("city.m3m2.pass", 1, None, 0),
    # state exceedance
    ("state.m1m0.median", 0.126, "0.126", TOL3),
    ("state.m1m0.positive", 4, None, 0),
    ("state.m1m0.max", 0.212, "0.212", TOL3),
    ("state.m1m0.pass", 2, None, 0),
    ("state.m1m0.width", 0.26, "0.26", TOL2),
    ("state.m2m1.median", 0.027, "0.027", TOL3),
    ("state.m3m2.median", 0.000, "0.000", TOL3),
    ("state.m3m2.pass", 0, "no origin in which the block improves", 0),
    # size ladder, state
    ("size.m1.skill", 0.30, "0.30", TOL2),
    ("size.m1.skill_low", -0.06, "-0.06", TOL2),
    ("size.m1.skill_high", 0.86, "0.86", TOL2),
    ("size.m1.positive", 3, None, 0),
    ("size.m1m0.pass", 2, None, 0),
    ("size.m2.skill", 0.12, "0.12", TOL2),
    ("size.m3.skill", 0.15, "0.15", TOL2),
    ("size.m3m2.tau0.5.median_rel", -5.2, "-5.2", TOL1),
    ("size.m3m2.tau0.5.pass", 0, None, 0),
    ("size.m3m2.tau0.5.excl", 2, None, 0),
    ("size.m3m2.tau0.95.median_rel", 4.9, "4.9", TOL1),
    ("size.m3m2.tau0.95.positive", 4, None, 0),
    ("size.m3m2.tau0.95.excl", 2, None, 0),
    ("size.grid.median_low", -13.6, "-13.6", TOL1),
    ("size.grid.median_high", 7.2, "7.2", TOL1),
    ("size.grid.quantiles_favouring", 0,
     "no quantile has three of the five origins agreeing in favour", 0),
    ("size.grid.quantiles_against", 2, "2 of the 19 levels", 0),
]

#: Quoted in main.tex, no committed artefact behind it. Reported, never counted
#: as OK: the number cannot be audited from this repository.
NOT_AUDITABLE = [
    ("auc pooled threshold", "0.87", "0.84--0.93",
     "classifier_results_metrics.csv carries threshold_scope='per_unit' only; "
     "rerun run_classifier.py with the pooled threshold to produce it"),
]


def main(strict: bool = False) -> int:
    tex = _flat(TEX.read_text())
    v = recompute()

    rows = []
    n_ok = n_drift = n_stale = 0
    for key, quoted, literal, tol in CLAIMS:
        got = v[key]
        in_text = literal is None or _flat(literal) in tex
        if not in_text:
            verdict, n_stale = "STALE", n_stale + 1
        elif tol is None:
            verdict, n_ok = "OK", n_ok + 1
        elif abs(float(got) - float(quoted)) > tol:
            verdict, n_drift = "DRIFT", n_drift + 1
        else:
            verdict, n_ok = "OK", n_ok + 1
        rows.append({"claim": key, "quoted": round(float(quoted), 4),
                     "recomputed": round(float(got), 4), "verdict": verdict})

    print("=" * 96)
    print("PREDICTIVE-LAYER AUDIT -- committed artefacts, main.tex, no refit")
    print("=" * 96)
    for name in ("classifier_results_metrics.csv", "ablation_city.csv",
                 "ablation_state_exceedance.csv", "ablation_results.csv"):
        d = _read(name)
        print(f"  {name:34s} {len(d):5d} rows")
    print()
    print(pd.DataFrame(rows).to_string(index=False))

    print()
    print("-" * 96)
    print("Direction check on the counting rule (intervals excluding zero)")
    print("-" * 96)
    for key in ("city.m1m0", "state.m1m0"):
        print(f"  {key:12s} in favour {v[key + '.pass']:2d}   "
              f"against {v[key + '.against']:2d}   of 5 origins")
    print(f"  size M3-M2 at tau=0.5: excludes zero in "
          f"{v['size.m3m2.tau0.5.excl']} of 5, in favour in "
          f"{v['size.m3m2.tau0.5.pass']}")
    print(f"  quantile grid ({v['size.grid.n_tau']} levels): three origins "
          f"agreeing in favour with an interval excluding zero at "
          f"{v['size.grid.quantiles_favouring']} quantiles; three agreeing "
          f"against at {v['size.grid.quantiles_against']}")

    print()
    print("-" * 96)
    print("NOT AUDITABLE (quoted in main.tex, no artefact in this repository)")
    print("-" * 96)
    for name, value, rng, why in NOT_AUDITABLE:
        mark = "in text" if _flat(value) in tex else "not in text"
        print(f"  {name:22s} {value:>6s} (range {rng})  [{mark}]")
        print(f"    {why}")

    print()
    if n_drift or n_stale:
        print(f"SUMMARY: {n_ok} OK, {n_drift} DRIFT, {n_stale} STALE")
        if n_drift:
            print("  DRIFT -- the text quotes a value the artefact no longer "
                  "produces: regenerate the number or fix the artefact.")
        if n_stale:
            print("  STALE -- the artefact supports a value the text no longer "
                  "contains: main.tex moved on and this gate did not.")
    else:
        print(f"SUMMARY: {n_ok} OK, 0 DRIFT, 0 STALE "
              f"({len(NOT_AUDITABLE)} quoted value(s) not auditable)")
    if strict and (n_drift or n_stale):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(strict="--strict" in sys.argv))
