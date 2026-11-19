#!/usr/bin/env python
"""Does the size forecast get worse below 100,000 inhabitants? Print the answer.

This is the audit behind Figure 14 and behind the three sentences of
``main.tex`` that lean on it. It prints the two size classes side by side, the
ratios, a bootstrap interval on the one that decides, and a verdict from a rule
that was fixed before the run (``size_error.DECISION_Q3_RATIO``).

Why a script and not just the figure
------------------------------------
The committed sidecar (``Manuscript/figures/fig14_pop_bands.json``) says the two
medians are 62.7% and 62.7%, and that the third quartiles are 85.0% and 80.9%.
Those numbers are a ratio of 1.05 -- no effect. If a fresh run reproduces them,
the figure is not something to redraw: it has nothing to show, and the sentences
that quote it ("the median relative error is larger below the threshold", "the
forecast is least accurate in that majority") are false and must go. If a run on
refreshed data gives a ratio of 1.3 or more with an interval excluding 1, the
claim can be restated about the tail -- where the effect actually lives -- and
the figure earns its place.

Usage
-----
    # from saved out-of-fold predictions (fast, needs only pandas/numpy)
    python check_size_error.py --oof predictions.parquet

    # recompute them with the estimator the paper reports (needs data/)
    python check_size_error.py --refit --dump-oof /tmp/oof.parquet

    # and draw the candidate panel somewhere harmless
    python check_size_error.py --refit --figure /tmp/fig14_ecdf.png

    # no data at all: check that the arithmetic and the rule behave
    python check_size_error.py --demo

    # --refit failed on my machine: what is it actually looking for?
    python check_size_error.py --doctor
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import size_error as se


def _fmt(x, spec=".1f"):
    return format(x, spec)


def _yn(flag: bool) -> str:
    return "yes" if flag else "no"


def print_doctor(rep: dict) -> None:
    """Every path the refit resolves, so a failure names its own cause."""
    print()
    print(f"  python        {rep['python']}  ({rep['executable']})")
    print(f"  cwd           {rep['cwd']}")
    print(f"  repository    {rep['repo_root']}")
    print(f"  sys.path[:4]  {rep['sys_path_head']}")
    print()
    print(f"  {'source file':<20}{'exists':>8}{'importable':>12}")
    print("  " + "-" * 38)
    for name, d in rep["local_modules"].items():
        print(f"  {name:<20}{_yn(d['exists']):>8}{_yn(d['resolves']):>12}")
    print()
    absent = [n for n, ok in rep["third_party"].items() if not ok]
    print("  third-party   "
          + ("all present" if not absent else "MISSING: " + ", ".join(absent)))
    data = rep["data"]
    print(f"  data/         "
          + (f"{data['parquet_files']:,} parquet files" if data["exists"]
             else "ABSENT -- --refit cannot run without it"))
    print(f"  panel cache   "
          + ("present" if rep["panel_cache"]["exists"]
             else "absent (rebuilt on the first --refit, slowly)"))
    print()
    print(f"  ready to --refit: {_yn(rep['ready_to_refit'])}")
    if not rep["ready_to_refit"]:
        print("  Note: the source files above are files of this repository, not "
              "packages.")
        print("        `uv sync` installs dependencies and cannot supply them.")
    print()


def print_table(res: dict) -> None:
    s = res["stats"]
    b, a = s["below"], s["above"]
    print()
    print(f"  size classes at the {s['cut']:,.0f}-inhabitant cut "
          f"({100 * s['share_below']:.1f}% of units below)")
    print()
    head = f"  {'':<22}{'below':>12}{'above':>12}{'ratio':>10}"
    print(head)
    print("  " + "-" * (len(head) - 2))
    rows = (
        ("municipalities", f"{b['n']:,}", f"{a['n']:,}", ""),
        ("median APE (%)", _fmt(b["median"]), _fmt(a["median"]),
         _fmt(s["median_ratio"], ".2f")),
        ("Q1 APE (%)", _fmt(b["q1"]), _fmt(a["q1"]), ""),
        # The deciding row, marked so it cannot be looked past.
        ("Q3 APE (%)   <-- rule", _fmt(b["q3"]), _fmt(a["q3"]),
         _fmt(s["q3_ratio"], ".2f")),
        ("Q90 APE (%)", _fmt(b["p90"]), _fmt(a["p90"]), ""),
        ("share APE > 100%", f"{100 * b['share_over_big']:.1f}%",
         f"{100 * a['share_over_big']:.1f}%",
         _fmt(s["share_over_big_ratio"], ".2f")),
    )
    for name, x, y, r in rows:
        print(f"  {name:<22}{x:>12}{y:>12}{r:>10}")
    print()
    n_exact = res["n_exact_forecasts"]
    if n_exact == 1:
        print("  note: 1 municipality was forecast exactly, so its relative error "
              "is 0 and it cannot be shown on a log axis; it is counted in every "
              "statistic above.")
        print()
    elif n_exact:
        print(f"  note: {n_exact} municipalities were forecast exactly, so their "
              "relative error is 0 and they cannot be shown on a log axis; they "
              "are counted in every statistic above.")
        print()


def print_verdict(res: dict) -> None:
    d, boot = res["decision"], res["bootstrap"]
    print(f"  Q3 ratio          {d['q3_ratio']:.3f}   "
          f"95% CI [{boot['ci_low']:.2f}, {boot['ci_high']:.2f}]   "
          f"({boot['n_boot']:,} bootstrap replicates over municipalities)")
    print(f"  median ratio      {d['median_ratio']:.3f}   "
          "(the statistic the caption used, and why it was the wrong one)")
    print(f"  share > 100%      {d['share_over_big_ratio']:.3f}x")
    print(f"  rule              Q3 ratio >= {d['threshold']:.1f} and CI excludes 1")
    print()
    tag = "KEEP" if d["verdict"] == "keep" else "DROP"
    print(f"  VERDICT: {tag}")
    print(f"  {d['reason']}.")
    print()
    if d["verdict"] == "drop":
        print("  Consequence: Figure 14 has no effect to draw. Remove it, and "
              "remove the sentences that quote it -- the caption's \"the median "
              "relative error is larger below the threshold\" included. What "
              "remains defensible, if anything, is a statement about x_min: a "
              "small municipality reaches 300 cases only in its exceptional "
              "seasons, so the tail that would order its next season is fitted "
              "from a handful of points. That is an argument, not a figure.")
    else:
        print("  Consequence: state the claim about the tail, not the median -- "
              "\"the typical error is the same; a large error is N times more "
              "likely\" -- and wire the ECDF panel into fig14().")
    print()


def make_demo(seed: int = 0):
    """Two synthetic populations, so the arithmetic can be checked offline.

    The effect is *planted* here (the small class has a heavier tail by
    construction), which is the point: it shows the script finds a gap when there
    is one. It says nothing about dengue.
    """
    rng = np.random.default_rng(seed)
    n_small, n_large = 3300, 320
    pop_small = rng.uniform(800, 99_000, n_small)
    pop_large = rng.uniform(100_000, 6e6, n_large)
    faces_small = rng.lognormal(mean=np.log(60), sigma=0.95, size=n_small)
    faces_large = rng.lognormal(mean=np.log(58), sigma=0.55, size=n_large)
    import pandas as pd
    unit = pd.DataFrame({
        "geocode": [f"{i:07d}" for i in range(n_small + n_large)],
        "population": np.concatenate([pop_small, pop_large]),
        "ape": np.concatenate([faces_small, faces_large]),
        "n_origins": 3,
    })
    return unit


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Audit the size-dependence of the forecast's relative error.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--oof", type=Path,
                     help="parquet or CSV with observed and predicted columns")
    src.add_argument("--refit", action="store_true",
                     help="recompute the out-of-fold predictions (needs data/)")
    src.add_argument("--demo", action="store_true",
                     help="run on synthetic data with a planted effect")
    src.add_argument("--doctor", action="store_true",
                     help="report where every piece the refit needs resolves to, "
                          "and stop (use this when --refit refuses to start)")
    ap.add_argument("--cut", type=float, default=se.SIZE_CUT,
                    help=f"population threshold (default {se.SIZE_CUT:,.0f})")
    ap.add_argument("--figure", type=Path, default=None,
                    help="write the candidate ECDF panel here (PNG)")
    ap.add_argument("--json", type=Path, default=None,
                    help="write the full result as JSON here")
    ap.add_argument("--dump-oof", type=Path, default=None,
                    help="save the row-level predictions here for later runs")
    ap.add_argument("--origins", type=int, nargs="+", default=[2023, 2024, 2025],
                    help="forecast origins used by --refit")
    ap.add_argument("--n-boot", type=int, default=se.N_BOOT)
    args = ap.parse_args(argv)

    if args.doctor:
        print_doctor(se.doctor())
        return 0

    if args.demo:
        unit = make_demo()
        print("\n  --demo: synthetic data with an effect planted in the tail by "
              "construction.\n  The verdict below says the script works, not "
              "anything about dengue.")
    else:
        oof = se.load_oof(args.oof, refit=args.refit, origins=tuple(args.origins))
        if args.dump_oof:
            args.dump_oof.parent.mkdir(parents=True, exist_ok=True)
            oof.to_parquet(args.dump_oof)
            print(f"\n  row-level predictions written to {args.dump_oof}")
        unit = se.unit_errors(oof)
        n_org = int(unit["n_origins"].max())
        tail = f" (median over {n_org} origins)" if n_org > 1 else ""
        print(f"\n  {len(oof):,} row-level predictions reduced to "
              f"{len(unit):,} municipalities{tail}.")

    boot = se.bootstrap_q3_ratio(unit, args.cut, n_boot=args.n_boot)
    res = se.report(unit, args.cut, boot=boot)
    print_table(res)
    print_verdict(res)

    if args.figure:
        import matplotlib
        matplotlib.use("Agg")
        fig, _ax = se.draw_ecdf(unit, args.cut)
        args.figure.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.figure, dpi=160, bbox_inches="tight")
        print(f"  candidate panel written to {args.figure}\n")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(res, indent=2) + "\n")
        print(f"  full result written to {args.json}\n")

    # The exit status reports the rule, so the check can gate a release: 0 when
    # the figure is justified, 1 when it is not. `--demo` always returns 0.
    if args.demo:
        return 0
    return 0 if res["decision"]["verdict"] == "keep" else 1


if __name__ == "__main__":
    sys.exit(main())
