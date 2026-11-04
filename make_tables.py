"""Generate Table 1, the derived facts, and audit the manuscript against them.

The manuscript used to carry a hand-typed copy of Table 1 while Methods claimed
that *no statistic is transcribed by hand*. The database then gained fits and
the table silently fell behind -- L2 alone was short by ~1000 city-years, and
every level's row was stale. This script removes the possibility: every number
is emitted from the database, and ``--check`` fails when ``main.tex`` still
asserts a value that the database no longer supports.

    python make_tables.py            # write Manuscript/generated/{tab_main.tex,facts.json}
    python make_tables.py --check    # audit only, exit 1 on staleness
    python make_tables.py --print    # dump the computed table

Levels are defined once, as (key, label, SQL). Adding a level means adding a
line here, not editing LaTeX.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys

DB_DEFAULT = "powerlaw_results.db"
GEN_DIR = os.path.join("Manuscript", "generated")

#: ``year = 0`` rows are pooled all-years fits, not a calendar year. Including
#: them inflates every per-year level by one row per unit (27 states). The
#: authors' own reported counts (330 state-years, 15 national years) only
#: reproduce with them excluded, so that is the convention used here.
LEVELS = [
    ("L1", "L1 weekly, city",             "SELECT * FROM powerlaw_fits"),
    ("L2", "L2 weekly, city-year",        "SELECT * FROM powerlaw_fits_yearly WHERE year <> 0"),
    ("L3", "L3 annual total, city",       "SELECT * FROM powerlaw_fits_total_cases"),
    ("L4", "L4 size, state-year",         "SELECT * FROM episcanner_state_fits WHERE metric='total_cases' AND year <> 0"),
    ("L5", "L5 duration, state-year",     "SELECT * FROM episcanner_state_fits WHERE metric='ep_dur' AND year <> 0"),
    ("L6", "L6 size, national-year",      "SELECT * FROM episcanner_fits WHERE metric='total_cases' AND year <> 0"),
    ("L7", "L7 duration, national-year",  "SELECT * FROM episcanner_fits WHERE metric='ep_dur' AND year <> 0"),
]


def _fmt(n: int) -> str:
    """Thousands separator as LaTeX-safe ``{,}``."""
    return f"{int(n):,}".replace(",", "{,}")


def _f(x: float, d: int = 2) -> str:
    return f"{x:.{d}f}"


def compute_facts(db_path: str = DB_DEFAULT) -> dict:
    """Every quantity the manuscript states about the fits."""
    import pandas as pd

    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    levels, warnings = {}, []
    for key, label, sql in LEVELS:
        d = pd.read_sql(sql, con)
        d = d.dropna(subset=["alpha", "xmin"])
        if d.empty:
            raise ValueError(f"{key}: no usable fits ({sql})")
        a, x = d["alpha"].astype(float), d["xmin"].astype(float)
        R, p = d["R"].astype(float), d["p"].astype(float)
        levels[key] = {
            "label": label,
            "n": int(len(d)),
            "alpha": round(float(a.median()), 2),
            "iqr_low": round(float(a.quantile(0.25)), 2),
            "iqr_high": round(float(a.quantile(0.75)), 2),
            "xmin": int(round(float(x.median()))),
            "xmin_max": int(round(float(x.max()))),
            "pl_pref": int(round(100 * float((R > 0).mean()))),
            "pl_sig": int(round(100 * float(((R > 0) & (p < 0.05)).mean()))),
            "exp_sig": int(round(100 * float(((R < 0) & (p < 0.05)).mean()))),
        }

    nat = pd.read_sql(LEVELS[5][2], con).dropna(subset=["alpha", "xmin"])
    st = pd.read_sql(LEVELS[3][2], con).dropna(subset=["alpha", "xmin"])
    a_lo, a_hi = float(nat["alpha"].min()), float(nat["alpha"].max())

    # Exceedance above k*x_min for a Pareto tail: P(X > k x_min) = k^-(alpha-1).
    # Evaluated at the two extremes of the national exponent, which is the
    # range the manuscript quotes.
    exceed = {str(k): [round(100 * k ** -(a_hi - 1), 1), round(100 * k ** -(a_lo - 1), 1)]
              for k in (10, 100, 1000)}

    alphas = [d["alpha"] for d in levels.values()]
    xmins = [d["xmin"] for d in levels.values()]

    return {
        "source_db": db_path,
        "levels": levels,
        "total_fits": sum(v["n"] for v in levels.values()),
        "national": {
            "n": int(len(nat)),
            "alpha": round(float(nat["alpha"].median()), 2),
            "alpha_min": round(a_lo, 2),
            "alpha_max": round(a_hi, 2),
            "xmin": int(round(float(nat["xmin"].median()))),
            "alpha_sd": round(float(nat["alpha"].std(ddof=1)), 3),
        },
        "state": {
            "n": int(len(st)),
            "alpha": round(float(st["alpha"].median()), 2),
            "xmin": int(round(float(st["xmin"].median()))),
            "pl_pref": int(round(100 * float((st["R"] > 0).mean()))),
        },
        "duration_ceiling": {
            "n": levels["L5"]["n"],
            "at_ceiling": int((pd.read_sql(LEVELS[4][2], con)["alpha"].dropna().round(6) == 3.0).sum()),
            "pct": round(100 * float((pd.read_sql(LEVELS[4][2], con)["alpha"].dropna().round(6) == 3.0).mean()), 1),
        },
        "alpha_span_levels": [min(alphas), max(alphas)],
        "xmin_span_levels": [min(xmins), max(xmins)],
        "exceedance_pct": exceed,
        "warnings": warnings,
    }


def render_table(f: dict) -> str:
    """Complete LaTeX float for Table 1, byte-stable across runs."""
    body = "\n".join(
        f"{d['label']:<27} & {_fmt(d['n'])} & {d['alpha']:.2f} & "
        f"{d['iqr_low']:.2f}--{d['iqr_high']:.2f} & {_fmt(d['xmin'])} & "
        f"{d['pl_pref']} & {d['pl_sig']} & {d['exp_sig']}\\\\"
        for d in f["levels"].values()
    )
    return (
        "% GENERATED by make_tables.py from powerlaw_results.db -- do not edit.\n"
        "% Regenerate with: python make_tables.py\n"
        "\\begin{table}[htbp]\n\\centering\n\\small\n"
        "\\setlength{\\tabcolsep}{3.5pt}\n"
        "\\caption{Power-law fits by aggregation level. $\\alpha$ median with\n"
        "interquartile range; $x_{\\min}$ median; PL$_{\\text{pref}}$ = proportion with\n"
        "$R>0$; PL$_{\\text{sig}}$ = proportion with $R>0$ and $p<0.05$;\n"
        "Exp$_{\\text{sig}}$ = proportion with $R<0$ and $p<0.05$. Shares are\n"
        "percentages. The levels that survive the $x_{\\min}$ diagnostic \\emph{and} the\n"
        "model comparison are the cross-sectional size constructions L4 and L6, which are\n"
        "the object of the rest of the paper.}\n"
        "\\label{tab:main}\n"
        "\\begin{tabular}{@{}lrrrrrrr@{}}\n\\toprule\n"
        "Level & $n$ & $\\alpha$ & IQR & $x_{\\min}$ & PL$_{\\text{pref}}$ & "
        "PL$_{\\text{sig}}$ & Exp$_{\\text{sig}}$\\\\\n\\midrule\n"
        f"{body}\n\\bottomrule\n" "\\end{tabular}\n\\end{table}\n"
    )


def claims(f: dict) -> list[dict]:
    """Prose statements in the manuscript that the facts must satisfy.

    Each claim pairs the string that *should* appear with the stale literal
    that must *not*. ``--check`` reports both directions, so a value that
    quietly disappears is as visible as one that is wrong.
    """
    L, N, S, E = f["levels"], f["national"], f["state"], f["exceedance_pct"]
    return [
        {"name": "total fits", "want": _fmt(f["total_fits"]),
         "stale": ["47{,}494", "47,494"], "count": 2},
        {"name": "national cross-sections", "want": f"{N['n']} of {N['n']}",
         "stale": ["in 15 of 15 national cross-sections"]},
        {"name": "state-years", "want": _fmt(S["n"]),
         "stale": ["across 330 state-years", "of 330 state-years",
                   "330 state-year fits, significant"]},
        {"name": "alpha national median", "want": _f(N["alpha"]),
         "stale": ["$\\alpha=2.00$"]},
        {"name": "alpha state median", "want": _f(S["alpha"]),
         "stale": ["$1.97$"]},
        {"name": "xmin national median", "want": _fmt(N["xmin"]),
         "stale": ["$x_{\\min}$ of 762", "median $x_{\\min}$ of 762"]},
        {"name": "xmin state median", "want": _fmt(S["xmin"]),
         "stale": ["and 159 cases"]},
        {"name": "span of median exponents",
         "want": f"{_f(f['alpha_span_levels'][0])}$ to ${_f(f['alpha_span_levels'][1])}",
         "stale": ["$1.95$ to $2.95$"]},
        {"name": "L7 exponent", "want": _f(L["L7"]["alpha"]),
         "stale": ["piling up at $2.95$"]},
        {"name": "exceedance 10x",
         "want": f"{E['10'][0]}--{E['10'][1]}\\%",
         "stale": ["5.4--16.5\\%"]},
        {"name": "exceedance 100x",
         "want": f"{E['100'][0]}--{E['100'][1]}\\%",
         "stale": ["0.3--2.6\\%"]},
        {"name": "national alpha range",
         "want": f"${_f(N['alpha_min'])}$ and ${_f(N['alpha_max'])}$",
         "stale": ["1.78 and 2.27"]},
        {"name": "L6 alpha/IQR/xmin",
         "want": f"$\\alpha={L['L6']['alpha']:.2f}$ (IQR "
                 f"{L['L6']['iqr_low']:.2f}--{L['L6']['iqr_high']:.2f})",
         "stale": ["(IQR 1.92--2.11)"]},
        {"name": "national alpha min..max",
         "want": f"{_f(N['alpha_min'])} to {_f(N['alpha_max'])}",
         "stale": ["1.78 to 2.27"]},
        {"name": "national alpha sd",
         "want": f"SD ${N['alpha_sd']:.3f}$",
         "stale": ["SD $0.154$"]},
        {"name": "national alpha max",
         "want": f"exceeds ${_f(N['alpha_max'])}",
         "stale": ["exceeds $2.27$"]},
        {"name": "duration ceiling count",
         "want": f"{f['duration_ceiling']['at_ceiling']} of them",
         "stale": ["11 of them"]},
    ]


def audit_prose(tex_path: str, f: dict) -> dict:
    """Report stale literals still present and wanted strings still absent."""
    if not os.path.exists(tex_path):
        return {"error": f"{tex_path} not found", "stale": [], "missing": [], "ok": []}
    tex = open(tex_path, encoding="utf-8", errors="replace").read()
    out = {"stale": [], "missing": [], "ok": []}
    for c in claims(f):
        found = [s for s in c["stale"] if s in tex]
        n = tex.count(c["want"])
        # A value that is merely *present* is not enough: if the manuscript is
        # supposed to state it twice and states it once, one site was missed.
        bad_count = c.get("count") is not None and n != c["count"]
        if found or bad_count:
            out["stale"].append(
                {"name": c["name"], "found": found or [f"{n} occurrence(s)"],
                 "want": f"{c['want']} x{c.get('count', '>=1')}"})
        elif n == 0:
            out["missing"].append({"name": c["name"], "want": c["want"]})
        else:
            out["ok"].append(c["name"])
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", default=DB_DEFAULT)
    ap.add_argument("--tex", default=os.path.join("Manuscript", "main.tex"))
    ap.add_argument("--check", action="store_true", help="audit, do not write")
    ap.add_argument("--print", dest="dump", action="store_true")
    a = ap.parse_args(argv)

    if not os.path.exists(a.db):
        print(f"database not found: {a.db}", file=sys.stderr)
        return 2
    f = compute_facts(a.db)

    if a.dump:
        print(f"{'level':<26}{'n':>8}{'alpha':>7}{'IQR':>14}{'xmin':>7}{'PLp':>5}{'PLs':>5}{'EXPs':>6}")
        for d in f["levels"].values():
            print(f"{d['label']:<26}{d['n']:>8,}{d['alpha']:>7.2f}"
                  f"{d['iqr_low']:>7.2f}-{d['iqr_high']:<6.2f}{d['xmin']:>7,}"
                  f"{d['pl_pref']:>5}{d['pl_sig']:>5}{d['exp_sig']:>6}")
        print(f"{'TOTAL':<26}{f['total_fits']:>8,}")
        print(f"\nnational n={f['national']['n']} alpha {f['national']['alpha_min']}-"
              f"{f['national']['alpha_max']} med {f['national']['alpha']} xmin {f['national']['xmin']}")
        print(f"state    n={f['state']['n']} alpha med {f['state']['alpha']} "
              f"xmin {f['state']['xmin']} PLpref {f['state']['pl_pref']}%")
        print(f"exceedance {f['exceedance_pct']}")
        return 0

    audit = audit_prose(a.tex, f)
    if a.check:
        for s in audit["stale"]:
            print(f"STALE   {s['name']}: found {s['found']}, want {s['want']!r}", file=sys.stderr)
        for m in audit["missing"]:
            print(f"MISSING {m['name']}: {m['want']!r} not present", file=sys.stderr)
        if not audit["stale"] and not audit["missing"]:
            print(f"manuscript in step with {a.db} ({len(audit['ok'])} claims verified)")
            return 0
        return 1

    os.makedirs(GEN_DIR, exist_ok=True)
    with open(os.path.join(GEN_DIR, "tab_main.tex"), "w", encoding="utf-8") as fh:
        fh.write(render_table(f))
    with open(os.path.join(GEN_DIR, "facts.json"), "w", encoding="utf-8") as fh:
        json.dump(f, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"wrote {GEN_DIR}/tab_main.tex + facts.json ({f['total_fits']:,} fits)")
    for w in f["warnings"]:
        print(f"  note: {w}")
    if audit["stale"]:
        print("\nmain.tex still carries stale values:", file=sys.stderr)
        for s in audit["stale"]:
            print(f"  {s['name']:<26} found {s['found']} -> want {s['want']!r}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
