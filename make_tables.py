"""Generate Table 1, the derived facts, and audit the manuscript against them.

The manuscript used to carry a hand-typed copy of Table 1 while Methods claimed
that *no statistic is transcribed by hand*. The database then gained fits and
the table silently fell behind -- W-seas alone was short by ~1000 city-years, and
every level's row was stale. This script removes the possibility: every number
is emitted from the database, and ``--check`` fails when ``main.tex`` still
asserts a value that the database no longer supports.

    python make_tables.py            # write Manuscript/generated/{tab_main.tex,facts.json}
    python make_tables.py --check    # audit only, exit 1 on staleness
    python make_tables.py --print    # dump the computed table

Constructions are defined once, as (key, label, SQL). Adding one means adding a
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
#: them inflates every per-year level by one row per unit (26 states). The
#: year counts the manuscript reports (353 state-years, 16 national years) only
#: reproduce with them excluded, so that is the convention used here.
#: Two variables are fitted -- the weekly count at two temporal windows, and the
#: epidemic size at three dimensions of pooling -- under the five constructions
#: of Table 1. The duration and peak-timing constructions were removed: a season
#: is bounded above by the length of the year, so a scale-free law cannot apply
#: to duration a priori, and the fits confirmed it by running into the
#: estimator's ceiling. Keeping them in the table invited the reader to compare
#: an exponent that means something with one that does not. Duration and peak
#: timing survive in the paper as covariates of size.
LEVELS = [
    ("W-mun",   "W-mun, weekly, city",
     "SELECT * FROM powerlaw_fits"),
    ("W-seas",  "W-seas, weekly, city-year",
     "SELECT * FROM powerlaw_fits_yearly WHERE year <> 0"),
    ("S-mun",   "S-mun, size, city",
     "SELECT * FROM powerlaw_fits_total_cases"),
    ("S-state", "S-state, size, state-year",
     "SELECT * FROM episcanner_state_fits WHERE metric='total_cases' AND year <> 0"),
    ("S-nat",   "S-nat, size, national-year",
     "SELECT * FROM episcanner_fits WHERE metric='total_cases' AND year <> 0"),
]

#: Constructions are addressed by key, never by position. The old code indexed
#: ``LEVELS[5]`` and ``LEVELS[3]``; dropping a construction shifted every index
#: silently and the national summary would have been computed from the state
#: table.
SQL_BY_KEY = {key: sql for key, _label, sql in LEVELS}


def _fmt(n: int) -> str:
    """Thousands separator as LaTeX-safe ``{,}``."""
    return f"{int(n):,}".replace(",", "{,}")


def _f(x: float, d: int = 2) -> str:
    return f"{x:.{d}f}"


def _sci(value: float) -> str:
    """LaTeX for a round 'about N x 10^k', e.g. 40738 -> ``$4\\times10^{4}$``."""
    import math

    k = int(math.floor(math.log10(float(value))))
    mantissa = int(round(float(value) / 10 ** k))
    return f"${mantissa}\\times10^{{{k}}}$"


def _exceedance_pct(nat) -> dict:
    """Exceedance percentages implied by each national fit, via Equation (2)."""
    from scipy.special import zeta as hurwitz

    out = {}
    for k in (10, 100, 1000):
        vals = [
            100.0 * float(hurwitz(a, k * x) / hurwitz(a, x))
            for a, x in zip(nat["alpha"].astype(float), nat["xmin"].astype(float))
        ]
        out[str(k)] = [round(min(vals), 1), round(max(vals), 1)]
    return out


def _coupling(con) -> dict:
    """Rank association between the two fitted parameters (Table 4).

    Computed rather than typed: these are the numbers that moved when the
    extraction was refreshed, and Table 4 had no artefact tying it to the
    database at all.
    """
    import numpy as np
    import pandas as pd
    from scipy.stats import spearmanr

    def _rho(a, b) -> float:
        return float(spearmanr(np.asarray(a, float), np.asarray(b, float)).statistic)

    nat = pd.read_sql(SQL_BY_KEY["S-nat"], con).dropna(subset=["alpha", "xmin"])
    st = pd.read_sql(SQL_BY_KEY["S-state"], con).dropna(subset=["alpha", "xmin"])

    per_rho, per_p = [], []
    for _state, g in st.groupby("state"):
        # A rank correlation needs a handful of points; states with fewer than
        # six seasons are dropped rather than reported as if they were stable.
        if len(g) < 6:
            continue
        r = spearmanr(g["alpha"].astype(float),
                      np.log10(g["xmin"].astype(float)))
        per_rho.append(float(r.statistic))
        per_p.append(float(r.pvalue))
    per_rho = np.asarray(per_rho, float)
    per_p = np.asarray(per_p, float)

    means = pd.DataFrame({
        "alpha": st.groupby("state")["alpha"].mean(),
        "lx": np.log10(st.groupby("state")["xmin"].mean()),
    })

    return {
        "national": {
            "rho": round(_rho(nat["alpha"], np.log10(nat["xmin"])), 2),
            "n": int(len(nat)),
        },
        "within_state": {
            "median_rho": round(float(np.median(per_rho)), 2),
            "n_states": int(len(per_rho)),
            "min_years": 6,
            "pct_positive": int(round(100 * float((per_rho > 0).mean()))),
            "pct_sig_positive": int(
                round(100 * float(((per_rho > 0) & (per_p < 0.05)).mean()))),
        },
        "between_state": {
            "rho": round(_rho(means["alpha"], means["lx"]), 2),
            "n": int(len(means)),
        },
    }


def _size_correlation(con, column: str) -> dict:
    """Pearson/Spearman of alpha against a municipal covariate (Section 5.3)."""
    import numpy as np
    import pandas as pd

    d = pd.read_sql(
        "SELECT f.alpha AS alpha, c." + column + " AS x "
        "FROM powerlaw_fits f JOIN city_stats c USING (geocode) "
        "WHERE c.population > 0", con).dropna()
    d["x"] = np.log10(d["x"].astype(float)) if column == "population" \
        else d["x"].astype(float)
    return {
        "pearson": round(float(d["x"].corr(d["alpha"])), 3),
        "spearman": round(float(d["x"].corr(d["alpha"], method="spearman")), 3),
        "n": int(len(d)),
    }


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

    nat = pd.read_sql(SQL_BY_KEY["S-nat"], con).dropna(subset=["alpha", "xmin"])
    st = pd.read_sql(SQL_BY_KEY["S-state"], con).dropna(subset=["alpha", "xmin"])
    a_lo, a_hi = float(nat["alpha"].min()), float(nat["alpha"].max())

    # Exceedance above k*x_min, from Equation (2) of the manuscript:
    #     P(S >= s) = zeta(alpha, s) / zeta(alpha, x_min),  s = k * x_min
    # with zeta the Hurwitz (generalised) zeta function -- the same expression
    # fig09 plots. The continuous Pareto limit k^-(alpha-1) was used here
    # before and disagrees in the first decimal for the small national lower
    # bounds (16.6\% against 16.5\% at ten times x_min), which put the
    # generated facts at odds with the paper's own equation.
    exceed = _exceedance_pct(nat)

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
        #: Proportion of fits whose x_min collapsed into the bulk -- the
        #: diagnostic that decides whether the tail language is vacuous.
        "xmin_le5_pct": {
            key: round(100 * float((pd.read_sql(sql, con)["xmin"]
                                    .dropna() <= 5).mean()), 0)
            for key, _label, sql in LEVELS
        },
        #: Largest selected lower bound anywhere, and where the smallest sits.
        "xmin_extremes": {
            "min": int(round(min(
                float(pd.read_sql(sql, con)["xmin"].dropna().min())
                for _k, _l, sql in LEVELS))),
            "max": int(round(max(
                float(pd.read_sql(sql, con)["xmin"].dropna().max())
                for _k, _l, sql in LEVELS))),
        },
        "alpha_span_levels": [min(alphas), max(alphas)],
        "xmin_span_levels": [min(xmins), max(xmins)],
        "exceedance_pct": exceed,
        "coupling": _coupling(con),
        "correlations": {
            "population": _size_correlation(con, "population"),
            "incidence": _size_correlation(con, "incidence"),
        },
        "warnings": warnings,
    }


def render_table(f: dict) -> str:
    """Complete LaTeX float for Table 1, byte-stable across runs."""
    body = "\n".join(
        f"{d['label']:<28} & {_fmt(d['n'])} & {d['alpha']:.2f} & "
        f"{d['iqr_low']:.2f}--{d['iqr_high']:.2f} & {_fmt(d['xmin'])} & "
        f"{d['pl_pref']} & {d['pl_sig']} & {d['exp_sig']}\\\\"
        for d in f["levels"].values()
    )
    return (
        "% GENERATED by make_tables.py from powerlaw_results.db -- do not edit.\n"
        "% Regenerate with: python make_tables.py\n"
        "\\begin{table}[htbp]\n\\centering\n\\small\n"
        "\\setlength{\\tabcolsep}{3.5pt}\n"
        "\\caption{Power-law fits under five constructions of the same municipal\n"
        "series. $\\alpha$ median with interquartile range; $x_{\\min}$\n"
        "median; PL$_{\\text{pref}}$ = proportion with $R>0$;\n"
        "PL$_{\\text{sig}}$ = proportion with $R>0$ and $p<0.05$;\n"
        "Exp$_{\\text{sig}}$ = proportion with $R<0$ and $p<0.05$. Shares are\n"
        "percentages. The two weekly constructions produce apparent support that\n"
        "the $x_{\\min}$ diagnostic removes; the cross-sectional constructions\n"
        "\\lvl{S-state} and \\lvl{S-nat} are the ones where the support survives\n"
        "both tests, and they are the object of the rest of the paper.}\n"
        "\\label{tab:main}\n"
        "\\begin{tabular}{@{}lrrrrrrr@{}}\n\\toprule\n"
        "Construction & $n$ & $\\alpha$ & IQR & $x_{\\min}$ & PL$_{\\text{pref}}$ & "
        "PL$_{\\text{sig}}$ & Exp$_{\\text{sig}}$\\\\\n\\midrule\n"
        f"{body}\n\\bottomrule\n" "\\end{tabular}\n\\end{table}\n"
    )


def claims(f: dict) -> list[dict]:
    """Prose statements in the manuscript that the facts must satisfy.

    Each claim pairs the string that *should* appear with the stale literal
    that must *not*. ``--check`` reports both directions, so a value that
    quietly disappears is as visible as one that is wrong.

    The stale literals are not decoration: each one is a number that was
    actually published in an earlier draft of this manuscript, so a regression
    to it is caught by name.
    """
    L, N, S, E, X = (
        f["levels"], f["national"], f["state"], f["exceedance_pct"],
        f["xmin_extremes"],
    )
    C, R = f["coupling"], f["correlations"]
    return [
        {"name": "total fits", "want": _fmt(f["total_fits"]),
         "stale": ["47{,}494", "47,494", "48{,}681", "48,681"], "count": 2},
        {"name": "number of constructions", "want": "five aggregation levels",
         "stale": ["seven aggregation levels", "seven nested constructions",
                   "the seven aggregation levels"]},
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
         "stale": ["$1.95$ to $2.95$", "$1.95$ to $2.93$"]},
        {"name": "exceedance 10x",
         "want": f"{E['10'][0]}--{E['10'][1]}\\%",
         "stale": ["5.4--16.5\\%", "5.6--16.6\\%"]},
        {"name": "exceedance 100x",
         "want": f"{E['100'][0]}--{E['100'][1]}\\%",
         "stale": ["0.3--2.6\\%"]},
        {"name": "national alpha range",
         "want": f"${_f(N['alpha_min'])}$ and ${_f(N['alpha_max'])}$",
         "stale": ["1.78 and 2.27"]},
        {"name": "S-nat alpha/IQR/xmin",
         "want": f"$\\alpha={L['S-nat']['alpha']:.2f}$ (IQR "
                 f"{L['S-nat']['iqr_low']:.2f}--{L['S-nat']['iqr_high']:.2f})",
         "stale": ["(IQR 1.92--2.11)"]},
        {"name": "national alpha min..max",
         "want": f"{_f(N['alpha_min'])} to {_f(N['alpha_max'])}",
         "stale": ["1.78 to 2.27"]},
        {"name": "national alpha sd",
         "want": f"SD ${N['alpha_sd']:.3f}$",
         "stale": ["SD $0.154$"]},
        {"name": "national alpha max",
         # stated twice: in the Results as the top of the observed range and in
         # the Limitations as the evidence that the estimator ceiling never
         # binds for size.
         "want": f"and ${_f(N['alpha_max'])}$ nationally",
         "stale": ["exceeds $2.27$"]},
        {"name": "xmin extremes",
         # 40{,}738 cases is "about 4 x 10^4"; derive it rather than typing it
         # so the phrase cannot drift from the database.
         "want": _sci(X["max"]), "stale": ["$1.5\\times10^{5}$"]},
        {"name": "share of weekly fits at xmin<=5 (W-mun)",
         "want": f"{f['xmin_le5_pct']['W-mun']:.0f}\\% of \\lvl{{W-mun}}",
         "stale": ["76\\% of L1", "77\\% of L1"]},
        {"name": "share of weekly fits at xmin<=5 (W-seas)",
         "want": f"{f['xmin_le5_pct']['W-seas']:.0f}\\% of \\lvl{{W-seas}}",
         "stale": ["82\\% of L2", "81\\% of L2"]},
        # Table 4 and the coupling paragraph. Every entry here is a number a
        # reader could check, and none of them existed as an artefact before.
        {"name": "coupling rho, national",
         "want": f"at ${C['national']['rho']:.2f}$ nationally",
         "stale": ["at $0.87$ nationally"]},
        {"name": "coupling rho, within state",
         "want": f"${C['within_state']['median_rho']:.2f}$ "
                 f"(median over {C['within_state']['n_states']} states)",
         "stale": ["$0.65$ (median over"]},
        {"name": "coupling rho, between states",
         "want": f"& ${C['between_state']['rho']:.2f}$ & "
                 f"{C['between_state']['n']}",
         "stale": ["& $0.22$ & 26"]},
        {"name": "share of states with a positive within-state rho",
         "want": f"positive in {C['within_state']['pct_positive']}\\% of the "
                 f"{C['within_state']['n_states']} states",
         "stale": ["positive in 88\\% of the 24 states"]},
        {"name": "alpha vs city size",
         "want": f"Pearson $r={R['population']['pearson']:.3f}$, "
                 f"Spearman ${R['population']['spearman']:.3f}$",
         "stale": ["Pearson $r=-0.065$"]},
        {"name": "alpha vs city size, n",
         "want": f"$n={_fmt(R['population']['n'])}$",
         "stale": ["$n=5{,}244$"]},
        {"name": "alpha vs incidence",
         "want": f"Pearson ${R['incidence']['pearson']:.3f}$, "
                 f"Spearman ${R['incidence']['spearman']:.3f}$; "
                 f"$n={_fmt(R['incidence']['n'])}$",
         "stale": ["Pearson $-0.136$"]},
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
