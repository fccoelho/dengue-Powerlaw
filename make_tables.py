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
import re
import sqlite3
import sys

from study_window import SEASON_YEARS

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

    # The manuscript counts these cross-sections, so the database and the
    # declared window have to agree. Without this the two can part company in
    # silence: a rebuild that loses a year moves every national fact by one
    # rank of the sample, and the prose gate is the only thing that would
    # notice -- after the number has already been written into the text.
    observed = sorted({int(y) for y in nat["year"]})
    if observed != list(SEASON_YEARS):
        missing = sorted(set(SEASON_YEARS) - set(observed))
        extra = sorted(set(observed) - set(SEASON_YEARS))
        raise ValueError(
            f"national cross-sections {observed} disagree with the declared "
            f"window {SEASON_YEARS} (missing {missing}, unexpected {extra}). "
            "Either restore the data for the missing years, or change "
            "study_window.py and re-run fit_episcanner_pl.py -- both, not one."
        )

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

    # Municipal size, for the resolution limit of the forecast. ``city_stats``
    # sums the platform's seasonal totals per municipality and carries the latest
    # population; it enters no fit, and the shares below exist so that the
    # Discussion can state them as quantities the database re-derives rather than
    # as numbers typed from a session.
    cs = pd.read_sql(
        "SELECT total_cases, population FROM city_stats "
        "WHERE total_cases IS NOT NULL AND population > 0", con)

    def _below(thr: float) -> dict:
        """Share of municipalities and of cases under a population threshold."""
        sel = cs["population"] < thr
        return {
            "n": int(sel.sum()),
            "pct_units": round(100 * float(sel.mean()), 1),
            "pct_cases": round(100 * float(
                cs.loc[sel, "total_cases"].sum() / cs["total_cases"].sum()), 1),
        }

    pop = cs["population"].astype(float)
    median_pop = float(pop.median())
    x_min_mun = int(levels["S-mun"]["xmin"])
    municipal_size = {
        "n": int(len(cs)),
        "median_population": int(round(median_pop)),
        "below_50k": _below(50_000),
        "below_100k": _below(100_000),
        #: What the tail threshold costs in attack-rate terms, which is the
        #: argument of the resolution-limit section: the same 300 cases are a
        #: large fraction of a small municipality and a rounding error of a large
        #: one, so a small unit enters the fitted tail only in its bad years.
        "xmin_attack_rate": {
            "xmin": x_min_mun,
            "at_median_pop": round(100.0 * x_min_mun / median_pop, 1),
            "at_100k": round(100.0 * x_min_mun / 100_000, 1),
        },
    }

    return {
        "source_db": db_path,
        "municipal_size": municipal_size,
        "levels": levels,
        "total_fits": sum(v["n"] for v in levels.values()),
        "national": {
            "n": int(len(nat)),
            "alpha": round(float(nat["alpha"].median()), 2),
            "alpha_min": round(a_lo, 2),
            "alpha_max": round(a_hi, 2),
            "xmin": int(round(float(nat["xmin"].median()))),
            "alpha_sd": round(float(nat["alpha"].std(ddof=1)), 3),
            "year_first": int(nat["year"].min()),
            "year_last": int(nat["year"].max()),
            "years_declared": list(SEASON_YEARS),
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
    M = f["municipal_size"]
    return [
        {"name": "total fits", "want": _fmt(f["total_fits"]),
         "stale": ["47{,}494", "47,494", "48{,}681", "48,681"], "count": 2},
        {"name": "number of constructions", "want": "five aggregation levels",
         "stale": ["seven aggregation levels", "seven nested constructions",
                   "the seven aggregation levels"]},
        {"name": "national cross-sections", "want": f"{N['n']} of {N['n']}",
         "stale": ["in 15 of 15 national cross-sections"]},
        # The window the fits were run over, which the reader is told in
        # Methods. It is the one statement the row count alone cannot check:
        # the database could agree with "16 of 16" and still have run over
        # 1995--2010.
        {"name": "national year window",
         "want": f"{N['n']} years, {N['year_first']} to {N['year_last']}",
         "stale": ["15 years, 2011 to 2025", "2011 to 2025"]},
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
         "want": f"$\\alpha_S={L['S-nat']['alpha']:.2f}$ (IQR "
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
        # The resolution limit of the forecast. These are sizes, not fits, so
        # they are the only claims here that do not come from the power-law
        # tables; they come from ``city_stats`` and are what the Discussion
        # rests on when it says the forecast is least accurate in small
        # municipalities.
        {"name": "municipal units with a season",
         "want": f"Of the {_fmt(M['n'])} municipalities",
         "stale": ["Of the 5{,}571 municipalities"]},
        {"name": "share of municipalities below 100k",
         "want": (f"{M['below_100k']['pct_units']:.1f}\\% have fewer than "
                  "100{,}000 inhabitants"),
         "stale": ["93.4\\% have fewer than"]},
        {"name": "share of cases below 100k",
         "want": f"{M['below_100k']['pct_cases']:.1f}\\% of the cases in the series",
         "stale": ["37.9\\% of the cases in the series"]},
        {"name": "municipalities below 50k",
         "want": (f"{M['below_50k']['pct_units']:.1f}\\% have fewer than "
                  f"50{{,}}000 and carry {M['below_50k']['pct_cases']:.1f}\\%"),
         "stale": ["86.3\\% have fewer than"]},
        {"name": "median municipal population",
         "want": f"median municipality has {_fmt(M['median_population'])} inhabitants",
         "stale": ["median municipality has 13{,}900 inhabitants"]},
        {"name": "municipal xmin",
         "want": f"$x_{{\\min}}={_fmt(M['xmin_attack_rate']['xmin'])}$ cases",
         "stale": ["$x_{\\min}=295$ cases"]},
        {"name": "xmin as an attack rate",
         "want": (f"{M['xmin_attack_rate']['at_median_pop']:.1f}\\% in the median "
                  "municipality"),
         "stale": ["2.4\\% in the median municipality"]},
        {"name": "xmin attack rate at 100k",
         "want": (f"{M['xmin_attack_rate']['at_100k']:.1f}\\% in one of "
                  "100{,}000 inhabitants"),
         "stale": ["0.4\\% in one of 100{,}000 inhabitants"]},
    ]


#: Artefacts of the survival reading (``survival_analysis.py``). They are not in
#: the database: they are computed from the panel the database summarises, so the
#: audit reads them from disk. When they are absent -- a reader who has not run
#: the script -- the block is skipped and the skip is printed, rather than failed.
SURV_DIR = os.path.join("analysis", "survival")

SURV_FILES = ["exceedance_within_season.csv", "exceedance_transport.csv",
              "duration_models.csv", "duration_summary.json"]


def survival_facts(root: str = ".") -> dict | None:
    """Every quantity the survival section states, or ``None`` if unrun."""
    import pandas as pd

    d = os.path.join(root, SURV_DIR)
    if not all(os.path.exists(os.path.join(d, n)) for n in SURV_FILES):
        return None
    within = pd.read_csv(os.path.join(d, SURV_FILES[0]))
    trans = pd.read_csv(os.path.join(d, SURV_FILES[1]))
    models = pd.read_csv(os.path.join(d, SURV_FILES[2]))
    with open(os.path.join(d, SURV_FILES[3]), encoding="utf-8") as fh:
        dur = json.load(fh)

    seasons = within[within["year"] > 0]
    pooled = within[within["year"] == 0].iloc[0]
    cox = {row["term"]: row for row in dur["cox"]}
    # The manuscript states the Cox effect of a *doubling* of the case count,
    # which is one log-unit effect raised to ln 2. Derive it rather than typing
    # it, so the two can never disagree.
    log_cases = cox["log_cases"]
    return {
        "within_max": float(seasons["departure_in_bands"].max()),
        "within_min": float(seasons["departure_in_bands"].min()),
        "within_median": float(seasons["departure_in_bands"].median()),
        "within_n": int(len(seasons)),
        "tail_min": int(seasons["n_above_xmin"].min()),
        "tail_max": int(seasons["n_above_xmin"].max()),
        "share_min": 100 * float(seasons["share_above_xmin"].min()),
        "share_max": 100 * float(seasons["share_above_xmin"].max()),
        "pooled_alpha": float(pooled["alpha"]),
        "pooled_xmin": float(pooled["xmin"]),
        "pooled_dep": float(pooled["departure_in_bands"]),
        "pooled_n": int(pooled["n_units"]),
        "pairs_possible": len(SEASON_YEARS) - 1,
        "pairs": int(len(trans)),
        "inside": int(trans["inside_band"].sum()),
        "trans_median": float(trans["departure_in_bands"].median()),
        "trans_max": float(trans["departure_in_bands"].max()),
        "trans_max_years": (int(trans.loc[trans["departure_in_bands"].idxmax(),
                                          "fitted_year"]),
                            int(trans.loc[trans["departure_in_bands"].idxmax(),
                                          "target_year"])),
        "e10_median": float(((trans["pred_10x"] - trans["obs_10x"]).abs()
                             * 100).median()),
        "e10_below": int((trans["pred_10x"] < trans["obs_10x"]).sum()),
        "pred100_median": float(trans["pred_100x"].median() * 100),
        "obs100_median": float(trans["obs_100x"].median() * 100),
        "over100": int((trans["pred_100x"] > trans["obs_100x"]).sum()),
        "obs100_zero": int((trans["obs_100x"] == 0).sum()),
        "dur_median": float(dur["km_median_duration"]),
        "dur_n": int(dur["n_units"]),
        "censored": 100 * float(dur["censored_share"]),
        "weibull": float(dur["weibull_shape"]),
        "weibull_lo": float(dur["weibull_shape_ci"][0]),
        "weibull_hi": float(dur["weibull_shape_ci"][1]),
        "exp_aic": float(models.loc[models["model"] == "exponential",
                                    "delta_aic"].iloc[0]),
        "weibull_aic": float(models.loc[models["model"] == "weibull",
                                        "delta_aic"].iloc[0]),
        "logrank_p": float(dur["logrank_p"]),
        "cox_hr": float(log_cases["hazard_ratio"]),
        "cox_lo": float(log_cases["ci_low"]),
        "cox_hi": float(log_cases["ci_high"]),
        "cox_p": float(log_cases["p"]),
        # The prose also says what that per-log-unit effect means for a season
        # twice the size. Derived from the same coefficient so the two cannot
        # disagree the way they did when the interval was quoted on one scale
        # and the point estimate on the other.
        "cox_doubling_pct": 100 * (float(log_cases["hazard_ratio"]) ** 0.6931471805599453 - 1),
        "cox_season": float(cox["season"]["hazard_ratio"]),
        "cox_season_lo": float(cox["season"]["ci_low"]),
        "cox_season_hi": float(cox["season"]["ci_high"]),
    }


def survival_claims(root: str = ".") -> list[dict]:
    """Prose statements about Section~\\ref{sec:survivalres}.

    Same contract as ``claims`` -- the string that should appear, and optionally
    literals from earlier drafts that must not -- except that ``stale`` is
    omitted here. These quantities enter the manuscript with the section that
    audits them, so there is no earlier draft whose values could be named; the
    check is that each stated number is still derived. ``audit_prose`` treats
    the key as optional and validates the key set instead.
    """
    f = survival_facts(root)
    if f is None:
        return []
    y0, y1 = f["trans_max_years"]
    return [
        {"name": "survival: within-season departure",
         "want": f"{f['within_max']:.2f} of the 95\\% simultaneous band"},
        {"name": "survival: within-season spread",
         "want": (f"the smallest {f['within_min']:.2f}, with a median of "
                  f"{f['within_median']:.2f}")},
        {"name": "survival: tail sizes",
         "want": f"between {f['tail_min']} and {f['tail_max']} municipalities"},
        {"name": "survival: left-truncation share",
         "want": f"ranges from {f['share_min']:.0f}\\% to {f['share_max']:.0f}\\%"},
        {"name": "survival: pooled fit",
         "want": (f"$\\hat\\alpha_S={f['pooled_alpha']:.2f}$ and "
                  f"$x_{{\\min}}={f['pooled_xmin']:.0f}$")},
        {"name": "survival: pooled departure",
         "want": f"by {f['pooled_dep']:.2f} bands"},
        {"name": "survival: pooled sample",
         "want": f"all {_fmt(f['pooled_n'])} municipality-seasons"},
        {"name": "survival: consecutive pairs",
         "want": f"Of the {f['pairs_possible']} consecutive pairs, {f['pairs']}"},
        {"name": "survival: pairs inside the band",
         "want": f"in {f['inside']} of those {f['pairs']}"},
        {"name": "survival: transport median",
         "want": f"the median departure is {f['trans_median']:.2f} bands"},
        {"name": "survival: transport worst pair",
         "want": f"and the largest is {f['trans_max']:.2f}, from {y0} to {y1}"},
        {"name": "survival: ten times the bound",
         "want": (f"the median absolute error is {f['e10_median']:.1f} percentage "
                  f"points")},
        {"name": "survival: ten times, sign balance",
         "want": f"{f['e10_below']} pairs of {f['pairs']} each way"},
        {"name": "survival: hundred times, observed",
         "want": f"delivers {f['obs100_median']:.2f}\\%"},
        {"name": "survival: hundred times, predicted",
         "want": f"implies a median share of {f['pred100_median']:.1f}\\%"},
        {"name": "survival: hundred times, overstatement",
         "want": f"overstates the observed share in {f['over100']} of the "
                 f"{f['pairs']} pairs"},
        {"name": "survival: seasons with no exceedance",
         "want": f"meets {_spell(f['obs100_zero'])} seasons"},
        {"name": "survival: duration median",
         "want": f"the median season lasts {f['dur_median']:.0f} weeks"},
        {"name": "survival: censoring",
         "want": f"{f['censored']:.1f}\\% of spells are censored"},
        {"name": "survival: weibull shape",
         "want": (f"the Weibull shape is {f['weibull']:.2f} (95\\% CI "
                  f"{f['weibull_lo']:.2f}--{f['weibull_hi']:.2f})")},
        {"name": "survival: exponential rejected",
         "want": f"rejected by {_fmt(round(f['exp_aic']))} AIC units"},
        {"name": "survival: gamma improvement",
         "want": f"improves on the Weibull by {round(f['weibull_aic'])}"},
        {"name": "survival: cox on size",
         "want": (f"is {f['cox_hr']:.3f} per log unit of the case count "
                  f"({f['cox_lo']:.3f}--{f['cox_hi']:.3f}, $p={f['cox_p']:.2f}$)")},
        {"name": "survival: cox size, doubling",
         "want": f"{f['cox_doubling_pct']:.1f}\\% change in the hazard"},
        {"name": "survival: cox on season",
         "want": (f"(${f['cox_season']:.3f}$ per season, "
                  f"${f['cox_season_lo']:.3f}$--${f['cox_season_hi']:.3f}$)")},
    ]


def _spell(n: int) -> str:
    """Small integers in the prose are words, not digits."""
    words = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
             7: "seven", 8: "eight", 9: "nine", 10: "ten"}
    return words.get(int(n), str(int(n)))


#: Keys a claim may carry. ``stale`` is optional -- see ``survival_claims``.
#: Validated rather than assumed: reading ``c["stale"]`` unconditionally turned
#: the first run of the survival block into a KeyError, and reading it with a
#: bare ``.get`` would let a misspelled key pass as a claim with nothing to
#: check.
CLAIM_KEYS = frozenset({"name", "want", "stale", "count"})


def audit_prose(tex_path: str, f: dict) -> dict:
    """Report stale literals still present and wanted strings still absent."""
    if not os.path.exists(tex_path):
        return {"error": f"{tex_path} not found", "stale": [], "missing": [], "ok": []}
    tex = open(tex_path, encoding="utf-8", errors="replace").read()
    # Collapse whitespace before matching. The manuscript is reflowed between
    # drafts, so a window such as "2011 to 2026" is one line in one version
    # and two in the next; a claim that only survives the original line breaks
    # is not a check but a coincidence.
    tex = re.sub(r"\s+", " ", tex)
    out = {"stale": [], "missing": [], "ok": []}
    # The survival block is audited too: those numbers are typed into the
    # prose the same way, and until now nothing checked them. Skipped, not
    # failed, when the artefacts are absent.
    for c in claims(f) + survival_claims():
        unknown = set(c) - CLAIM_KEYS
        if unknown:
            raise ValueError(
                f"claim {c.get('name', '<unnamed>')!r} carries unknown key(s) "
                f"{sorted(unknown)}; allowed are {sorted(CLAIM_KEYS)}")
        found = [s for s in c.get("stale", []) if s in tex]
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
    try:
        f = compute_facts(a.db)
    except ValueError as exc:
        print(f"cannot derive the facts: {exc}", file=sys.stderr)
        return 2

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
            skip = "" if survival_facts() is not None else (
                " -- survival artefacts absent, that block not audited")
            print(f"manuscript in step with {a.db} "
                  f"({len(audit['ok'])} claims verified){skip}")
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
