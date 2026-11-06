"""Derive Tables 3 and 4, and the ablation facts, from the forecasting artefacts.

Why this exists
---------------
``make_tables.py`` derives everything the manuscript says about the *fits*, and
``--check`` fails when ``main.tex`` disagrees with the database. Nothing did the
same for the *predictive* results, and that gap produced a concrete defect: after
the estimator change and the data refresh, Figure 11 was regenerated from
``ablation_results.csv`` while the Table 3 cells and the figure caption were
still the hand-typed values of the superseded run. The figure and its own caption
then contradicted each other on the same page -- the panel said the scaling block
*helped* the 95th percentile by 4.9% in median, the caption said it *hurt* by
18.3%.

So the rule is the same here as there: every ablation number in the manuscript is
emitted from an artefact, never typed.

    python make_ablation_tables.py            # write generated/{tab_ablation,tab_exceed}.tex
    python make_ablation_tables.py --check    # audit only, exit 1 on staleness
    python make_ablation_tables.py --print    # dump the facts
    python make_ablation_tables.py --sources  # which artefacts exist, and why not

"Table 3 cannot be rendered: no size ladder" is therefore not a rendering bug --
it means ``ablation_results.csv`` is missing or holds something else, usually
because a run with a different ``--level``/``--tasks`` wrote over it. Run
``--sources`` first; it names the file and the reason.

Three verdicts, because a missing artefact is neither fresh nor stale:

* ``STALE``      -- ``main.tex`` carries a number an artefact contradicts.
* ``MISSING``    -- ``main.tex`` omits a number an artefact supplies.
* ``UNVERIFIED`` -- a source artefact is absent, so the number cannot be checked
                    either way. This is the state the state-level exceedance
                    results are in: they were produced by a run whose output was
                    overwritten by the size-only refresh, so the manuscript
                    quotes them, the repository cannot regenerate them, and no
                    one can tell whether they still hold after the new
                    extraction. It fails the check by default: a number nobody
                    can reproduce should not ship quietly.

Sign conventions, fixed once here because they are the easiest thing to get
backwards: a ``contrast`` row carries ``value = mean(L_reference - L_rung)`` and
``rel_improvement = value / mean(L_reference)``, so **positive means the later
rung has the lower loss** -- the later rung is better. ``improves`` is the
stricter flag (same direction *and* a bootstrap interval excluding zero);
``ci_excludes_zero`` alone also fires on intervals that exclude zero downwards.
Table 4's ``Origins CI excl. 0`` column reports ``improves``, not
``ci_excludes_zero``, and is labelled accordingly.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

GEN_DIR = os.path.join("Manuscript", "generated")

#: The primary loss for the decision rule: pinball at the median. Section 2.10
#: fixes this, so it is not a free parameter here.
PRIMARY_TAU = 0.5

#: The quantile the scaling block was expected to earn its keep at: a tail
#: exponent concerns extreme seasons, so any gain should concentrate here.
EXTREME_TAU = 0.95

#: Adjacent rungs of the ladder, as contrasts.
ADJACENT = ("M1-M0", "M2-M1", "M3-M2")

#: Every artefact the tables depend on. ``columns`` is the minimum a file must
#: have to be used at all: a file that exists but carries the wrong shape is
#: worse than a missing one, because it reads as present.
SOURCES = {
    "state_size": {
        "path": "ablation_results.csv",
        "meta": "ablation_results.meta.json",
        "columns": ("record_type", "task", "loss", "tau", "contrast", "rung"),
        # Content, not just shape. A file with the right columns but the wrong
        # ``task`` is the signature of one run overwriting another, and it used
        # to pass every check here while Table 3 quietly lost its source.
        "values": {"task": ("size",), "level": ("state",)},
        "what": "state-level size ablation",
        "how": "python run_ablation.py --level state --attach-climate",
        # Keys written by the current run_ablation.py. Their absence means the
        # file was produced before the estimator change, so its numbers describe
        # an experiment the manuscript no longer reports.
        "design": ("extrapolating_regressor", "classifier_model"),
    },
    "state_exceedance": {
        "path": "ablation_state_exceedance.csv",
        "meta": "ablation_state_exceedance.meta.json",
        "columns": ("record_type", "task", "loss", "contrast", "rung"),
        "values": {"task": ("exceedance",), "level": ("state",)},
        "what": "state-level exceedance classifier",
        "design": ("extrapolating_classifier", "classifier_model"),
        # Not produced by any script in this repository as committed: the
        # size-only refresh overwrote ablation_results.csv without the
        # exceedance task. Left as a declared source so its absence is loud.
        "how": ("python run_ablation.py --level state --tasks exceedance "
                "--output ablation_state_exceedance.csv"),
    },
    "city_exceedance": {
        "path": "ablation_city.csv",
        "meta": "ablation_city.meta.json",
        "columns": ("record_type", "task", "loss", "contrast", "rung"),
        "values": {"task": ("exceedance",), "level": ("city",)},
        "what": "municipal-level exceedance ablation",
        "design": ("extrapolating_classifier", "classifier_model"),
        "how": ("python run_ablation.py --level city --tasks exceedance "
                "--output ablation_city.csv"),
    },
    "city_classifier": {
        "path": "classifier_results_metrics.csv",
        "meta": "classifier_results.meta.json",
        "columns": ("origin", "rung", "brier", "auc"),
        "values": {"level": ("city",)},
        "what": "municipal-level out-of-fold classifier metrics",
        "design": ("classifier_model",),
    },
}


def facts_get(objs=()):
    """Keys present in the classifier facts dict (a tiny readability shim)."""
    return dict(objs or {})


def _pct(x: float, d: int = 1) -> str:
    """Signed percentage for LaTeX prose, e.g. ``+4.9\\%``."""
    return f"{100 * float(x):+.{d}f}\\%"


def _num(x: float, d: int = 2) -> str:
    return f"{float(x):.{d}f}"


def _load(name: str, root: str = ".") -> tuple[pd.DataFrame | None, dict, str | None]:
    """Load one artefact. Returns ``(df, meta, problem)``; df is None if absent."""
    spec = SOURCES[name]
    path = os.path.join(root, spec["path"])
    if not os.path.exists(path):
        return None, {}, f"{spec['path']} absent ({spec['what']})"
    try:
        df = pd.read_csv(path)
    except Exception as e:  # noqa: BLE001
        return None, {}, f"{spec['path']} unreadable: {e}"
    miss = [c for c in spec["columns"] if c not in df.columns]
    if miss:
        return None, {}, (f"{spec['path']} lacks the columns {miss}: it is not a "
                          f"{spec['what']} artefact")
    for col, allowed in spec.get("values", {}).items():
        if col not in df.columns:
            continue
        got = sorted(str(v) for v in df[col].dropna().unique())
        absent = [v for v in allowed if v not in got]
        if absent:
            return None, {}, (
                f"{spec['path']} holds {col}={got or ['nothing']} but a "
                f"{spec['what']} artefact needs {col} in {list(allowed)}: "
                f"another run has written over it")
    meta = {}
    mpath = os.path.join(root, spec["meta"])
    if os.path.exists(mpath):
        try:
            meta = json.loads(open(mpath, encoding="utf-8").read())
        except Exception:  # noqa: BLE001
            meta = {}
    return df, meta, None


def _contrast_summary(df: pd.DataFrame, task: str, tau: float | None = None) -> dict:
    """Per-contrast medians and counts over origins, for one task.

    ``tau`` selects one quantile of the grid. The size ladder writes a contrast
    row at each of the 19 quantiles, so a summary that forgets to filter reads
    95 rows and reports a median over the grid instead of over origins -- which
    would look plausible and be wrong. Exceedance rows carry no ``tau`` at all,
    so ``tau=None`` selects them.
    """
    sub = df[(df["record_type"] == "contrast") & (df["task"] == task)]
    if tau is None:
        sub = sub[sub["tau"].isna()]
    else:
        sub = sub[np.isclose(sub["tau"].astype(float), float(tau))]
    out = {}
    for c in ADJACENT:
        d = sub[sub["contrast"] == c]
        if d.empty:
            continue
        out[c] = {
            "n_origins": int(len(d)),
            "median_diff": float(d["value"].median()),
            "median_rel": (float(d["rel_improvement"].median())
                           if "rel_improvement" in d else None),
            "n_diff_positive": int((d["value"] > 0).sum()),
            "n_same_direction_ci": int(d["improves"].sum()),
            "n_ci_excludes_zero": int(d["ci_excludes_zero"].sum()),
        }
        if "rel_improvement" in d:
            out[c]["n_rel_positive"] = int((d["rel_improvement"] > 0).sum())
    dec = df[(df["record_type"] == "decision") & (df["task"] == task)]
    for _, r in dec.iterrows():
        if r.get("contrast") in out:
            out[r["contrast"]]["verdict"] = r.get("verdict")
            out[r["contrast"]]["decision_origins"] = (
                int(r["value"]) if pd.notna(r.get("value")) else None)
    return out


def compute_facts(root: str = ".") -> dict:
    """Every quantity Tables 3 and 4 state, plus the warnings about them."""
    facts: dict = {
        "primary_tau": PRIMARY_TAU,
        "extreme_tau": EXTREME_TAU,
        "sources": {},
        "size": {},
        "exceed": {"state": None, "city": None},
        "classifier": {},
        "warnings": [],
    }

    for name, spec in SOURCES.items():
        _df, _meta, problem = _load(name, root)
        fresh = problem is None
        if fresh:
            missing = [k for k in spec.get("design", ()) if k not in _meta]
            if missing:
                fresh = False
                problem = (f"{spec['path']} predates the current design (meta has no "
                           f"{missing}): its numbers describe a superseded experiment")
        facts["sources"][name] = {"path": spec["path"], "what": spec["what"],
                                  "present": fresh, "problem": problem,
                                  "how": spec.get("how")}
        if problem:
            facts["warnings"].append(
                problem + (f" -- regenerate with: {spec['how']}" if spec.get("how") else ""))

    # ---- Table 3: the size ladder at state level -------------------------
    df, meta, problem = _load("state_size", root)
    if df is not None and facts["sources"]["state_size"]["present"]:
        prim = df[(df["task"] == "size") & (df["loss"] == "pinball")
                  & np.isclose(df["tau"].astype(float), PRIMARY_TAU)]
        skill = prim[prim["record_type"] == "skill"]
        piv = skill.pivot_table(index="origin", columns="rung", values="value")
        facts["size"]["skill"] = {
            r: {"median": float(piv[r].median()), "min": float(piv[r].min()),
                "max": float(piv[r].max()),
                "n_positive": int((piv[r] > 0).sum()),
                "n_origins": int(piv[r].notna().sum())}
            for r in ("M1", "M2", "M3") if r in piv
        }
        facts["size"]["contrasts"] = _contrast_summary(df, "size", PRIMARY_TAU)

        # The quantile grid, for the extreme-tail claim.
        c = df[(df["record_type"] == "contrast") & (df["task"] == "size")
               & (df["contrast"] == "M3-M2") & (df["loss"] == "pinball")]
        if not c.empty:
            g = c.groupby("tau").agg(
                median_rel=("rel_improvement", "median"),
                n_same_direction_ci=("improves", "sum"),
                n_ci=("ci_excludes_zero", "sum"),
            )
            ext = c[np.isclose(c["tau"].astype(float), EXTREME_TAU)]
            facts["size"]["scaling_vs_tau"] = {
                "n_taus": int(len(g)),
                "grid_median_min": float(g["median_rel"].min()),
                "grid_median_max": float(g["median_rel"].max()),
                "taus_with_three_same_direction": [
                    float(t) for t in g.index[g["n_same_direction_ci"] >= 3]],
                "at_extreme": {
                    "median_rel": float(ext["rel_improvement"].median()),
                    "n_positive": int((ext["rel_improvement"] > 0).sum()),
                    "n_negative": int((ext["rel_improvement"] < 0).sum()),
                    "n_same_direction_ci": int(ext["improves"].sum()),
                    "n_ci_excludes_zero": int(ext["ci_excludes_zero"].sum()),
                    "per_origin": {int(o): float(v) for o, v in
                                   ext.set_index("origin")["rel_improvement"].items()},
                },
            }
        facts["size"]["model"] = meta.get("model")
        facts["size"]["estimator_extrapolates"] = bool(
            meta.get("extrapolating_regressor", False))
        facts["size"]["decision_rule"] = meta.get("decision_rule")

    # ---- Table 4, municipal block ---------------------------------------
    cdf, cmeta, _ = _load("city_exceedance", root)
    if cdf is not None and facts["sources"]["city_exceedance"]["present"]:
        facts["exceed"]["city"] = {
            "contrasts": _contrast_summary(cdf, "exceedance"),
            "threshold_scope": cmeta.get("classifier_threshold_scope"),
            "n_units": cmeta.get("panel_units"),
            "auc_median": {r: float(v) for r, v in
                           cdf[cdf["record_type"] == "loss"]
                           .groupby("rung")["auc"].median().items()},
        }
        # AUC lives on the per-unit ``loss`` rows of the exceedance artefact.
        # The scaling block's gain and its rank among the rungs are quoted in
        # the manuscript from *this* artefact, not from the classifier one,
        # because this is the run that carries the bootstrap intervals and the
        # verdicts; the classifier artefact exists for figures 12 and 13.
        loss = cdf[cdf["record_type"] == "loss"]
        if "auc" in loss and not loss.empty:
            piv_auc = loss.pivot_table(index="origin", columns="rung", values="auc")
            if {"M1", "M2", "M3"}.issubset(piv_auc.columns):
                gain = piv_auc["M3"] - piv_auc["M2"]
                facts["exceed"]["city"]["auc_gain_scaling_median"] = float(gain.median())
                facts["exceed"]["city"]["n_origins_scaling_best"] = int(
                    (piv_auc[["M1", "M2", "M3"]].idxmax(axis=1) == "M3").sum())
        pops = cdf[cdf["record_type"] == "diagnostic"]
        if not pops.empty:
            facts["exceed"]["city"]["population_only_auc_median"] = float(
                pops["auc"].median())
            facts["exceed"]["city"]["population_only_auc_range"] = [
                float(pops["auc"].min()), float(pops["auc"].max())]

    # ---- Table 4, state block (may be absent) ---------------------------
    sdf, _smeta, _ = _load("state_exceedance", root)
    if sdf is not None and facts["sources"]["state_exceedance"]["present"]:
        facts["exceed"]["state"] = {"contrasts": _contrast_summary(sdf, "exceedance")}

    # ---- The classifier artefact behind figures 12 and 13 ---------------
    kdf, kmeta, _ = _load("city_classifier", root)
    if kdf is not None and facts["sources"]["city_classifier"]["present"]:
        piv_auc = kdf.pivot_table(index="origin", columns="rung", values="auc")
        piv_b = kdf.pivot_table(index="origin", columns="rung", values="brier")
        facts["classifier"] = {
            "auc_median": {r: float(piv_auc[r].median()) for r in piv_auc},
            "brier_median": {r: float(piv_b[r].median()) for r in piv_b},
            "prevalence_min": float(kdf["prevalence"].min()),
            "prevalence_max": float(kdf["prevalence"].max()),
            "threshold_scope": kmeta.get("threshold_scope"),
            "n_repeats": kmeta.get("n_repeats"),
            "brier_by_origin": {
                str(int(o)): {r: float(v) for r, v in row.items() if pd.notna(v)}
                for o, row in piv_b.iterrows()},
        }
        if {"M1", "M2", "M3"}.issubset(piv_auc.columns):
            gain = piv_auc["M3"] - piv_auc["M2"]
            facts["classifier"]["auc_gain_scaling_median"] = float(gain.median())
            facts["classifier"]["n_origins_scaling_best"] = int(
                (piv_auc[["M1", "M2", "M3"]].idxmax(axis=1) == "M3").sum())
        if "train_prevalence" in kdf.columns:
            facts["classifier"]["train_prevalence"] = {
                int(o): float(v) for o, v in
                kdf.drop_duplicates("origin").set_index("origin")["train_prevalence"].items()}

        # Two artefacts of the same municipal classifier disagree slightly on
        # AUC. They are different runs (one with bootstrap intervals, one with
        # five repeated out-of-fold fits), so a small gap is expected; a large
        # one would mean they are not the same experiment, which is what this
        # bounds.
        city = facts["exceed"]["city"]
        if city and "M3" in city["auc_median"]:
            gap = abs(city["auc_median"]["M3"] - facts["classifier"]["auc_median"].get("M3", np.nan))
            facts["classifier"]["auc_gap_vs_ablation_city"] = float(gap)
            if np.isfinite(gap) and gap > 0.05:
                facts["warnings"].append(
                    f"the two municipal artefacts disagree on median AUC(M3) by "
                    f"{gap:.3f}; they may not describe the same experiment")
            elif np.isfinite(gap) and gap > 0.005:
                facts["warnings"].append(
                    f"the two municipal artefacts give median AUC(M3) as "
                    f"{city['auc_median']['M3']:.3f} (ablation_city) and "
                    f"{facts['classifier']['auc_median']['M3']:.3f} "
                    f"(classifier_results); the manuscript quotes the first and "
                    f"figure 12 draws the second")

    # ---- Fallback: municipal block from the fresh classifier artefact ----
    # The exceedance ablation carries the bootstrap intervals and therefore the
    # verdicts, so it is preferred. When it is stale, the out-of-fold classifier
    # artefact still gives the differentials and their signs -- just not the
    # intervals. Reporting the signs and marking the rule ``not tested`` is
    # honest; reporting the stale differentials would not be, because they were
    # produced by a tree classifier that the manuscript no longer uses (in 2024
    # it emits the training base rate, which is why every rung scores a Brier of
    # 0.40 in that artefact but 0.25 in the logistic one).
    if facts["exceed"]["city"] is None and facts["classifier"]:
        piv = kdf.pivot_table(index="origin", columns="rung", values="brier")
        contrasts = {}
        for later, ref in (("M1", "M0"), ("M2", "M1"), ("M3", "M2")):
            if later in piv and ref in piv:
                d = piv[ref] - piv[later]
                contrasts[f"{later}-{ref}"] = {
                    "n_origins": int(d.notna().sum()),
                    "median_diff": float(d.median()),
                    "n_diff_positive": int((d > 0).sum()),
                    "n_same_direction_ci": None,
                    "n_ci_excludes_zero": None,
                    "verdict": None,
                }
        facts["exceed"]["city"] = {
            "source": "classifier_results_metrics.csv",
            "no_intervals": True,
            "contrasts": contrasts,
            "auc_median": {r: v for r, v in facts["classifier"]["auc_median"].items()
                           if r not in ("POP",)},
            "population_only_auc_median": facts["classifier"]["auc_median"].get("POP"),
            "n_units": kmeta.get("panel_units"),
            "threshold_scope": kmeta.get("threshold_scope"),
        }
        for k in ("auc_gain_scaling_median", "n_origins_scaling_best"):
            if k in facts["classifier"]:
                facts["exceed"]["city"][k] = facts["classifier"][k]
        facts["warnings"].append(
            "the municipal block is drawn from the classifier artefact because "
            "ablation_city.csv is stale; the CI column and the pre-registered "
            "verdicts need the re-run")

    return facts


# ---------------------------------------------------------------- rendering

def render_ablation_table(f: dict) -> str:
    """LaTeX float for Table 3, built from the size ladder."""
    if not f.get("size", {}).get("skill"):
        prob = (f.get("sources", {}).get("state_size") or {}).get("problem")
        raise ValueError(
            "no size ladder in the artefacts; cannot render Table 3\n"
            f"    ablation_results.csv: {prob or 'absent'}\n"
            "    rebuild it with: python run_ablation.py --level state "
            "--attach-climate")
    s, c = f["size"]["skill"], f["size"]["contrasts"]
    cells = []
    for r in ("M1", "M2", "M3"):
        d = s[r]
        cells.append(f"${_num(d['median'])}$ [${_num(d['min'])}$, ${_num(d['max'])}$]")
    d32 = c.get("M3-M2", {})
    diff = (f"${_pct(d32['median_rel'])}$ "
            f"({d32['n_rel_positive']}/{d32['n_origins']})") if d32 else "--"
    return (
        "% GENERATED by make_ablation_tables.py from ablation_results.csv -- do not edit.\n"
        "% Regenerate with: python make_ablation_tables.py\n"
        "\\begin{table}[htbp]\n\\centering\n\\small\n"
        "\\setlength{\\tabcolsep}{4pt}\n"
        "\\caption{Ablation ladder over the five forecast origins 2021--2025, state\n"
        "level (26 units, 23--25 evaluated per origin), for the single target. Cells\n"
        "give the median across origins of the skill score\n"
        "$1-L(\\text{rung})/L(\\text{M0})$ against calibrated persistence, with the\n"
        "range over origins in brackets; the primary loss is the pinball loss at\n"
        "$\\tau=0.5$. The last column is the M3$-$M2 differential at that loss:\n"
        "median relative change, positive meaning that the scaling block improves\n"
        "the forecast, and in parentheses the number of origins in which M3 does so\n"
        "by sign alone. The classifier is reported separately in\n"
        "Table~\\ref{tab:exceed}.}\n"
        "\\label{tab:ablation_results}\n"
        "\\begin{tabular}{@{}lrrrr@{}}\n\\toprule\n"
        "Target & M1 & M2 & M3 & M3$-$M2 \\\\\n\\midrule\n"
        f"Size $S_t$ & {cells[0]} & {cells[1]} & {cells[2]} & {diff}\\\\\n"
        "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )


def render_exceed_table(f: dict) -> str:
    """LaTeX float for Table 4, state and municipal blocks.

    A block whose artefact is absent is rendered as a single conspicuous row
    rather than omitted. Dropping it would silently delete the paper's only
    ``supported`` verdict; keeping the old numbers would be worse. A placeholder
    cannot be overlooked in the compiled PDF and disappears the moment the run
    happens.
    """
    def rows(block: dict, label: str) -> list[str]:
        names = ("M1$-$M0 (lagged metrics)", "M2$-$M1 (climate)",
                 "M3$-$M2 (scaling)")
        out = []
        for i, (c, name) in enumerate(zip(ADJACENT, names)):
            d = block["contrasts"][c]
            head = label if i == 0 else ""
            if d.get("verdict") == "supported":
                verdict = "\\textbf{supported}"
            elif d.get("verdict") == "not_supported":
                verdict = "not supported"
            else:
                verdict = "\\emph{not tested}"
            ci = ("--" if d.get("n_same_direction_ci") is None
                  else f"{d['n_same_direction_ci']}/{d['n_origins']}")
            out.append(
                f"{head} & {name} & ${d['median_diff']:+.3f}$ & "
                f"{d['n_diff_positive']}/{d['n_origins']} & {ci} & {verdict}\\\\")
        return out

    lines: list[str] = []
    state = f["exceed"].get("state")
    if state:
        lines += rows(state, "State (26 units)")
    else:
        lines.append(
            "\\multicolumn{6}{@{}l}{\\emph{State level: no committed artefact; run "
            "\\texttt{run\\_ablation.py --level state --tasks exceedance}}}\\\\")
    lines.append("\\addlinespace")

    city = f["exceed"].get("city")
    if city:
        n_units = city.get("n_units")
        label = (f"Municipal ({int(n_units):,} units)".replace(",", "{,}")
                 if n_units else "Municipal")
        lines += rows(city, label)
    else:
        lines.append("\\multicolumn{6}{@{}l}{\\emph{Municipal level: no artefact}}\\\\")

    pop = (city or {}).get("population_only_auc_median")
    aucs = [v for v in ((city or {}).get("auc_median") or {}).values() if v > 0.5]
    auc_txt = f"${min(aucs):.2f}$--${max(aucs):.2f}$" if aucs else "the fitted rungs"

    caption = (
        "\\caption{The large-season classifier at both aggregation levels, five\n"
        "forecast origins 2021--2025. The target is whether a unit exceeds the 75th\n"
        "percentile of its own history; the loss is the Brier score. Positive values\n"
        "mean the later rung has the lower Brier score, and ``median'' is over the\n"
        "five origins. ``Origins CI excl.\\ 0'' counts the origins with a\n"
        "same-direction improvement whose 95\\% unit-bootstrap interval excludes\n"
        "zero, which is what the decision rule of Section~\\ref{sec:ablation}\n"
        "requires in at least 3 of 5. The municipal rows use a within-unit\n"
        "threshold, without which the label is a population indicator; there the\n"
        "population-only reference reaches a median AUC of "
        f"${pop:.2f}$ against {auc_txt} for the fitted rungs.}}"
        if pop else
        "\\caption{The large-season classifier at both aggregation levels.}"
    )
    return (
        "% GENERATED by make_ablation_tables.py -- do not edit.\n"
        "% Regenerate with: python make_ablation_tables.py\n"
        "\\begin{table}[htbp]\n\\centering\n\\small\n"
        "\\setlength{\\tabcolsep}{5pt}\n" + caption + "\n"
        "\\label{tab:exceed}\n"
        "\\begin{tabular}{@{}llrrrr@{}}\n\\toprule\n"
        "Level & Contrast & $\\Delta$Brier, median & Origins $\\Delta>0$ & "
        "Origins CI excl.\\ 0 & Verdict\\\\\n\\midrule\n"
        + "\n".join(lines) +
        "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )


# ------------------------------------------------------------------- claims

def claims(f: dict) -> list[dict]:
    """Prose statements the artefacts must satisfy.

    ``want`` must appear in ``main.tex``; each ``stale`` entry is a value that
    was actually published in an earlier draft, so a regression to it is caught
    by name rather than by re-deriving it. ``needs`` names the artefact a claim
    depends on: when that artefact is missing the claim is reported as
    UNVERIFIED instead of being silently skipped.
    """
    s = f.get("size", {}).get("skill") or {}
    c = f.get("size", {}).get("contrasts") or {}
    tau = f.get("size", {}).get("scaling_vs_tau") or {}
    ext = tau.get("at_extreme") or {}
    clf = f.get("classifier") or {}
    city = f.get("exceed", {}).get("city") or {}
    out: list[dict] = []

    if s:
        m1 = s["M1"]
        out += [
            {"name": "M1 median skill",
             "want": f"median skill ${_num(m1['median'])}$",
             "stale": ["median skill $0.10$"], "needs": "state_size"},
            {"name": "M1 skill range",
             "want": f"${_num(m1['min'])}$ to ${_num(m1['max'])}$",
             "stale": ["$-0.59$ to $0.71$"], "needs": "state_size"},
            {"name": "M2 median skill",
             "want": f"M2 has median ${_num(s['M2']['median'])}$",
             "stale": ["M2 has median $-0.02$"], "needs": "state_size"},
            {"name": "M3 median skill",
             "want": f"and M3 ${_num(s['M3']['median'])}$",
             "stale": ["M3 $-0.14$"], "needs": "state_size"},
        ]
    if "M3-M2" in c:
        d = c["M3-M2"]
        out.append(
            {"name": "M3-M2 differential at primary tau",
             "want": f"of ${_pct(d['median_rel'])}$",
             "stale": ["$-1.5\\%$"], "needs": "state_size"})
    if ext:
        out += [
            {"name": "extreme-tau median differential",
             "want": f"$\\tau={f['extreme_tau']:.2f}$ the median differential is "
                     f"${_pct(ext['median_rel'])}$",
             "stale": ["median $-18.3\\%$", "median of $18.3\\%$"],
             "needs": "state_size"},
            {"name": "extreme-tau sign count",
             "want": f"positive in {ext['n_positive']} of "
                     f"{ext['n_positive'] + ext['n_negative']} origins",
             "stale": ["negative in \\emph{all five} origins",
                       "in every origin tested"],
             "needs": "state_size"},
            {"name": "extreme-tau interval count",
             "want": f"2 of {ext['n_positive'] + ext['n_negative']}",
             "stale": [], "needs": "state_size"},
            {"name": "no tau clears the rule",
             "want": f"{len(tau.get('taus_with_three_same_direction', []))} "
                     f"of {tau.get('n_taus')} quantiles",
             "stale": [], "needs": "state_size"},
        ]
    if city:
        auc = city.get("auc_median") or {}
        d32 = city["contrasts"]["M3-M2"]
        # Which artefact backs the municipal claims: the ablation when it is
        # fresh (it has the intervals), otherwise the classifier.
        src = ("city_exceedance" if not city.get("no_intervals")
               else "city_classifier")
        if {"M1", "M2", "M3"}.issubset(auc):
            out.append(
                {"name": "municipal AUC ladder",
                 "want": f"{auc['M1']:.2f}$, ${auc['M2']:.2f}$ and ${auc['M3']:.2f}$",
                 "stale": ["$0.59$, $0.63$ and $0.64$"], "needs": src})
        out.append(
            {"name": "municipal scaling Brier",
             "want": f"median of ${d32['median_diff']:+.3f}$",
             "stale": [], "needs": src})
        out.append(
            {"name": "municipal origins where the scaling block helps",
             "want": f"in {d32['n_diff_positive']} of {d32['n_origins']} origins",
             "stale": ["improves the Brier score in 4 of\n5 origins"],
             "needs": src})
        if "auc_gain_scaling_median" in city:
            out.append(
                {"name": "municipal AUC gain of the scaling block",
                 "want": f"median ${city['auc_gain_scaling_median']:+.3f}$ over M2",
                 "stale": ["median $+0.008$ over M2"], "needs": src})
        if "n_origins_scaling_best" in city:
            out.append(
                {"name": "municipal origins where M3 ranks best",
                 "want": f"best of the three in {city['n_origins_scaling_best']} of 5",
                 "stale": ["best of the three in 4 of 5"], "needs": src})

    # Numbers the manuscript quotes about state-level exceedance. They are
    # unverifiable while ablation_state_exceedance.csv is missing, and that is
    # the point of listing them.
    out += [
        {"name": "state exceedance supported in 3 of 5",
         "want": "3 of the 5 origins", "stale": [],
         "needs": "state_exceedance"},
        {"name": "state exceedance per-origin differentials",
         "want": "$-0.047$, $+0.074$, $+0.134$, $+0.175$ and $+0.159$",
         "stale": [], "needs": "state_exceedance"},
    ]
    # The record-year paragraph: its numbers come from the classifier artefact,
    # and its base rate needs a column that only a re-run of run_classifier.py
    # writes.
    if clf:
        b24 = (clf.get("brier_by_origin") or {}).get("2024") if clf else None
        if b24:
            out.append(
                {"name": "2024 Brier by rung",
                 "want": f"the fitted rungs about ${b24['M3']:.2f}$",
                 "stale": ["every rung scores a Brier score of about $0.40$"],
                 "needs": "city_classifier"})
        out.append(
            {"name": "2024 training base rate",
             "want": "training base rate of 26.4\\%", "stale": [],
             "needs": "train_prevalence"})

    for o in out:
        if o["needs"] == "train_prevalence":
            o["available"] = "train_prevalence" in (
                facts_get(objs=(f.get("classifier") or {})))
        else:
            o["available"] = f["sources"].get(o["needs"], {}).get("present", False)
    return [o for o in out if o["want"]]


def audit_prose(tex_path: str, f: dict) -> dict:
    """Sort every claim into ok / stale / missing / unverified."""
    res = {"stale": [], "missing": [], "unverified": [], "ok": []}
    if not os.path.exists(tex_path):
        res["stale"].append({"name": "main.tex", "found": ["absent"],
                             "want": tex_path})
        return res
    tex = open(tex_path, encoding="utf-8", errors="replace").read()
    for c in claims(f):
        if not c["available"]:
            res["unverified"].append({"name": c["name"], "want": c["want"],
                                      "source": c["needs"]})
            continue
        stale = [s for s in c["stale"] if s in tex]
        if stale:
            res["stale"].append({"name": c["name"], "found": stale,
                                 "want": c["want"]})
        elif c["want"] not in tex:
            res["missing"].append({"name": c["name"], "want": c["want"]})
        else:
            res["ok"].append(c["name"])
    # Stale literals are checked even when their source is missing: a retired
    # value is a retired value whether or not the replacement is available.
    seen = {x["name"] for x in res["stale"]}
    for c in claims(f):
        if c["available"] or c["name"] in seen or not c["stale"]:
            continue
        hit = [s for s in c["stale"] if s in tex]
        if hit:
            res["stale"].append({"name": c["name"], "found": hit,
                                 "want": "value of the superseded run"})
    return res


#: Sources whose absence blocks a table outright, as opposed to one that only
#: withdraws a column. ``state_size`` is Table 3; ``state_exceedance`` is the
#: paper's only ``supported`` verdict, and it is currently reproducible by no
#: committed script -- so it must fail loudly rather than read as an omission.
HARD_SOURCES = ("state_size", "state_exceedance")


def format_sources(f: dict) -> str:
    """One line per source artefact, with the reason when it cannot be used."""
    rows = []
    for name, s in f.get("sources", {}).items():
        hard = " (required)" if name in HARD_SOURCES else ""
        rows.append(f"{'ok     ' if s.get('present') else 'MISSING'}  "
                    f"{s['path']:32s}  {s['what']}{hard}")
        if not s.get("present"):
            rows.append(f"         -> {s.get('problem')}")
            if s.get("how"):
                rows.append(f"         -> rebuild: {s['how']}")
    return "\n".join(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=".")
    ap.add_argument("--tex", default=os.path.join("Manuscript", "main.tex"))
    ap.add_argument("--check", action="store_true", help="audit, do not write")
    ap.add_argument("--print", dest="dump", action="store_true")
    ap.add_argument("--sources", action="store_true",
                    help="report the state of the ablation artefacts and exit; "
                         "exit 1 when a source a table needs is unusable")
    ap.add_argument("--allow-unverified", action="store_true",
                    help="treat unverifiable claims as warnings, not failures")
    a = ap.parse_args(argv)

    f = compute_facts(a.root)

    if a.dump:
        print(json.dumps(f, indent=2, sort_keys=True, default=float))
        return 0

    if a.sources:
        print(format_sources(f))
        for w in f["warnings"]:
            print(f"note: {w}")
        bad = [n for n in HARD_SOURCES
               if not f["sources"].get(n, {}).get("present")]
        if bad:
            print(f"unusable required source(s): {', '.join(bad)}", file=sys.stderr)
        return 1 if bad else 0

    audit = audit_prose(os.path.join(a.root, a.tex), f)

    if a.check:
        for s in audit["stale"]:
            print(f"STALE      {s['name']}: found {s['found']} -> want {s['want']!r}",
                  file=sys.stderr)
        for m in audit["missing"]:
            print(f"MISSING    {m['name']}: {m['want']!r} not present", file=sys.stderr)
        for u in audit["unverified"]:
            print(f"UNVERIFIED {u['name']}: needs {u['source']}", file=sys.stderr)
        for w in f["warnings"]:
            print(f"NOTE       {w}", file=sys.stderr)
        bad = bool(audit["stale"] or audit["missing"]
                   or (audit["unverified"] and not a.allow_unverified))
        print(f"ablation tables: {len(audit['ok'])} claims verified, "
              f"{len(audit['stale'])} stale, {len(audit['missing'])} missing, "
              f"{len(audit['unverified'])} unverifiable")
        return 1 if bad else 0

    gen = os.path.join(a.root, GEN_DIR)
    os.makedirs(gen, exist_ok=True)
    # Render both before writing either. Writing them in a loop meant a failure
    # on the second left a fresh Table 3 beside a stale Table 4, which is worse
    # than writing neither: the manuscript would be internally inconsistent and
    # nothing would say so.
    rendered = {}
    for name, fn in (("tab_ablation.tex", render_ablation_table),
                     ("tab_exceed.tex", render_exceed_table)):
        try:
            rendered[name] = fn(f)
        except ValueError as e:
            print(f"cannot write {name}: {e}", file=sys.stderr)
            print("no table was written, so Manuscript/generated/ is unchanged",
                  file=sys.stderr)
            return 2
    for name, tex in rendered.items():
        open(os.path.join(gen, name), "w", encoding="utf-8").write(tex)
        print(f"wrote {GEN_DIR}/{name}")
    with open(os.path.join(gen, "ablation_facts.json"), "w", encoding="utf-8") as fh:
        json.dump(f, fh, indent=2, sort_keys=True, default=float)
        fh.write("\n")
    print(f"wrote {GEN_DIR}/ablation_facts.json")
    for w in f["warnings"]:
        print(f"  note: {w}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
