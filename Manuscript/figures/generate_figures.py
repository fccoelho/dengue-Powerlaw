"""Generate candidate figures for every figure environment in main.tex.

Usage:
    python Manuscript/figures/generate_figures.py            # all 14
    python Manuscript/figures/generate_figures.py 2 4 6      # subset

Every figure is built from the repository's own artefacts (powerlaw_results.db,
data/episcanner_*.parquet, data/<geocode>.parquet, muni_br.gpkg) using the same
estimators as the pipeline, so each PNG/PDF is a *candidate* for its figure
environment, not a placeholder: captions in main.tex describe exactly what is
drawn. Predictive figures (10-15) retrain the models described in the Methods;
their numbers therefore move when the database is refreshed, and the figures
should be regenerated after every data update.

Three exceptions, deliberately. Figures 11, 12 and 13 are *read* from
artefacts, not recomputed here:

* Figure 11 reads ``ablation_results.csv`` / ``.meta.json`` from
  ``run_ablation.py``. Recomputing it was the defect: the previous version ran
  its own ladder at city level with a random forest, a raw persistence
  baseline and a substring-discovered climate block, so it could disagree with
  Table 4 -- and did, once the climate block silently came up empty.
* Figures 12 and 13 read ``classifier_results*`` from ``run_classifier.py``.
  The previous version fitted its own classifier with a single split and, at
  municipality level, a Q_75 pooled across every municipality in the country.
  That label is answerable from population size alone -- the population-only
  reference reaches AUC 0.94 against 0.96 for the full model -- so the ROC
  panel was reporting the ability to tell a large municipality from a small
  one. ``_load_classifier`` refuses such an artefact instead of drawing it.

A figure that is a view of the same artefact cannot disagree with it.
"""
import json
import sqlite3
import sys
import time
import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
import os  # noqa: E402

os_cwd = Path.cwd()
if str(ROOT) != str(os_cwd):
    os.chdir(ROOT)  # the data layer uses relative paths ("data/...", "muni_br.gpkg")

warnings.filterwarnings("ignore")

OUT = HERE
PANEL_CACHE = HERE / "_panel_city_BR.parquet"

#: The ablation artefacts written by ``run_ablation.py`` at the repository root.
#: Figure 11 is a *view* of these files, not a recomputation: the numbers in the
#: manuscript, in the CSV and in the figure are then the same numbers by
#: construction. Set ``EPIDEMIC_ABLATION_CSV`` to point elsewhere.
ABLATION_CSV = Path(os.environ.get("EPIDEMIC_ABLATION_CSV",
                                   ROOT / "ablation_results.csv"))
ABLATION_META = ABLATION_CSV.with_suffix("").with_suffix(".meta.json")

#: The rarity-classifier artefacts written by ``run_classifier.py``. Figures 12
#: and 13 are *views* of these files, never a recomputation: a figure that fits
#: its own classifier can disagree with the table it is meant to illustrate.
CLASSIFIER_PREFIX = Path(os.environ.get("EPIDEMIC_CLASSIFIER_PREFIX",
                                        ROOT / "classifier_results"))

mpl.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
})

#: Five constructions of the same municipal series: the weekly count at two
#: temporal windows and the epidemic size at three dimensions of pooling. The
#: duration and peak-timing constructions are gone: a season is bounded above by
#: the length of the year, so a scale-free law cannot apply to duration a priori,
#: and the fits said so by running into the estimator's ceiling. Duration and
#: peak timing survive as covariates of size.
LEVELS = {
    "W-mun":   ("powerlaw_fits", "", "weekly, city"),
    "W-seas":  ("powerlaw_fits_yearly", "", "weekly, city-season"),
    "S-mun":   ("powerlaw_fits_total_cases", "", "size, city"),
    "S-state": ("episcanner_state_fits", "metric='total_cases' AND year>0",
                "size, state-year"),
    "S-nat":   ("episcanner_fits", "metric='total_cases' AND year>0",
                "size, national-year"),
}
SIZE_LEVELS = list(LEVELS)
COLORS = plt.cm.tab10.colors


def _read(sql):
    with sqlite3.connect("powerlaw_results.db") as conn:
        return pd.read_sql_query(sql, conn)


def _level_alpha(where_extra=""):
    out = {}
    for lvl, (table, where, _label) in LEVELS.items():
        sql = f"SELECT alpha, xmin, R, p FROM {table}"
        conds = [c for c in (where, where_extra) if c]
        if conds:
            sql += " WHERE " + " AND ".join(conds)
        df = _read(sql).dropna(subset=["alpha"])
        out[lvl] = df
    return out


def _episcanner_all():
    import glob

    files = sorted(glob.glob("data/episcanner_*.parquet"))
    return pd.concat(
        [pd.read_parquet(f, columns=["geocode", "year", "total_cases", "ep_dur"])
         for f in files],
        ignore_index=True,
    ).dropna(subset=["total_cases", "ep_dur"])


def _city_panel():
    if PANEL_CACHE.exists():
        return pd.read_parquet(PANEL_CACHE)
    from predictive_model import prepare_lagged_data

    df = prepare_lagged_data("BR", target_year=None, level="city")
    df.to_parquet(PANEL_CACHE)
    return df


# ---------------------------------------------------------------------------
# shared: the ablation artefact (fig 11)
# ---------------------------------------------------------------------------
ABL_REQUIRED = ("record_type", "task", "target", "origin", "rung", "contrast",
                "loss", "tau", "value", "ci_low", "ci_high")

#: Epidemic size is the only forecasting target. Duration and peak timing are
#: covariates in the model, not outcomes.
TASK_LABEL = {"total_cases": "size $S_t$"}
TARGETS_ALLOWED = set(TASK_LABEL)
RUNG_COLORS = {"M1": "#9ecae1", "M2": "#4292c6", "M3": "#08519c"}


def _load_ablation(csv_path=None):
    """Read and *validate* the ablation artefact. Never recompute it here.

    The validation is the point of this function. The defect that produced a
    full column of zeros -- M2 silently identical to M1 because the climate
    block was discovered by substring and came up empty -- is exactly the kind
    of failure that is invisible in a figure. Each check below converts it into
    a loud abort with an actionable message.
    """
    csv_path = Path(csv_path or ABLATION_CSV)
    meta_path = csv_path.with_suffix("").with_suffix(".meta.json")
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. Run `python run_ablation.py` first; this "
            "figure is a view of that artefact, it does not recompute it.")

    raw = pd.read_csv(csv_path)
    missing = [c for c in ABL_REQUIRED if c not in raw.columns]
    if missing:
        raise ValueError(f"{csv_path} is missing columns {missing}")

    meta = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
    else:
        warnings.warn(f"{meta_path} not found; cannot check the ladder.", stacklevel=2)

    # -- guard 1: M2 must add something to M1 ------------------------------
    ladder = meta.get("ladder", {})
    if ladder:
        m1, m2, m3 = (list(ladder.get(k, [])) for k in ("M1", "M2", "M3"))
        if m2 and m1 and set(m2) == set(m1):
            raise RuntimeError(
                "ablation artefact is defective: rung M2 has exactly the same "
                f"features as M1 ({len(m1)} columns). The M2-M1 contrast is "
                "then identically zero and the figure would read 'climate adds "
                "nothing' when climate never entered. Re-run `python "
                "run_ablation.py --attach-climate`.")
        if m3 and m2 and set(m3) == set(m2):
            raise RuntimeError(
                "ablation artefact is defective: M3 adds nothing to M2, so the "
                "scaling block this figure exists to report is absent.")
        if m1 and m2 and not set(m1) < set(m2):
            raise RuntimeError("ladder is not nested: M1 is not a subset of M2.")

    # -- guard 2: the climate block must be populated -----------------------
    climate = meta.get("climate", {})
    if climate and not climate.get("available"):
        raise RuntimeError(
            "ablation artefact reports an empty climate block "
            f"(missing: {climate.get('missing')}). M2 is undefined; the figure "
            "would misattribute the M3-M2 contrast to scaling alone.")

    # -- guard 3: the M2-M1 contrast must actually vary ----------------------
    m21 = raw[(raw.record_type == "contrast") & (raw.contrast == "M2-M1")]
    if len(m21) and float(np.nanmax(np.abs(m21.value))) == 0.0:
        raise RuntimeError(
            "every M2-M1 contrast is exactly 0.0 with a [0, 0] interval: this "
            "is the 'climate never entered' signature, not a result.")

    # -- guard 4: no stale duration/peak target may survive -----------------
    stale = sorted(set(raw.loc[raw.record_type == "loss", "target"].dropna())
                   - TARGETS_ALLOWED)
    if stale:
        raise RuntimeError(
            f"the ablation artefact still reports the target(s) {stale}. "
            "Duration and peak timing are covariates now, not targets, and "
            "their duration scaling level no longer exists. Re-run "
            "`python run_ablation.py` -- a view of a superseded design is "
            "worse than no figure."
        )
    stale_cols = [c for rung in meta.get("ladder", {}).values() for c in rung
                  if c.startswith(("prev_alpha_dur", "prev_xmin_dur"))]
    if stale_cols:
        raise RuntimeError(
            f"the ablation ladder still carries duration scaling descriptors "
            f"{sorted(set(stale_cols))}; they no longer exist in the database."
        )

    return {"raw": raw, "meta": meta, "path": csv_path}


def _ablation_skill(abl, tau=0.5):
    """Median skill vs M0 per (target, rung), plus the per-origin scatter."""
    sk = abl["raw"]
    sk = sk[(sk.record_type == "skill") & (sk.loss == "pinball")
            & np.isclose(sk.tau, tau)]
    return sk


def _ablation_relative(abl):
    """100 * (L_M2 - L_M3) / L_M2 per (target, tau, origin); positive = M3 better."""
    lo = abl["raw"]
    lo = lo[(lo.record_type == "loss") & (lo.loss == "pinball")]
    piv = lo.pivot_table(index=["target", "tau", "origin"],
                         columns="rung", values="value").reset_index()
    piv = piv.dropna(subset=["M2", "M3"])
    piv["rel"] = (piv["M2"] - piv["M3"]) / piv["M2"] * 100.0
    return piv


# ---------------------------------------------------------------------------
# Fig 01 -- schematic of the seven aggregation levels
# ---------------------------------------------------------------------------
def _load_classifier():
    """Read and validate the rarity-classifier artefacts. Never recompute them.

    The checks are the point. A municipality-level artefact built with a pooled
    threshold carries a label that a population count answers on its own -- the
    population-only reference reaches AUC ~0.94 against ~0.96 for the full
    model -- so the ROC panel would look excellent while measuring nothing but
    municipality size. Such an artefact is refused here instead of drawn.
    """
    oof_path = CLASSIFIER_PREFIX.with_suffix(".parquet")
    meta_path = CLASSIFIER_PREFIX.with_suffix(".meta.json")
    metrics_path = CLASSIFIER_PREFIX.parent / f"{CLASSIFIER_PREFIX.name}_metrics.csv"
    imp_path = CLASSIFIER_PREFIX.parent / f"{CLASSIFIER_PREFIX.name}_importance.csv"

    if not oof_path.exists() or not meta_path.exists():
        raise RuntimeError(
            f"classifier artefact not found at {oof_path} / {meta_path}. "
            "Build it first:\n"
            "    python run_classifier.py --level city --origins "
            "2021,2022,2023,2024,2025\n"
            "Figures 12 and 13 do not fit their own classifier: a figure that "
            "recomputes the experiment can disagree with the table it is "
            "supposed to illustrate."
        )

    meta = json.loads(meta_path.read_text())
    oof = pd.read_parquet(oof_path)
    metrics = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()
    importance = pd.read_csv(imp_path) if imp_path.exists() else pd.DataFrame()

    missing = {"M2", "M3"} - set(oof["rung"].unique())
    if missing:
        raise RuntimeError(
            f"the OOF artefact has no {sorted(missing)} predictions; it cannot "
            "support a scaling-vs-climate comparison"
        )
    if meta.get("level") == "city" and meta.get("threshold_scope") == "pooled":
        raise RuntimeError(
            "refusing a municipality-level artefact built with "
            "--threshold-scope pooled: a single Q_p across thousands of "
            "municipalities of very different size labels a small municipality "
            "0 in every year it exists, so the label is a population indicator. "
            "Rebuild with --threshold-scope per_unit (the default at --level city)."
        )
    if oof["y_true"].nunique() < 2:
        raise RuntimeError("the OOF artefact carries a single class; nothing to plot")
    if oof["origin"].nunique() < 2:
        raise RuntimeError(
            f"the OOF artefact covers {oof['origin'].nunique()} forecast origin; "
            "an out-of-fold curve needs at least two"
        )

    if not metrics.empty and {"POP", "M3"} <= set(metrics["rung"]):
        gap = float((metrics[metrics.rung == "M3"]["auc"].mean()
                     - metrics[metrics.rung == "POP"]["auc"].mean()))
        if gap < 0.05:
            print(f"  [classifier] note: M3 beats the population-only reference "
                  f"by only {gap:+.3f} AUC", flush=True)

    return {"oof": oof, "meta": meta, "metrics": metrics,
            "importance": importance, "path": oof_path}


# ---------------------------------------------------------------------------
# Fig 02 -- alpha by level (boxes), size above, duration below
# ---------------------------------------------------------------------------
def fig02():
    """Distribution of the estimated exponent at each construction."""
    data = _level_alpha()
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    vals = [data[lvl]["alpha"].values for lvl in SIZE_LEVELS]
    bp = ax.boxplot(vals, tick_labels=SIZE_LEVELS, whis=1.5, showfliers=False,
                    patch_artist=True, widths=0.55)
    for patch in bp["boxes"]:
        patch.set_facecolor("#aec7e8")
    ax.axhline(2.0, ls="--", lw=0.9, color="0.4")
    ax.text(0.02, 0.92, r"$\alpha=2$", transform=ax.transAxes, fontsize=9,
            color="0.35")
    ax.set_xlabel("construction")
    ax.set_ylabel(r"$\alpha$")
    ax.set_title("estimated exponent by construction (dashed: $\\alpha=2$)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "fig02_alpha_by_level.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig02_alpha_by_level.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 03 -- verdict proportions by level
# ---------------------------------------------------------------------------
def fig03():
    rows = []
    for lvl, (table, where, label) in LEVELS.items():
        sql = f"SELECT alpha, xmin, R, p FROM {table}"
        if where:
            sql += " WHERE " + where
        d = _read(sql).dropna(subset=["R", "p"])
        rows.append({
            "level": lvl,
            "PL pref": 100 * (d["R"] > 0).mean(),
            "PL pref+sig": 100 * ((d["R"] > 0) & (d["p"] < 0.05)).mean(),
            "Exp sig": 100 * ((d["R"] < 0) & (d["p"] < 0.05)).mean(),
        })
    df = pd.DataFrame(rows)
    x = np.arange(len(df))
    w = 0.27
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    ax.bar(x - w, df["PL pref"], w, label="power law preferred ($R>0$)", color="#8fbf8f")
    ax.bar(x, df["PL pref+sig"], w, label="power law, $p<0.05$", color="#2e7d32")
    ax.bar(x + w, df["Exp sig"], w, label="exponential, $p<0.05$", color="#c62828")
    ax.set_xticks(x, df["level"])
    ax.set_ylabel("share of fits (%)")
    ax.legend(ncol=3, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    fig.tight_layout()
    fig.savefig(OUT / "fig03_verdict_by_level.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig03_verdict_by_level.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 04 -- CCDFs at the national cross-section, pooled 15 years
# ---------------------------------------------------------------------------
def fig04():
    """CCDF of epidemic size at the national cross-section, with both models.

    The figure used to carry a second panel: the same CCDF for epidemic
    duration, drawn to show that it bends downwards while size does not. That
    contrast is gone with the duration aggregation levels -- duration is
    bounded by the length of the season, so the comparison was between a
    question the model can answer and one it cannot. What remains is the
    claim the paper actually uses, with the exponential alternative overlaid
    on the same axes.
    """
    from powerlaw import Fit
    from scipy.special import zeta

    df = _episcanner_all()
    nat = _read(
        "SELECT metric, alpha, xmin FROM episcanner_fits WHERE year=0"
    ).set_index("metric")

    v = df["total_cases"].values
    v = v[v > 0]
    xmin = float(nat.loc["total_cases", "xmin"])
    alpha = float(nat.loc["total_cases", "alpha"])

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    fit = Fit(v, discrete=True, xmin=xmin)
    fit.plot_ccdf(ax=ax, color="0.55", label="empirical (pooled 2011–2026)")

    s = np.logspace(np.log10(xmin), np.log10(v.max()), 200)
    # discrete power law tail: Pr(X >= s) = zeta(alpha, s) / zeta(alpha, xmin)
    ax.plot(s, zeta(alpha, s) / zeta(alpha, xmin), "k-", lw=1.5,
            label=rf"power law ($\alpha={alpha:.2f}$, $x_\min={xmin:.0f}$)")

    # exponential alternative fitted on the same tail
    lam = 1.0 / v[v >= xmin].mean()
    ax.plot(s, np.exp(-lam * (s - xmin)), "k--", lw=1.4,
            label=f"exponential (mean {1 / lam:.0f} cases)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("epidemic size $s$ (cases)")
    ax.set_ylabel(r"$\Pr(X \geq x)$")
    ax.legend(fontsize=8, loc="lower left")
    ax.set_title("epidemic size over two decades above $x_\\min$", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "fig04_size_ccdf.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig04_size_ccdf.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 05 -- xmin diagnostic: alpha vs xmin by level
# ---------------------------------------------------------------------------
def fig05():
    data = _level_alpha()
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    ax.axvspan(0.5, 5.5, color="0.9", zorder=0)
    ax.text(1.4, 0.02, r"$x_{\min}\leq5$: fitting the bulk, not a tail",
            transform=ax.get_xaxis_transform(), fontsize=8, color="0.35")
    for i, lvl in enumerate(LEVELS):
        d = data[lvl]
        ax.scatter(d["xmin"].clip(lower=1), d["alpha"], s=3, alpha=0.25,
                   color=COLORS[i], label=f"{lvl} {LEVELS[lvl][2]}")
    ax.set_xscale("log")
    ax.set_xlabel(r"selected lower bound $x_{\min}$")
    ax.set_ylabel(r"estimated $\alpha$")
    ax.legend(fontsize=6.5, ncol=2, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "fig05_xmin_diagnostic.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig05_xmin_diagnostic.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 06 -- four individual fits with both alternatives
# ---------------------------------------------------------------------------
def fig06():
    """Three individual fits, with the power-law and exponential alternatives.

    The former panel (d) showed the same cross-section for epidemic duration,
    to make the point that the curve bends downwards there. With the duration
    constructions removed there is no duration fit to contrast, and the point
    survives without it: what panels (a), (b) and (c) share is that the same
    estimator is being applied to three different objects, and only the third
    one describes a tail.
    """
    from powerlaw import Fit

    small = pd.read_parquet("data/3117836.parquet")["casos_est"]
    large = pd.read_parquet("data/3550308.parquet")["casos_est"]
    epi = pd.read_parquet("data/episcanner_SP_2024.parquet")

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5))
    panels = [
        (axes[0], small.dropna().values,
         "C\u00f4nego Marinho (3117836)\nweekly cases, 2010\u20132026"),
        (axes[1], large.dropna().values,
         "S\u00e3o Paulo (3550308)\nweekly cases, 2010\u20132026"),
        (axes[2], epi["total_cases"].dropna().values,
         "SP state-year cross-section 2024\nepidemic size"),
    ]
    for ax, v, title in panels:
        v = v[v > 0]
        fit = Fit(v, discrete=True)
        fit.power_law.plot_ccdf(ax=ax, color="#2e7d32", lw=1.4, label="power law")
        try:
            fit.exponential.plot_ccdf(ax=ax, color="k", ls="--", lw=1.2,
                                      label="exponential")
        except Exception:
            pass
        fit.plot_ccdf(ax=ax, color="0.55", label="empirical")
        pl_pref = ""
        try:
            R, p = fit.distribution_compare("power_law", "exponential")
            pl_pref = f"  R={R:.1f}, p={p:.3f}"
        except Exception:
            pass
        ax.set_title(title + pl_pref, fontsize=8)
        ax.set_xlabel("")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "fig06_example_fits.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig06_example_fits.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 07 -- the W-mun/W-seas window flip
# ---------------------------------------------------------------------------
def fig07():
    l1 = _read("SELECT alpha, geocode FROM powerlaw_fits").dropna(subset=["alpha"])
    l2 = _read(
        "SELECT alpha FROM powerlaw_fits_yearly WHERE geocode IN "
        f"(SELECT geocode FROM powerlaw_fits)"
    ).dropna(subset=["alpha"])
    l1R = _read("SELECT R, p FROM powerlaw_fits").dropna(subset=["R", "p"])
    l2R = _read(
        "SELECT R, p FROM powerlaw_fits_yearly WHERE geocode IN "
        f"(SELECT geocode FROM powerlaw_fits)"
    ).dropna(subset=["R", "p"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.3))
    bins = np.linspace(0.5, 6, 60)
    ax1.hist(l1["alpha"], bins=bins, density=True, alpha=0.65,
             label=f"W-mun full series (n={len(l1):,})", color="#8fbf8f")
    ax1.hist(l2["alpha"], bins=bins, density=True, alpha=0.65,
             label=f"W-seas single seasons (n={len(l2):,})", color="#e8a09a")
    ax1.axvline(2.0, ls="--", lw=0.9, color="0.4")
    ax1.set_xlabel(r"estimated $\alpha$")
    ax1.set_ylabel("density")
    ax1.legend(fontsize=8)

    stats = pd.DataFrame({
        "W-mun, full series": [100 * (l1R["R"] > 0).mean(), 100 * ((l1R["R"] < 0) & (l1R["p"] < .05)).mean()],
        "W-seas, single seasons": [100 * (l2R["R"] > 0).mean(), 100 * ((l2R["R"] < 0) & (l2R["p"] < .05)).mean()],
    }, index=["power law preferred", "exponential, $p<0.05$"])
    stats.T.plot.bar(ax=ax2, rot=0, color=["#2e7d32", "#c62828"], legend=True)
    ax2.set_ylabel("share of fits (%)")
    for c in ax2.containers:
        ax2.bar_label(c, fmt="%.0f", fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "fig07_window_flip.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig07_window_flip.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 08 -- stability by year per state + alpha vs population
# ---------------------------------------------------------------------------
def fig08():
    st = _read(
        "SELECT state, year, alpha FROM episcanner_state_fits "
        "WHERE metric='total_cases' AND year>0"
    )
    l1 = _read(
        "SELECT f.alpha, c.population FROM powerlaw_fits f "
        "JOIN city_stats c USING (geocode) WHERE c.population>0"
    ).dropna()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.5))
    pivot = st.pivot_table(index="year", columns="state", values="alpha")
    ax1.plot(pivot.index, pivot.values, lw=0.7, alpha=0.6)
    ax1.plot(pivot.index, pivot.median(axis=1), lw=2.2, color="k",
             label="national median")
    ax1.set_xlabel("year")
    ax1.set_ylabel(r"$\alpha$ (size, state cross-section)")
    ax1.legend(fontsize=8)
    ax1.axhline(2.0, ls="--", lw=0.8, color="0.5")

    ax2.scatter(np.log10(l1["population"]), l1["alpha"], s=3, alpha=0.2, color="0.4")
    ax2.set_xlabel(r"$\log_{10}$ municipal population")
    ax2.set_ylabel(r"$\alpha$ (weekly, city)")
    from scipy.stats import pearsonr, spearmanr

    pr = pearsonr(np.log10(l1["population"]), l1["alpha"])
    sr = spearmanr(np.log10(l1["population"]), l1["alpha"])
    ax2.set_title(f"Pearson r={pr[0]:.3f}, Spearman $\\rho$={sr[0]:.3f} (n={len(l1):,})",
                  fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig08_alpha_stability.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig08_alpha_stability.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 09 -- fitted exceedance curve vs empirical, national cross-sections
# ---------------------------------------------------------------------------
def fig09():
    from scipy.special import zeta

    epi = _episcanner_all()
    nat = _read(
        "SELECT year, alpha, xmin FROM episcanner_fits "
        "WHERE metric='total_cases' AND year>0"
    )
    show = [2016, 2019, 2024]  # high / typical / recent
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    for i, year in enumerate(show):
        row = nat[nat["year"] == year]
        if row.empty:
            continue
        alpha, xmin = float(row["alpha"].iloc[0]), float(row["xmin"].iloc[0])
        v = epi[epi["year"] == year]["total_cases"].values
        v = v[v > 0]
        s = np.logspace(np.log10(xmin), np.log10(v.max() * 1.5), 200)
        theo = zeta(alpha, s) / zeta(alpha, xmin)
        # the fitted law is conditional on x_min; rescale by the empirical
        # fraction of the cross-section exceeding it so curve and points
        # live on the same unconditional exceedance scale
        theo = theo * (v >= xmin).mean()
        ax.plot(s, theo, color=COLORS[i],
                label=rf"{year}: fit $\alpha={alpha:.2f}$, $x_\min={xmin:.0f}$")
        srt = np.sort(v)[::-1]
        emp = np.arange(1, len(srt) + 1) / len(srt)
        q95 = np.quantile(v, 0.95)
        hi = srt >= q95
        ax.plot(srt[~hi], emp[~hi], ".", ms=3, color=COLORS[i])
        ax.plot(srt[hi], emp[hi], "x", ms=5, color=COLORS[i],
                label="upper 5%: sampling noise dominates" if i == len(show) - 1 else None)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("epidemic size $s$ (cases)")
    ax.set_ylabel(r"$\Pr(S\geq s)$")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "fig09_exceedance.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig09_exceedance.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# shared: out-of-sample regression predictions for figs 10, 11, 14
# ---------------------------------------------------------------------------
def _regression_oof(origins=(2023, 2024, 2025)):
    from predictive_model import train_predictive_models
    from sklearn.ensemble import RandomForestRegressor

    df = _city_panel()
    # Declared block, not a substring scan: a silent empty match would make M2
    # a copy of M1 in this pass too. run_ablation owns the definition.
    from run_ablation import CLIMATE_BLOCK

    climate = [c for c in CLIMATE_BLOCK if c in df.columns]
    if not climate:
        warnings.warn(
            "figs 10/14: the panel carries none of the declared climate "
            f"columns {CLIMATE_BLOCK}; the M2 block of this pass equals M1. "
            "These figures do not report the ablation, so this does not "
            "invalidate Table 4, but the M2 labels are not climate-controlled.",
            stacklevel=2)
    static = ["prev_log_pop"]
    lags = ["prev_total_cases", "prev_ep_dur", "prev_peak_week", "prev_R0"]
    scaling = ["prev_alpha_size", "prev_xmin_size", "prev_alpha_trend"]
    blocks = {
        "M1": lags,
        "M2": lags + static + climate,
        "M3": lags + static + climate + scaling,
    }
    rows = []
    pinball = {m: {} for m in ("M2", "M3")}
    import lightgbm as lgb

    for origin in origins:
        train = df[df["year"] < origin]
        test = df[df["year"] == origin]
        if train.empty or test.empty:
            continue
        for name, feats in blocks.items():
            use = [f for f in feats if f in df.columns
                   and not df[f].isnull().all()]
            clean = train.dropna(subset=use + ["total_cases"])
            model = RandomForestRegressor(
                n_estimators=150, oob_score=True, random_state=42, n_jobs=-1
            )
            ytr = np.log1p(clean["total_cases"])
            model.fit(clean[use], ytr)
            preds_log = model.predict(test[use])
            preds = np.expm1(preds_log)
            tr_preds = np.expm1(model.predict(clean[use]))
            bias = (np.expm1(ytr).mean() / tr_preds.mean()
                    if tr_preds.mean() > 0 else 1.0)
            for geo, yt, yp in zip(test["geocode"], test["total_cases"],
                                   preds * bias):
                rows.append({"origin": origin, "model": name, "geocode": geo,
                             "target": "total_cases",
                             "y_true": yt, "y_pred": yp,
                             "population": test.loc[test['geocode'] == geo, 'population'].iloc[0]
                             if 'population' in test.columns else np.nan})
        # quantile models for the pinball curve (size only)
        for name in ("M2", "M3"):
            use = [f for f in blocks[name] if f in df.columns
                   and not df[f].isnull().all()]
            clean = train.dropna(subset=use + ["total_cases"])
            for tau in (0.1, 0.25, 0.5, 0.75, 0.9):
                qm = lgb.LGBMRegressor(
                    objective="quantile", alpha=tau, n_estimators=200,
                    learning_rate=0.05, num_leaves=16, random_state=42,
                    verbose=-1, n_jobs=-1)
                qm.fit(clean[use], np.log1p(clean["total_cases"]))
                p = np.expm1(qm.predict(test[use]))
                pinball[name].setdefault(tau, []).append(
                    np.mean(np.maximum(tau * (test["total_cases"].values - p),
                                       (tau - 1) * (test["total_cases"].values - p))))
    oof = pd.DataFrame(rows)
    return oof, pinball


def fig10():
    """Predicted against observed epidemic size, for the estimator the paper uses.

    One panel and one estimator: the spline-basis quantile regression at
    tau = 0.5, which continues the boundary trend outside the observed range and
    can therefore forecast a season larger than any in its training window. A
    tree ensemble bounded by its training responses was drawn here previously;
    it was removed because for a target whose tail lies beyond the training
    support, a downward bias at the top is a property of the estimator and not
    of the data.

    The dotted red line marks the largest annual total in the training window,
    so it separates the seasons the training data covers from those it does
    not; the share of test units beyond it is stated on the panel, since that
    share is the whole reason the estimator had to be chosen for its ability to
    extrapolate.
    """
    from run_ablation import (
        CLIMATE_BLOCK, COVARIATE_DUR, COVARIATE_PEAK, DEMO, EPI,
        LAGGED_SIZE, SCALING_SIZE, fit_quantile_model,
    )

    df = _city_panel()
    train_max = float(df[df["year"] < 2023]["total_cases"].max())

    def _panel(ax, y_true, y_pred, title):
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        ok = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true > 0) & (y_pred > 0)
        ax.scatter(y_true[ok], y_pred[ok], s=4, alpha=0.25, color="0.4")
        lims = [max(1e1, min(y_true[ok].min(), y_pred[ok].min())),
                10 ** np.ceil(np.log10(max(y_true[ok].max(), y_pred[ok].max())))]
        ax.plot(lims, lims, "k--", lw=1, label="identity")
        ax.fill_between(lims, [l / 10 for l in lims], [l * 10 for l in lims],
                        alpha=0.10, color="0.3", label="10:1 / 1:10 band")
        ax.axhline(train_max, color="#c62828", ls=":", lw=1.2,
                   label="largest training response")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel("observed annual cases")
        ax.set_title(title, fontsize=8.5)
        ax.legend(fontsize=6.5, loc="upper left")

    feats = (LAGGED_SIZE + COVARIATE_DUR + COVARIATE_PEAK + EPI + DEMO
             + list(CLIMATE_BLOCK) + SCALING_SIZE)
    use = [c for c in feats if c in df.columns and not df[c].isnull().all()]
    tr = df[df["year"] < 2023].dropna(subset=use + ["total_cases"])
    te = df[df["year"].isin([2023, 2024, 2025])].dropna(
        subset=use + ["total_cases"])
    est = fit_quantile_model(tr[use], np.log1p(tr["total_cases"]), 0.5)
    pred = np.expm1(est.predict(te[use]))
    beyond = 100.0 * float((te["total_cases"] > train_max).mean())

    fig, ax = plt.subplots(figsize=(5.8, 4.6))
    _panel(ax, te["total_cases"], pred,
           "spline-basis quantile regression, $\\tau=0.5$\n"
           f"origins 2023--2025, n={len(te):,}; "
           f"{beyond:.1f}% of test units above the training maximum")
    ax.set_ylabel("predicted")

    fig.tight_layout()
    fig.savefig(OUT / "fig10_pred_vs_obs.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig10_pred_vs_obs.png", bbox_inches="tight")
    plt.close(fig)


def fig11(abl):
    """The ablation ladder, read from ablation_results.csv (not recomputed).

    (a) skill against the calibrated-persistence benchmark M0 for epidemic
        size, bar = median over the five forecast origins,
        dot = one origin, at the primary loss (pinball, tau = 0.5);
    (b) the quantity the decision rule turns on: the relative change in
        pinball loss from M2 to M3 against tau. A scaling block that carries
        tail information should help at high tau; negative values mean it
        hurts. No tau reaches the 3/5 threshold in this run, which is stated
        on the panel rather than left to the reader.
    """
    sk = _ablation_skill(abl)
    rel = _ablation_relative(abl)
    targets = [t for t in TASK_LABEL if t in set(sk["target"])]
    rungs = ["M1", "M2", "M3"]
    origins = sorted(sk["origin"].unique())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.4, 3.8),
                                   gridspec_kw={"width_ratios": [1.15, 1]})

    # ---------------- (a) skill vs M0 -------------------------------------
    width, gap = 0.24, 0.26
    rng = np.random.RandomState(0)
    for i, tgt in enumerate(targets):
        for j, rung in enumerate(rungs):
            v = sk[(sk.target == tgt) & (sk.rung == rung)]["value"].dropna()
            if v.empty:
                continue
            x = i + (j - 1) * gap
            ax1.bar(x, v.median(), width=width, color=RUNG_COLORS[rung],
                    edgecolor="0.25", linewidth=0.5,
                    label=rung if i == 0 else None, zorder=2)
            ax1.scatter(np.full(len(v), x) + rng.uniform(-0.05, 0.05, len(v)),
                        v.values, s=9, color="0.15", zorder=3,
                        linewidths=0)
    ax1.axhline(0, color="k", lw=0.9, zorder=1)
    ax1.set_xticks(range(len(targets)),
                   [TASK_LABEL.get(t, t) for t in targets])
    ax1.set_ylabel(r"skill score $1-L(\mathrm{rung})/L(\mathrm{M0})$")
    ax1.set_title("(a) skill vs persistence, pinball at $\\tau=0.5$", fontsize=9)
    ax1.legend(fontsize=7, ncol=3, loc="upper left")
    ax1.text(0.99, 0.02, f"bar = median of {len(origins)} origins, dot = origin",
             transform=ax1.transAxes, fontsize=6.5, color="0.35",
             ha="right", va="bottom")

    # ---------------- (b) M3 - M2 vs tau ----------------------------------
    colors = {"total_cases": "#08519c"}
    for tgt in targets:
        d = rel[rel.target == tgt]
        med = d.groupby("tau")["rel"].median().sort_index()
        ax2.plot(med.index, med.values, "o-", ms=3.5, lw=1.4,
                 color=colors.get(tgt, "0.3"),
                 label=TASK_LABEL.get(tgt, tgt))
        if tgt == "total_cases":
            lo = d.groupby("tau")["rel"].min().sort_index()
            hi = d.groupby("tau")["rel"].max().sort_index()
            ax2.fill_between(med.index, lo.values, hi.values,
                             color=colors[tgt], alpha=0.12, lw=0,
                             label="range over origins")
    ax2.axhline(0, color="k", lw=0.9)
    # a tiny L_M2 in one origin makes the relative change explode (to
    # several hundred percent), squashing a linear axis; symlog keeps the
    # region around zero linear and compresses the tails instead of
    # hiding them
    ax2.set_yscale("symlog", linthresh=50)
    ax2.axhspan(ax2.get_ylim()[0], 0, color="0.9", zorder=0)
    ax2.set_xlabel(r"quantile level $\tau$")
    ax2.set_ylabel(r"$100\,(L_{M2}-L_{M3})/L_{M2}$  (%)")
    ax2.set_title("(b) scaling block vs $\\tau$: negative = M3 worse",
                  fontsize=9)
    ax2.legend(fontsize=7, loc="upper right")

    # the headline number, stated on the panel
    d95 = rel[(rel.target == "total_cases") & np.isclose(rel.tau, 0.95)]["rel"]
    if len(d95):
        ax2.annotate(
            f"$\\tau=0.95$: median {d95.median():+.1f}%\n"
            f"({int((d95 < 0).sum())}/{len(d95)} origins worse)",
            xy=(0.95, d95.median()), xytext=(0.55, 0.12),
            textcoords="axes fraction", fontsize=6.5,
            arrowprops=dict(arrowstyle="->", lw=0.7, color="0.3"),
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", lw=0.5))
    ax2.text(0.02, 0.97, "no $\\tau$ meets the 3/5 criterion",
             transform=ax2.transAxes, fontsize=6.5, color="0.35", va="top")

    fig.tight_layout()
    fig.savefig(OUT / "fig11_ablation_skill.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig11_ablation_skill.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 12 -- binary ROC / PR / calibration / F1(threshold), M2 vs M3
# ---------------------------------------------------------------------------
def _binary_oof(df, feature_cols, percentile=0.75, test_year=2024, n_splits=5):
    import xgboost as xgb
    from sklearn.impute import SimpleImputer
    from temporal_split import year_forward_chain_folds

    targets = ["total_cases"]
    dfc = df.dropna(subset=feature_cols + targets).copy()
    df_train = dfc[dfc["year"] < test_year]
    cases = df_train["total_cases"].to_numpy(dtype=float)
    years = df_train["year"].to_numpy()
    threshold = np.percentile(cases, percentile * 100)
    folds = year_forward_chain_folds(years, n_splits=n_splits, min_train_years=2)
    y_true_all, y_prob_all = [], []
    X = df_train[feature_cols].to_numpy(dtype=float)
    for tr_idx, val_idx in folds:
        fthr = np.percentile(cases[tr_idx], percentile * 100)
        y_tr = (cases[tr_idx] > fthr).astype(int)
        y_val = (cases[val_idx] > fthr).astype(int)
        if len(np.unique(y_tr)) < 2:
            continue
        imp = SimpleImputer(strategy="median")
        Xtr, Xva = imp.fit_transform(X[tr_idx]), imp.transform(X[val_idx])
        pos_w = float(np.sum(y_tr == 0) / max(np.sum(y_tr == 1), 1))
        clf = xgb.XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            scale_pos_weight=pos_w, eval_metric="logloss",
            random_state=42, tree_method="hist", n_jobs=-1)
        clf.fit(Xtr, y_tr)
        y_true_all.append(y_val)
        y_prob_all.append(clf.predict_proba(Xva)[:, 1])
    return np.concatenate(y_true_all), np.concatenate(y_prob_all)


def fig12(art):
    """ROC / PR / calibration / F1 of the rarity classifier, read from the artefact.

    Curves are averaged over forecast origins rather than pooled across them.
    Pooling is not neutral here: the positive rate ranges from 0.21 to 0.70
    between origins, so a pooled curve ranks a 2021 observation against a 2024
    one and a predictor that knows only the calendar year scores a pooled AUC of
    0.60 while having no within-origin discrimination at all (0.50 by
    construction). The M0 curve is drawn for exactly that reason.

    The population-only curve is drawn for the other reason: if it tracks the
    full model, the task is separating large municipalities from small ones.
    """
    from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score

    oof, meta = art["oof"], art["meta"]
    scope = meta.get("threshold_scope", "?")
    n_orig = int(oof["origin"].nunique())
    q = int(round(float(meta.get("percentile", 0.75)) * 100))

    style = {
        "M0": dict(color="#444444", ls="-.", label="M0 (no skill)"),
        "M2": dict(color="#1f77b4", ls="-", label="M2 (no scaling)"),
        "M3": dict(color="#d62728", ls="--", label="M3 (with scaling)"),
        "POP": dict(color="#7f7f7f", ls=":", label="population only"),
    }
    present = set(oof["rung"])
    order = [r for r in ("M0", "M2", "M3", "POP") if r in present]

    fpr_grid = np.linspace(0, 1, 201)
    rec_grid = np.linspace(0, 1, 201)
    thr_grid = np.linspace(0.01, 0.99, 99)

    def _macro(rung):
        """Per-origin curves, interpolated onto a common grid and averaged."""
        roc, pr, cal, f1 = [], [], [], []
        aucs = []
        sub = oof[oof["rung"] == rung]
        for _, d in sub.groupby("origin"):
            y = d["y_true"].to_numpy()
            p = d["y_prob"].to_numpy()
            if len(np.unique(y)) < 2:
                continue
            aucs.append(roc_auc_score(y, p))

            f, t, _ = roc_curve(y, p)
            roc.append(np.interp(fpr_grid, f, t))

            prec, rec, _ = precision_recall_curve(y, p)
            o = np.argsort(rec)
            pr.append(np.interp(rec_grid, rec[o], prec[o]))

            qs = np.quantile(p, np.linspace(0, 1, 11))
            bins = np.clip(np.digitize(p, qs[1:-1]), 0, 9)
            exp, obs = [], []
            for k in range(10):
                m = bins == k
                if m.any():
                    exp.append(p[m].mean())
                    obs.append(y[m].mean())
            cal.append((np.array(exp), np.array(obs)))

            f1s = []
            for t0 in thr_grid:
                pred = p >= t0
                tp = float(np.sum(pred & (y == 1)))
                den = float(np.sum(pred) + np.sum(y == 1))
                f1s.append(2 * tp / den if den else np.nan)
            f1.append(f1s)
        return roc, pr, cal, f1, float(np.mean(aucs))

    fig, axes = plt.subplots(1, 4, figsize=(12.4, 3.2))
    for rung in order:
        st = style[rung]
        roc, pr, cal, f1, auc_v = _macro(rung)

        if roc:
            axes[0].plot(fpr_grid, np.nanmean(roc, axis=0), ls=st["ls"],
                         color=st["color"],
                         label=f"{st['label']} (AUC {auc_v:.3f})")
            axes[0].fill_between(fpr_grid, np.nanmin(roc, axis=0),
                                 np.nanmax(roc, axis=0), color=st["color"],
                                 alpha=0.12, lw=0)
        if pr:
            axes[1].plot(rec_grid, np.nanmean(pr, axis=0), ls=st["ls"],
                         color=st["color"], label=st["label"])
        if cal:
            xs = np.array([c[0].mean() for c in cal])
            ys = np.array([c[1].mean() for c in cal])
            axes[2].plot(xs, ys, "o", ls=st["ls"], color=st["color"],
                         label=st["label"], ms=5)
        if f1:
            axes[3].plot(thr_grid, np.nanmean(f1, axis=0), ls=st["ls"],
                         color=st["color"], label=st["label"])

    axes[0].plot([0, 1], [0, 1], "k:", lw=0.8)
    axes[0].set_xlabel("false positive rate")
    axes[0].set_ylabel("true positive rate")
    axes[0].set_title("(a) ROC, averaged over origins", fontsize=9)
    axes[1].set_xlabel("recall")
    axes[1].set_ylabel("precision")
    axes[1].set_title("(b) precision-recall", fontsize=9)
    axes[2].plot([0, 1], [0, 1], "k:", lw=0.8)
    axes[2].set_xlabel("mean predicted probability")
    axes[2].set_ylabel("observed frequency")
    axes[2].set_title("(c) calibration (deciles within origin)", fontsize=9)
    axes[3].set_xlabel("decision threshold")
    axes[3].set_ylabel(r"$F_1$")
    axes[3].set_title(r"(d) $F_1$ vs threshold", fontsize=9)
    for ax in axes:
        ax.legend(fontsize=7)
    fig.suptitle(
        f"$Q_{{{q}}}$ rarity, {scope} threshold, {n_orig} origins averaged, "
        f"{meta.get('panel_units', oof['unit'].nunique())} units strictly "
        f"out-of-fold", fontsize=9, y=1.06)
    fig.tight_layout()
    fig.savefig(OUT / "fig12_roc_pr_calibration.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig12_roc_pr_calibration.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 13: what the classifier uses
# ---------------------------------------------------------------------------


def fig13(art):
    """Held-out permutation importance, with the scaling block scored jointly.

    Permuting one column of a collinear group while its partners stay in place
    lets the model read the information off the partners, so per-column
    importance collapses towards zero even when the group carries signal. The
    scaling descriptors are rank-correlated at about 0.65 within units, so the
    block is also scored as a unit -- that joint row is what the ablation's
    M3-M2 contrast measures, and the two must agree.
    """
    imp = art["importance"]
    if imp is None or len(imp) == 0:
        raise RuntimeError(
            f"no importance table for {CLASSIFIER_PREFIX}. Rebuild the artefact "
            "with run_classifier.py (it writes <out>_importance.csv for the "
            "rungs named in --importance-rungs, M3 by default)."
        )

    scaling = ["prev_alpha_size", "prev_xmin_size", "prev_alpha_trend"]
    joint = imp[imp["feature"] == "SCALING_BLOCK"]
    per_feat = imp[imp["feature"] != "SCALING_BLOCK"]

    agg = (per_feat.groupby(["feature", "is_scaling"])["importance"]
           .agg(["mean", "std"]).reset_index().sort_values("mean"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.4),
                                   gridspec_kw={"width_ratios": [2, 1.2]})
    labels = [str(f).replace("prev_", "").replace("_", " ") for f in agg["feature"]]
    colors = ["#d62728" if bool(s) else "#7f7f7f" for s in agg["is_scaling"]]
    ax1.barh(np.arange(len(agg)), agg["mean"], xerr=agg["std"], color=colors,
             error_kw={"lw": 0.8, "capsize": 2})
    ax1.set_yticks(np.arange(len(agg)), labels, fontsize=7)
    if len(joint):
        jm = float(joint["importance"].mean())
        jd = float(joint["importance"].std())
        ax1.errorbar(jm, len(agg) + 0.4, xerr=jd, fmt="D", color="#d62728",
                     ms=6, capsize=2)
        ax1.set_yticks(np.append(np.arange(len(agg)), len(agg) + 0.4),
                       labels + ["scaling block (joint)"], fontsize=7)
    ax1.axvline(0, color="k", lw=0.8)
    ax1.set_xlabel(r"permutation importance ($\Delta$ Brier, held-out)")
    ax1.set_title("(a) all features (red = scaling block)", fontsize=9)

    plotted = 0
    for i, feat in enumerate(scaling):
        d = imp[imp["feature"] == feat].sort_values("origin")
        if d.empty:
            continue
        ax2.plot(d["origin"], d["importance"], "o-", color=COLORS[plotted % len(COLORS)],
                 ms=4, label=feat.replace("prev_", "").replace("_", " "))
        plotted += 1
    if len(joint):
        d = joint.sort_values("origin")
        ax2.plot(d["origin"], d["importance"], "D-", color="#d62728", ms=6,
                 lw=2, label="block (joint)", zorder=5)
    ax2.axhline(0, color="k", lw=0.8)
    ax2.set_xlabel("forecast origin")
    ax2.set_ylabel(r"$\Delta$ Brier when shuffled")
    ax2.set_title("(b) scaling descriptors: stability", fontsize=9)
    ax2.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "fig13_importance.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig13_importance.png", bbox_inches="tight")
    plt.close(fig)


def fig14(oof):
    from scipy.stats import norm

    d = oof[oof["model"] == "M3"].copy()
    d = d[(d["y_true"] > 0) & (d["population"] > 0)]
    d["ape"] = np.abs(d["y_pred"] - d["y_true"]) / d["y_true"] * 100
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.6))
    ax1.scatter(np.log10(d["population"]), d["ape"].clip(upper=500), s=4,
                alpha=0.2, color="0.4")
    ax1.set_xlabel(r"$\log_{10}$ municipal population")
    ax1.set_ylabel("absolute percentage error (%)")
    med = d.groupby(pd.cut(np.log10(d["population"]), 12), observed=True)["ape"].median()
    xs = [iv.mid for iv in med.index]
    ax1.plot(xs, med.values, "o-", color="#d62728", lw=1.4, label="median APE")
    ax1.legend(fontsize=8)
    ax1.set_title("(a) APE vs population", fontsize=9)

    df = _city_panel()
    # Epidemic size only: duration and peak timing are bounded by the length
    # of the season and are covariates now, not targets.
    targets = ["total_cases"]
    for i, tgt in enumerate(targets):
        prev = f"prev_{tgt}"
        sub = df.dropna(subset=[prev, tgt])
        resid = sub[tgt] - sub[prev]  # persistence residual
        z = (resid - resid.mean()) / resid.std()
        grid = np.linspace(-4, 4, 200)
        ax2.hist(z, bins=60, density=True, alpha=0.4,
                 color=COLORS[i], label=tgt)
    xs = np.linspace(-4, 4, 200)
    ax2.plot(xs, norm.pdf(xs), "k-", lw=1.2, label="standard normal")
    ax2.set_xlabel("standardised persistence residual")
    ax2.set_ylabel("density")
    ax2.set_xlim(-5, 5)
    ax2.legend(fontsize=8)
    ax2.set_title("(b) residuals vs normal reference", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig14_error_vs_pop.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig14_error_vs_pop.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 15 -- predicted-probability map for one forecast origin
# ---------------------------------------------------------------------------
def fig15():
    """Map of the predicted probability of a large season, one origin.

    The classifier is the one the ablation reports (``run_ablation.make_classifier``
    with its default), on the M3 feature set. It used to be a separate XGBoost
    fit with its own feature list, which is the same defect the ablation
    figures had: a second, undeclared experiment drawn as if it were a view of
    the first.
    """
    from run_ablation import (
        CLIMATE_BLOCK, COVARIATE_DUR, COVARIATE_PEAK, DEMO, EPI,
        LAGGED_SIZE, SCALING_SIZE, make_classifier,
    )

    df = _city_panel()
    feats = (LAGGED_SIZE + COVARIATE_DUR + COVARIATE_PEAK + EPI + DEMO
             + list(CLIMATE_BLOCK) + SCALING_SIZE + ["log_pop"])
    m3 = [c for c in feats if c in df.columns and not df[c].isnull().all()]
    train = df[df["year"] < 2024].dropna(subset=m3 + ["total_cases"])
    y = (train["total_cases"] > train["total_cases"].quantile(0.75)).astype(int)
    clf = make_classifier()
    clf.fit(train[m3], y)
    test = df[df["year"] == 2024].copy()
    test_ok = test.dropna(subset=m3)
    prob = clf.predict_proba(test_ok[m3])[:, 1]
    test_ok["prob"] = prob

    gdf = gpd.read_file("muni_br.gpkg")
    gdf["code_muni"] = gdf["code_muni"].astype(int)
    merged = gdf[["code_muni", "abbrev_state", "geometry"]].merge(
        test_ok[["geocode", "prob"]], left_on="code_muni",
        right_on="geocode", how="left")
    q = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    cats = np.where(merged["prob"].notna(),
                    np.digitize(merged["prob"], merged["prob"].quantile([.2, .4, .6, .8])), -1)
    from matplotlib.colors import ListedColormap

    cmap = ListedColormap(["#f7f4f9", "#e7e1ef", "#d4b9da", "#c994c7",
                           "#df65b0", "#dd1c77"])
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    merged.plot(column=cats, cmap=cmap, ax=ax, linewidth=0.05, edgecolor="0.4",
                missing_kwds={"color": "#d0d0d0", "label": "no forecast"},
                categorical=True, legend=False)
    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=mpl.colors.BoundaryNorm(range(6), 6))
    cbar = fig.colorbar(sm, ax=ax, shrink=0.55, ticks=range(6))
    cbar.ax.set_yticklabels(["lowest 20%", "", "middle", "", "highest 20%", ""],
                            fontsize=8)
    ax.set_title("predicted probability of a large 2024 season\n"
                 "(exceeds own 75th percentile; quintile classes; grey = no forecast)",
                 fontsize=9)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(OUT / "fig15_pred_map.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig15_pred_map.png", bbox_inches="tight")
    plt.close(fig)


def main(which=None):
    everything = {
        "fig02": fig02, "fig03": fig03, "fig04": fig04, "fig05": fig05,
        "fig06": fig06, "fig07": fig07, "fig08": fig08,
        "fig09": fig09, "fig10": fig10, "fig11": fig11, "fig12": fig12,
        "fig13": fig13, "fig14": fig14, "fig15": fig15,
    }
    todo = which or list(everything)

    # fig 14 uses the city-level out-of-sample pass. It is a separate
    # experiment from the ablation and must stay that way: the artefact only
    # carries aggregate losses, not row-level predictions, so it cannot drive a
    # scatter of predictions. NOTE: that pass is still fitted with a random
    # forest (see _regression_oof), so fig14 panel (a) inherits the estimator
    # removed from fig 10 and should be moved to the spline quantile model.
    need_oof = "fig14" in todo
    oof = None
    if need_oof:
        t0 = time.time()
        print("retraining out-of-sample regression pass (origins 2023-2025, "
              "city level, random forest) -- this is NOT the ablation...",
              flush=True)
        oof, _pinball_unused = _regression_oof()
        print(f"  done in {time.time() - t0:.0f}s", flush=True)

    # fig 11 reads the ablation artefact written by run_ablation.py.
    abl = None
    if "fig11" in todo:
        abl = _load_ablation()
        print(f"[fig11] reading {abl['path'].name} "
              f"({len(abl['raw'])} rows); not recomputing the ladder.", flush=True)

    # figs 12 and 13 read the classifier artefacts written by run_classifier.py.
    clf = None
    if {"fig12", "fig13"} & set(todo):
        clf = _load_classifier()
        print(f"[fig12/13] reading {clf['path'].name} "
              f"({len(clf['oof'])} OOF rows, "
              f"{clf['oof']['rung'].nunique()} rungs, "
              f"{clf['oof']['origin'].nunique()} origins); not refitting.",
              flush=True)

    for name in todo:
        t0 = time.time()
        print(f"[{name}] generating...", flush=True)
        if name == "fig11":
            everything[name](abl)
        elif name in ("fig12", "fig13"):
            everything[name](clf)
        elif name == "fig14":
            everything[name](oof)
        else:
            everything[name]()
        print(f"[{name}] done in {time.time() - t0:.0f}s -> {OUT}/{name}_*.png", flush=True)


if __name__ == "__main__":
    args = []
    for a in sys.argv[1:]:
        if a.startswith("-"):
            continue
        n = a.removeprefix("fig").lstrip("0") or "0"
        args.append(f"fig{int(n):02d}")
    main(args or None)
