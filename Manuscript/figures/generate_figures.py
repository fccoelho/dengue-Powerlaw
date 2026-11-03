"""Generate candidate figures for every figure environment in main.tex.

Usage:
    python Manuscript/figures/generate_figures.py            # all 15
    python Manuscript/figures/generate_figures.py 2 4 6      # subset

Every figure is built from the repository's own artefacts (powerlaw_results.db,
data/episcanner_*.parquet, data/<geocode>.parquet, muni_br.gpkg) using the same
estimators as the pipeline, so each PNG/PDF is a *candidate* for its figure
environment, not a placeholder: captions in main.tex describe exactly what is
drawn. Predictive figures (10-15) retrain the models described in the Methods;
their numbers therefore move when the database is refreshed, and the figures
should be regenerated after every data update.
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
os_cwd = Path.cwd()
if str(ROOT) != str(os_cwd):
    import os

    os.chdir(ROOT)  # the data layer uses relative paths ("data/...", "muni_br.gpkg")

warnings.filterwarnings("ignore")

OUT = HERE
PANEL_CACHE = HERE / "_panel_city_BR.parquet"

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

LEVELS = {
    "L1": ("powerlaw_fits", "", "weekly, city"),
    "L2": ("powerlaw_fits_yearly", "", "weekly, city-year"),
    "L3": ("powerlaw_fits_total_cases", "", "size, city"),
    "L4": ("episcanner_state_fits", "metric='total_cases' AND year>0", "size, state-year"),
    "L5": ("episcanner_state_fits", "metric='ep_dur' AND year>0", "duration, state-year"),
    "L6": ("episcanner_fits", "metric='total_cases' AND year>0", "size, natl-year"),
    "L7": ("episcanner_fits", "metric='ep_dur' AND year>0", "duration, natl-year"),
}
SIZE_LEVELS = ["L1", "L2", "L3", "L4", "L6"]
DUR_LEVELS = ["L5", "L7"]
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
# Fig 01 -- schematic of the seven aggregation levels
# ---------------------------------------------------------------------------
def fig01():
    fig, ax = plt.subplots(figsize=(8.4, 3.4))
    ax.axis("off")

    def box(x, y, w, h, text, fc):
        ax.add_patch(mpl.patches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.02",
            facecolor=fc, edgecolor="0.3", linewidth=0.8))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8)

    box(0.02, 0.72, 0.30, 0.16, "municipal weekly series\n(Infodengue, 2010–2025)", "#d9e8f4")
    ax.annotate("", xy=(0.17, 0.52), xytext=(0.17, 0.70),
                arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.text(0.18, 0.62, "temporal window:\nfull series / season / year",
            fontsize=7, va="center")
    box(0.02, 0.34, 0.30, 0.16, "random variable $X$:\ncounts, sizes or durations", "#d9e8f4")

    ax.annotate("", xy=(0.45, 0.42), xytext=(0.33, 0.42),
                arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.text(0.39, 0.47, "unit of observation:\nmunicipality / cross-section",
            fontsize=7, ha="center")

    labels = [
        ("L1 weekly, city\n(full series)", 0.46, 0.80),
        ("L2 weekly, city\n(single years)", 0.46, 0.56),
        ("L3 size, city\n(annual totals)", 0.46, 0.32),
        ("L4 size, state-year\ncross-section", 0.74, 0.80),
        ("L5 duration, state-year", 0.74, 0.56),
        ("L6 size, national-year", 0.74, 0.32),
        ("L7 duration, national-year", 0.74, 0.08),
    ]
    for i, (t, x, y) in enumerate(labels):
        fc = "#f6d6d0" if "duration" in t else "#d8ecd8"
        box(x, y, 0.24, 0.14, t, fc)
    ax.text(0.50, 0.02, "green: constructions whose verdict is 'power law'   "
            "red: constructions whose verdict is 'exponential'",
            fontsize=7, color="0.3")
    fig.savefig(OUT / "fig01_aggregation_levels.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig01_aggregation_levels.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 02 -- alpha by level (boxes), size above, duration below
# ---------------------------------------------------------------------------
def fig02():
    data = _level_alpha()
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7.0, 4.2),
        gridspec_kw={"height_ratios": [5, 2]})
    for ax, levels in ((ax1, SIZE_LEVELS), (ax2, DUR_LEVELS)):
        vals = [data[l]["alpha"].values for l in levels]
        bp = ax.boxplot(vals, tick_labels=levels, whis=1.5, showfliers=False,
                        patch_artist=True, widths=0.55)
        for patch in bp["boxes"]:
            patch.set_facecolor("#aec7e8")
        ax.axhline(2.0, ls="--", lw=0.9, color="0.4")
        ax.set_ylabel(r"$\alpha$")
    ax2.set_xlabel("aggregation level")
    ax1.text(0.02, 0.92, "size constructions", transform=ax1.transAxes, fontsize=9)
    ax2.text(0.02, 0.8, "duration constructions", transform=ax2.transAxes, fontsize=9)
    fig.suptitle(r"estimated exponent by aggregation level (dashed: $\alpha=2$)", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
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
    from powerlaw import Fit

    df = _episcanner_all()
    nat = _read(
        "SELECT metric, alpha, xmin FROM episcanner_fits WHERE year=0"
    ).set_index("metric")
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4))

    for ax, col, metric, lab in (
        (axes[0], "total_cases", "total_cases", "epidemic size (cases)"),
        (axes[1], "ep_dur", "ep_dur", "epidemic duration (weeks)"),
    ):
        v = df[col].values
        v = v[v > 0]
        fit = Fit(v, discrete=True, xmin=float(nat.loc[metric, "xmin"]))
        fit.plot_ccdf(ax=ax, color="0.55", label="empirical (pooled 2011–2025)")
        xmin = float(nat.loc[metric, "xmin"])
        alpha = float(nat.loc[metric, "alpha"])
        s = np.logspace(np.log10(xmin), np.log10(v.max()), 100)
        if metric == "total_cases":
            # discrete PL tail: Pr(X>=s) = zeta(alpha, s)/zeta(alpha, xmin)
            from scipy.special import zeta

            ccdf = zeta(alpha, s) / zeta(alpha, xmin)
            ax.plot(s, ccdf, "k-", lw=1.4,
                    label=rf"power law ($\alpha={alpha:.2f}$, $x_\min={xmin:.0f}$)")
            ax.set_xscale("log")
            ax.set_yscale("log")
        else:
            # exponential alternative: Pr(X>=s) = exp(-lambda (s - xmin))
            lam = 1.0 / v[v >= xmin].mean()
            ax.plot(s, np.exp(-lam * (s - xmin)), "k--", lw=1.4,
                    label=f"exponential (mean {1/lam:.1f} wk)")
            ax.set_xscale("log")
        ax.set_xlabel(lab)
        ax.set_ylabel(r"$\Pr(X \geq x)$")
        ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "fig04_size_vs_duration.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig04_size_vs_duration.png", bbox_inches="tight")
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
    from powerlaw import Fit

    small = pd.read_parquet("data/3117836.parquet")["casos_est"]
    large = pd.read_parquet("data/3550308.parquet")["casos_est"]
    epi = pd.read_parquet("data/episcanner_SP_2024.parquet")

    fig, axes = plt.subplots(2, 2, figsize=(8.6, 6.2))
    panels = [
        (axes[0][0], small.dropna().values, "Cônego Marinho (3117836), weekly cases, 2010–2025"),
        (axes[0][1], large.dropna().values, "São Paulo (3550308), weekly cases, 2010–2025"),
        (axes[1][0], epi["total_cases"].dropna().values, "SP state-year cross-section 2024, epidemic size"),
        (axes[1][1], epi["ep_dur"].dropna().values, "SP state-year cross-section 2024, epidemic duration"),
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
# Fig 07 -- the L1/L2 window flip
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
             label=f"L1 full series (n={len(l1):,})", color="#8fbf8f")
    ax1.hist(l2["alpha"], bins=bins, density=True, alpha=0.65,
             label=f"L2 single years (n={len(l2):,})", color="#e8a09a")
    ax1.axvline(2.0, ls="--", lw=0.9, color="0.4")
    ax1.set_xlabel(r"estimated $\alpha$")
    ax1.set_ylabel("density")
    ax1.legend(fontsize=8)

    stats = pd.DataFrame({
        "L1 full": [100 * (l1R["R"] > 0).mean(), 100 * ((l1R["R"] < 0) & (l1R["p"] < .05)).mean()],
        "L2 yearly": [100 * (l2R["R"] > 0).mean(), 100 * ((l2R["R"] < 0) & (l2R["p"] < .05)).mean()],
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
    climate = [c for c in df.columns if c.startswith("prev_avg_")]
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


def fig10(oof):
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0), sharey=True)
    # RF panel
    d = oof[oof["model"] == "M3"]
    ax = axes[0]
    ax.scatter(d["y_true"], d["y_pred"], s=4, alpha=0.25, color="0.4")
    lims = [1e1, 10 ** np.ceil(np.log10(max(d["y_true"].max(), d["y_pred"].max())))]
    ax.plot(lims, lims, "k--", lw=1, label="identity")
    ax.fill_between(lims, [l / 10 for l in lims], [l * 10 for l in lims],
                    alpha=0.10, color="0.3", label="10:1 / 1:10 band")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("observed annual cases")
    ax.set_ylabel("predicted")
    ax.set_title(f"random forest (origins 2023–2025, n={len(d):,})", fontsize=9)
    ax.legend(fontsize=7, loc="upper left")
    # quantile ensemble: retrain tau=0.5 pooled for the right panel
    from predictive_model import train_predictive_models

    df = _city_panel()
    use = [c for c in df.columns if c.startswith("prev_") and not df[c].isnull().all()]
    use = [c for c in use if not c.startswith(("prev_alpha_dur", "prev_xmin_dur"))]
    train = df[df["year"] < 2023].dropna(subset=use + ["total_cases"])
    import lightgbm as lgb

    qm = lgb.LGBMRegressor(objective="quantile", alpha=0.5, n_estimators=300,
                           learning_rate=0.05, num_leaves=16, random_state=42,
                           verbose=-1, n_jobs=-1)
    qm.fit(train[use], np.log1p(train["total_cases"]))
    test = df[df["year"].isin([2023, 2024, 2025])].dropna(subset=use)
    p = np.expm1(qm.predict(test[use]))
    ax = axes[1]
    ax.scatter(test["total_cases"], p, s=4, alpha=0.25, color="0.4")
    lims = [1e1, 10 ** np.ceil(np.log10(max(test["total_cases"].max(), p.max())))]
    ax.plot(lims, lims, "k--", lw=1, label="identity")
    ax.fill_between(lims, [l / 10 for l in lims], [l * 10 for l in lims],
                    alpha=0.10, color="0.3", label="10:1 / 1:10 band")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("observed annual cases")
    ax.set_title(f"LightGBM median quantile (origin 2022, n={len(test):,})", fontsize=9)
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "fig10_pred_vs_obs.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig10_pred_vs_obs.png", bbox_inches="tight")
    plt.close(fig)


def fig11(oof, pinball):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.6))
    # (a) skill vs M0 (persistence: predict S_{t-1})
    df = _city_panel()
    m0_rows = []
    for origin in (2023, 2024, 2025):
        d = df[df["year"] == origin].dropna(subset=["prev_total_cases", "total_cases"])
        m0_rows.append(d.assign(model="M0", y_pred=d["prev_total_cases"],
                                y_true=d["total_cases"])[
            ["model", "y_true", "y_pred"]])
    m0 = pd.concat(m0_rows)
    mae = oof.groupby("model").apply(
        lambda g: np.mean(np.abs(g["y_pred"] - g["y_true"])),
        include_groups=False,
    )
    mae0 = np.mean(np.abs(m0["y_pred"] - m0["y_true"]))
    skill = {m: 1 - mae[m] / mae0 for m in mae.index}
    ax1.bar(skill.keys(), skill.values(),
            color=["#9ecae1", "#4292c6", "#08519c"])
    ax1.axhline(0, color="k", lw=0.8)
    ax1.set_ylabel("skill score vs persistence M0")
    ax1.set_xlabel("predictor blocks")
    ax1.set_title("(a) epidemic size, MAE skill", fontsize=9)
    for i, (k, v) in enumerate(skill.items()):
        ax1.text(i, v + (0.01 if v >= 0 else -0.02), f"{v:.2f}", ha="center",
                 fontsize=8)
    # (b) pinball vs tau
    for name, color in (("M2", "#fdae6b"), ("M3", "#e6550d")):
        taus = sorted(pinball[name].keys())
        vals = [np.mean(pinball[name][t]) for t in taus]
        ax2.plot(taus, vals, "o-", color=color, label=name)
    ax2.set_xlabel(r"quantile level $\tau$")
    ax2.set_ylabel("pinball loss (cases)")
    ax2.set_title("(b) quantile loss, epidemic size", fontsize=9)
    ax2.legend()
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


def fig12():
    from sklearn.metrics import roc_curve, precision_recall_curve, auc

    df = _city_panel()
    scaling = ["prev_alpha_size", "prev_xmin_size", "prev_alpha_trend"]
    m3 = [c for c in df.columns if c.startswith("prev_")
          and not df[c].isnull().all()] + ["log_pop"]
    m3 = [c for c in m3 if not c.startswith(("prev_alpha_dur", "prev_xmin_dur"))]
    m2 = [c for c in m3 if c not in scaling]
    variants = {"M2 (no scaling)": m2, "M3 (with scaling)": m3}
    fig, axes = plt.subplots(1, 4, figsize=(12.4, 3.1))
    for name, feats in variants.items():
        y, p = _binary_oof(df, feats)
        color = "#1f77b4" if "M2" in name else "#d62728"
        ls = "-" if "M2" in name else "--"
        fpr, tpr, th = roc_curve(y, p)
        axes[0].plot(fpr, tpr, ls=ls, color=color,
                     label=f"{name} (AUC {auc(fpr, tpr):.3f})")
        prec, rec, _ = precision_recall_curve(y, p)
        axes[1].plot(rec, prec, ls=ls, color=color,
                     label=f"{name} (AP {auc(rec, prec):.3f})")
        # calibration in deciles
        qs = np.quantile(p, np.linspace(0, 1, 11))
        bins = np.clip(np.digitize(p, qs[1:-1]), 0, 9)
        obs = [y[bins == b].mean() for b in range(10)]
        exp = [p[bins == b].mean() for b in range(10)]
        axes[2].plot(exp, obs, "o-", ls=ls, color=color, label=name, ms=4)
        # F1 vs threshold
        grid = np.linspace(0.01, 0.99, 99)
        f1s = []
        for t in grid:
            pred = p >= t
            tp = np.sum(pred & (y == 1))
            f1s.append(2 * tp / (np.sum(pred) + np.sum(y == 1)))
        axes[3].plot(grid, f1s, ls=ls, color=color, label=name)
        j = np.argmax(tpr - fpr)
        axes[3].axvline(th[j], color=color, ls=":", lw=1)
    axes[0].plot([0, 1], [0, 1], "k:", lw=0.8)
    axes[0].set_xlabel("false positive rate")
    axes[0].set_ylabel("true positive rate")
    axes[0].set_title("(a) ROC (out-of-fold)", fontsize=9)
    axes[1].set_xlabel("recall")
    axes[1].set_ylabel("precision")
    axes[1].set_title("(b) precision–recall", fontsize=9)
    axes[2].plot([0, 1], [0, 1], "k:", lw=0.8)
    axes[2].set_xlabel("mean predicted probability")
    axes[2].set_ylabel("observed frequency")
    axes[2].set_title("(c) calibration (deciles)", fontsize=9)
    axes[3].set_xlabel("decision threshold")
    axes[3].set_ylabel(r"$F_1$")
    axes[3].set_title("(d) $F_1$ vs threshold (dotted: Youden)", fontsize=9)
    for ax in axes:
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "fig12_roc_pr_calibration.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig12_roc_pr_calibration.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 13 -- SHAP ranking with the scaling block highlighted
# ---------------------------------------------------------------------------
def fig13():
    import shap
    import xgboost as xgb
    from sklearn.impute import SimpleImputer

    df = _city_panel()
    m3 = [c for c in df.columns if c.startswith("prev_")
          and not df[c].isnull().all()] + ["log_pop"]
    m3 = [c for c in m3 if not c.startswith(("prev_alpha_dur", "prev_xmin_dur"))]
    scaling = ["prev_alpha_size", "prev_xmin_size", "prev_alpha_trend"]
    dfc = df.dropna(subset=m3 + ["total_cases"])
    df_train = dfc[dfc["year"] < 2024]
    y = (df_train["total_cases"] > df_train["total_cases"].quantile(0.75)).astype(int)
    X = SimpleImputer(strategy="median").fit_transform(df_train[m3])
    pos_w = float(np.sum(y == 0) / max(np.sum(y == 1), 1))
    clf = xgb.XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        scale_pos_weight=pos_w, eval_metric="logloss",
        random_state=42, tree_method="hist", n_jobs=-1)
    clf.fit(X, y)
    explainer = shap.TreeExplainer(clf)
    sv = explainer.shap_values(X)
    if isinstance(sv, list):
        sv = sv[1]
    mean_abs = np.abs(sv).mean(axis=0)
    order = np.argsort(mean_abs)
    _label_map = {"log_pop": "log pop (static)"}
    labels = [
        _label_map.get(c, c.replace("prev_", "").replace("_", " "))
        for c in np.array(m3)[order]
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.0, 4.4),
                                   gridspec_kw={"width_ratios": [2, 1.2]})
    colors = ["#d62728" if m3[i] in scaling else "#7f7f7f" for i in order]
    ax1.barh(range(len(order)), mean_abs[order], color=colors)
    ax1.set_yticks(range(len(order)), labels, fontsize=7)
    ax1.set_xlabel("mean |SHAP value|")
    ax1.set_title("(a) all features (red = scaling block)", fontsize=9)

    # (b) direction of the scaling block: SHAP value vs feature value
    for i, feat in enumerate(scaling):
        col = m3.index(feat)
        sub = np.random.RandomState(0).choice(len(X), min(3000, len(X)),
                                              replace=False)
        ax2.scatter(X[sub, col], sv[sub, col], s=4, alpha=0.3,
                    color=COLORS[i],
                    label=feat.replace("prev_", "").replace("_", " "))
    ax2.axhline(0, color="k", lw=0.8)
    ax2.set_xlabel("feature value (median-imputed)")
    ax2.set_ylabel("SHAP value")
    ax2.set_title("(b) scaling descriptors: direction", fontsize=9)
    ax2.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "fig13_shap.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig13_shap.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 14 -- error structure
# ---------------------------------------------------------------------------
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
    targets = ["total_cases", "ep_dur", "peak_week"]
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
    import xgboost as xgb
    from sklearn.impute import SimpleImputer

    df = _city_panel()
    m3 = [c for c in df.columns if c.startswith("prev_")
          and not df[c].isnull().all()] + ["log_pop"]
    m3 = [c for c in m3 if not c.startswith(("prev_alpha_dur", "prev_xmin_dur"))]
    train = df[df["year"] < 2024].dropna(subset=m3 + ["total_cases"])
    y = (train["total_cases"] > train["total_cases"].quantile(0.75)).astype(int)
    imp = SimpleImputer(strategy="median")
    X = imp.fit_transform(train[m3])
    pos_w = float(np.sum(y == 0) / max(np.sum(y == 1), 1))
    clf = xgb.XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        scale_pos_weight=pos_w, eval_metric="logloss",
        random_state=42, tree_method="hist", n_jobs=-1)
    clf.fit(X, y)
    test = df[df["year"] == 2024].copy()
    test_ok = test.dropna(subset=m3)
    prob = clf.predict_proba(imp.transform(test_ok[m3]))[:, 1]
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
        "fig01": fig01, "fig02": fig02, "fig03": fig03, "fig04": fig04,
        "fig05": fig05, "fig06": fig06, "fig07": fig07, "fig08": fig08,
        "fig09": fig09, "fig10": fig10, "fig11": fig11, "fig12": fig12,
        "fig13": fig13, "fig14": fig14, "fig15": fig15,
    }
    todo = which or list(everything)
    # figs 10, 11, 14 share one out-of-sample pass
    need_oof = any(f in todo for f in ("fig10", "fig11", "fig14"))
    oof = pinball = None
    if need_oof:
        t0 = time.time()
        print("retraining out-of-sample regression pass (origins 2023–2025)...",
              flush=True)
        oof, pinball = _regression_oof()
        print(f"  done in {time.time() - t0:.0f}s", flush=True)
    for name in todo:
        t0 = time.time()
        print(f"[{name}] generating...", flush=True)
        if name == "fig10":
            everything[name](oof)
        elif name == "fig11":
            everything[name](oof, pinball)
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
