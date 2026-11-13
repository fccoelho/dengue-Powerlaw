# survival_analysis.py

"""Read the epidemic-size tail as a survival curve, and season duration as a
time-to-event outcome.

Why this file exists
--------------------
Section 3.3 states the size exponent as a risk curve: with the pair fixed,

    Pr(S >= s) = zeta(alpha, s) / zeta(alpha, x_min),

and Reports the exceedance probabilities as a single number per season. Two
things that expression licenses are not carried out in the manuscript, and one
thing it was given up on is better done in this vocabulary.

**The curve is a survival function.** The Pareto hazard is
``h(s) = 1 - zeta(alpha, s+1)/zeta(alpha, s)``, which declines roughly as
``(alpha-1)/s``. "Heavy tail" and "declining hazard" are one statement in two
languages, and the second is the one a reader can act on: the longer a season
has already run relative to the bound, the *less* likely it is to end. That is
a sentence, not a computation, so nothing below estimates it.

**The non-parametric competitor.** Because every unit is left-truncated at the
same point x_min, the Kaplan-Meier estimator collapses to the empirical CCDF of
the units that clear the bound -- the product of (1 - d/n) is the count ratio,
exactly. Figure 9 already draws this against the fitted curve. What it does not
draw is a band, so agreement is an impression. We supply the DKW band, which is
simultaneous over the whole curve and distribution-free.

**But the in-season comparison cannot be a test, and the reason matters.**
``powerlaw.Fit`` chooses x_min by minimising the Kolmogorov-Smirnov distance
between the empirical and the fitted CDF. The quantity a band would test is
therefore the quantity that was minimised to produce the fit. Any Pareto family,
true or false, will land inside a band drawn around the data it was tuned on.
So the in-season departure is reported as a *magnitude*, in band units, never as
a p-value, and the honest test is the out-of-season one: take the pair estimated
in season t and ask whether it describes the sizes of season t+1. That is not
circular, and it is the question a risk curve has to answer to be worth
reporting -- a curve that describes only the season it was fitted to is a
description, not a risk instrument. The manuscript's own coupling result
(alpha and x_min rank-correlated at 0.89) predicts that the pair travels better
than either parameter alone; this script measures whether it does.

**Duration.** The manuscript dropped duration as a scaling object because a
season is bounded above by the length of the year, so a scale-free law cannot
apply a priori. That argument is right, and it is not an argument against
survival models -- it is an argument for them. The paper's evidence was that the
exponential beat the power law, but "the power law loses" is not "duration is
memoryless": the exponential is itself a model, with a constant hazard, and it
can be rejected in turn. A Weibull shape above one says the hazard of a season
ending *rises* with its age -- the season has an internal clock. That converts a
negative result into a mechanism, which is what a bounded support should look
like.

The last season is right-censored by the data cut-off. Ignoring that biases
duration downwards, and the bias lands entirely on the most recent year, which
is the one most likely to be quoted. ``--no-censor`` reruns without it so the
size of the correction is visible rather than asserted.

Usage
-----
    python survival_analysis.py --part all
    python survival_analysis.py --part duration --no-censor

Requires ``lifelines`` (not currently in pyproject.toml):

    uv add lifelines        # or: pip install lifelines

Outputs land in ``analysis/survival/`` and are not committed automatically: they
are a diagnostic, and nothing here is cited by the manuscript yet.
"""

from __future__ import annotations

import argparse
import glob
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from study_window import LAST_SEASON_YEAR, POOLED_YEAR, SEASON_YEARS

#: Columns read from the EpiScanner extracts. Kept explicit so a schema change
#: upstream fails here rather than silently producing a frame of NaNs.
PANEL_COLUMNS = ["geocode", "year", "total_cases", "ep_dur"]

OUT_DIR = Path("analysis") / "survival"
DB_PATH = "powerlaw_results.db"

#: Family-wise confidence level for the DKW band.
DKW_DELTA = 0.05

#: A CCDF point enters the distance only while this many units remain above it.
#: Below that the empirical curve is one or two observations, and its departure
#: measures sampling noise, not lack of fit.
MIN_AT_RISK = 10

#: A cross-section smaller than this is not compared at all.
MIN_UNITS = 50


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def load_city_panel(columns=PANEL_COLUMNS, pattern="data/episcanner_*.parquet"):
    """Concatenate the EpiScanner city extracts into one panel.

    The files are one per (state, season); a municipality appears once per
    season. Rows are de-duplicated on (geocode, year) because summing a
    duplicate would inflate the cross-section, and an inflated cross-section
    moves every quantile of the tail it is meant to describe.
    """
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"no files match {pattern!r}. The EpiScanner extracts are not in "
            "the repository; run `python fit_episcanner_pl.py` once to populate "
            "data/, or pass --data-glob pointing at where they live."
        )

    frames = []
    for path in files:
        try:
            frames.append(pd.read_parquet(path, columns=list(columns)))
        except Exception as exc:
            raise RuntimeError(
                f"{path} could not be read with columns {list(columns)}: {exc}"
            ) from exc
    df = pd.concat(frames, ignore_index=True)

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    bad = df["year"].isna()
    if bad.any():
        raise TypeError(
            f"{int(bad.sum())} row(s) carry a non-numeric 'year' in the "
            "EpiScanner extracts; the season key is unusable."
        )
    df["year"] = df["year"].astype(int)

    before = len(df)
    df = df.drop_duplicates(subset=["geocode", "year"])
    if before != len(df):
        print(f"[load] dropped {before - len(df)} duplicate (geocode, year) rows")

    outside = df.loc[~df["year"].isin(SEASON_YEARS), "year"].unique()
    if len(outside):
        print(f"[load] ignoring seasons outside the study window: {sorted(outside)}")
    return df


def load_national_fits(db_path=DB_PATH):
    """The national size fits, one row per season plus the pooled sentinel."""
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(
            "SELECT year, alpha, xmin FROM episcanner_fits "
            "WHERE metric='total_cases'", conn)
    if df.empty:
        raise RuntimeError(
            f"{db_path} holds no national size fit "
            "(episcanner_fits WHERE metric='total_cases'). Run "
            "`python fit_episcanner_pl.py` first."
        )
    df = df.drop_duplicates(subset=["year"]).set_index("year")
    return df


def _integer_counts(values, what="total_cases"):
    """Return the counts as int64, refusing non-integer input.

    The fits were made with ``discrete=True`` and the zeta survival function is
    defined on integers. Silently rounding a continuous series would compare the
    data to a different object than the one that produced alpha.
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size and not np.allclose(arr, np.round(arr), atol=1e-9):
        examples = arr[~np.isclose(arr, np.round(arr), atol=1e-9)][:5]
        raise ValueError(
            f"{what} is not integer-valued (examples: {examples}). The zeta "
            "survival function and the discrete fit both assume integer "
            "counts; the comparison would not be to the same distribution."
        )
    return np.round(arr).astype(np.int64)


# ---------------------------------------------------------------------------
# survival machinery
# ---------------------------------------------------------------------------
def empirical_ccdf(s):
    """Empirical CCDF, and the number at risk, at each observed size.

    ``at_risk`` is the Kaplan-Meier risk set. With a single common truncation
    point the product-limit estimator equals this count ratio exactly, so it is
    written directly -- there is no approximation to audit.
    """
    s = np.sort(np.asarray(s, dtype=np.int64))
    values, counts = np.unique(s, return_counts=True)
    at_risk = s.size - np.cumsum(counts) + counts
    return values, at_risk / s.size, at_risk


def pareto_ccdf(s, alpha, xmin):
    """``Pr(S >= s)`` for the discrete power law: zeta(alpha,s)/zeta(alpha,xmin)."""
    from scipy.special import zeta

    s = np.asarray(s, dtype=float)
    return zeta(alpha, s) / zeta(alpha, float(xmin))


def pareto_hazard(s, alpha):
    """Discrete Pareto hazard ``1 - zeta(a,s+1)/zeta(a,s)``; declines like (a-1)/s."""
    from scipy.special import zeta

    s = np.asarray(s, dtype=float)
    return 1.0 - zeta(alpha, s + 1.0) / zeta(alpha, s)


def dkw_epsilon(n, delta=DKW_DELTA):
    """Half-width of the simultaneous DKW band around an empirical CDF.

    ``P(sup|F_n - F| > eps) <= 2 exp(-2 n eps^2)`` holds for every distribution,
    discrete included, so the band needs no assumption about the tail -- which
    is the point, since the tail is what is in question.
    """
    return float(np.sqrt(np.log(2.0 / delta) / (2.0 * n)))


def _departure(sizes, alpha, xmin, delta=DKW_DELTA):
    """Max |empirical - fitted| above x_min, in DKW-band units."""
    kept = sizes[sizes >= xmin]
    if kept.size < MIN_UNITS:
        return None
    values, emp, at_risk = empirical_ccdf(kept)
    theo = pareto_ccdf(values, alpha, xmin)
    keep = at_risk >= MIN_AT_RISK
    if keep.sum() < 3:
        return None
    dev = float(np.abs(emp[keep] - theo[keep]).max())
    eps = dkw_epsilon(kept.size, delta)
    return dict(
        n_units=int(sizes.size),
        n_above_xmin=int(kept.size),
        share_above_xmin=float(kept.size / sizes.size),
        max_abs_departure=dev,
        dkw_eps=eps,
        departure_in_bands=dev / eps,
        n_grid_points=int(keep.sum()),
    )


# ---------------------------------------------------------------------------
# part A -- the exceedance curve
# ---------------------------------------------------------------------------
def _checkpoints(kept, alpha, xmin, multiples=(10, 100)):
    """Predicted against observed share above ``k * x_min``.

    This is the manuscript's own risk-curve reading -- "5.6--16.5% of
    municipalities exceed ten times the bound" -- evaluated against what the
    target season actually did. The predicted share is unconditional given
    ``S >= x_min``, so the observed share is taken over the same units.
    """
    out = {}
    for k in multiples:
        threshold = k * xmin
        out[f"pred_{k}x"] = float(pareto_ccdf(np.array([threshold]), alpha, xmin)[0])
        out[f"obs_{k}x"] = float((kept >= threshold).mean()) if kept.size else np.nan
    return out


def exceedance_tables(panel, fits):
    """Within-season departure, and the out-of-season transport of the pair.

    The first table is descriptive and is labelled so: x_min was chosen by
    minimising this departure, so it cannot test the family. The second table
    carries the test that is not circular.

    The transport window is ``[x_min fitted, max size in the target season]``,
    which means a pair whose x_min moved up is judged on a *narrower* window
    and has correspondingly less power. ``n_grid_points`` and ``window_ratio``
    are reported so that a small departure on a narrow window is not read as a
    better transport than a large one on a wide window.
    """
    within, transport = [], []

    for year in SEASON_YEARS:
        if year not in fits.index:
            continue
        block = panel.loc[panel["year"] == year, "total_cases"]
        if block.empty:
            continue
        sizes = _integer_counts(block)
        alpha, xmin = float(fits.loc[year, "alpha"]), float(fits.loc[year, "xmin"])
        result = _departure(sizes, alpha, xmin)
        if result is None:
            within.append(dict(year=year, alpha=alpha, xmin=xmin, status="skipped"))
            continue
        within.append(dict(year=year, alpha=alpha, xmin=xmin, status="ok", **result))

    if POOLED_YEAR in fits.index:
        block = panel.loc[panel["year"].isin(SEASON_YEARS), "total_cases"]
        if not block.empty:
            sizes = _integer_counts(block)
            alpha = float(fits.loc[POOLED_YEAR, "alpha"])
            xmin = float(fits.loc[POOLED_YEAR, "xmin"])
            result = _departure(sizes, alpha, xmin)
            if result is not None:
                within.append(dict(year=POOLED_YEAR, alpha=alpha, xmin=xmin,
                                   status="ok", **result))

    for y0, y1 in zip(SEASON_YEARS, SEASON_YEARS[1:]):
        if y0 not in fits.index or y1 not in fits.index:
            continue
        block = panel.loc[panel["year"] == y1, "total_cases"]
        if block.empty:
            continue
        sizes = _integer_counts(block)

        alpha0 = float(fits.loc[y0, "alpha"])
        xmin0 = float(fits.loc[y0, "xmin"])
        moved = _departure(sizes, alpha0, xmin0)
        if moved is None:
            continue
        kept = sizes[sizes >= xmin0]

        # Reference: the same pair re-estimated inside the target season. The
        # gap between the two rows is what season t fails to anticipate.
        alpha1 = float(fits.loc[y1, "alpha"])
        xmin1 = float(fits.loc[y1, "xmin"])
        local = _departure(sizes, alpha1, xmin1)

        transport.append(dict(
            fitted_year=y0, target_year=y1,
            alpha_fitted=alpha0, xmin_fitted=xmin0,
            alpha_target=alpha1, xmin_target=xmin1,
            alpha_abs_change=abs(alpha1 - alpha0),
            xmin_ratio=xmin1 / xmin0,
            departure_in_bands=moved["departure_in_bands"],
            departure_in_bands_local=(local["departure_in_bands"]
                                      if local else np.nan),
            inside_band=bool(moved["departure_in_bands"] <= 1.0),
            n_above_xmin=moved["n_above_xmin"],
            n_grid_points=moved["n_grid_points"],
            window_ratio=float(sizes.max() / xmin0),
            **_checkpoints(kept, alpha0, xmin0),
        ))

    return pd.DataFrame(within), pd.DataFrame(transport)


def plot_exceedance(panel, fits, within, transport, out_dir=OUT_DIR):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

    target = POOLED_YEAR if POOLED_YEAR in fits.index else SEASON_YEARS[-1]
    block = (panel.loc[panel["year"].isin(SEASON_YEARS), "total_cases"]
             if target == POOLED_YEAR
             else panel.loc[panel["year"] == target, "total_cases"])
    sizes = _integer_counts(block)
    alpha = float(fits.loc[target, "alpha"])
    xmin = float(fits.loc[target, "xmin"])
    kept = sizes[sizes >= xmin]
    values, emp, at_risk = empirical_ccdf(kept)
    theo = pareto_ccdf(values, alpha, xmin)
    eps = dkw_epsilon(kept.size)

    ax = axes[0]
    ax.fill_between(values, np.clip(emp - eps, 0, 1), np.clip(emp + eps, 0, 1),
                    color="0.75", alpha=0.5, lw=0,
                    label=f"DKW 95% band (half-width {eps:.3f})")
    ax.plot(values, emp, drawstyle="steps-post", color="#08519c", lw=1.4,
            label="empirical (Kaplan-Meier)")
    ax.plot(values, theo, color="#cb181d", lw=1.2, ls="--",
            label=fr"fitted $\zeta(\alpha,s)/\zeta(\alpha,x_{{\min}})$, "
                  fr"$\alpha={alpha:.2f}$")
    ax.axvline(xmin, color="0.3", lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"epidemic size $s$ (cases)")
    ax.set_ylabel(r"$\Pr(S \geq s)$")
    ax.set_title("(a) one cross-section: fit against the data it was fitted to")
    ax.legend(loc="lower left", fontsize=7)

    ax = axes[1]
    if not transport.empty:
        x = transport["target_year"].to_numpy()
        ax.plot(x, transport["departure_in_bands"], "o-", color="#cb181d", lw=1.3,
                ms=5, label=r"pair $(\alpha_{t},x_{\min,t})$ applied to season $t{+}1$")
        ax.plot(x, transport["departure_in_bands_local"], "s--", color="0.45",
                lw=1.0, ms=4, label=r"pair re-fitted inside season $t{+}1$")
        ax.axhline(1.0, color="black", lw=0.9, ls=":",
                   label="edge of the DKW band")
        ax.set_yscale("log")
        ax.set_xlabel("target season")
        ax.set_ylabel("departure, in DKW-band units")
        ax.set_title("(b) out-of-season transport -- the test that is not circular")
        ax.legend(loc="upper left", fontsize=7)
    else:
        ax.text(0.5, 0.5, "no consecutive pair met the minimum size",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()

    fig.tight_layout()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(out_dir / f"fig_survival_exceedance.{suffix}", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# part B -- duration as time to event
# ---------------------------------------------------------------------------
def duration_tables(panel, censor_last=True):
    """Parametric survival fits for season duration, plus a Cox model.

    Returns ``(models, summary)``. ``models`` is the AIC comparison; ``summary``
    holds the KM median, the censoring fraction, the Weibull shape with its
    interval, the size-tercile log-rank test and the Cox coefficients.
    """
    from lifelines import (CoxPHFitter, ExponentialFitter, GeneralizedGammaFitter,
                           KaplanMeierFitter, LogLogisticFitter, LogNormalFitter,
                           WeibullFitter)
    from lifelines.statistics import multivariate_logrank_test

    d = panel.dropna(subset=["ep_dur", "total_cases"]).copy()
    d = d[d["ep_dur"] > 0]
    if d.empty:
        raise RuntimeError("no positive ep_dur values in the panel")
    d["T"] = d["ep_dur"].astype(float)

    if d["T"].median() > 60:
        print(
            f"[duration] median ep_dur is {d['T'].median():.1f}. If the "
            "EpiScanner series is in days rather than weeks the axis label and "
            "every hazard reading below must be rescaled; check before quoting."
        )

    if censor_last:
        d["E"] = (d["year"] < LAST_SEASON_YEAR).astype(int)
        censored_share = float(1 - d["E"].mean())
        if censored_share == 0:
            print("[duration] --censor given but nothing was censored")
    else:
        d["E"] = 1
        censored_share = 0.0

    km = KaplanMeierFitter().fit(d["T"], event_observed=d["E"])
    km_median = float(km.median_survival_time_)

    families = {
        "exponential": ExponentialFitter,
        "weibull": WeibullFitter,
        "lognormal": LogNormalFitter,
        "loglogistic": LogLogisticFitter,
        "generalized_gamma": GeneralizedGammaFitter,
    }
    fitted, rows = {}, []
    for name, cls in families.items():
        f = cls().fit(d["T"], event_observed=d["E"])
        fitted[name] = f
        rows.append(dict(model=name, aic=float(f.AIC_),
                         log_likelihood=float(f.log_likelihood_),
                         n_params=int(len(f.params_))))
    models = pd.DataFrame(rows).sort_values("aic").reset_index(drop=True)
    models["delta_aic"] = models["aic"] - models["aic"].min()

    w = fitted["weibull"]
    shape = float(w.rho_)
    try:
        lo = float(w.summary.loc["rho_", "coef lower 95%"])
        hi = float(w.summary.loc["rho_", "coef upper 95%"])
    except Exception:
        lo = hi = float("nan")

    # Size terciles. Repeat seasons of one municipality are not independent, so
    # the log-rank p-value is read as a screen, and the Cox model below clusters
    # on the municipality where it can.
    d["size_tercile"] = pd.qcut(np.log1p(d["total_cases"]), 3,
                                labels=["small", "medium", "large"],
                                duplicates="drop")
    lr = multivariate_logrank_test(d["T"], d["size_tercile"], d["E"])

    cox_rows = []
    try:
        cox_df = pd.DataFrame({
            "T": d["T"], "E": d["E"],
            "log_cases": np.log1p(d["total_cases"].astype(float)),
            "season": d["year"].astype(float),
            "geocode": d["geocode"],
        })
        cph = CoxPHFitter()
        cph.fit(cox_df, duration_col="T", event_col="E", cluster_col="geocode")
        for term in ("log_cases", "season"):
            cox_rows.append(dict(
                term=term,
                hazard_ratio=float(np.exp(cph.params_[term])),
                ci_low=float(np.exp(cph.confidence_intervals_.loc[term].iloc[0])),
                ci_high=float(np.exp(cph.confidence_intervals_.loc[term].iloc[1])),
                p=float(cph.summary.loc[term, "p"]),
            ))
    except Exception as exc:
        print(f"[duration] Cox model skipped: {exc}")

    summary = dict(
        n_units=int(len(d)),
        n_seasons=int(d["year"].nunique()),
        censored_share=censored_share,
        censoring=censor_last,
        km_median_duration=km_median,
        mean_duration=float(d["T"].mean()),
        weibull_shape=shape,
        weibull_shape_ci=[lo, hi],
        exponential_rejected=bool(not (lo <= 1.0 <= hi)),
        logrank_p=float(lr.p_value),
        logrank_by="total_cases tercile",
        cox=cox_rows,
    )
    return models, summary, d, fitted


def plot_duration(d, fitted, models, summary, out_dir=OUT_DIR):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

    ax = axes[0]
    for label, color in (("small", "#9ecae1"), ("medium", "#4292c6"),
                         ("large", "#08519c")):
        sub = d[d["size_tercile"] == label]
        if sub.empty:
            continue
        km = __import__("lifelines").KaplanMeierFitter().fit(
            sub["T"], event_observed=sub["E"], label=f"{label} season")
        km.plot_survival_function(ax=ax, ci_show=False, color=color, lw=1.3)
    w = fitted["weibull"]
    grid = np.linspace(0, float(d["T"].max()), 200)
    ax.plot(grid, w.survival_function_at_times(grid).to_numpy(), "k--", lw=1.1,
            label=fr"Weibull $k={summary['weibull_shape']:.2f}$")
    ax.set_xlabel("season duration (weeks, EpiScanner `ep_dur`)")
    ax.set_ylabel(r"$\Pr(\text{duration} > t)$")
    ax.set_title("(a) duration by epidemic size tercile")
    ax.legend(fontsize=7)

    ax = axes[1]
    ax.plot(models["model"], models["delta_aic"], "o", color="#08519c", ms=7)
    for _, row in models.iterrows():
        ax.annotate(f"{row['delta_aic']:.0f}", (row["model"], row["delta_aic"]),
                    textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=7)
    ax.set_ylabel(r"$\Delta$AIC (0 = best)")
    ax.set_title("(b) which family describes duration")
    ax.tick_params(axis="x", rotation=20)
    ax.set_ylim(bottom=-0.5)

    fig.tight_layout()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(out_dir / f"fig_survival_duration.{suffix}", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# cli
# ---------------------------------------------------------------------------
def run_exceedance(panel, fits, out_dir):
    within, transport = exceedance_tables(panel, fits)
    out_dir.mkdir(parents=True, exist_ok=True)
    within.to_csv(out_dir / "exceedance_within_season.csv", index=False)
    transport.to_csv(out_dir / "exceedance_transport.csv", index=False)

    print("\n" + "=" * 78)
    print("A1. WITHIN-SEASON DEPARTURE  (descriptive, NOT a test: x_min was")
    print("    chosen by minimising this very quantity)")
    print("=" * 78)
    cols = ["year", "alpha", "xmin", "n_above_xmin", "share_above_xmin",
            "max_abs_departure", "departure_in_bands"]
    show = within[within["status"] == "ok"] if "status" in within else within
    if not show.empty:
        print(show[[c for c in cols if c in show]].round(4).to_string(index=False))

    print("\n" + "=" * 78)
    print("A2. OUT-OF-SEASON TRANSPORT  (this one is a test)")
    print("=" * 78)
    if transport.empty:
        print("no consecutive pair met the minimum size")
    else:
        cols = ["fitted_year", "target_year", "alpha_fitted", "xmin_fitted",
                "xmin_ratio", "window_ratio", "n_grid_points",
                "departure_in_bands", "departure_in_bands_local", "inside_band"]
        print(transport[[c for c in cols if c in transport]]
              .round(3).to_string(index=False))

        print("\n   the risk curve as the manuscript states it, carried forward "
              "one season:")
        print(f"   {'pair':>11} {'pred 10x':>9} {'obs 10x':>9} {'pred 100x':>10} {'obs 100x':>10}")
        for _, row in transport.iterrows():
            print(f"   {int(row.fitted_year)}->{int(row.target_year):<5} "
                  f"{row.get('pred_10x', float('nan')):>8.2%} "
                  f"{row.get('obs_10x', float('nan')):>9.2%} "
                  f"{row.get('pred_100x', float('nan')):>9.2%} "
                  f"{row.get('obs_100x', float('nan')):>10.2%}")

        inside = int(transport["inside_band"].sum())
        print(f"\n{inside} of {len(transport)} consecutive pairs transport inside "
              "the band")

    plot_exceedance(panel, fits, within, transport, out_dir)
    return within, transport


def run_duration(panel, out_dir, censor_last=True):
    models, summary, d, fitted = duration_tables(panel, censor_last=censor_last)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "" if censor_last else "_nocensor"
    models.to_csv(out_dir / f"duration_models{tag}.csv", index=False)
    (out_dir / f"duration_summary{tag}.json").write_text(
        json.dumps(summary, indent=2, default=float))

    print("\n" + "=" * 78)
    print(f"B. DURATION AS TIME TO EVENT  (censoring {'on' if censor_last else 'off'})")
    print("=" * 78)
    print(models.round(2).to_string(index=False))
    print(f"\nunits {summary['n_units']:,}   seasons {summary['n_seasons']}   "
          f"censored {summary['censored_share']:.1%}")
    print(f"KM median duration   {summary['km_median_duration']:.2f} weeks")
    lo, hi = summary["weibull_shape_ci"]
    print(f"Weibull shape        {summary['weibull_shape']:.3f}  "
          f"[{lo:.3f}, {hi:.3f}]  -> exponential "
          f"{'REJECTED' if summary['exponential_rejected'] else 'not rejected'}")
    print(f"log-rank by size     p = {summary['logrank_p']:.3g}")
    for row in summary["cox"]:
        print(f"  Cox {row['term']:<10} HR {row['hazard_ratio']:.3f} "
              f"[{row['ci_low']:.3f}, {row['ci_high']:.3f}]  p={row['p']:.3g}")

    plot_duration(d, fitted, models, summary, out_dir)
    return models, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--part", choices=("exceedance", "duration", "all"),
                        default="all")
    parser.add_argument("--db", default=DB_PATH)
    parser.add_argument("--data-glob", default="data/episcanner_*.parquet")
    parser.add_argument("--out", default=str(OUT_DIR))
    parser.add_argument("--no-censor", action="store_true",
                        help="treat the last season's durations as observed")
    args = parser.parse_args(argv)

    panel = load_city_panel(pattern=args.data_glob)
    out_dir = Path(args.out)
    print(f"[load] {len(panel):,} municipality-seasons, "
          f"{panel['year'].nunique()} seasons, "
          f"{panel['geocode'].nunique():,} municipalities")

    if args.part in ("exceedance", "all"):
        fits = load_national_fits(args.db)
        run_exceedance(panel, fits, out_dir)
    if args.part in ("duration", "all"):
        run_duration(panel, out_dir, censor_last=not args.no_censor)

    print(f"\nwritten to {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
