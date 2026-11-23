r"""Is the alert threshold a per-state quantity? The audit that answers no.

A state secretariat reads its own list, and Table~\ref{tab:state_lists} shows the
declared rule $p\ge0.5$ returning lists whose lengths bear no relation to the
number of exceedances they contain: no alert at all in three states, twice as
many alerts as exceedances in Cear{\'a}, one alert for every four exceedances in
Bahia. The obvious repair is to estimate the threshold, or the level, inside
each state. This script tests that repair instead of asserting it, and the answer
is that the quantity to repair is not a state property.

Three measurements, and they are separate objects.

*Calibration* is the part a threshold operates on: whether $p=0.5$ means "more
likely than not" for this state in this season. We decompose the state-season
offset $\\bar p - \\bar y$ into a season component and a state-within-season
component, and then ask whether the state component survives from one season to
the next. It barely does: $r\\approx0.30$, interval touching zero.

*Recalibration* is the repair that follows from it, and it is tested out of
sample, which is the only way the question is worth asking. A per-state logit
shift is fitted on seasons strictly earlier than the target season, shrunk
toward the global shift by 20 pseudo-observations, and applied to the target
season. It makes the next season *worse* than leaving the probabilities alone,
in every paired comparison the bootstrap keeps, and the global shift is worse
too. The reason is in the decomposition read the other way round: the season
component is the larger half of the offset in the seasons that matter, and the
season is the thing being forecast. A model trained on earlier seasons cannot
know that 2024 will sit 23 points above what it predicts, and a per-state
correction fitted on 2021--2023 carries the wrong sign into it.

*Threshold* is the second repair, and it is a wash: choosing a Youden threshold
inside each state and applying it to the next season returns the same mean $F_1$
as the declared 0.5, at the price of about fourteen per cent more alerts, and
the estimate is unstable across target seasons. Youden's $J$ is prevalence-
invariant by construction, so per-state fitting has little left to find and adds
variance while looking for it.

What follows for the paper: the state may act on the *order* of its list, which
is what a within-state AUC of the order of 0.6 measures, and the *length* of the
list is a capacity input rather than an estimated threshold. No per-state
threshold is issued, and that is a result rather than an omission.

Run from the repository root:

    python calibration_diagnosis.py

writes ``calibration_diagnosis.json`` and ``calibration_diagnosis.meta.json``,
which ``make_ablation_tables.py`` reads and whose numbers the prose gate checks.
The bootstrap is seeded, so the intervals are reproducible.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.metrics import f1_score, roc_curve

SEED = 20261123
N_BOOT_LEVEL = 500
N_BOOT_LEVEL_RECAL = 400
N_BOOT_PERSIST = 2000
MIN_UNITS = 20          # the same cut at which a per-state list is issued
MIN_FIT_UNITS = 20      # rows needed before a state gets its own shift
SHRINK = 20.0           # pseudo-observations pulling a state toward the global shift
MIN_THRESHOLD_ROWS = 40 # rows before a state gets its own Youden threshold
FIXED_RULE = 0.5
TARGET_SEASONS = (2022, 2023, 2024, 2025)
REF_RUNG = "M3"


def _logit(p):
    q = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(q / (1 - q))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _brier(p, y):
    return float(np.mean((p - y) ** 2))


def _fit_shift(lp, y):
    """Logit shift maximising the Bernoulli likelihood, one scalar."""
    def neg(d):
        s = _sigmoid(lp + d)
        return -np.mean(y * np.log(np.clip(s, 1e-12, 1))
                        + (1 - y) * np.log(np.clip(1 - s, 1e-12, 1)))
    return float(minimize_scalar(neg, bounds=(-5, 5), method="bounded").x)


def _youden(y, p):
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[int(np.argmax(tpr - fpr))])


def _state_of(unit):
    from make_ablation_tables import _state_of as f
    return f(unit)


def _prepare(root):
    path = os.path.join(root, "classifier_results.parquet")
    oof = pd.read_parquet(path)
    oof["uf"] = _state_of(oof["unit"])
    d = oof[oof["rung"] == REF_RUNG].dropna(subset=["y_prob", "y_true"]).copy()
    d["lp"] = _logit(d["y_prob"].values)
    return d


def _decomposition(d):
    per = (d.groupby(["origin", "uf"])
             .agg(n=("y_true", "size"), base=("y_true", "mean"),
                  mean_p=("y_prob", "mean")).reset_index())
    per = per[per["n"] >= MIN_UNITS].copy()
    per["offset"] = per["mean_p"] - per["base"]
    per["season_mean"] = per.groupby("origin")["offset"].transform("mean")
    per["resid"] = per["offset"] - per["season_mean"]
    w = per["n"].to_numpy(float)
    sm = per["season_mean"].to_numpy()
    grand = float(np.average(per["offset"].to_numpy(), weights=w))
    ss_b = float((w * (sm - grand) ** 2).sum())
    ss_w = float((w * (per["offset"].to_numpy() - sm) ** 2).sum())
    by_season = (per.groupby("origin")["offset"].agg(["mean", "std", "size"])
                   .reset_index())
    return per, {
        "ss_between_season": ss_b,
        "ss_within_state": ss_w,
        "share_between_season": ss_b / (ss_b + ss_w),
        "sd_offset": float(per["offset"].std()),
        "min_offset": float(per["offset"].min()),
        "max_offset": float(per["offset"].max()),
        "n_state_seasons": int(len(per)),
        "n_states": int(per["uf"].nunique()),
        "seasons": [{"season": int(r.origin), "mean_offset": float(r.mean),
                     "sd_offset": float(r.std), "n_states": int(r.size)}
                    for r in by_season.itertuples()],
    }


def _persistence(per, rng):
    pairs = []
    seasons = sorted(per["origin"].unique())
    for o1, o2 in zip(seasons, seasons[1:]):
        a = per[per["origin"] == o1].set_index("uf")["resid"]
        b = per[per["origin"] == o2].set_index("uf")["resid"]
        common = a.index.intersection(b.index)
        for u in common:
            pairs.append((u, int(o1), int(o2), float(a[u]), float(b[u])))
    pp = pd.DataFrame(pairs, columns=["uf", "s1", "s2", "x", "y"])
    r = float(np.corrcoef(pp["x"], pp["y"])[0, 1])
    boot = []
    for _ in range(N_BOOT_PERSIST):
        idx = rng.integers(0, len(pp), len(pp))
        s = pp.iloc[idx]
        if s["x"].std() > 0 and s["y"].std() > 0:
            boot.append(np.corrcoef(s["x"], s["y"])[0, 1])
    boot = np.asarray(boot)
    return pp, {
        "r": r,
        "lo": float(np.percentile(boot, 2.5)),
        "hi": float(np.percentile(boot, 97.5)),
        "n_pairs": int(len(pp)),
        "replicates": N_BOOT_PERSIST,
    }


def _recalibration(d, rng):
    """Nested: fit the correction on earlier seasons, score the target season."""
    point = []
    for t in TARGET_SEASONS:
        tr, te = d[d["origin"] < t], d[d["origin"] == t]
        if len(tr) < 200 or len(te) < 100:
            continue
        dg = _fit_shift(tr["lp"].to_numpy(), tr["y_true"].to_numpy())
        shift = {}
        for uf, g in tr.groupby("uf"):
            if len(g) >= MIN_FIT_UNITS:
                raw = _fit_shift(g["lp"].to_numpy(), g["y_true"].to_numpy())
                shift[uf] = (len(g) * raw + SHRINK * dg) / (len(g) + SHRINK)
        p_state = np.asarray([_sigmoid(l + shift.get(u, dg))
                              for l, u in zip(te["lp"].to_numpy(),
                                              te["uf"].to_numpy())])
        p_glob = _sigmoid(te["lp"].to_numpy() + dg)
        y = te["y_true"].to_numpy()
        point.append({
            "season": int(t), "n": int(len(te)), "base_rate": float(y.mean()),
            "brier_raw": _brier(te["y_prob"].to_numpy(), y),
            "brier_global": _brier(p_glob, y),
            "brier_state": _brier(p_state, y),
            "fitted_global_shift": float(dg),
            "states_with_own_shift": int(len(shift)),
        })
    point = pd.DataFrame(point)

    states = np.asarray(sorted(d["uf"].unique()))
    cache = {t: (d[d["origin"] < t], d[(d["origin"] == t)]) for t in TARGET_SEASONS}
    deltas = []
    for _ in range(N_BOOT_LEVEL_RECAL):
        samp = rng.choice(states, size=len(states), replace=True)
        acc = []
        for t, (tr, te) in cache.items():
            tr = tr[tr["uf"].isin(samp)]
            te = te[te["uf"].isin(samp)]
            if len(tr) < 150 or len(te) < 80:
                continue
            dg = _fit_shift(tr["lp"].to_numpy(), tr["y_true"].to_numpy())
            shift = {}
            for uf, g in tr.groupby("uf"):
                if len(g) >= MIN_FIT_UNITS:
                    raw = _fit_shift(g["lp"].to_numpy(), g["y_true"].to_numpy())
                    shift[uf] = (len(g) * raw + SHRINK * dg) / (len(g) + SHRINK)
            p_state = np.asarray([_sigmoid(l + shift.get(u, dg))
                                  for l, u in zip(te["lp"].to_numpy(),
                                                  te["uf"].to_numpy())])
            acc.append(_brier(p_state, te["y_true"].to_numpy())
                       - _brier(te["y_prob"].to_numpy(), te["y_true"].to_numpy()))
        if acc:
            deltas.append(float(np.mean(acc)))
    deltas = np.asarray(deltas)
    return {
        "per_season": point.to_dict(orient="records"),
        "brier_raw_mean": float(point["brier_raw"].mean()),
        "brier_global_mean": float(point["brier_global"].mean()),
        "brier_state_mean": float(point["brier_state"].mean()),
        "delta_state_mean": float(deltas.mean()),
        "delta_state_lo": float(np.percentile(deltas, 2.5)),
        "delta_state_hi": float(np.percentile(deltas, 97.5)),
        "share_resamples_worse": float((deltas > 0).mean()),
        "replicates": N_BOOT_LEVEL_RECAL,
    }


def _thresholds(d):
    rows = []
    for t in TARGET_SEASONS:
        tr, te = d[d["origin"] < t], d[d["origin"] == t]
        if len(te) < 100:
            continue
        tg = _youden(tr["y_true"].to_numpy(), tr["y_prob"].to_numpy())
        own = {}
        for uf, g in tr.groupby("uf"):
            if len(g) >= MIN_THRESHOLD_ROWS and g["y_true"].nunique() == 2:
                own[uf] = _youden(g["y_true"].to_numpy(), g["y_prob"].to_numpy())
        y = te["y_true"].to_numpy()
        p = te["y_prob"].to_numpy()
        p_state = np.asarray([1 if v >= own.get(u, tg) else 0
                              for v, u in zip(p, te["uf"].to_numpy())])
        rows.append({
            "season": int(t), "n": int(len(te)),
            "n_true": int(y.sum()),
            "states_with_own_threshold": int(len(own)),
            "f1_fixed": float(f1_score(y, (p >= FIXED_RULE).astype(int), zero_division=0)),
            "f1_global_threshold": float(f1_score(y, (p >= tg).astype(int), zero_division=0)),
            "f1_state_threshold": float(f1_score(y, p_state, zero_division=0)),
            "alerts_fixed": int((p >= FIXED_RULE).sum()),
            "alerts_state_threshold": int(p_state.sum()),
            "global_threshold": float(tg),
        })
    tab = pd.DataFrame(rows)
    # dispersion of the per-state thresholds, most recent fit
    tr = d[d["origin"] < TARGET_SEASONS[-1]]
    vals = []
    for _, g in tr.groupby("uf"):
        if len(g) >= MIN_THRESHOLD_ROWS and g["y_true"].nunique() == 2:
            vals.append(_youden(g["y_true"].to_numpy(), g["y_prob"].to_numpy()))
    vals = np.asarray(vals)
    return {
        "per_season": tab.to_dict(orient="records"),
        "f1_fixed_mean": float(tab["f1_fixed"].mean()),
        "f1_global_mean": float(tab["f1_global_threshold"].mean()),
        "f1_state_mean": float(tab["f1_state_threshold"].mean()),
        "alerts_fixed_mean": float(tab["alerts_fixed"].mean()),
        "alerts_state_mean": float(tab["alerts_state_threshold"].mean()),
        "threshold_min": float(vals.min()),
        "threshold_median": float(np.median(vals)),
        "threshold_max": float(vals.max()),
        "threshold_fit_states": int(len(vals)),
        "fixed_rule": FIXED_RULE,
    }


def main(root: str = ".") -> int:
    rng = np.random.default_rng(SEED)
    d = _prepare(root)
    per, dec = _decomposition(d)
    pp, pers = _persistence(per, rng)
    rec = _recalibration(d, rng)
    thr = _thresholds(d)

    n_alarm_worse = rec["share_resamples_worse"]
    facts = {
        "seed": SEED,
        "ref_rung": REF_RUNG,
        "target_seasons": list(TARGET_SEASONS),
        "min_units": MIN_UNITS,
        "decomposition": dec,
        "persistence": pers,
        "recalibration": rec,
        "threshold": thr,
        "verdict": {
            "per_state_threshold_issued": False,
            "reason_level_not_stable": pers["r"] < 0.5,
            "reason_recalibration_hurts": rec["delta_state_lo"] > 0,
            "share_boot_worse": float(n_alarm_worse),
        },
    }
    out = os.path.join(root, "calibration_diagnosis.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(facts, fh, indent=2)
    meta = {
        "argv": [], "level": "city", "region": "BR",
        "source": "classifier_results.parquet",
        "ref_rung": REF_RUNG, "fixed_rule": FIXED_RULE,
        "target_seasons": list(TARGET_SEASONS),
        "min_units": MIN_UNITS, "shrink_pseudo_obs": SHRINK,
        "seed": SEED,
        "bootstrap_replicates": {
            "persistence": N_BOOT_PERSIST,
            "recalibration": N_BOOT_LEVEL_RECAL,
        },
        "bootstrap_unit": "state",
        "decision_rule": ("a per-state correction is issued only if its 95% "
                          "state-bootstrap interval for the Brier difference "
                          "against the uncorrected probability lies below zero "
                          "on seasons held out from the fit"),
        "what": "whether the alert threshold is a per-state quantity",
    }
    with open(os.path.join(root, "calibration_diagnosis.meta.json"), "w",
              encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    print(f"decomposition: {100*dec['share_between_season']:.1f}% season, "
          f"{100*(1-dec['share_between_season']):.1f}% state (n={dec['n_state_seasons']})")
    print(f"persistence r = {pers['r']:+.3f} [{pers['lo']:+.3f}, {pers['hi']:+.3f}]")
    print(f"Brier raw {rec['brier_raw_mean']:.4f} | global {rec['brier_global_mean']:.4f} "
          f"| state {rec['brier_state_mean']:.4f}")
    print(f"delta per-state = {rec['delta_state_mean']:+.4f} "
          f"[{rec['delta_state_lo']:+.4f}, {rec['delta_state_hi']:+.4f}], "
          f"worse in {100*n_alarm_worse:.0f}% of resamples")
    print(f"F1 fixed {thr['f1_fixed_mean']:.3f} | global {thr['f1_global_mean']:.3f} "
          f"| state {thr['f1_state_mean']:.3f} | alerts {thr['alerts_fixed_mean']:.0f} "
          f"vs {thr['alerts_state_mean']:.0f}")
    print(f"thresholds across states: {thr['threshold_min']:.3f} to "
          f"{thr['threshold_max']:.3f} (median {thr['threshold_median']:.3f})")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
