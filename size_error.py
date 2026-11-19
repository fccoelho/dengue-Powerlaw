"""How the size forecast's relative error depends on municipality size.

One question, one place to answer it. The manuscript reports that a municipality
below 100,000 inhabitants is a different forecasting problem from one above it
(Section ``sec:smallmunic``), and both Figure 14 and the three sentences of the
text that lean on it rest on the same quantity: the distribution of the relative
error in each size class, computed from the out-of-fold predictions of the
reported estimator.

Keeping that computation here rather than inside the figure has two
consequences. The first is that the numbers the figure draws and the numbers the
text quotes cannot drift apart, which they did: the caption of Figure 14 said
"the median relative error is larger below the threshold" over an image whose two
medians were the *same number*. The second is that the claim can be audited
without drawing anything -- ``check_size_error.py`` prints the table and the
verdict, and the figure is one optional flag away.

The decision rule is fixed here, before any run, so that a figure that does not
support the claim is removed instead of re-drawn until it does.

Why the median is the wrong statistic for this question
-------------------------------------------------------
93.2% of the municipalities sit below the cut. The median of the "small" class is
therefore, to within the 6.8% above it, the median of the whole sample: the two
medians are the same number by construction, and only an implausible effect could
separate them. Any panel built on medians will show no difference *whatever the
data say*, which is why the claim has to be stated about the tail of the error
distribution -- the interquartile range and the share of units that miss by more
than a factor of two -- where the two classes genuinely differ if they differ at
all.

The epidemiological reading is also the better one. A small municipality does not
have a biased forecast; it has an *unstable* one: the same method that is usually
within a factor of two is, twice as often, not.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

#: The population threshold the manuscript separates the two classes at. It is a
#: round number chosen for interpretability, not estimated from the data: a cut
#: fitted to the errors would let the split choose the answer it reports.
SIZE_CUT = 100_000.0

#: A large relative error, used for the "misses by more than a factor of two"
#: share. 100% is the natural scale here -- the forecast is off by the size of
#: the season itself -- and it needs no fitting.
BIG_ERROR_APE = 100.0

#: Pre-registered decision rule. The figure survives only if the third quartile
#: of the relative error is at least this many times larger in the class below
#: the threshold, with a bootstrap interval that excludes 1. Written down before
#: the run so that the outcome is a verdict, not a judgement call.
DECISION_Q3_RATIO = 1.3

#: Bootstrap replicates for the interval on that ratio.
N_BOOT = 2000

__all__ = [
    "SIZE_CUT", "BIG_ERROR_APE", "DECISION_Q3_RATIO",
    "unit_errors", "class_stats", "decide", "bootstrap_q3_ratio",
    "draw_ecdf", "load_oof", "report",
    "repo_root", "ensure_local_module", "doctor",
]


# ---------------------------------------------------------------------------
# from row-level predictions to one row per municipality
# ---------------------------------------------------------------------------
def _pick(df, *names):
    """First of ``names`` present in ``df``, else raise with the columns it saw."""
    for n in names:
        if n in df.columns:
            return n
    raise KeyError(
        f"none of {names} found; the frame has {sorted(df.columns)}")


def unit_errors(oof: pd.DataFrame) -> pd.DataFrame:
    """One row per municipality: its population and its median relative error.

    A unit evaluated in three forecast origins would otherwise enter three times,
    inflating every count and thickening any cloud with correlated copies of
    itself. The reduction is the median over the origins, which is what
    Figure 14 in the manuscript draws and what the text calls "a unit's error".

    Predictions that are not strictly positive, or against a season of zero
    cases, carry no relative error and are dropped before the reduction; the
    count is reported by :func:`class_stats` so that the exclusion is visible
    rather than silent.
    """
    tcol = _pick(oof, "y_true", "obs", "observed", "total_cases")
    pcol = _pick(oof, "y_pred", "pred", "prediction", "y_hat")
    gcol = _pick(oof, "geocode", "id_municip", "unit", "id")
    popcol = _pick(oof, "population", "pop", "populacao")

    d = oof[[gcol, popcol, tcol, pcol] + (
        ["origin"] if "origin" in oof.columns else [])].copy()
    d.columns = ["geocode", "population", "y_true", "y_pred"] + (
        ["origin"] if "origin" in oof.columns else [])

    for c in ("population", "y_true", "y_pred"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d[np.isfinite(d["y_true"]) & np.isfinite(d["y_pred"])
          & np.isfinite(d["population"])]
    d = d[(d["y_true"] > 0) & (d["y_pred"] > 0) & (d["population"] > 0)].copy()
    if d.empty:
        raise ValueError(
            "no usable prediction: every row has a non-positive observed season, "
            "a non-positive forecast or a missing population. Relative error is "
            "undefined there, so there is nothing to report.")
    d["ape"] = (d["y_pred"] - d["y_true"]).abs() / d["y_true"] * 100.0

    agg = {"population": ("population", "first"), "ape": ("ape", "median")}
    if "origin" in d.columns:
        agg["n_origins"] = ("origin", "nunique")
    unit = d.groupby("geocode").agg(**agg).reset_index()
    if "n_origins" not in unit.columns:
        unit["n_origins"] = 1
    return unit


# ---------------------------------------------------------------------------
# the table, and the rule
# ---------------------------------------------------------------------------
def _describe(x: np.ndarray) -> dict:
    return {
        "n": int(x.size),
        "median": round(float(np.median(x)), 1),
        "q1": round(float(np.percentile(x, 25)), 1),
        "q3": round(float(np.percentile(x, 75)), 1),
        "p90": round(float(np.percentile(x, 90)), 1),
        "share_over_big": round(float((x > BIG_ERROR_APE).mean()), 4),
    }


def class_stats(unit: pd.DataFrame, cut: float | None = None) -> dict:
    """The two classes side by side, with the ratios the claim needs.

    Returns medians (reported, because the text quotes them) *and* the tail
    statistics the verdict is based on. Reporting both is deliberate: it lets the
    reader see that the medians agree, which is the reason the claim had to be
    restated about the tail.
    """
    cut = SIZE_CUT if cut is None else float(cut)
    small = unit.loc[unit["population"] < cut, "ape"].to_numpy(float)
    large = unit.loc[unit["population"] >= cut, "ape"].to_numpy(float)
    if small.size == 0 or large.size == 0:
        raise ValueError(
            f"both size classes must be populated at the {cut:,.0f}-inhabitant "
            f"cut; got {small.size} below and {large.size} above. The "
            "municipal panel of the paper has 93.2% of its units below the cut, "
            "so an empty side means this is not that panel.")

    below, above = _describe(small), _describe(large)

    def ratio(key):
        a, b = below[key], above[key]
        return round(float(a / b), 3) if b else float("nan")

    share = float(small.size) / float(small.size + large.size)
    return {
        "cut": cut,
        "n_units": int(small.size + large.size),
        "share_below": round(share, 4),
        "below": below,
        "above": above,
        "median_ratio": ratio("median"),
        "q3_ratio": ratio("q3"),
        "share_over_big_ratio": ratio("share_over_big"),
    }


def bootstrap_q3_ratio(unit: pd.DataFrame, cut: float | None = None,
                       n_boot: int = N_BOOT, seed: int = 0,
                       alpha: float = 0.05) -> dict:
    """Percentile interval on the Q3 ratio, resampling municipalities.

    The unit of resampling is the municipality, not the municipality-origin: the
    reduction in :func:`unit_errors` already made the unit the observation, and
    resampling origins would treat three correlated errors of the same
    municipality as three independent ones.
    """
    cut = SIZE_CUT if cut is None else float(cut)
    small = unit.loc[unit["population"] < cut, "ape"].to_numpy(float)
    large = unit.loc[unit["population"] >= cut, "ape"].to_numpy(float)
    rng = np.random.default_rng(seed)
    rs = np.empty(n_boot)
    for i in range(n_boot):
        a = np.percentile(rng.choice(small, small.size, replace=True), 75)
        b = np.percentile(rng.choice(large, large.size, replace=True), 75)
        rs[i] = a / b if b > 0 else np.nan
    rs = rs[np.isfinite(rs)]
    return {
        "point": round(float(np.percentile(small, 75) / np.percentile(large, 75)), 3),
        "ci_low": round(float(np.percentile(rs, 100 * alpha / 2)), 3),
        "ci_high": round(float(np.percentile(rs, 100 * (1 - alpha / 2))), 3),
        "n_boot": int(rs.size),
        "excludes_one": bool(np.percentile(rs, 100 * alpha / 2) > 1.0
                             or np.percentile(rs, 100 * (1 - alpha / 2)) < 1.0),
    }


def decide(stats: dict, boot: dict | None = None) -> dict:
    """Apply the pre-registered rule, and say which clause decided.

    Two clauses, both required:

    1. the Q3 ratio reaches ``DECISION_Q3_RATIO`` -- the tail of the error is
       markedly heavier below the cut; and
    2. its bootstrap interval excludes 1 -- the gap is not sampling noise.

    Failing either one kills the figure and the sentences that lean on it. That
    is the point of writing the rule down: a ratio of 1.05 with an interval
    around 1 is not a weak result to be softened in a caption, it is the absence
    of the effect the figure exists to show.
    """
    r = float(stats["q3_ratio"])
    big_enough = r >= DECISION_Q3_RATIO
    ci_ok = bool(boot["excludes_one"]) if boot else True

    if big_enough and ci_ok:
        verdict, why = "keep", (
            f"the third quartile of the relative error is {r:.2f} times larger "
            f"below the {stats['cut']:,.0f}-inhabitant cut, above the "
            f"{DECISION_Q3_RATIO:.1f}x threshold"
            + (f", and the bootstrap interval [{boot['ci_low']:.2f}, "
               f"{boot['ci_high']:.2f}] excludes 1" if boot else ""))
    elif not big_enough:
        verdict, why = "drop", (
            f"the third quartile is only {r:.2f} times larger below the cut, "
            f"below the {DECISION_Q3_RATIO:.1f}x threshold. The medians are "
            f"{stats['below']['median']:.1f}% and {stats['above']['median']:.1f}% "
            f"(ratio {stats['median_ratio']:.2f}) and the tail of the error "
            "distribution separates the classes no better: there is no size "
            "effect for this figure to draw, and the claim it was drawn to "
            "support is not supported by these predictions")
    else:
        verdict, why = "drop", (
            f"the ratio is {r:.2f}x but its interval "
            f"[{boot['ci_low']:.2f}, {boot['ci_high']:.2f}] does not exclude 1, "
            "so the gap cannot be separated from sampling noise")
    return {"verdict": verdict, "reason": why,
            "q3_ratio": r, "threshold": DECISION_Q3_RATIO,
            "median_ratio": stats["median_ratio"],
            "share_over_big_ratio": stats["share_over_big_ratio"]}


# ---------------------------------------------------------------------------
# the figure the claim would need
# ---------------------------------------------------------------------------
def draw_ecdf(unit: pd.DataFrame, cut: float | None = None, ax=None,
              figsize=(6.4, 4.4)):
    """Two empirical CDFs of the relative error, on a logarithmic error axis.

    An ECDF rather than a box, because the claim is about the whole distribution
    and the difference lives in its upper part: two curves that separate only
    after the 75th percentile say exactly that, while two boxes hide it in the
    whiskers. On a log axis the curves become near-parallel lines whose vertical
    offset is a constant factor, which is the quantity the ratio reports.
    """
    import matplotlib.pyplot as plt

    cut = SIZE_CUT if cut is None else float(cut)
    small = np.sort(unit.loc[unit["population"] < cut, "ape"].to_numpy(float))
    large = np.sort(unit.loc[unit["population"] >= cut, "ape"].to_numpy(float))

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    styles = ((small, f"$<$ {cut:,.0f} (n = {small.size:,})", "#c0392b"),
              (large, f"$\\geq$ {cut:,.0f} (n = {large.size:,})", "#2471a3"))
    for x, label, colour in styles:
        y = np.arange(1, x.size + 1) / x.size
        ax.step(x, y, where="post", color=colour, lw=1.6, label=label)

    for q in (0.50, 0.75, 0.90):
        ax.axhline(q, color="0.86", lw=0.9, zorder=0)
        ax.text(ax.get_xlim()[0], q, f" {q:.0%}", va="bottom", ha="left",
                fontsize=7, color="0.45")
        for x, colour in ((small, "#c0392b"), (large, "#2471a3")):
            v = float(np.percentile(x, 100 * q))
            ax.plot([v], [q], "o", ms=4.5, mfc="white", mec=colour, mew=1.4,
                    zorder=4)

    stats = class_stats(unit, cut)
    txt = (f"median  {stats['below']['median']:.0f}%  vs  "
           f"{stats['above']['median']:.0f}%   ($\\times${stats['median_ratio']:.2f})\n"
           f"Q3      {stats['below']['q3']:.0f}%  vs  "
           f"{stats['above']['q3']:.0f}%   ($\\times${stats['q3_ratio']:.2f})\n"
           f"$>$ 100%  {100 * stats['below']['share_over_big']:.0f}%  vs  "
           f"{100 * stats['above']['share_over_big']:.0f}%")
    ax.text(0.02, 0.97, txt, transform=ax.transAxes, va="top", ha="left",
            fontsize=7.5, family="monospace", zorder=5,
            bbox=dict(fc="white", ec="0.85", lw=0.6, alpha=0.92, pad=2.5))

    ax.set_xscale("log")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("absolute percentage error of the size forecast (%)")
    ax.set_ylabel("cumulative share of municipalities")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------
_OOF_COLUMNS = ("origin", "geocode", "y_true", "y_pred", "population")

#: Modules of this checkout that the refit imports by bare name.
#:
#: ``pyproject.toml`` lists the dependencies but declares no build backend, so
#: ``uv sync`` installs the third-party set and never the project: ``run_ablation``
#: is a *file of the checkout*, not a package in site-packages. It resolves because
#: ``generate_figures`` puts the repository root on ``sys.path`` when it is
#: imported -- an accident of the import order. The refit below no longer depends
#: on that accident, but the distinction is what the previous error message lost:
#: it reported an absent source file as a missing dependency, and answered it with
#: ``uv sync``, which installs packages and cannot supply a source file.
_LOCAL_MODULES = ("run_ablation", "predictive_model", "data_utils", "fitpl")


def repo_root() -> Path:
    """The checkout holding this file and the modules the refit imports."""
    return Path(__file__).resolve().parent


def _load_from_file(name: str, path: Path):
    """Import ``path`` under ``name`` and register it in ``sys.modules``.

    Registration is half the job. ``predictive_model`` does
    ``from run_ablation import fit_quantile_model`` lazily, so a second copy
    compiled from a different path would hand that import a different module
    object -- or fail outright.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"{path} cannot be imported as {name!r}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)  # never leave a half-executed module behind
        raise
    return module


def ensure_local_module(name: str, root: Path | None = None):
    """Make one of the checkout's own modules importable, or say what is missing.

    ``sys.path`` is tried first because that is what normally happens; loading
    from the file is the fallback for interpreters started without the script's
    directory (``python -P``), for runners that rewrite ``sys.path``, and for
    import hooks that answer ``find_spec`` with ``None``.
    """
    import importlib.util

    root = Path(root or repo_root())
    try:
        if importlib.util.find_spec(name) is not None:
            return sys.modules.get(name)
    except (ImportError, ValueError):
        pass  # a broken parent package is not a reason to stop looking

    path = root / f"{name}.py"
    if not path.exists():
        raise RuntimeError(
            f"{name!r} does not resolve and there is no {path} either, so this "
            "checkout cannot refit the estimator.\n"
            f"    repository root : {root}\n"
            f"    python          : {sys.executable}\n"
            f"    sys.path[:3]    : {sys.path[:3]}\n"
            f"    {name}.py is a file of the repository, not a dependency: "
            "`uv sync` installs packages and cannot supply a missing source "
            f"file. Confirm the checkout is complete (`git ls-files {name}.py`) "
            "and that you are on the branch you think you are on. Or skip the "
            "refit: `--oof FILE` needs nothing but pandas.")
    return _load_from_file(name, path)


def _module_error(exc: ModuleNotFoundError, root: Path) -> str:
    """The right advice for what is actually missing."""
    name = exc.name or str(exc)
    if name in _LOCAL_MODULES:
        return (
            f"{name!r} is a file of this checkout ({root / (name + '.py')}) and "
            "Python could not import it, so the estimator cannot be refitted. It "
            "is a source file, not a package: `uv sync` cannot supply it. Check "
            f"`git ls-files {name}.py` in {root}, or pass the predictions you "
            "already have with `--oof FILE`.")
    return (f"the data layer is missing the dependency {name!r}, so the "
            "estimator cannot be refitted in this environment. Run `uv sync` in "
            "the repository root and retry, or pass the predictions you already "
            "have with `--oof FILE` -- that path needs nothing but pandas.")


def load_oof(path: str | Path | None = None, refit: bool = False,
             origins=(2023, 2024, 2025), tau: float = 0.5) -> pd.DataFrame:
    """Row-level out-of-fold predictions, from a file or from a refit.

    ``path`` reads a parquet/CSV with at least an observed and a predicted
    column; ``refit`` recomputes the panel with the estimator the manuscript
    reports (spline-basis quantile regression at ``tau``), which needs ``data/``
    and the municipal panel -- i.e. a machine where the pipeline has been run.
    """
    if path is not None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(p)
        return pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)

    if not refit:
        raise ValueError(
            "pass `path=` to read saved predictions, or `refit=True` to recompute "
            "them. Refitting needs data/<geocode>.parquet and the municipal panel; "
            "from the command line use `--oof FILE` or `--refit`.")

    root = repo_root()
    for cand in (str(root), str(root / "Manuscript" / "figures")):
        if cand not in sys.path:
            sys.path.insert(0, cand)
    try:
        from Manuscript.figures.generate_figures import _size_oof
    except ImportError:
        # ``Manuscript`` has no __init__.py; the namespace-package import can
        # fail depending on how the interpreter was started. The module itself is
        # importable from its directory.
        try:
            from generate_figures import _size_oof
        except Exception as exc:
            raise RuntimeError(
                f"could not import the figure module that refits the estimator "
                f"({type(exc).__name__}: {exc}). `--refit` needs the project "
                "environment (`uv sync`, then `python check_size_error.py "
                "--refit`) plus data/<geocode>.parquet, because it retrains the "
                "reported estimator. If you already have the predictions saved, "
                "use `--oof FILE` instead -- it needs nothing but pandas."
            ) from exc

    # Before the fit, not inside it. ``_size_oof`` imports the estimator by bare
    # name, so a source file absent from the checkout used to surface as a bare
    # ModuleNotFoundError from deep inside the figure module, was reported as a
    # missing dependency of the data layer, and was answered with `uv sync` --
    # which cannot install a file of the repository.
    for name in _LOCAL_MODULES:
        ensure_local_module(name, root)

    try:
        return _size_oof(origins=origins, tau=tau)
    except ModuleNotFoundError as exc:
        raise RuntimeError(_module_error(exc, root)) from exc


def doctor(root: Path | None = None) -> dict:
    """Where every piece the refit needs resolves to, present or not.

    Written because the failure that prompted it could not be diagnosed from its
    own message: it named a dependency that was really a source file, and the
    remedy it printed could not have worked. One run here settles which of the
    two it is.
    """
    import importlib.util
    import platform

    root = Path(root or repo_root())

    def resolves(name: str) -> bool:
        try:
            return importlib.util.find_spec(name) is not None
        except (ImportError, ValueError):
            return False

    local = {}
    for name in _LOCAL_MODULES:
        path = root / f"{name}.py"
        local[name] = {"file": str(path), "exists": path.is_file(),
                       "resolves": resolves(name)}

    third = {n: resolves(n) for n in
             ("numpy", "pandas", "pyarrow", "sklearn", "matplotlib",
              "geopandas", "duckdb", "dotenv")}

    data_dir = root / "data"
    panel = root / "Manuscript" / "figures" / "_panel_city_BR.parquet"
    n_parquet = (len(list(data_dir.glob("*.parquet")))
                 if data_dir.is_dir() else 0)
    return {
        "python": platform.python_version(),
        "executable": sys.executable,
        "cwd": str(Path.cwd()),
        "repo_root": str(root),
        "sys_path_head": list(sys.path[:4]),
        "local_modules": local,
        "third_party": third,
        "data": {"dir": str(data_dir), "exists": data_dir.is_dir(),
                 "parquet_files": n_parquet},
        "panel_cache": {"path": str(panel), "exists": panel.is_file()},
        # A refit needs the source files, every third-party import, and the data
        # directory. It does *not* need the panel cache, which is rebuilt.
        "ready_to_refit": (all(v["exists"] for v in local.values())
                           and all(third.values())
                           and data_dir.is_dir()),
    }


def report(unit: pd.DataFrame, cut: float | None = None,
           boot: dict | None = None) -> dict:
    """The table, the rule and the verdict, as one JSON-able dictionary."""
    stats = class_stats(unit, cut)
    if boot is None:
        boot = bootstrap_q3_ratio(unit, cut)
    verdict = decide(stats, boot)
    return {"stats": stats, "bootstrap": boot, "decision": verdict,
            "n_exact_forecasts": int((unit["ape"] <= 0).sum())}
