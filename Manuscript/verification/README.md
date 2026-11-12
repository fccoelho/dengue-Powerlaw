# Verification of reported quantities

Every number in `Manuscript/main.tex` and `Seminar.md` is derived from
`powerlaw_results.db`, which is committed at the repository root. Nothing is
transcribed by hand.

## Run it

```bash
python Manuscript/verification/verify_results.py
```

Optionally pass a path to another copy of the database:

```bash
python Manuscript/verification/verify_results.py path/to/powerlaw_results.db
```

Expected output: Table 1 of the manuscript (five constructions), the
national year-by-year fits, the per-state exponents, and the α-vs-population
correlations. If the database is regenerated with different settings, this
script will report different numbers — that is the point.

## The predictive layer

```bash
python Manuscript/verification/verify_predictive.py           # audit
python Manuscript/verification/verify_predictive.py --strict  # exit 1 on any drift
```

`verify_results.py` covers the descriptive half of the paper. This script
covers the half that used to be unauditable: every predictive number in
`main.tex` is recomputed from the four committed artefacts
(`classifier_results_metrics.csv`, `ablation_city.csv`,
`ablation_state_exceedance.csv`, `ablation_results.csv`) and nothing is
refitted, so it runs without the data directory.

It is a **two-sided** check. A gate that only compares against artefacts
drifts away from the manuscript: it keeps re-deriving a value that a rewrite
removed, reports drift for ever, and stops being read. So each claim has to
pass twice — the quoted value must still occur in `main.tex` (`STALE` if not)
and the artefact must still produce it (`DRIFT` if not). The two verdicts mean
opposite things.

Two counting rules are worth knowing before reading its output:

- **`ci_excludes_zero` is direction-agnostic.** An interval can exclude zero
  while pointing *against* the rung. Every "satisfies the rule" count is taken
  jointly with `improves`; the script prints the counts in both directions, so
  a pass count can never be read off the raw column.
- **Tolerance follows the printed precision.** Two-decimal values are compared
  at 0.0051, three-decimal at 0.0006, percentages at 0.051. One tolerance for
  everything would flag a correctly rounded 0.5825 quoted as 0.58.

Numbers quoted in `main.tex` with no artefact behind them are printed under
**NOT AUDITABLE** rather than silently skipped. Currently one: the
population-only AUC under a pooled threshold (0.87, range 0.84--0.93), which
needs `run_classifier.py` rerun with `threshold_scope='pooled'` — the
committed classifier artefact carries `per_unit` only.

`verify_tables.sql` contains the same computations as plain SQL, for readers
who prefer to query the database directly with `sqlite3` or DuckDB.

## Conventions that matter

These are easy to get wrong and they change the reported percentages:

- **`R` is a log-likelihood ratio, not a KS distance.** It is the output of
  `powerlaw.Fit.distribution_compare("power_law", "exponential",
  normalized_ratio=True)` (see `fitpl.py`). Sign convention: `R > 0` favours
  the power law, `R < 0` favours the exponential. The KS distance is a
  different quantity and lives in [0, 1]; if you see `R` outside that range,
  you are looking at the comparison statistic.
- **A positive `R` is not evidence.** "PL supported" throughout the manuscript
  means `R > 0 AND p < 0.05`. Counting positive-but-non-significant `R` as
  support inflates the power-law share substantially, especially at `W-seas`.
- **`year = 0` in the `episcanner` tables is a pooled all-years fit**, not a
  calendar year (documented in `fitpl.py`: "Use 0 for combined/all-years
  fits"). All year counts therefore exclude it; the national construction has 16
  annual fits, not 17.
- **`alpha` is not clipped at 3.** Only 12 of the 48,937 weekly fits return
  α ≥ 2.9999 (0.02%). The apparent wall at 3.0 in the summary statistics is a
  handful of boundary cases, not a constraint imposed by the estimator.
- **`xmin` is the diagnostic, not a nuisance parameter.** In `W-mun` and
  `W-seas` the KS
  selection drives `xmin` down to 1–5 cases in most fits, which means the
  fitted power law is describing the whole distribution. Those fits are
  reported but not interpreted as tails.

## Table correspondence

| Construction | Table | Filter |
|---|---|---|
| `W-mun` | `powerlaw_fits` | none |
| `W-seas` | `powerlaw_fits_yearly` | none |
| `S-mun` | `powerlaw_fits_total_cases` | none |
| `S-state` | `episcanner_state_fits` | `metric='total_cases' AND year > 0` |
| `S-nat` | `episcanner_fits` | `metric='total_cases' AND year > 0` |
| -- | *(duration and peak-timing constructions removed)* | -- |
