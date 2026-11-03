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

Expected output: Table 1 of the manuscript (7 aggregation levels), the
national year-by-year fits, the per-state exponents, and the α-vs-population
correlations. If the database is regenerated with different settings, this
script will report different numbers — that is the point.

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
  support inflates the power-law share substantially, especially at L2.
- **`year = 0` in the `episcanner` tables is a pooled all-years fit**, not a
  calendar year (documented in `fitpl.py`: "Use 0 for combined/all-years
  fits"). All year counts therefore exclude it; the national levels have 15
  annual fits, not 16.
- **`alpha` is not clipped at 3.** Only 0.02% of the 45,860 weekly fits return
  α ≥ 2.9999. The apparent wall at 3.0 in the summary statistics is a handful
  of boundary cases, not a constraint imposed by the estimator.
- **`xmin` is the diagnostic, not a nuisance parameter.** In L1/L2 the KS
  selection drives `xmin` down to 1–5 cases in most fits, which means the
  fitted power law is describing the whole distribution. Those fits are
  reported but not interpreted as tails.

## Table correspondence

| Level | Table | Filter |
|---|---|---|
| L1 | `powerlaw_fits` | none |
| L2 | `powerlaw_fits_yearly` | none |
| L3 | `powerlaw_fits_total_cases` | none |
| L4 | `episcanner_state_fits` | `metric='total_cases' AND year > 0` |
| L5 | `episcanner_state_fits` | `metric='ep_dur' AND year > 0` |
| L6 | `episcanner_fits` | `metric='total_cases' AND year > 0` |
| L7 | `episcanner_fits` | `metric='ep_dur' AND year > 0` |
