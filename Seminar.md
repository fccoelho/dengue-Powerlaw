# Seminar.md

# Seminar Title and Abstract

## Title

**Aggregation Determines Scaling: Why Power-Law Claims About Dengue Epidemics
Depend on How You Cut the Data**

---

## Abstract

Dengue is one of the most frequently cited examples of power-law scaling in
infectious disease epidemiology, and the claim carries weight: a heavy tail
implies that catastrophic outbreaks are structurally more likely than any
tractable count model admits. We re-examine that claim on a large and
deliberately uniform dataset — weekly dengue surveillance from January 2010 to
January 2026 for the 5,571 Brazilian municipalities, obtained from the
InfoDengue/Mosqlimate platform — by fitting the *same* discrete maximum-likelihood estimator under
seven nested aggregation levels. Because the estimator and the underlying data
are held fixed, any difference in the inferred regime is attributable to the
construction of the distribution rather than to differing sources, periods or
software.

The verdict splits cleanly. For the cross-sectional distribution of **epidemic
size** across municipalities within a year, the power law is preferred in
100% of the 15 national fits — all statistically significant, with
α ≈ 2.0 and a lower bound x_min of 762 cases — and in 96% of 330 state-year
fits, with state-level exponents tightly clustered around a median of 1.98
(state medians spanning 1.70–2.58). For the distribution of **epidemic
duration** over the same municipal epidemics, the power law is rejected in
all 15 national fits (and in 95% of the 330 state-year fits) in favour of an
exponential, with α piling up at 2.90–2.96 — the signature of a truncated
body forced through a power-law estimator.

For **weekly case counts** the picture is more subtle and, we argue, where much
of the published confusion originates. The power law appears preferred in 75%
of city-level full-series fits, but the selected lower bound collapses to
x_min ≤ 5 cases in 77% of them, meaning the model describes the bulk of the
distribution rather than a tail. The same counts, pooled over a single season
instead of a decade, drop the power-law preference from 75% to 23% and raise
significant support for an exponential from 4% to 21%. Nothing about dengue
changed; only the window did.

We conclude that (i) the heavy-tail literature on dengue conflates at least
three distinct random variables (weekly counts, epidemic size, epidemic
duration) with very different tail behaviour; (ii) scale claims are only
interpretable where x_min lies well above the bulk of the data, a diagnostic
that should be reported routinely and generally is not; and (iii) reporting an
exponent α without the (unit, x_min, comparison model) tuple is uninformative.
We also report a negative result: the municipal exponent is essentially
unrelated to city size (r = −0.065 against log population, n = 5,244), so the
data do not support the view that larger cities occupy a different scaling
regime.

The seminar closes with what follows for forecasting: if the tail is real for
epidemic size but absent for duration, then extreme-event prediction should
target size conditional on a season having started, rather than attempting to
predict which seasons will be extreme from a scale-free premise.

---

**Keywords:** Power law distribution, Dengue epidemics, Heavy tails,
Aggregation, Model comparison, Statistical physics, Epidemiological
surveillance, Brazil

**Target Audience:** Epidemiologists, public health researchers,
biostatisticians, and computational scientists interested in outbreak dynamics
and predictive modelling

**Duration:** 45–60 minutes (including Q&A)

---

## Changelog — corrections relative to the previous draft

The previous version of this abstract made four claims that the committed
results database (`powerlaw_results.db`) does not support. They are recorded
here rather than silently deleted, so the record of what changed is auditable.

| Previous claim | Verified position |
|---|---|
| "significant spatial heterogeneity … some states thin-tailed (high α), others heavy-tailed (low α)" | All 26 states are heavy-tailed. State-median α spans 1.70–2.58 (SD 0.20, CV ≈ 10%), i.e. a narrow band around α ≈ 2, not two distinct regimes. |
| "(1) longitudinal tracking of α to detect temporal shifts in epidemic regimes" | No temporal shift found. National annual α for epidemic size ranges 1.78–2.27 over 15 years with no trend. |
| "(2) spatial clustering analysis using Local Moran's I" | Not implemented. No such results exist in the database. Removed. |
| "(3) correlation analysis linking scaling parameters to R₀, duration, timing" | Only the population correlation was computable, and it is null (r = −0.065). R₀ and timing are not in the database. |
| "(4) ML model … improved out-of-sample forecasting for extreme events" | Not demonstrated. `predictive_model.py` and `binary_classifier.py` exist in the repository but no out-of-sample results are stored. Claimed nowhere in this abstract. |

All quantitative statements in the abstract above are reproduced by
`Manuscript/verification/verify_results.py`.
