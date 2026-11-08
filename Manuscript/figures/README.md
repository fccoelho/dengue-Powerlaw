# Figure specifications

The manuscript (`main.tex`) contains 15 `figure` environments with final
captions and labels. The image files are **not** in the repository: generate
them locally and drop them in this directory using the filenames below.

All environments use `\includegraphics` **without an extension**, so either
`.pdf` (preferred for vector plots) or `.png` at >= 300 dpi will resolve.

| File | Section | Content |
|---|---|---|
| `fig01_aggregation_levels` | Methods 2.2 | Schematic of the five aggregation levels (`W-mun`, `W-seas`, `S-mun`, `S-state`, `S-nat`) as nested constructions: same weekly series, different window and different unit of observation. Shaded by the $x_{\min}$ diagnostic. |
| `fig02_alpha_by_level` | Results 3.1 | Box plot of $\alpha$ per construction, single panel; dashed line at $\alpha=2$. |
| `fig03_verdict_by_level` | Results 3.1 | Grouped bars per construction: share with $R>0$, share with $R>0$ and $p<0.05$, share with $R<0$ and $p<0.05$. |
| `fig04_size_ccdf` | Results 3.2 | Single-panel empirical CCDF of epidemic size on log--log axes, national cross-section pooled over 16 years, with the fitted power law (solid) and the exponential alternative (dashed). |
| `fig05_xmin_diagnostic` | Results 3.1 | Scatter of $\alpha$ vs $x_{\min}$ (log x), one point per fit, coloured by construction; shaded band at $x_{\min}\le5$. |
| `fig06_example_fits` | Results 3.1 | Three individual fits side by side: (a) small city weekly, (b) state capital weekly, (c) state-year size cross-section. PL solid, exponential dashed. |
| `fig07_window_flip` | Results 3.1 | Left: density of $\alpha$ for `W-mun` vs `W-seas`. Right: verdict shares for `W-mun` vs `W-seas`. |
| `fig08_alpha_stability` | Results 3.1 | (a) $\alpha$ by year, one line per state; (b) $\alpha$ vs $\log_{10}$ population with fitted slope and $r$. |
| `fig09_exceedance` | Results 3.3 | Implied $\Pr(S\ge s)=\zeta(\alpha,s)/\zeta(\alpha,x_{\min})$ vs empirical exceedance frequency, per national year. |
| `fig10_pred_vs_obs` | Results 3.6 | Predicted vs observed annual cases, log--log, one point per unit-origin; identity line; 10:1 and 1:10 bands; dotted line at the largest training response. Two panels: random forest and the spline-basis quantile regression the ablation uses. |
| `fig11_ablation_skill` | Results 3.6 | (a) Skill score vs M0 for epidemic size, M1/M2/M3; (b) pinball loss vs $\tau$ for M2 vs M3, with the range over origins. |
| `fig12_roc_pr_calibration` | Results 3.5 | 2x2: ROC, PR, calibration, F1 vs threshold; out-of-fold predictions averaged over the five origins (not pooled); M0 dotted-dash reference, M2 solid, M3 dashed, population-only dotted. |
| `fig13_importance` | Results 3.5 | Held-out permutation importance ($\Delta$ Brier), power-law block highlighted; joint block row plus per-origin stability of the three scaling features. |
| `fig14_error_vs_pop` | Results 3.7 | (a) absolute percentage error vs population; (b) standardised persistence residuals for epidemic size vs N(0,1). |
| `fig15_pred_map` | Results 3.7 | Choropleth of predicted probability of exceeding the 75th percentile, one forecast origin, quintile classes, using `muni_br.gpkg`. The classifier is the one the ablation reports. |

## Where the numbers come from

- Construction descriptive statistics (`W-mun`, `W-seas`, `S-mun`, `S-state`,
  `S-nat`): `powerlaw_results.db`
  (`powerlaw_fits`, `powerlaw_fits_yearly`, `powerlaw_fits_total_cases`,
  `episcanner_state_fits`, `episcanner_fits`). Queries in
  `Manuscript/verification/verify_tables.sql` reproduce Table 2 exactly.
- Predictive panels (fig10--fig15): outputs of `predictive_model.py` and
  `binary_classifier.py`, as exposed in `dashboard.py`
  (tabs *Predictive* and *Predicting Large Epidemics*).
- Population and geography: `parquet_cache/city_stats.parquet`,
  `parquet_cache/geography.parquet`, `muni_br.gpkg`.

## Two existing assets that can seed panels

- `fig06_example_fits` panel (a)/(b): `Manuscript/figures/Ituiutaba_PL.png`
  and `Manuscript/figures/SP_weekly.png`.
- `fig02_alpha_by_level`: `report/report_assets/pl_alpha_dist.png` gives a
  first version, but it pools all constructions and must be split by construction.

## Note on the ablation figures

`fig10`--`fig15` and `tab:ablation_results` are the only parts of the
manuscript whose numbers are not yet in the database. They require the
rolling-origin ablation runs described in Methods 2.11 (rungs M0--M3,
origins 2021--2025). Everything else in the paper is already reproducible from
`powerlaw_results.db`.
