# Figure specifications

The manuscript (`main.tex`) contains 14 `figure` environments with final
captions and labels. Both renderings of each figure are committed in this
directory, as `.pdf` and as `.png`; regenerate them with
`generate_figures.py` after a data update.

All environments use `\includegraphics` **without an extension**, so either
`.pdf` (preferred for vector plots) or `.png` at >= 300 dpi will resolve.

| File | Section | Content |
|---|---|---|
| `fig02_alpha_by_level` | Results 3.1 | Box plot of $\alpha$ per construction, single panel; dashed line at $\alpha=2$. |
| `fig03_verdict_by_level` | Results 3.1 | Grouped bars per construction: share with $R>0$, share with $R>0$ and $p<0.05$, share with $R<0$ and $p<0.05$. |
| `fig04_size_ccdf` | Results 3.2 | Single-panel empirical CCDF of epidemic size on log--log axes, national cross-section pooled over 16 years, with the fitted power law (solid) and the exponential alternative (dashed). |
| `fig05_xmin_diagnostic` | Results 3.1 | Scatter of $\alpha$ vs $x_{\min}$ (log x), one point per fit, coloured by construction; shaded band at $x_{\min}\le5$. |
| `fig06_example_fits` | Results 3.1 | Three individual fits side by side: (a) small city weekly, (b) state capital weekly, (c) state-year size cross-section. PL solid, exponential dashed. |
| `fig07_window_flip` | Results 3.1 | Left: density of $\alpha$ for `W-mun` vs `W-seas`. Right: verdict shares for `W-mun` vs `W-seas`. |
| `fig08_alpha_stability` | Results 3.3 | (a) $\alpha$ by year, one line per state; (b) $\alpha$ vs $\log_{10}$ population with fitted slope and $r$. |
| `fig09_exceedance` | Results 3.4 | Implied $\Pr(S\ge s)=\zeta(\alpha,s)/\zeta(\alpha,x_{\min})$ vs empirical exceedance frequency, per national year. |
| `fig10_pred_vs_obs` | Results 3.7 | Predicted vs observed annual cases for 2025, log--log, one point per municipality; identity line; 10:1 and 1:10 bands; dotted line at the largest training response. Upper panel pools the country; lower panels are the four states with the smallest median absolute percentage error, among those with at least twenty municipalities. Estimator: the spline-basis quantile regression at $\tau=0.5$, the one the ablation uses. Writes `fig10_states.json` (selection rule, full ranking, PNG pixel shape) which `verify_snapshot` checks against the committed image. |
| `fig11_ablation_skill` | Results 3.7 | (a) Skill score vs M0 for epidemic size, M1/M2/M3; (b) pinball loss vs $\tau$ for M2 vs M3, with the range over origins. |
| `fig12_roc_pr_calibration` | Results 3.6 | 2x2: ROC, PR, calibration, F1 vs threshold; out-of-fold predictions averaged over the five origins (not pooled); M0 dotted-dash reference, M2 solid, M3 dashed, population-only dotted. |
| `fig13_importance` | Results 3.6 | Held-out permutation importance ($\Delta$ Brier), power-law block highlighted; joint block row plus per-origin stability of the three scaling features. |
| `fig14_error_vs_pop` | Results 3.8 | (a) absolute percentage error of the median ($\tau=0.5$) spline-quantile size forecast vs municipal population, with the median within population bins and the 100,000-inhabitant threshold marked; (b) standardised persistence residuals for epidemic size vs N(0,1). Writes `fig14_pop_bands.json` with the median error either side of the threshold. |
| `fig15_pred_map` | Results 3.8 | Three panels for the record season, at the M3 rung: (a) choropleth of the predicted probability of exceeding the unit's own 75th percentile, quintile classes; (b) the realised outcome on the same municipalities; (c) observed share against mean prediction by predicted quintile. Grey = no forecast (fewer than five seasons of history). Uses `muni_br.gpkg` and writes `fig15_map.json` recording the origin, the rung and the pixel shape. |

## Where the numbers come from

- Construction descriptive statistics (`W-mun`, `W-seas`, `S-mun`, `S-state`,
  `S-nat`): `powerlaw_results.db`
  (`powerlaw_fits`, `powerlaw_fits_yearly`, `powerlaw_fits_total_cases`,
  `episcanner_state_fits`, `episcanner_fits`). Queries in
  `Manuscript/verification/verify_tables.sql` reproduce Table 2 exactly.
- Predictive panels: `fig10` and `fig14` refit the reported model
  (`run_ablation.fit_quantile_model`, spline quantile at the median) on the
  pooled city panel, because the ablation artefact holds aggregate losses and
  not row-level predictions. `fig11`--`fig13` and `fig15` are *read* from
  `ablation_results.csv` and `classifier_results*`, so they cannot disagree
  with the tables; `fig15` in particular draws the out-of-fold predictions
  rather than refitting a classifier of its own.
- Population and geography: `parquet_cache/city_stats.parquet`,
  `parquet_cache/geography.parquet`, `muni_br.gpkg`.

## Two existing assets that can seed panels

- `fig06_example_fits` panel (a)/(b): `Manuscript/figures/Ituiutaba_PL.png`
  and `Manuscript/figures/SP_weekly.png`.
- `fig02_alpha_by_level`: `report/report_assets/pl_alpha_dist.png` gives a
  first version, but it pools all constructions and must be split by construction.

## Note on the ablation figures

`fig11`--`fig13` are views of committed artefacts and are never recomputed
here. `fig10`, `fig14` and `fig15` do refit, because the artefacts carry
aggregate losses only; they fit the same estimators the ablation reports, in
one place (`_m3_features`), which aborts rather than draw a model without the
climate block.
