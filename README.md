# Power Law Dynamics in Dengue Epidemics

This project implements a comprehensive computational framework for the statistical analysis of power law scaling in dengue reported cases across Brazilian municipalities. The analysis integrates methods from statistical physics, epidemic modeling, machine learning, and longitudinal data analysis to characterize the heavy-tailed distribution of outbreak magnitudes and their relationship with underlying epidemiological drivers.

## Reproducing the manuscript

Every number in the manuscript is derived from `powerlaw_results.db`. The
dependency chain is:

```
Mosqlimate API --> data/*.parquet --> powerlaw_results.db
                                          |
                 +------------------------+-----------------------+
                 |                        |                       |
                 v                        v                       v
   Manuscript/generated/        ablation_results.csv     classifier_results.*
   tab_main.tex, facts.json     ablation_city.csv        _metrics / _importance
                 |                        |                       |
                 +-------------> Manuscript/figures/*.png|pdf <--+
                                          |
                                          v
                                   Manuscript/main.pdf
```

Nothing downstream of the database may be computed twice. Figures 11--13 are
*views* of the ablation and classifier artefacts, not second implementations
of them. A figure that re-derives the same quantity can disagree with the
table it illustrates, and once did.

### Which script produces which artefact

| artefact | produced by | consumed by |
| --- | --- | --- |
| `powerlaw_results.db` | `update_data.py` | everything below |
| `city_stats` table | `aggregate_city_stats.py` | `generate_figures.py`, dashboard |
| `Manuscript/generated/tab_main.tex`, `facts.json` | `make_tables.py` | `main.tex` (Table 1) |
| `Manuscript/generated/tab_ablation.tex`, `tab_exceed.tex`, `ablation_facts.json` | `make_ablation_tables.py` | `main.tex` (ablation + classifier tables) |
| `ablation_results.csv` + `.meta.json` | `run_ablation.py --level state` | `tab_ablation.tex`, Figure 11 |
| `ablation_state_exceedance.csv` + `.meta.json` | `run_ablation.py --level state --tasks exceedance` | `tab_exceed.tex` (state block) |
| `ablation_city.csv` + `.meta.json` | `run_ablation.py --level city --tasks exceedance --output ablation_city.csv` | `tab_exceed.tex` (municipal block) |
| `classifier_results.{parquet,metrics.csv,importance.csv,meta.json}` | `run_classifier.py` | Figures 12, 13, `tab_exceed.tex` fallback |
| `Manuscript/figures/fig*.png` / `.pdf` | `Manuscript/figures/generate_figures.py` | `main.tex` |
| `snapshot_manifest.json` | `provenance.py --write` | `verify_snapshot.py` |

`--output` defaults to a name derived from `--level` and `--tasks`, and
`run_ablation.py` refuses to overwrite a file holding a different experiment.
This is not tidiness: the default used to be `ablation_results.csv` for every
run, so `--level city --tasks exceedance` replaced the state size ladder in
place. The columns still matched, every check stayed green, and the only
symptom was `make_ablation_tables.py` reporting that the repository contained
no size ladder -- which it did, right up until the wrong run overwrote it.

### Full refresh after a new data extraction

Run in this order. Steps 1 and 2 are network-bound; the rest are local.

```bash
# 0. environment (see Getting Started)
uv sync && source .venv/bin/activate

# 1. re-extract and re-fit everything (phases 0-4, resumable, hours)
python update_data.py

# 2. rebuild the population/case join table used by figures and dashboard
python aggregate_city_stats.py

# 3. ablation ladder - state level, size (Table 3, Figure 11)
python run_ablation.py                     # --dry-run first to see the plan

# 4. ablation ladder - state level, exceedance classifier (Table 4, state block)
python run_ablation.py --level state --tasks exceedance \
    --output ablation_state_exceedance.csv

# 5. ablation ladder - municipality level, exceedance (Table 4, municipal block)
python run_ablation.py --level city --tasks exceedance --output ablation_city.csv

# 6. out-of-fold rarity classifier (Figures 12 and 13)
python run_classifier.py --level city

# 7. regenerate Table 1 and the derived facts from the database
python make_tables.py

# 8. regenerate the ablation and classifier tables from the artefacts
python make_ablation_tables.py

# 9. regenerate every figure (or a subset: ... 2 4 6)
python Manuscript/figures/generate_figures.py

# 10. freeze the new state as THE snapshot
python provenance.py --write

# 11. prove the repository is internally consistent
python verify_snapshot.py
```

Step 11 must exit `0` before the PDF is rebuilt:

```bash
cd Manuscript && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

A partial refresh is fine as long as it is followed by steps 7--11: the
artefacts are hashed, so a stale table is a hard error rather than a subtle
one.

`verify_snapshot.py` failing is normal right after a partial run and is the
point of steps 8 and 11: it names the artefact that is behind. If it reports
a number as *unverifiable*, the artefact that should back it does not exist --
run the command in the message rather than editing the prose.

### Snapshot invariants

These are not style preferences. Each one corresponds to a defect that
already reached the manuscript once.

1. **Never hand-edit generated files.** `Manuscript/generated/*`,
   `ablation_results.csv`, `ablation_city.csv` and `classifier_results.*` are
   outputs. Edit the script and re-run it. Table 1 used to be typed by hand
   while Methods claimed no statistic was transcribed by hand.
2. **`year = 0` rows are pooled all-years fits, not a calendar year.** Every
   per-year level in `make_tables.py` excludes them with `WHERE year <> 0`.
   Including them inflates each level by one row per unit and makes the
   reported counts irreproducible.
3. **A rung that equals its parent is a bug, not a null result.** Each rung of
   the ablation ladder must add at least one column. `run_ablation.py` aborts
   when `M2` would equal `M1` (the climate block came up empty), and writes
   `NaN` with an explanatory `note` rather than `0.0` when a contrast cannot
   be evaluated. A ladder that silently degrades to `M2 = M1` produced 95
   confident zeros before anyone noticed.
4. **`--threshold-scope` is not symmetric.** At `--level city` the default is
   `per_unit`: each municipality's Q75 comes from its own history. A pooled
   Q75 across 5,000 municipalities of very different sizes is answerable from
   population size alone (AUC 0.94 population-only versus 0.96 full model),
   and half the municipalities never cross it in any year. `run_classifier.py`
   and `generate_figures.py` both refuse an artefact built that way.
5. **The classifier is judged out of fold.** Prevalence ranges from 0.21 to
   0.70 across origins, so pooling the five origins into a single ROC hands a
   discrimination-free model an AUC near 0.60. Figures 12 and 13 plot
   per-origin curves and the permutation importance is computed on the
   held-out fold.
6. **The forecasting estimator must be the one the manuscript describes.**
   The default is `--model spline_quantile` (linear in the covariates beyond
   the boundary knot, so it can forecast outside its training range) and
   `--classifier-model logistic`. An artefact produced before the estimator
   change is byte-valid but experimentally different, so
   `verify_snapshot.py` refuses meta files without `extrapolating_regressor`
   and `classifier_model`.
7. **Epidemic size is the only target and the only fitted variable.**
   `TARGETS` in `run_ablation.py` has one entry, `FIT_METRICS` in
   `fit_episcanner_pl.py` has one entry, and duration/peak timing appear only
   as `COVARIATE_DUR`/`COVARIATE_PEAK` on the right-hand side. Reintroducing
   them as outcomes means re-introducing a scaling claim about a bounded
   variable.
8. **No ablation number is typed, and no number is drawn from a superseded
   run.** `make_ablation_tables.py` renders both predictive tables from
   `ablation_results.csv`, `ablation_city.csv` and
   `classifier_results_metrics.csv`, and `--check` compares every ablation
   figure in the prose against them. An artefact whose `.meta.json` lacks
   `extrapolating_regressor` / `classifier_model` predates the current design
   and is refused even though it is byte-valid: `ablation_city.csv` carried a
   tree classifier into the published table for exactly that reason. A figure
   and its own caption once reported opposite signs for the same contrast on
   the same page.

9. **`parquet_cache/` must not come back.** It held Parquet exports of tables
   that no script read; its row counts silently disagreed with the database.
   Dead copies of the source of truth are how this class of bug survives.
   `verify_snapshot.py` reports them.

### Before every commit that touches data or numbers

```bash
pytest -q                          # test modules, including the ablation guards
python make_tables.py --check      # fit literals still match the database
python make_ablation_tables.py --check  # ablation literals still match the artefacts
python make_ablation_tables.py --sources # every source artefact present and of the right kind
python provenance.py --check       # no artefact came from a different run
python verify_snapshot.py          # all five checks; exit 1 on any drift
```

The same checks belong in CI. They are cheap (seconds) and they are the
difference between a manuscript that describes its database and one that
merely agrees with it today.

### Troubleshooting

| symptom | cause | fix |
| --- | --- | --- |
| `DRIFT: ... changed since the snapshot was frozen` | `powerlaw_results.db` was re-fitted but the tables were not regenerated | run steps 6--8 above, or `python make_tables.py` then `python provenance.py --write` if the new database is the intended one |
| `[2/5] prose STALE` lists a value with the wrong occurrence count | a number appears twice in `main.tex` and only one site was updated | grep for the stale literal; both sites must change together |
| `[5/5] ablation INCOMPLETE` names a source as absent | the artefact that should back a quoted number does not exist (e.g. the state-level exceedance run was never committed) | run the command in the message; do not edit the prose to match the gap |
| `[5/5] ablation STALE` names a contrast | the artefact was regenerated and the prose still carries the old median, or the reverse | re-run `make_ablation_tables.py` first: the tables fix themselves. Then update the prose the message names |
| `[3/5] design STALE` names `ablation_city.meta.json` | the municipal exceedance run predates the estimator change | `python run_ablation.py --level city --tasks exceedance --output ablation_city.csv` |
| `0/8 block columns available` | the climate block is missing from the panel, so `M2` would equal `M1` | `python run_ablation.py --attach-climate` (rebuilds from `data/[0-9]*.parquet`) or `--climate-parquet <file>` |
| every `M2-M1` contrast is exactly `0.0` with CI `[0, 0]` | a pre-fix artefact; the rungs were identical | regenerate the ablation; the current script aborts instead of writing zeros |
| `RuntimeError: ... refuses to plot` from `generate_figures.py` | a figure was pointed at a defective artefact | fix the artefact, do not relax the check |
| `cannot write tab_ablation.tex: no size ladder in the artefacts` | `ablation_results.csv` is missing, or holds a *different* experiment because a run with another `--level`/`--tasks` wrote over it | `python make_ablation_tables.py --sources` names the file and the reason; rebuild with `python run_ablation.py --level state --attach-climate` |
| `refusing to overwrite ablation_results.csv` | the run you launched writes a different `level`/`tasks` combination than the file already holds | keep the derived filename (drop `--output`), or pass `--force` if the replacement is intended |

## Technical Analysis Overview

### 1. Power Law Parameter Estimation
The central analysis involves the estimation of the scaling parameter ($\alpha$) for the probability distribution of weekly reported cases ($X$). We assume a discrete power law distribution:
$$P(X \ge x) \propto x^{-(\alpha - 1)}$$
The estimation follows the methodology proposed by Clauset et al. (2009):
- **Maximum Likelihood Estimation (MLE)**: Numerical optimization to find the $\alpha$ that maximizes the probability of the observed counts.
- **Scaling Regime Identification**: Selection of the lower bound $x_{\text{min}}$ that minimizes the Kolmogorov-Smirnov (KS) distance between the empirical data and the model.
- **Distributional Comparison**: Likelihood ratio tests (Vuong's test) are performed to compare the power law fit against alternative heavy-tailed and non-scaling distributions, specifically the exponential distribution, to validate the scale-invariant hypothesis.

### 2. Longitudinal Scaling Trends
To capture the temporal evolution of epidemic dynamics, the framework performs **Yearly Power Law Fitting**. This allows for the characterization of the "scaling trajectory" of individual cities over a 15-year span (2010–2025). 
- **Alpha Dynamics**: The variation in $\alpha$ over time serves as a proxy for changes in the regularity and intensity of transmission across successive seasons.
- **Trend Analysis**: OLS regression on yearly $\alpha$ values identifies long-term shifts in the structural behavior of outbreaks within specific geographic nodes.

### 3. Spatial Analysis
- **Choropleth Mapping**: Interactive visualization of power law parameters across Brazilian states and municipalities.
- **Alpha Trend Mapping**: Spatial representation of temporal trends in scaling parameters to identify regions with increasing or decreasing heavy-tailed behavior.
- **Local Moran Clustering**: Spatial autocorrelation analysis using Local Indicators of Spatial Association (LISA) to identify statistically significant clusters of high/high and low/low alpha values.

### 4. Epidemiological Correlation and Exploratory Analysis
The project bridges statistical scaling with mechanistic epidemiological indicators derived from the **Episcanner** engine. We perform cross-correlation analysis between the scaling parameter ($\alpha$) and several key indicators:
- **Baseline Transmission Intensity ($R_0$)**: Investigating the correlation between the basic reproduction number and the thickness of the distribution's tail.
- **Epidemic Intensity**: Correlation with total seasonal case counts (attack rates).
- **Outbreak Temporal Phenology**: Analysis of $\alpha$ in relation to the epidemic onset ($ep_{\text{ini}}$) and total epidemic duration ($ep_{\text{dur}}$).
- **Model Residuals**: Relationship between alpha values and sum of residuals from epidemic models.

### 5. Epidemic Metrics Power Laws (Episcanner Integration)
Power law analysis applied to epidemic-level metrics. **Only epidemic size is
fitted.** Duration and peak timing are bounded above by the length of the
season, so a scale-free law cannot apply to them a priori; they are carried
through the analysis as *covariates* of size instead. `fit_episcanner_pl.py`
fits `FIT_METRICS = ["total_cases"]` and offers `--purge-ep-dur` to remove any
`metric='ep_dur'` rows left over from earlier runs.
- **Epidemic Size**: Power law fitting of total case counts per epidemic event.
- **Combined CCDF Analysis**: Multi-year cumulative distribution analysis with year-stratified visualization.
- **Dispersion-Alpha Relationships**: Boxplot analysis of metric distributions overlaid with yearly alpha trends.
- **State-Level Aggregation**: Comparative analysis of power law parameters across all Brazilian states.

### 6. Predictive Modeling
A machine learning framework for forecasting epidemic characteristics using historical power law patterns:
- **Model Architecture**: a quantile regression on a cubic-spline basis with
  linear boundary extrapolation (`run_ablation.py --model spline_quantile`), the
  default. A tree ensemble predicts a weighted average of its training
  responses and therefore cannot produce a value outside their range, which
  under a heavy tail is exactly where the interesting season sits; the spline
  basis continues the boundary trend instead. Boosted alternatives
  (`lightgbm`, `lightgbm_linear_tree`, `hist_gradient_boosting`) are kept as a
  comparison arm, and a Random Forest is retained for the illustrative scatter
  of figure 10.
- **Target Variable**: Epidemic Size ($S_t$) only. Duration ($D_t$) and peak
  week ($P_t$) are *predictors*, not outcomes.
- **Feature Engineering**:
  - Lagged epidemic metrics from previous season ($S_{t-1}, D_{t-1}, P_{t-1}$)
  - Power law scaling factors ($\alpha_{S, t-1}, x_{\text{min}, S, t-1}$)
  - Historical trend in alpha ($\dot{\alpha}_{S, t-1}$)
  - Basic reproduction number ($R_{0, t-1}$)
  - Demographics ($\log_{10}(\text{Pop})$)
  - Climate variables: Quarterly averaged minimum temperature and humidity ($\bar{T}_{Q1...Q4}, \bar{H}_{Q1...Q4}$)
- **Model Evaluation**: Out-of-bag R², MAE, RMSE metrics with feature importance analysis.
- **Error Analysis**: Population distribution analysis for predictions with high relative error (>20%).
- **Aggregation Levels**: Support for both state-level and city-level predictions.

## Interactive Dashboard Features

The Gradio-based dashboard provides four main analysis tabs:

### General Overview
- **Alpha Parameter Map**: Choropleth visualization at state or city level
- **Alpha Trend Map**: Spatial representation of temporal scaling trends
- **Local Moran Clustering**: Spatial autocorrelation hotspot analysis
- **Alpha Distribution**: Histogram of power law parameters across selected region
- **Alpha vs Incidence**: Scatter plot correlating scaling with cumulative incidence

### City Details & Yearly Trends
- **Power Law Fit**: CCDF visualization with fitted power law curve for selected city
- **Timeseries**: Weekly case estimates over the full observation period
- **Yearly Alpha Trend**: Temporal evolution of scaling parameters with trend line
- **Epidemiological Correlations**: Five-panel correlation analysis ($\alpha$ vs $R_0$, cases, onset, duration, residuals)
- **Data Tables**: General city statistics and yearly fit parameters

### Epidemic Metrics Power Laws
- **Combined Fits**: Multi-year CCDF plots for epidemic size
- **Dispersion Analysis**: Boxplots overlaid with yearly alpha trends
- **State Alpha Map**: Choropleth map of the epidemic-size exponent
- **Regional Timeseries**: Weekly case trends at state or national level
- **Residuals Analysis**: Alpha vs residuals scatter plots
- **Fit Details Table**: Parameter table for the epidemic-size fits

### Predictive Modeling
- **Forecast Configuration**: Select prediction level (State/City), region, and target year (2012-2025)
- **Predicted vs Actual**: Scatter plots comparing model predictions with observed values
- **Feature Importance**: Bar charts showing relative importance of predictive features
- **Performance Metrics**: Model evaluation table (MAE, RMSE, OOB R²)
- **Error Analysis**: Population distribution histograms for high-error predictions
- **Forecast Output**: Detailed predictions with error margins for selected year

## Computational Implementation
- **Data Source**: Integrated with the Mosqlimate API for Infodengue and Episcanner data retrieval.
- **Fitting Engine**: Core logic implemented in the `FitPL` class using the `powerlaw` Python library.
- **Database**: SQLite with DuckDB for efficient aggregation of large-scale epidemiological data.
- **Asynchronous Workflow**: High-performance data fetching and processing using `asyncio` and `ThreadPoolExecutor` for exhaustive scanning of all 5,500+ municipalities.
- **Geospatial Analysis**: PyGeoDa integration for spatial autocorrelation and clustering analysis.
- **Machine Learning**: Scikit-learn Random Forest implementation with cross-validation.
- **Interactive Analytics**: Multi-tab Gradio dashboard with real-time visualization of empirical CCDFs, longitudinal trends, multi-indicator correlations, and predictive modeling.

## Getting Started

### 1. Environment Setup

This project uses `uv` for fast, reliable Python package management.

1.  **Install `uv`** (if you haven't already):
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
2.  **Create a virtual environment and install dependencies**:
    ```bash
    uv sync
    ```
3.  **Activate the environment**:
    ```bash
    source .venv/bin/activate
    ```

### 2. Configuration

Create a `.env` file in the root directory and add your Mosqlimate API key:
```env
MOSQLIMATE_API_KEY=your_api_key_here
```

### 3. Data Caching and Fitting

The extraction and fitting pipeline is documented in
[Reproducing the manuscript](#reproducing-the-manuscript). For a first run,
`python update_data.py` followed by `python aggregate_city_stats.py`
populates `data/` and `powerlaw_results.db`.

### 4. Running the Dashboard

Once the data is cached, launch the interactive dashboard:
```bash
python dashboard.py
```
The dashboard will be available at `http://127.0.0.1:7860`.

## Project Structure

### Data pipeline
- `update_data.py`: one-shot refresh (phases 0-4: Episcanner download, weekly fits, yearly fits, state/national metric fits, total-cases fits); resumable
- `fitpl.py`: `FitPL` class; Mosqlimate/Infodengue fetching, parquet caching and every power-law fit
- `fit_episcanner_pl.py`: state- and national-level Episcanner metric fits
- `aggregate_city_stats.py`: builds the `city_stats` table (population, case totals) used for joins
- `provenance.py`: snapshot manifest; SHA-256 of the database and of every derived artefact (`--write` freezes, `--check` raises on drift)
- `make_tables.py`: emits `Manuscript/generated/tab_main.tex` and `facts.json` from the database; `--check` audits the prose against them
- `verify_snapshot.py`: manifest + prose + orphan-export checks in one command; exit 1 on any drift

### Predictive layer
- `run_ablation.py`: rolling-origin ablation ladder M0 to M3, state and city; writes `ablation_results.csv` / `ablation_city.csv`
- `run_classifier.py`: out-of-fold rarity classifier feeding Figures 12 and 13
- `temporal_split.py`, `predictive_model.py`: shared panel construction and model definitions
- `binary_classifier.py`, `binary_viz_utils.py`: classifier helpers and plotting

### Manuscript
- `Manuscript/main.tex`: manuscript source; Table 1 is `\input{generated/tab_main}`
- `Manuscript/figures/generate_figures.py`: all 15 figures, or a subset (`python Manuscript/figures/generate_figures.py 2 4 6`)
- `Manuscript/verification/`: result verification scripts and SQL
- `report/`, `slides/`, `slides_beamer/`: presentation material

### Dashboard and shared utilities
- `dashboard.py`, `viz_utils.py`, `data_utils.py`, `profile_dashboard.py`: Gradio dashboard and helpers
- `powerlaw_results.db`: SQLite, the source of truth for every reported number
- `data/`: cached parquet files (git-ignored)
