# Power Law Dynamics in Dengue Epidemics

This project implements a comprehensive computational framework for the statistical analysis of power law scaling in dengue reported cases across Brazilian municipalities. The analysis integrates methods from statistical physics, epidemic modeling, machine learning, and longitudinal data analysis to characterize the heavy-tailed distribution of outbreak magnitudes and their relationship with underlying epidemiological drivers.

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
Extended power law analysis applied to epidemic-level metrics:
- **Epidemic Size**: Power law fitting of total case counts per epidemic event.
- **Epidemic Duration**: Power law analysis of outbreak duration in weeks.
- **Combined CCDF Analysis**: Multi-year cumulative distribution analysis with year-stratified visualization.
- **Dispersion-Alpha Relationships**: Boxplot analysis of metric distributions overlaid with yearly alpha trends.
- **State-Level Aggregation**: Comparative analysis of power law parameters across all Brazilian states.

### 6. Predictive Modeling
A machine learning framework for forecasting epidemic characteristics using historical power law patterns:
- **Model Architecture**: Random Forest Regressor ($f_{RF}$) trained on lagged features from year $t-1$ to predict year $t$ metrics.
- **Target Variables**: Epidemic Size ($S_t$), Duration ($D_t$), and Peak Week ($P_t$).
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
- **Combined Fits**: Multi-year CCDF plots for epidemic size and duration
- **Dispersion Analysis**: Boxplots overlaid with yearly alpha trends
- **State Alpha Maps**: Choropleth maps for epidemic size and duration parameters
- **Regional Timeseries**: Weekly case trends at state or national level
- **Residuals Analysis**: Alpha vs residuals scatter plots
- **Fit Details Tables**: Comprehensive parameter tables for size and duration

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

Before running the dashboard, you need to populate the local cache and SQLite database. This process may take some time depending on the number of municipalities.

1.  **Fetch Infodengue data and fit power laws**:
    ```bash
    python fitpl.py
    ```
    *This script processes all Brazilian municipalities and saves results to `powerlaw_results.db`.*

2.  **Fetch Episcanner data and fit power laws**:
    ```bash
    python fit_episcanner_pl.py
    ```
    *This script processes state-level and national-level Episcanner metrics.*

### 4. Running the Dashboard

Once the data is cached, launch the interactive dashboard:
```bash
python dashboard.py
```
The dashboard will be available at `http://127.0.0.1:7860`.

## Project Structure

- `dashboard.py`: Main Gradio dashboard application
- `fitpl.py`: Core power law fitting implementation
- `fit_episcanner_pl.py`: Episcanner data integration and fitting
- `viz_utils.py`: Visualization functions for all dashboard components
- `data_utils.py`: Data fetching, caching, and utility functions
- `predictive_model.py`: Machine learning models for epidemic forecasting
- `powerlaw_results.db`: SQLite database storing all fit results and parameters
- `data/`: Cached parquet files for Infodengue and Episcanner data
