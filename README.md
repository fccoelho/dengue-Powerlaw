# Power Law Dynamics in Dengue Epidemics

This project implements a comprehensive computational framework for the statistical analysis of power law scaling in dengue reported cases across Brazilian municipalities. The analysis integrates methods from statistical physics, epidemic modeling, and longitudinal data analysis to characterize the heavy-tailed distribution of outbreak magnitudes and their relationship with underlying epidemiological drivers.

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

### 3. Epidemiological Correlation and Exploratory Analysis
The project bridges statistical scaling with mechanistic epidemiological indicators derived from the **Episcanner** engine. We perform cross-correlation analysis between the scaling parameter ($\alpha$) and several key indicators:
- **Baseline Transmission Intensity ($R_0$)**: Investigating the correlation between the basic reproduction number and the thickness of the distribution's tail.
- **Epidemic Intensity**: Correlation with total seasonal case counts (attack rates).
- **Outbreak Temporal Phenology**: Analysis of $\alpha$ in relation to the epidemic onset ($ep_{\text{ini}}$) and total epidemic duration ($ep_{\text{dur}}$).

### 4. Spatio-Temporal aggregation
Given the vast number of administrative units in Brazil (>5,500), the exploratory analysis utilizes **State-Level Spatial Aggregation** and **Geometry Dissolution** for regional comparative analysis. This provides:
- **Regional Scaling Profiles**: Average $\alpha$ values per state to identify macroeconomic or climatic zones with distinct scaling regimes.
- **Choropleth Visualizations**: Spatially smoothed representation of scaling behavior across the Brazilian territory.

## Computational Implementation
- **Data Source**: Integrated with the Mosqlimate API for Infodengue and Episcanner data retrieval.
- **Fitting Engine**: Core logic implemented in the `FitPL` class using the `powerlaw` Python library.
- **Asynchronous Workflow**: High-performance data fetching and processing using `asyncio` for exhaustive scanning of all 5,500+ municipalities.
- **Interactive Analytics**: A multi-tab Gradio dashboard providing real-time visualization of empirical CCDFs, longitudinal trends, and multi-indicator correlations.

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
