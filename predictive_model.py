import pandas as pd
import numpy as np
import sqlite3
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
import duckdb
import glob
import os
from data_utils import get_duckdb_episcanner_data, get_episcanner_fit_results, ensure_episcanner_files, GEO_STATE_MAP, get_city_stats
from scipy import stats
try:
    from _mun_by_geocode import NAME_BY_GEOCODE
except ImportError:
    NAME_BY_GEOCODE = {}

def get_quarterly_climate(region, level="state"):
    """
    Fetches and aggregates tempmin and umidmin by quarter from Infodengue parquet files.
    """
    con = duckdb.connect()
    data_path = "data/[0-9]*.parquet"
    
    # Filter by region if not BR
    where_clause = ""
    if region != "BR":
        state_code = None
        for k, v in GEO_STATE_MAP.items():
            if v == region:
                state_code = k
                break
        if state_code:
            min_geo = state_code * 100000
            max_geo = (state_code + 1) * 100000
            where_clause = f"WHERE municipio_geocodigo >= {min_geo} AND municipio_geocodigo < {max_geo}"
    
    try:
        # Extract quarter and year from data_iniSE
        # umidmin and tempmin are weekly. We average by quarter.
        query = f"""
            SELECT 
                municipio_geocodigo as geocode,
                year,
                EXTRACT(quarter FROM data_iniSE) as quarter,
                AVG(tempmin) as avg_temp,
                AVG(umidmin) as avg_umid
            FROM read_parquet('{data_path}')
            {where_clause}
            GROUP BY geocode, year, quarter
        """
        df_q = con.execute(query).df()
        
        if df_q.empty:
            return pd.DataFrame()
            
        # Pivot to have Year, Geocode as rows and (Metric, Quarter) as columns
        df_pivot = df_q.pivot_table(
            index=['geocode', 'year'], 
            columns='quarter', 
            values=['avg_temp', 'avg_umid']
        )
        
        # Flatten columns
        df_pivot.columns = [f"{col[0]}_Q{int(col[1])}" for col in df_pivot.columns]
        df_pivot = df_pivot.reset_index()
        
        if level == 'state':
            # Map to states and aggregate
            def get_state(code):
                try: return GEO_STATE_MAP.get(int(str(code)[:2]), 'Unknown')
                except: return 'Unknown'
            
            df_pivot['state'] = df_pivot['geocode'].apply(get_state)
            climate_cols = [c for c in df_pivot.columns if 'avg_' in c]
            df_res = df_pivot.groupby(['state', 'year'])[climate_cols].mean().reset_index()
            return df_res
        else:
            return df_pivot
            
    except Exception as e:
        print(f"Error fetching climate data: {e}")
        return pd.DataFrame()

def prepare_lagged_data(region="BR", target_year=None, level="state"):
    """
    Prepares a dataset with lagged features for predicting target_year metrics.
    If target_year is None, returns a dataset with all possible year pairs.
    level: "state" or "city"
    """
    
    # 1. Fetch all available data
    years = list(range(2010, 2026)) 
    ensure_episcanner_files(region, years)
    
    # Get raw metrics (Size, Duration, Peak) for all available years
    # We need aggregated city-level or state-level data? 
    # The prompt implies predicting for a "given year", likely at the regional level since we use "scaling factors used for previous years"
    # Let's target STATE-LEVEL prediction for now, using state-level features.
    
    df_metrics = get_duckdb_episcanner_data(region, years) # This is usually city-level. 
    # But usually we predict the NEXT epidemic characteristics based on PAST.
    # Actually, let's try to predict STATE aggregated metrics first as it's simpler and more robust.
    
    # Wait, get_duckdb_episcanner_data returns city-level data if region != BR?
    # No, it returns whatever is in the parquet files. The parquet files contain city-level data.
    
    # Let's aggregate to State level first
    if df_metrics.empty:
        return pd.DataFrame()
    
    if level == 'state':
        # Aggregate by State and Year
        
        # Helper to map geocodes to states
        def get_state(code):
            try:
                prefix = int(str(code)[:2])
                return GEO_STATE_MAP.get(prefix, 'Unknown')
            except:
                return 'Unknown'
        
        # Only add state column if not present (it might be if we re-use this df)
        if 'state' not in df_metrics.columns:
            df_metrics['state'] = df_metrics['geocode'].apply(get_state)
        
        state_agg = df_metrics.groupby(['state', 'year']).agg({
            'total_cases': 'sum',
            'ep_dur': 'mean',
            'peak_week': 'median',
            'R0': 'mean'
        }).reset_index()
        
        # For state level, we use the state-aggregated fits (episcanner_state_fits)
        import sqlite3
        with sqlite3.connect("powerlaw_results.db") as conn:
            try:
                df_fits = pd.read_sql_query("SELECT * FROM episcanner_state_fits", conn)
            except:
                df_fits = pd.DataFrame()

        if df_fits.empty:
            return pd.DataFrame()
            
        df_fits_size = df_fits[df_fits['metric'] == 'total_cases'][['state', 'year', 'alpha', 'xmin']].rename(columns={'alpha': 'alpha_size', 'xmin': 'xmin_size'})
        df_fits_dur = df_fits[df_fits['metric'] == 'ep_dur'][['state', 'year', 'alpha', 'xmin']].rename(columns={'alpha': 'alpha_dur', 'xmin': 'xmin_dur'})
        
        df_features = state_agg.merge(df_fits_size, on=['state', 'year'], how='left')
        df_features = df_features.merge(df_fits_dur, on=['state', 'year'], how='left')
        
        group_col = 'state'
        
        # Add Population for State
        df_pop = get_city_stats()
        if not df_pop.empty:
            df_pop['state'] = df_pop['geocode'].apply(get_state)
            state_pop = df_pop.groupby('state')['population'].sum().reset_index()
            df_features = df_features.merge(state_pop, on='state', how='left')
        
    elif level == 'city':
        # Use city-level data (df_metrics is already city level, but check uniqueness)
        # We need to filter by region if region != BR
        if region != "BR":
            # Filter df_metrics by state prefix
            state_code = None
            for k, v in GEO_STATE_MAP.items():
                if v == region:
                    state_code = k
                    break
            if state_code:
                min_geo = state_code * 100000
                max_geo = (state_code + 1) * 100000
                df_metrics = df_metrics[(df_metrics['geocode'] >= min_geo) & (df_metrics['geocode'] < max_geo)]
        
        # We need city-level Power Law fits (from powerlaw_fits_yearly table)
        import sqlite3
        with sqlite3.connect("powerlaw_results.db") as conn:
            try:
                # We need all years, not just the single comprehensive fit
                df_fits = pd.read_sql_query("SELECT geocode, year, alpha, xmin, xmax FROM powerlaw_fits_yearly", conn)
            except:
                df_fits = pd.DataFrame()
                
        if df_fits.empty:
            # Fallback or just continue with limited features?
            # Creating dummy fit features if missing to avoid crash, or return empty?
            pass
            
        # In city data, we only have one alpha (for cases usually).
        # The 'powerlaw_fits_yearly' usually tracks the alpha for Cases distribution (Size).
        # We don't typically have city-level Duration alpha fits in that table unless specified.
        # Assuming 'powerlaw_fits_yearly' is for Size.
        
        df_fits = df_fits.rename(columns={'alpha': 'alpha_size', 'xmin': 'xmin_size'})
        
        # Merge
        # Data types: geocode in fit might be int, in metrics int.
        df_features = df_metrics.merge(df_fits, on=['geocode', 'year'], how='left')
        
        # We might miss 'alpha_dur' at city level if not calculated. 
        # Fill with NaN or 0? XGBoost/RF handle NaNs, but sklearn Random Forest doesn't nativey support NaNs well without Imputer.
        # Let's add dummy alpha_dur if not present
        df_features['alpha_dur'] = np.nan
        df_features['xmin_dur'] = np.nan
        
        group_col = 'geocode'
        # Add city_name for reference if available?
        # df_features['city_name'] = ...
        
        # Add Population for City
        df_pop = get_city_stats()
        if not df_pop.empty:
            df_features = df_features.merge(df_pop[['geocode', 'population']], on='geocode', how='left')
            
    # Add Climate Data
    df_climate = get_quarterly_climate(region, level=level)
    if not df_climate.empty:
        df_features = df_features.merge(df_climate, on=[group_col, 'year'], how='left')
    
    def calculate_historical_slope(group):
        group = group.sort_values('year')
        slopes = []
        for i, row in group.iterrows():
            curr_y = row['year']
            # Historical alpha values (years < current)
            hist = group[group['year'] < curr_y].dropna(subset=['alpha_size'])
            if len(hist) >= 2:
                try:
                    res = stats.linregress(hist['year'], hist['alpha_size'])
                    slopes.append(res.slope)
                except:
                    slopes.append(np.nan)
            else:
                slopes.append(np.nan)
        group['alpha_trend'] = slopes
        return group

    df_features = df_features.groupby(group_col, group_keys=False).apply(calculate_historical_slope)
    
    # Calculate log population
    if 'population' in df_features.columns:
        df_features['log_pop'] = np.log10(df_features['population'].replace(0, np.nan))
    
    # Sort
    df_features = df_features.sort_values([group_col, 'year'])
    
    # Create Lagged Features
    # Basic metrics can be assumed 0 if missing (meaning no epidemic detected)
    basic_metrics = ['total_cases', 'ep_dur', 'peak_week', 'R0']
    fit_metrics = ['alpha_size', 'xmin_size', 'alpha_dur', 'xmin_dur', 'alpha_trend']
    static_features = ['log_pop'] if 'log_pop' in df_features.columns else []
    climate_features = [c for c in df_features.columns if 'avg_temp' in c or 'avg_umid' in c]
    feature_cols = basic_metrics + fit_metrics + static_features + climate_features
    
    df_lagged = df_features.copy()
    for col in feature_cols:
        # Note: alpha_trend for Year T already uses data from Years < T.
        # However, to keep it consistent with 'prev_' naming, 
        # we treat it like other features and shift if needed, or use it as is.
        # Let's shift it by 1 JUST to be safe so 'prev_alpha_trend' at Year T
        # uses the trend calculated at Year T-1 (which used data < T-1).
        # Actually, if we use the slope of < T, it's already a valid predictor for T.
        if col in static_features:
            df_lagged[f'prev_{col}'] = df_lagged[col] # Static features don't need real lag if we assume they don't change
        else:
            df_lagged[f'prev_{col}'] = df_lagged.groupby(group_col)[col].shift(1) if col != 'alpha_trend' else df_lagged['alpha_trend']
        
    mandatory_features = [f'prev_{c}' for c in basic_metrics + ['alpha_size', 'xmin_size']]
    # Filter mandatory to only those that actually exist in df_lagged
    mandatory_features = [c for c in mandatory_features if c in df_lagged.columns and not df_lagged[c].isnull().all()]
    
    # Drop rows missing mandatory features
    count_before = len(df_lagged)
    
    # We want to keep city_name/state for plotting even if they are not features
    id_cols = ['year', group_col]
    if group_col == 'geocode':
        df_lagged['city_name'] = df_lagged['geocode'].map(NAME_BY_GEOCODE)
        id_cols.append('city_name')
    
    df_model = df_lagged.dropna(subset=mandatory_features)
    
    # Optional features (trend) - impute with 0 if missing instead of dropping
    if 'prev_alpha_trend' in df_model.columns:
        df_model.loc[:, 'prev_alpha_trend'] = df_model['prev_alpha_trend'].fillna(0)
        
    print(f"[Model Data] {region} {level}: {count_before} -> {len(df_model)} rows after dropping missing mandatory lags")
    
    # If filling NaNs is needed for sklearn:
    # df_model = df_model.fillna(0) # Risks biasing
    # Better to just drop rows with missing history
    
    if target_year:
        df_model = df_model[df_model['year'] == target_year]
        
    return df_model

def train_predictive_models(train_df, test_year=None):
    """
    Trains Random Forest models for Size, Duration, and Peak Week.
    If test_year is provided, trains on data where year < test_year,
    and calculates metrics on data where year >= test_year.
    Otherwise uses random split.
    """
    # Identify available lagged columns that have at least some data
    all_prev_cols = [c for c in train_df.columns if c.startswith('prev_')]
    feature_cols = [c for c in all_prev_cols if not train_df[c].isnull().all()]
    
    targets = ['total_cases', 'ep_dur', 'peak_week']
    
    models = {}
    metrics = {}
    
    # Drop NaNs only for the features we actually intend to use and targets
    train_clean = train_df.dropna(subset=feature_cols + targets)
    
    if train_clean.empty:
        print(f"[Training] FAILED: Input dataframe is empty after dropping NaNs. Input size: {len(train_df)}")
        print(f"Features considered: {all_prev_cols}")
        print(f"Features with data: {feature_cols}")
        return {}, {}, feature_cols
    
    print(f"[Training] Starting for {len(train_clean)} clean rows. Test Year: {test_year}")
    print(f"Features used: {feature_cols}")
    
    X = train_clean[feature_cols]
    
    for target in targets:
        y = train_clean[target]
        
        if test_year:
            # Temporal Split
            mask_train = train_clean['year'] < test_year
            mask_test = train_clean['year'] >= test_year
            
            X_train = X[mask_train]
            y_train = y[mask_train]
            X_test = X[mask_test]
            y_test = y[mask_test]
            
            if X_train.empty:
                # Cannot train
                models[target] = None
                metrics[target] = {'MAE': 0, 'RMSE': 0, 'R2': 0, 'MAPE': 0}
                continue
        else:
            # Random Split
            if len(train_clean) > 10:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            else:
                X_train, X_test, y_train, y_test = X, X, y, y
        model = RandomForestRegressor(n_estimators=100, random_state=42)    
        model.fit(X_train, y_train)
        
        # Calculate metrics if test set exists
        if len(y_test) > 0:
            preds = model.predict(X_test)
            mae = mean_absolute_error(y_test, preds)
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            r2 = r2_score(y_test, preds) if len(y_test) > 1 else 0
            try:
                mape = mean_absolute_percentage_error(y_test, preds)
            except:
                mape = 0
        else:
            mae, rmse, r2, mape = 0, 0, 0, 0
            
        models[target] = model
        metrics[target] = {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'MAPE': mape}
        
    return models, metrics, feature_cols

def predict_future(models, current_data_df, feature_cols):
    """
    Predicts for the rows in current_data_df using trained models.
    """
    X = current_data_df[feature_cols]
    predictions = {}
    
    for target, model in models.items():
        if model:
            predictions[target] = model.predict(X)
            
    return predictions

def get_variable_importance(models, feature_cols):
    """
    Extracts variable importance from random forest models.
    """
    importance_df = pd.DataFrame({'Feature': feature_cols})
    
    for target, model in models.items():
        importance_df[f'Importance_{target}'] = model.feature_importances_
        
    return importance_df
