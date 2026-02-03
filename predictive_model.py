import pandas as pd
import numpy as np
import sqlite3
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
import duckdb
import glob
import os
from data_utils import get_duckdb_episcanner_data, get_episcanner_fit_results, ensure_episcanner_files, GEO_STATE_MAP

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
        
    else:
        return pd.DataFrame()
    
    # sort
    df_features = df_features.sort_values([group_col, 'year'])
    
    # Create Lagged Features
    feature_cols = ['total_cases', 'ep_dur', 'peak_week', 'R0', 'alpha_size', 'xmin_size', 'alpha_dur', 'xmin_dur']
    
    df_lagged = df_features.copy()
    for col in feature_cols:
        # Group by the entity (state or geocode)
        df_lagged[f'prev_{col}'] = df_lagged.groupby(group_col)[col].shift(1)
        
    # Drop rows with NaN in prev_ columns (lag creation)
    # However, for city level, alpha_dur might be always NaN. 
    # If a feature is fully NaN, we should drop the feature or fill it.
    # Check if 'alpha_dur' is fully NaN -> drop it from feature list
    
    cols_to_use = []
    for col in feature_cols:
        col_name = f'prev_{col}'
        if df_lagged[col_name].isnull().all():
            pass # Drop this feature
        else:
            cols_to_use.append(col_name)
            
    # Drop rows where used features are NaN
    df_model = df_lagged.dropna(subset=cols_to_use)
    
    # If filling NaNs is needed for sklearn:
    # df_model = df_model.fillna(0) # Risks biasing
    # Better to just drop rows with missing history
    
    if target_year:
        df_model = df_model[df_model['year'] == target_year]
        
    return df_model

def train_predictive_models(train_df):
    """
    Trains Random Forest models for Size, Duration, and Peak Week using lagged features.
    """
    # Identify available lagged columns
    feature_cols = [c for c in train_df.columns if c.startswith('prev_')]
    targets = ['total_cases', 'ep_dur', 'peak_week']
    
    models = {}
    metrics = {}
    
    # Drop any remaining NaNs in features or targets just in case
    train_clean = train_df.dropna(subset=feature_cols + targets)
    
    if train_clean.empty:
        return {}, {}, feature_cols
    
    X = train_clean[feature_cols]
    
    for target in targets:
        y = train_clean[target]
        
        if len(train_clean) > 10:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        else:
            X_train, X_test, y_train, y_test = X, X, y, y
            
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        preds = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds) if len(y_test) > 1 else 0
        mape = mean_absolute_percentage_error(y_test, preds)
        
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
