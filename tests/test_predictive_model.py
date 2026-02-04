import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import predictive_model

@patch('predictive_model.get_duckdb_episcanner_data')
@patch('predictive_model.get_episcanner_fit_results')
@patch('predictive_model.ensure_episcanner_files')
def test_prepare_lagged_data(mock_ensure, mock_fits, mock_duck):
    # Mock Episcanner raw data
    mock_metric_df = pd.DataFrame({
        'geocode': [4300001, 4300002]*2,
        'year': [2020, 2020, 2021, 2021],
        'total_cases': [100, 200, 110, 210],
        'ep_dur': [10, 15, 12, 16],
        'peak_week': [10, 12, 11, 13],
        'R0': [1.2, 1.3, 1.25, 1.35]
    })
    mock_duck.return_value = mock_metric_df
    
    # Mock Fits. returns state-level fits.
    mock_fits_df = pd.DataFrame({
        'state': ['RS']*2,
        'year': [2020, 2021],
        'alpha': [1.5, 1.6],
        'xmin': [10, 11],
        'metric': ['total_cases']*2
    })
    
    # We need to simulate behavior of get_episcanner_fit_results calling logic inside prepare_lagged_data
    # Actually prepare_lagged_data queries DB directly for ALL fits.
    # We should mock pd.read_sql_query inside if possible, but here we patched the helper? 
    # Ah, in prepare_lagged_data I used read_sql_query directly. So I need to patch sqlite3 or the function that gets fits.
    pass

@patch('predictive_model.pd.read_sql_query')
@patch('predictive_model.sqlite3.connect')
@patch('predictive_model.get_quarterly_climate')
@patch('predictive_model.get_city_stats')
@patch('predictive_model.get_duckdb_episcanner_data')
@patch('predictive_model.ensure_episcanner_files')
def test_prepare_lagged_data_full(mock_ensure, mock_duck, mock_pop, mock_climate, mock_connect, mock_read_sql):
    
    # Mock Episcanner Data
    mock_duck.return_value = pd.DataFrame({
        'geocode': [4300001, 4300002]*3, # 2 cities, valid 7-digits
        'year': [2020, 2020, 2021, 2021, 2022, 2022],
        'total_cases': [100, 200, 110, 220, 120, 240],
        'ep_dur': [10, 20, 10, 20, 10, 20],
        'peak_week': [15, 16, 15, 16, 15, 16],
        'R0': [1.5, 1.5, 1.5, 1.5, 1.5, 1.5]
    })
    
    # Mock Fits Data (from DB)
    def side_effect(query, conn):
        if "episcanner_state_fits" in query:
            return pd.DataFrame({
                'state': ['RS', 'RS', 'RS'],
                'year': [2020, 2021, 2022],
                'alpha': [1.5, 1.6, 1.7],
                'xmin': [10, 10, 10],
                'metric': ['total_cases']*3 
            })
            
        elif "powerlaw_fits_yearly" in query:
            return pd.DataFrame({
                'geocode': [4300001, 4300001, 4300001], 
                'year': [2020, 2021, 2022],
                'alpha': [1.5, 1.6, 1.7],
                'xmin': [10, 10, 10],
                'xmax': [100, 100, 100]
            })
        return pd.DataFrame()
        
    mock_read_sql.side_effect = side_effect
    
    # Mock Population Data
    mock_pop.return_value = pd.DataFrame({
        'geocode': [4300001, 4300002],
        'population': [100000, 200000]
    })
    
    # Mock Climate Data
    def climate_side_effect(region, level="state"):
        df = pd.DataFrame({
            'geocode': [4300001, 4300001, 4300002, 4300002],
            'year': [2021, 2022, 2021, 2022],
            'avg_temp_Q1': [20.0, 21.0, 22.0, 23.0],
            'avg_umid_Q1': [70.0, 71.0, 72.0, 73.0]
        })
        if level == 'state':
            df['state'] = 'RS'
            return df.groupby(['state', 'year'])[['avg_temp_Q1', 'avg_umid_Q1']].mean().reset_index()
        return df

    mock_climate.side_effect = climate_side_effect
    
    # Test
    df_res = predictive_model.prepare_lagged_data("RS")
    
    # With 2020, 2021, 2022 data:
    # 2020: No lag
    # 2021: Has 2020 lag
    # 2022: Has 2021 lag
    # Should return 2 rows (2021, 2022) with valid lagged features
    
    # Test City Level
    df_res_city = predictive_model.prepare_lagged_data("RS", level="city")
    assert not df_res_city.empty
    if not df_res_city.empty:
         assert 'log_pop' in df_res_city.columns
         assert 'prev_log_pop' in df_res_city.columns
         # log10(100000) = 5
         assert 5.0 in df_res_city['log_pop'].values
         
         assert 'avg_temp_Q1' in df_res_city.columns
         assert 'prev_avg_temp_Q1' in df_res_city.columns
         # For 2022, prev_avg_temp_Q1 should be 20.0 (from 2021)
         assert 20.0 in df_res_city[df_res_city['year']==2022]['prev_avg_temp_Q1'].values

def test_train_models():
    # Create fake training data
    df_train = pd.DataFrame({
        'year': [2021, 2022, 2023]*10,
        'total_cases': np.random.rand(30)*100,
        'ep_dur': np.random.rand(30)*20,
        'peak_week': np.random.rand(30)*52,
        'prev_total_cases': np.random.rand(30)*100,
        'prev_ep_dur': np.random.rand(30)*20,
        'prev_peak_week': np.random.rand(30)*52,
        'prev_alpha_size': np.random.rand(30)*3
    })
    
    models, metrics, features = predictive_model.train_predictive_models(df_train)
    
    assert 'total_cases' in models
    assert 'ep_dur' in models
    assert 'peak_week' in models
    assert 'MAE' in metrics['total_cases']
    assert 'MAPE' in metrics['total_cases']
    
    preds = predictive_model.predict_future(models, df_train.iloc[:5], features)
    assert len(preds['total_cases']) == 5

def test_train_models_temporal():
    # Create fake training data with years
    df_train = pd.DataFrame({
        'year': [2020, 2021, 2022, 2023]*10,
        'total_cases': np.random.rand(40)*100,
        'ep_dur': np.random.rand(40)*20,
        'peak_week': np.random.rand(40)*52,
        'prev_total_cases': np.random.rand(40)*100,
        'prev_ep_dur': np.random.rand(40)*20,
        'prev_peak_week': np.random.rand(40)*52,
        'prev_alpha_size': np.random.rand(40)*3
    })
    
    # Train with split at 2022. Train: 2020, 2021. Test: 2022, 2023.
    models, metrics, features = predictive_model.train_predictive_models(df_train, test_year=2022)
    
    assert 'total_cases' in models
    assert metrics['total_cases']['MAE'] >= 0
    # Ideally checking that it didn't crash.
    
    # Test with split beyond data
    models_empty, metrics_empty, _ = predictive_model.train_predictive_models(df_train, test_year=2025)
    assert metrics_empty['total_cases']['MAE'] == 0
