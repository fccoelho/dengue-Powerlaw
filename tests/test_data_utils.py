import pytest
import pandas as pd
import sqlite3
import os
import data_utils
from unittest.mock import patch, MagicMock

def test_extract_geocode():
    assert data_utils.extract_geocode("City (12345)") == 12345
    assert data_utils.extract_geocode("City (12345) - alpha: 1.23") == 12345
    assert data_utils.extract_geocode("No geocode here") is None
    assert data_utils.extract_geocode(None) is None

def test_get_db_data(tmp_path):
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    df = pd.DataFrame({'city_name': ['A'], 'geocode': [1], 'alpha': [1.5]})
    df.to_sql('powerlaw_fits', conn, index=False)
    conn.close()
    
    result = data_utils.get_db_data(db_path=str(db_path))
    assert len(result) == 1
    assert result.iloc[0]['city_name'] == 'A'

def test_get_city_stats_empty(tmp_path):
    db_path = tmp_path / "test_empty.db"
    # No table created
    result = data_utils.get_city_stats(db_path=str(db_path))
    assert result.empty

def test_get_alpha_trends(tmp_path):
    db_path = tmp_path / "test_trends.db"
    conn = sqlite3.connect(db_path)
    df = pd.DataFrame({
        'geocode': [1, 1, 2, 2],
        'year': [2020, 2021, 2020, 2021],
        'alpha': [1.0, 2.0, 3.0, 3.0]
    })
    df.to_sql('powerlaw_fits_yearly', conn, index=False)
    conn.close()
    
    # clear cache if any
    data_utils.get_alpha_trends.cache_clear()
    result = data_utils.get_alpha_trends(db_path=str(db_path))
    assert len(result) == 2
    # Slope for geocode 1 should be 1.0 (2.0 - 1.0 / 2021 - 2020)
    assert result.loc[result['geocode'] == 1, 'alpha_trend'].iloc[0] == pytest.approx(1.0)
    # Slope for geocode 2 should be 0.0
    assert result.loc[result['geocode'] == 2, 'alpha_trend'].iloc[0] == pytest.approx(0.0)

@patch('data_utils._fetch_episcanner')
def test_ensure_episcanner_files(mock_fetch, tmp_path):
    # Mock os.path.exists to simulate missing files
    with patch('os.path.exists', return_value=False):
        data_utils.ensure_episcanner_files("RS", years=[2024])
        assert mock_fetch.called

def test_get_yearly_state_data(tmp_path):
    db_path = tmp_path / "test_state.db"
    conn = sqlite3.connect(db_path)
    # 43 is RS
    df = pd.DataFrame({
        'geocode': [4300001, 3500001],
        'year': [2020, 2020],
        'alpha': [1.5, 2.0]
    })
    df.to_sql('powerlaw_fits_yearly', conn, index=False)
    conn.close()
    
    result = data_utils.get_yearly_state_data("RS", db_path=str(db_path))
    assert len(result) == 1
    assert result.iloc[0]['geocode'] == 4300001
