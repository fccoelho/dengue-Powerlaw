import pytest
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import viz_utils
from unittest.mock import patch, MagicMock
import pandas as pd

@patch('viz_utils.get_merged_data')
def test_plot_alpha_histogram(mock_get_data):
    # Mock empty data
    mock_get_data.return_value = (pd.DataFrame(), pd.DataFrame())
    fig = viz_utils.plot_alpha_histogram("RS")
    assert isinstance(fig, go.Figure)
    assert "No alpha data" in fig.layout.title.text

    # Mock some data
    df = pd.DataFrame({'alpha': [1.5, 2.0, 2.5]})
    mock_get_data.return_value = (df, pd.DataFrame())
    fig = viz_utils.plot_alpha_histogram("RS")
    assert isinstance(fig, go.Figure)
    assert "Distribution of Alpha" in fig.layout.title.text

@patch('viz_utils.extract_geocode')
@patch('sqlite3.connect')
@patch('viz_utils.fetch_infodengue')
def test_plot_fit(mock_fetch, mock_connect, mock_extract):
    mock_extract.return_value = 123
    mock_conn = MagicMock()
    mock_connect.return_value.__enter__.return_value = mock_conn
    mock_cursor = mock_conn.cursor.return_value
    mock_cursor.fetchone.return_value = (1.5, 10, 100) # alpha, xmin, xmax
    
    df = pd.DataFrame({'casos_est': [1, 10, 100]})
    mock_fetch.return_value = df
    
    fig = viz_utils.plot_fit("City (123)")
    assert isinstance(fig, plt.Figure)

@patch('viz_utils.get_yearly_state_data')
@patch('viz_utils.ensure_episcanner_files')
@patch('viz_utils.get_duckdb_episcanner_data')
def test_plot_state_residuals_vs_alpha(mock_duck, mock_ensure, mock_state_data):
    # Test with data
    mock_state_data.return_value = pd.DataFrame({'geocode': [123], 'year': [2020], 'alpha': [1.5], 'city_name': ['A']})
    mock_duck.return_value = pd.DataFrame({'geocode': [123], 'year': [2020], 'sum_res': [0.1]})
    
    fig = viz_utils.plot_state_residuals_vs_alpha("RS")
    assert isinstance(fig, go.Figure)
    assert "Alpha vs Sum of Residuals" in fig.layout.title.text
