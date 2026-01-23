import pandas as pd
import numpy as np
import pytest

# Mocking the fixed get_alpha_trends behavior
def get_alpha_trends_fixed(empty=False):
    if empty:
        trends = []
    else:
        trends = [{'geocode': 123, 'alpha_trend': 0.1}]
    return pd.DataFrame(trends, columns=['geocode', 'alpha_trend'])

def test_merge_fixed_empty_trends():
    """Test that merging with an empty trends DataFrame doesn't raise KeyError."""
    df_trends = get_alpha_trends_fixed(empty=True)
    assert df_trends.empty
    
    merged = pd.DataFrame({'code_muni': [123, 456], 'alpha': [1.5, 2.0]})
    # This should not raise KeyError
    result = merged.merge(df_trends, left_on='code_muni', right_on='geocode', how='left')
    
    assert 'alpha_trend' in result.columns
    assert result['alpha_trend'].isnull().all()

def test_merge_fixed_with_data():
    """Test merge with actual data."""
    df_trends = get_alpha_trends_fixed(empty=False)
    merged = pd.DataFrame({'code_muni': [123, 456], 'alpha': [1.5, 2.0]})
    
    result = merged.merge(df_trends, left_on='code_muni', right_on='geocode', how='left')
    
    assert 'alpha_trend' in result.columns
    assert result.loc[result['code_muni'] == 123, 'alpha_trend'].iloc[0] == 0.1
    assert result.loc[result['code_muni'] == 456, 'alpha_trend'].isnull().all()
