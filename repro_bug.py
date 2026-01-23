import pandas as pd
import numpy as np

def get_alpha_trends_mock(empty=False):
    if empty:
        trends = []
    else:
        trends = [{'geocode': 123, 'alpha_trend': 0.1}]
    return pd.DataFrame(trends)

def test_merge():
    df_trends = get_alpha_trends_mock(empty=True)
    print(f"Empty DataFrame columns: {df_trends.columns.tolist()}")
    
    merged = pd.DataFrame({'code_muni': [123, 456], 'alpha': [1.5, 2.0]})
    try:
        merged = merged.merge(df_trends, left_on='code_muni', right_on='geocode', how='left')
        print("Merge successful")
    except KeyError as e:
        print(f"Merge failed with KeyError: {e}")

if __name__ == "__main__":
    test_merge()
