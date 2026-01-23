import pandas as pd
import numpy as np

# Mocking the fixed get_alpha_trends behavior
def get_alpha_trends_fixed(empty=False):
    if empty:
        trends = []
    else:
        trends = [{'geocode': 123, 'alpha_trend': 0.1}]
    return pd.DataFrame(trends, columns=['geocode', 'alpha_trend'])

def test_merge_fixed():
    print("--- Testing empty trends with fixed return ---")
    df_trends = get_alpha_trends_fixed(empty=True)
    print(f"Empty DataFrame columns: {df_trends.columns.tolist()}")
    
    merged = pd.DataFrame({'code_muni': [123, 456], 'alpha': [1.5, 2.0]})
    try:
        merged = merged.merge(df_trends, left_on='code_muni', right_on='geocode', how='left')
        print("Merge successful")
        print(f"Merged columns: {merged.columns.tolist()}")
    except KeyError as e:
        print(f"Merge failed with KeyError: {e}")

if __name__ == "__main__":
    test_merge_fixed()
