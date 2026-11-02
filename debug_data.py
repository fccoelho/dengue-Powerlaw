import predictive_model
import pandas as pd
import numpy as np

def debug_region(region, level):
    print(f"--- Debugging {region} ({level}) ---")
    try:
        df = predictive_model.prepare_lagged_data(region, level=level)
        print(f"Shape: {df.shape}")
        if not df.empty:
            print(f"Years: {df['year'].unique()}")
            print(f"Columns: {df.columns.tolist()}")
            print(f"Rows per year:\n{df['year'].value_counts().sort_index()}")
        else:
            print("DataFrame is EMPTY.")
    except Exception as e:
        print(f"Error: {e}")
    print("\n")

if __name__ == "__main__":
    debug_region("RS", "state")
    debug_region("RS", "city")
    debug_region("BR", "state")
