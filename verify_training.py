import predictive_model
import pandas as pd
import numpy as np

def test_training_flow():
    print("--- Testing Training Flow for RS ---")
    region = "RS"
    level = "city"
    test_year = 2024
    
    # 1. Prepare
    df_all = predictive_model.prepare_lagged_data(region, level=level)
    print(f"Data prepared: {len(df_all)} rows.")
    
    # 2. Train
    models, metrics, feature_cols = predictive_model.train_predictive_models(df_all, test_year=test_year)
    
    print(f"Training complete. Models found for: {list(models.keys())}")
    for target, model in models.items():
        if model:
            print(f"  {target}: TRAINED. Metrics: {metrics[target]}")
        else:
            print(f"  {target}: NOT TRAINED (Insufficient data)")

if __name__ == "__main__":
    test_training_flow()
