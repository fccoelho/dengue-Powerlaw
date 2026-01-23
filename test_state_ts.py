from fitpl import fetch_infodengue
import pandas as pd

# Test for RJ ("RJ")
df = fetch_infodengue("RJ")
if df is not None:
    print("Columns:", df.columns.tolist())
    print("Head:\n", df.head())
else:
    print("Failed to fetch")
