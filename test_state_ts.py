from fitpl import fetch_infodengue
import pandas as pd

# Test for RJ (33)
df = fetch_infodengue(33)
if df is not None:
    print("Columns:", df.columns.tolist())
    print("Head:\n", df.head())
    print("Empty:", df.empty)
else:
    print("DF is None")
