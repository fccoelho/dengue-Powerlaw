import os
import pandas as pd
from fitpl import fetch_episcanner
import dotenv

dotenv.load_dotenv()

# Assuming RS 2024 as a trial
df = fetch_episcanner(state="RS", year=2024)
if df is not None:
    print(df.columns.tolist())
    print(df.head())
else:
    print("Failed to fetch episcanner data.")
