import dotenv
import os
from fitpl import fetch_episcanner

dotenv.load_dotenv()

state = "SP"
years_found = []
for year in range(2010, 2026):
    df = fetch_episcanner(state=state, year=year)
    if df is not None and not df.empty:
        print(f"Year {year}: FOUND")
        years_found.append(year)
    else:
        print(f"Year {year}: NOT FOUND")

print(f"Available years: {years_found}")
