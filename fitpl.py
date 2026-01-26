import pandas as pd
import numpy as np
from datetime import date
import os
import traceback
from functools import lru_cache
import dotenv
from mosqlient.datastore import Infodengue
from mosqlient import get_episcanner
from _mun_by_geocode import NAME_BY_GEOCODE
import powerlaw
import sqlite3
import asyncio
from concurrent.futures import ThreadPoolExecutor

dotenv.load_dotenv()

GEO_STATE_MAP = {
    11: "RO", 12: "AC", 13: "AM", 14: "RR", 15: "PA", 16: "AP", 17: "TO",
    21: "MA", 22: "PI", 23: "CE", 24: "RN", 25: "PB", 26: "PE", 27: "AL",
    28: "SE", 29: "BA", 31: "MG", 32: "ES", 33: "RJ", 35: "SP", 41: "PR",
    42: "SC", 43: "RS", 50: "MS", 51: "MT", 52: "GO", 53: "DF"
}

def fetch_infodengue(geocode, start_date="2010-01-01", end_date=None, disease="dengue", force_download=False):
    if end_date is None:
        end_date = date.today().strftime("%Y-%m-%d")

    file_path = f"data/{geocode}.parquet"
    
    if not force_download and os.path.exists(file_path):
        df = pd.read_parquet(file_path)
        # print(f"Loaded cached data for {geocode}")
        return df.reset_index()
    
    try:
        # Determine if geocode is a state abbreviation or a numeric geocode
        uf_val = None
        geocode_val = None
        if isinstance(geocode, str) and len(geocode) == 2 and geocode.isalpha():
            uf_val = geocode.upper()
        else:
            geocode_val = int(geocode)

        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        if (end_dt - start_dt).days > 365:
            # print(f"Downloading data for {uf_val or geocode_val} in yearly chunks...")
            dfs = []
            for year in range(start_dt.year, end_dt.year + 1):
                print(f"Downloading data for {uf_val or geocode_val} in {year}...")
                y_start = max(start_dt, pd.to_datetime(f"{year}-01-01")).strftime("%Y-%m-%d")
                y_end = min(end_dt, pd.to_datetime(f"{year}-12-31")).strftime("%Y-%m-%d")
                
                df_year = Infodengue.get(disease=disease,
                                    start=y_start,
                                    end=y_end, 
                                    uf=uf_val,
                                    geocode=geocode_val,
                                    api_key=os.getenv("MOSQLIMATE_API_KEY"))
                if df_year:
                    dfs.append(pd.DataFrame(df_year))
            
            if not dfs:
                return None
            df = pd.concat(dfs).drop_duplicates()
        else:
            # print(f"Downloading data for {uf_val or geocode_val}...")
            df = Infodengue.get(disease=disease,
                                start=start_date,
                                end=end_date, 
                                uf=uf_val,
                                geocode=geocode_val,
                                api_key=os.getenv("MOSQLIMATE_API_KEY"))
            df = pd.DataFrame(df)
        if df.empty:
            return None
        
        df['data_iniSE'] = pd.to_datetime(df['data_iniSE'])
        df.set_index('data_iniSE', inplace=True)
        
        # Numeric columns to sum
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df = df[numeric_cols].resample('W-SUN').sum()
        
        # Ensure 'SE' is present if it was in original columns, otherwise re-derive from index if possible
        # Actually SE is yyyyww. Let's re-derive year and EW from index.
        df['year'] = df.index.isocalendar().year
        df['EW'] = df.index.isocalendar().week
        
        os.makedirs("data", exist_ok=True)
        df.to_parquet(file_path)
        return df.reset_index()
    except Exception as e:
        print(f"Error fetching data for {geocode}: {traceback.print_exc()}")
        return None

def fetch_episcanner(disease: str="dengue", state: str="RS", year: int=2024):
    """
    Fetches episcanner data from the Mosqlimate API for a given state and year.

    Args:
        disease (str): The disease to fetch data for.
        state (str): The state to fetch data for.
        year (int): The year to fetch data for.
    
    Returns:
        pd.DataFrame: The fetched data.
    """
    file_path = f"data/episcanner_{state}_{year}.parquet"
    
    if os.path.exists(file_path):
        df = pd.read_parquet(file_path)
        return df
    
    try:
        df = get_episcanner(disease=disease, uf=state, year=year, api_key=os.getenv("MOSQLIMATE_API_KEY"))
        df = pd.DataFrame(df)
        if df.empty:
            return None
        
        os.makedirs("data", exist_ok=True)
        df.to_parquet(file_path)
        return df
    except Exception as e:
        # Check if it's a 404 error from requests (mosqlient usually uses requests)
        if "404" in str(e):
            # print(f"No Episcanner data for {state} in {year} (404)")
            return None
        print(f"Error fetching data for {state}: {e}")
        return None

class FitPL:
    def __init__(self, db_path="powerlaw_results.db"):
        self.start_date = "2010-01-01"
        self.end_date = date.today().strftime("%Y-%m-%d")
        self.disease = "dengue"
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # Add start_date and end_date columns
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS powerlaw_fits (
                    geocode INTEGER PRIMARY KEY,
                    city_name TEXT,
                    alpha REAL,
                    xmin REAL,
                    xmax REAL,
                    R REAL,
                    p REAL,
                    start_date TEXT,
                    end_date TEXT
                )
            ''')
            # Create yearly fits table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS powerlaw_fits_yearly (
                    geocode INTEGER,
                    city_name TEXT,
                    year INTEGER,
                    alpha REAL,
                    xmin REAL,
                    xmax REAL,
                    R REAL,
                    p REAL,
                    PRIMARY KEY (geocode, year)
                )
            ''')
            # Check if columns exist (for migration if table already exists)
            cursor.execute("PRAGMA table_info(powerlaw_fits)")
            columns = [info[1] for info in cursor.fetchall()]
            if 'start_date' not in columns:
                cursor.execute("ALTER TABLE powerlaw_fits ADD COLUMN start_date TEXT")
            if 'end_date' not in columns:
                cursor.execute("ALTER TABLE powerlaw_fits ADD COLUMN end_date TEXT")
            conn.commit()

    def fit_pl(self, df):
        try:
            data = df.casos_est.values
            data = data[data > 0]
            if len(data) < 10:
                return None
                
            fit = powerlaw.Fit(data, verbose=False, discrete=True)
            
            alpha = fit.power_law.alpha
            xmin = fit.power_law.xmin
            xmax = fit.power_law.xmax
            
            R, p = fit.distribution_compare('power_law', 'exponential', normalized_ratio=True)
            
            return alpha, xmin, xmax, R, p
        except Exception as e:
            return None

    def save_to_db(self, geocode, city_name, results):
        if not results:
            return
        
        alpha, xmin, xmax, R, p = results
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO powerlaw_fits (geocode, city_name, alpha, xmin, xmax, R, p, start_date, end_date)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (geocode, city_name, alpha, xmin, xmax, R, p, self.start_date, self.end_date))
                conn.commit()
        except Exception as e:
            print(f"Error saving to DB for {city_name}: {e}")

    def save_yearly_to_db(self, geocode, city_name, year, results):
        if not results:
            return
        
        alpha, xmin, xmax, R, p = results
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO powerlaw_fits_yearly (geocode, city_name, year, alpha, xmin, xmax, R, p)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (geocode, city_name, year, alpha, xmin, xmax, R, p))
                conn.commit()
        except Exception as e:
            print(f"Error saving yearly results to DB for {city_name} in {year}: {e}")

    async def process_city(self, geocode, city_name, executor, force_download=False):
        loop = asyncio.get_running_loop()
        try:
            df = await loop.run_in_executor(executor, fetch_infodengue, geocode, self.start_date, self.end_date, self.disease, force_download)
            
            if df is not None and not df.empty:
                results = await loop.run_in_executor(executor, self.fit_pl, df)
                
                if results:
                    self.save_to_db(geocode, city_name, results)
        except Exception as e:
            print(f"Failed to process {city_name} ({geocode}): {e}")

    async def process_city_yearly(self, geocode, city_name, executor, force_download=False):
        loop = asyncio.get_running_loop()
        try:
            df = await loop.run_in_executor(executor, fetch_infodengue, geocode, self.start_date, self.end_date, self.disease, force_download)
            
            if df is not None and not df.empty:
                # Group by year and fit power law for each year
                for year, year_df in df.groupby('year'):
                    results = await loop.run_in_executor(executor, self.fit_pl, year_df)
                    if results:
                        self.save_yearly_to_db(geocode, city_name, year, results)
        except Exception as e:
            print(f"Failed to process yearly data for {city_name} ({geocode}): {e}")

    async def _update_state_cache(self, executor, force_download=False):
        loop = asyncio.get_running_loop()
        print("--- Downloading Statewide Aggregated Data ---")
        state_tasks = []
        for state_uf in GEO_STATE_MAP.values():
            # Use fetch_infodengue directly in executor for caching
            task = loop.run_in_executor(executor, fetch_infodengue, state_uf, self.start_date, self.end_date, self.disease, force_download)
            state_tasks.append(task)
        
        if state_tasks:
            await asyncio.gather(*state_tasks)

    async def run_scan(self, geocodes=None, force_download=False, max_workers=5):
        if geocodes is None:
            geocodes = NAME_BY_GEOCODE
            
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            loop = asyncio.get_running_loop()
            tasks = []
            
            for geocode, city_name in geocodes.items():
                task = self.process_city(geocode, city_name, executor, force_download)
                tasks.append(task)
                
            await asyncio.gather(*tasks)

            # Process states
            await self._update_state_cache(executor, force_download)

    async def run_yearly_scan(self, geocodes=None, force_download=False, max_workers=5):
        if geocodes is None:
            geocodes = NAME_BY_GEOCODE
            
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            loop = asyncio.get_running_loop()
            tasks = []
            
            for geocode, city_name in geocodes.items():
                task = self.process_city_yearly(geocode, city_name, executor, force_download)
                tasks.append(task)
                
            await asyncio.gather(*tasks)

            # Process states
            await self._update_state_cache(executor, force_download)

if __name__ == "__main__":
    asyncio.run(FitPL().run_scan(max_workers=1))
    # asyncio.run(FitPL().run_yearly_scan(max_workers=10))