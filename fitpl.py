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
import duckdb

dotenv.load_dotenv()


"""
Power law fitting module for epidemic case data.

Database Tables
---------------
This module uses SQLite to store power law fit parameters. The tables are:

**Weekly Cases (from Infodengue):**
- ``powerlaw_fits``: Stores power law fits over weekly estimated cases (casos_est)
  across the full date range for each city. Fitted by ``fit_pl()`` via ``run_scan()``.
- ``powerlaw_fits_yearly``: Stores per-year power law fits over weekly cases.
  Fitted by ``fit_pl()`` via ``process_city_yearly()`` and ``run_yearly_scan()``.

**Total Annual Cases (from Episcanner):**
- ``powerlaw_fits_total_cases``: Stores city-level fits over total_cases across
  all years. Fitted by ``fit_pl_total_cases()`` via ``run_total_cases_scan()``.
- ``episcanner_state_fits``: Stores state-level fits over total_cases with
  metric='total_cases'. Fitted by ``fit_pl_total_cases()`` via ``run_total_cases_scan()``.
- ``episcanner_fits``: Stores national-level fits over total_cases with
  metric='total_cases'. Fitted by ``fit_pl_total_cases()`` via ``run_total_cases_scan()``.

Functions
---------
- ``fetch_infodengue()``: Downloads weekly case data from Infodengue API
- ``fetch_episcanner()``: Downloads annual episcanner data from Mosqlimate API
- ``FitPL`` class: Main class for fitting power laws and managing database storage
  - ``fit_pl()``: Fit power law over weekly cases
  - ``fit_pl_total_cases()``: Fit power law over total annual cases
  - ``run_scan()``: Fit weekly cases for multiple cities
  - ``run_yearly_scan()``: Fit weekly cases per year for multiple cities
  - ``run_total_cases_scan()``: Fit total annual cases (city/state/national levels)
"""

GEO_STATE_MAP = {
    11: "RO",
    12: "AC",
    13: "AM",
    14: "RR",
    15: "PA",
    16: "AP",
    17: "TO",
    21: "MA",
    22: "PI",
    23: "CE",
    24: "RN",
    25: "PB",
    26: "PE",
    27: "AL",
    28: "SE",
    29: "BA",
    31: "MG",
    32: "ES",
    33: "RJ",
    35: "SP",
    41: "PR",
    42: "SC",
    43: "RS",
    50: "MS",
    51: "MT",
    52: "GO",
    53: "DF",
}

STATE_TO_GEO = {v: k for k, v in GEO_STATE_MAP.items()}

STATES = list(GEO_STATE_MAP.values())


def fetch_infodengue(
    geocode,
    start_date="2010-01-01",
    end_date=None,
    disease="dengue",
    force_download=False,
):
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
                y_start = max(start_dt, pd.to_datetime(f"{year}-01-01")).strftime(
                    "%Y-%m-%d"
                )
                y_end = min(end_dt, pd.to_datetime(f"{year}-12-31")).strftime(
                    "%Y-%m-%d"
                )

                df_year = Infodengue.get(
                    disease=disease,
                    start=y_start,
                    end=y_end,
                    uf=uf_val,
                    geocode=geocode_val,
                    api_key=os.getenv("MOSQLIMATE_API_KEY"),
                )
                if df_year:
                    dfs.append(pd.DataFrame(df_year))

            if not dfs:
                return None
            df = pd.concat(dfs).drop_duplicates()
        else:
            # print(f"Downloading data for {uf_val or geocode_val}...")
            df = Infodengue.get(
                disease=disease,
                start=start_date,
                end=end_date,
                uf=uf_val,
                geocode=geocode_val,
                api_key=os.getenv("MOSQLIMATE_API_KEY"),
            )
            df = pd.DataFrame(df)
        if df.empty:
            return None

        df["data_iniSE"] = pd.to_datetime(df["data_iniSE"])
        df.set_index("data_iniSE", inplace=True)

        # Numeric columns to sum
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df = df[numeric_cols].resample("W-SUN").sum()

        # Ensure 'SE' is present if it was in original columns, otherwise re-derive from index if possible
        # Actually SE is yyyyww. Let's re-derive year and EW from index.
        df["year"] = df.index.isocalendar().year
        df["EW"] = df.index.isocalendar().week

        os.makedirs("data", exist_ok=True)
        df.to_parquet(file_path)
        return df.reset_index()
    except Exception as e:
        print(f"Error fetching data for {geocode}: {traceback.print_exc()}")
        return None


def fetch_episcanner(disease: str = "dengue", state: str = "RS", year: int = 2024):
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
        df = get_episcanner(
            disease=disease,
            uf=state,
            year=year,
            api_key=os.getenv("MOSQLIMATE_API_KEY"),
        )
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
            cursor.execute("""
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
            """)
            cursor.execute("""
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
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS powerlaw_fits_total_cases (
                    geocode INTEGER PRIMARY KEY,
                    city_name TEXT,
                    alpha REAL,
                    xmin REAL,
                    xmax REAL,
                    R REAL,
                    p REAL
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS episcanner_fits (
                    metric TEXT,
                    year INTEGER,
                    alpha REAL,
                    xmin REAL,
                    xmax REAL,
                    R REAL,
                    p REAL,
                    PRIMARY KEY (metric, year)
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS episcanner_state_fits (
                    state TEXT,
                    metric TEXT,
                    year INTEGER,
                    alpha REAL,
                    xmin REAL,
                    xmax REAL,
                    R REAL,
                    p REAL,Alpha Parameter
                    PRIMARY KEY (state, metric, year)
                )
            """)
            cursor.execute("PRAGMA table_info(powerlaw_fits)")
            columns = [info[1] for info in cursor.fetchall()]
            if "start_date" not in columns:
                cursor.execute("ALTER TABLE powerlaw_fits ADD COLUMN start_date TEXT")
            if "end_date" not in columns:
                cursor.execute("ALTER TABLE powerlaw_fits ADD COLUMN end_date TEXT")
            conn.commit()

    def fit_pl(self, df):
        """
        Fit a power law over weekly estimated cases.

        This method fits a discrete power law to the distribution of weekly
        estimated cases (casos_est) from Infodengue data. The data is resampled
        at weekly intervals (W-SUN) in the fetch_infodengue function.
        """
        try:
            data = df.casos_est.values
            data = data[data > 0]
            if len(data) < 10:
                return None

            fit = powerlaw.Fit(data, verbose=False, discrete=True)

            alpha = fit.power_law.alpha
            xmin = fit.power_law.xmin
            xmax = fit.power_law.xmax

            R, p = fit.distribution_compare(
                "power_law", "exponential", normalized_ratio=True
            )

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
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO powerlaw_fits (geocode, city_name, alpha, xmin, xmax, R, p, start_date, end_date)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        geocode,
                        city_name,
                        alpha,
                        xmin,
                        xmax,
                        R,
                        p,
                        self.start_date,
                        self.end_date,
                    ),
                )
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
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO powerlaw_fits_yearly (geocode, city_name, year, alpha, xmin, xmax, R, p)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (geocode, city_name, year, alpha, xmin, xmax, R, p),
                )
                conn.commit()
        except Exception as e:
            print(f"Error saving yearly results to DB for {city_name} in {year}: {e}")

    def fit_pl_total_cases(self, data):
        """
        Fit a power law distribution over total_cases data.

        This method fits a discrete power law to the distribution of total
        epidemic cases across observations (years or cities, depending on
        aggregation level).

        Parameters
        ----------
        data : array-like
            Array of total_cases values. Values <= 0 are filtered out
            before fitting.

        Returns
        -------
        tuple or None
            A tuple containing (alpha, xmin, xmax, R, p) where:
            - alpha: Power law scaling exponent
            - xmin: Minimum value of the power law tail
            - xmax: Maximum value in the fitted range
            - R: Log-likelihood ratio between power law and exponential
            - p: p-value for the comparison test

            Returns None if:
            - Less than 10 positive values available
            - Fitting procedure fails

        Notes
        -----
        Uses the `powerlaw` package with discrete=True since total_cases
        are count data. The power law is compared against an exponential
        distribution to assess goodness of fit.
        """
        try:
            data = np.array(data)
            data = data[data > 0]
            if len(data) < 10:
                return None

            fit = powerlaw.Fit(data, verbose=False, discrete=True)

            alpha = fit.power_law.alpha
            xmin = fit.power_law.xmin
            xmax = fit.power_law.xmax

            R, p = fit.distribution_compare(
                "power_law", "exponential", normalized_ratio=True
            )

            return alpha, xmin, xmax, R, p
        except Exception as e:
            return None

    def save_total_cases_city_to_db(self, geocode, city_name, results):
        """
        Save city-level total_cases power law fit results to the database.

        This method stores the power law fit parameters for the distribution
        of total_cases across all years for a single city.

        Parameters
        ----------
        geocode : int
            IBGE municipality code (7 digits)
        city_name : str
            Human-readable city name
        results : tuple
            Fit results tuple (alpha, xmin, xmax, R, p) from fit_pl_total_cases

        Returns
        -------
        None

        Notes
        -----
        Uses INSERT OR REPLACE to handle updates for existing entries.
        Results are stored in the 'powerlaw_fits_total_cases' table.
        """
        if not results:
            return

        alpha, xmin, xmax, R, p = results

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO powerlaw_fits_total_cases (geocode, city_name, alpha, xmin, xmax, R, p)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                    (geocode, city_name, alpha, xmin, xmax, R, p),
                )
                conn.commit()
        except Exception as e:
            print(f"Error saving total_cases city results to DB for {city_name}: {e}")

    def save_total_cases_state_to_db(self, state, year, results):
        """
        Save state-level total_cases power law fit results to the database.

        This method stores the power law fit parameters for the distribution
        of total_cases across all cities in a state for a given year.

        Parameters
        ----------
        state : str
            Two-letter state abbreviation (e.g., 'RS', 'SP')
        year : int
            Year of the fit. Use 0 for combined/all-years fits.
        results : tuple
            Fit results tuple (alpha, xmin, xmax, R, p) from fit_pl_total_cases

        Returns
        -------
        None

        Notes
        -----
        Results are stored in 'episcanner_state_fits' table with metric='total_cases'.
        Uses INSERT OR REPLACE to handle updates for existing entries.
        """
        if not results:
            return

        alpha, xmin, xmax, R, p = results

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO episcanner_state_fits (state, metric, year, alpha, xmin, xmax, R, p)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (state, "total_cases", year, alpha, xmin, xmax, R, p),
                )
                conn.commit()
        except Exception as e:
            print(
                f"Error saving total_cases state results to DB for {state} in {year}: {e}"
            )

    def save_total_cases_national_to_db(self, year, results):
        """
        Save national-level (Brazil) total_cases power law fit results.

        This method stores the power law fit parameters for the distribution
        of total_cases across all cities in Brazil for a given year.

        Parameters
        ----------
        year : int
            Year of the fit. Use 0 for combined/all-years fits.
        results : tuple
            Fit results tuple (alpha, xmin, xmax, R, p) from fit_pl_total_cases

        Returns
        -------
        None

        Notes
        -----
        Results are stored in 'episcanner_fits' table with metric='total_cases'.
        Uses INSERT OR REPLACE to handle updates for existing entries.
        """
        if not results:
            return

        alpha, xmin, xmax, R, p = results

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO episcanner_fits (metric, year, alpha, xmin, xmax, R, p)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                    ("total_cases", year, alpha, xmin, xmax, R, p),
                )
                conn.commit()
        except Exception as e:
            print(
                f"Error saving total_cases national results to DB for year {year}: {e}"
            )

    def run_total_cases_scan(self, states=None, years=None):
        """
        Scan all episcanner parquet files and fit power laws over total_cases.

        This method performs multiple types of power law fits over the
        distribution of epidemic total cases:

        1. **City-level (all years combined)**: For each city, collects
           total_cases across all available years and fits a power law
           over the distribution. This reveals how epidemic sizes are
           distributed for each city over time.

        2. **State-level by year**: For each state-year combination, fits
           a power law over the distribution of total_cases across all
           cities in that state. This reveals spatial heterogeneity in
           epidemic sizes for each year.

        3. **State-level combined (year=0)**: For each state, combines
           all years and fits over the full distribution of total_cases.

        4. **National-level by year**: Fits power law over total_cases
           distribution across all Brazilian cities for each year.

        5. **National-level combined (year=0)**: Fits power law over
           total_cases distribution across all cities and all years.

        Parameters
        ----------
        states : list of str, optional
            List of state abbreviations to process (e.g., ['RS', 'SP']).
            If None, processes all 27 Brazilian states.
        years : list of int, optional
            List of years to include in the analysis.
            If None, uses range(2011, current_year+1).

        Returns
        -------
        None
            Results are saved directly to the SQLite database.

        Raises
        ------
        sqlite3.Error
            If database operations fail.

        Notes
        -----
        - Uses DuckDB for efficient aggregation of parquet files
        - Requires episcanner parquet files to exist in the 'data/' directory
        - Files should follow the naming convention: episcanner_{STATE}_{YEAR}.parquet
        - Progress is printed to stdout during execution
        - Cities with fewer than 10 years of positive total_cases are skipped
        - State/year combinations with fewer than 10 cities are skipped

        Examples
        --------
        >>> fitter = FitPL()
        >>> fitter.run_total_cases_scan()  # Process all states and years
        >>> fitter.run_total_cases_scan(states=['RS', 'SP'], years=[2020, 2021, 2022])
        """
        if states is None:
            states = STATES
        if years is None:
            years = list(range(2011, date.today().year + 1))

        min_year = min(years)
        max_year = max(years)

        print("=" * 60)
        print("Power Law Fitting over Total Cases (from Episcanner data)")
        print("=" * 60)

        con = duckdb.connect()

        try:
            print("\nLoading episcanner parquet files...")
            all_data = con.query(f"""
                SELECT geocode, muni_name, year, total_cases
                FROM read_parquet('data/episcanner_*.parquet')
                WHERE year BETWEEN {min_year} AND {max_year}
            """).df()

            if all_data.empty:
                print("No episcanner data found. Ensure parquet files exist in data/")
                return

            print(
                f"Loaded {len(all_data)} records spanning {all_data['year'].min()}-{all_data['year'].max()}"
            )

            # 1. City-level fits (all years per city)
            print("\n--- Fitting City-level Total Cases (all years combined) ---")
            city_count = 0
            for geocode, group in all_data.groupby("geocode"):
                city_name = group["muni_name"].iloc[0]
                results = self.fit_pl_total_cases(group["total_cases"].values)
                if results:
                    self.save_total_cases_city_to_db(int(geocode), city_name, results)
                    city_count += 1
            print(f"Successfully fitted {city_count} cities")

            # 2. State-level fits
            print("\n--- Fitting State-level Total Cases ---")
            for state in states:
                state_code = STATE_TO_GEO.get(state)
                if state_code is None:
                    continue

                min_geo = state_code * 100000
                max_geo = (state_code + 1) * 100000
                state_data = all_data[
                    (all_data["geocode"] >= min_geo) & (all_data["geocode"] < max_geo)
                ]

                if state_data.empty:
                    continue

                # By year
                for year in years:
                    year_data = state_data[state_data["year"] == year]
                    if len(year_data) >= 10:
                        results = self.fit_pl_total_cases(
                            year_data["total_cases"].values
                        )
                        if results:
                            self.save_total_cases_state_to_db(state, year, results)

                # Combined (year=0)
                results = self.fit_pl_total_cases(state_data["total_cases"].values)
                if results:
                    self.save_total_cases_state_to_db(state, 0, results)
                    alpha = results[0]
                    print(f"  {state} combined: alpha={alpha:.2f}")

            # 3. National-level fits
            print("\n--- Fitting National-level Total Cases ---")
            for year in years:
                year_data = all_data[all_data["year"] == year]
                if len(year_data) >= 10:
                    results = self.fit_pl_total_cases(year_data["total_cases"].values)
                    if results:
                        self.save_total_cases_national_to_db(year, results)

            # National combined (year=0)
            results = self.fit_pl_total_cases(all_data["total_cases"].values)
            if results:
                self.save_total_cases_national_to_db(0, results)
                alpha = results[0]
                print(f"  Brazil combined: alpha={alpha:.2f}")

            print("\n" + "=" * 60)
            print("Total cases power law fitting complete!")
            print("=" * 60)

        except Exception as e:
            print(f"Error during total_cases scan: {e}")
            traceback.print_exc()
        finally:
            con.close()

    async def process_city(self, geocode, city_name, executor, force_download=False):
        """
        Process a single city to fit a power law over weekly cases.

        Fetches weekly case data from Infodengue for the city and fits a power
        law distribution over the weekly estimated cases (casos_est).
        """
        loop = asyncio.get_running_loop()
        try:
            df = await loop.run_in_executor(
                executor,
                fetch_infodengue,
                geocode,
                self.start_date,
                self.end_date,
                self.disease,
                force_download,
            )

            if df is not None and not df.empty:
                results = await loop.run_in_executor(executor, self.fit_pl, df)

                if results:
                    self.save_to_db(geocode, city_name, results)
        except Exception as e:
            print(f"Failed to process {city_name} ({geocode}): {e}")

    async def process_city_yearly(
        self, geocode, city_name, executor, force_download=False
    ):
        """
        Process a single city to fit power laws over weekly cases per year.

        Fetches weekly case data from Infodengue for the city and fits separate
        power law distributions for each year's worth of weekly estimated cases.
        """
        loop = asyncio.get_running_loop()
        try:
            df = await loop.run_in_executor(
                executor,
                fetch_infodengue,
                geocode,
                self.start_date,
                self.end_date,
                self.disease,
                force_download,
            )

            if df is not None and not df.empty:
                # Group by year and fit power law for each year
                for year, year_df in df.groupby("year"):
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
            task = loop.run_in_executor(
                executor,
                fetch_infodengue,
                state_uf,
                self.start_date,
                self.end_date,
                self.disease,
                force_download,
            )
            state_tasks.append(task)

        if state_tasks:
            await asyncio.gather(*state_tasks)

    async def run_scan(self, geocodes=None, force_download=False, max_workers=5):
        """
        Run a scan to fit power laws over weekly cases for multiple cities.

        This method fetches weekly case data from Infodengue for each city and
        fits a power law distribution over the weekly estimated cases (casos_est).
        Results are saved to the powerlaw_fits table in the database.
        """
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
        """
        Run a yearly scan to fit power laws over weekly cases for multiple cities.

        This method fetches weekly case data from Infodengue for each city and
        fits separate power law distributions for each year's worth of weekly
        estimated cases (casos_est). Results are saved to the powerlaw_fits_yearly
        table in the database.
        """
        if geocodes is None:
            geocodes = NAME_BY_GEOCODE

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            loop = asyncio.get_running_loop()
            tasks = []

            for geocode, city_name in geocodes.items():
                task = self.process_city_yearly(
                    geocode, city_name, executor, force_download
                )
                tasks.append(task)

            await asyncio.gather(*tasks)

            # Process states
            await self._update_state_cache(executor, force_download)


if __name__ == "__main__":
    # Fit power laws over weekly cases (from Infodengue)
    # asyncio.run(FitPL().run_scan(max_workers=1))
    # asyncio.run(FitPL().run_yearly_scan(max_workers=10))

    # Fit power laws over total_cases (from Episcanner data)
    # City-level: distribution of total_cases across years for each city
    # State-level: distribution of total_cases across cities per year
    # National-level: distribution of total_cases across all cities per year
    FitPL().run_total_cases_scan()
