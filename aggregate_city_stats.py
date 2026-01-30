import duckdb
import sqlite3
import pandas as pd
import os
import glob

def aggregate_stats(data_dir="data", db_path="powerlaw_results.db"):
    print(f"Aggregating city stats from {data_dir}...")
    
    # 1. Aggregate Total Cases from Episcanner files
    # These files have 'geocode' and 'total_cases' columns.
    episcanner_pattern = os.path.join(data_dir, "episcanner_*.parquet")
    print(f"Reading Episcanner files: {episcanner_pattern}")
    
    con = duckdb.connect()
    
    try:
        df_epi = con.execute(f"""
            SELECT geocode, SUM(total_cases) as total_cases
            FROM read_parquet('{episcanner_pattern}')
            GROUP BY geocode
        """).df()
        print(f"Found {len(df_epi)} cities in Episcanner data.")
    except Exception as e:
        print(f"Error reading Episcanner files: {e}")
        df_epi = pd.DataFrame(columns=['geocode', 'total_cases'])

    # 2. Extract Population from Infodengue files
    # We only need it for cities we have alpha or total_cases for.
    # But let's try to get it for all numeric parquet files.
    # Infodengue files are named <geocode>.parquet
    infodengue_files = glob.glob(os.path.join(data_dir, "[0-9]*.parquet"))
    print(f"Found {len(infodengue_files)} Infodengue cache files.")
    
    # We can process them in one go if they all have the same schema
    try:
        # Note: mapping geocode from filename if not in columns, 
        # but earlier we saw 'municipio_geocodigo' or 'geocode'?
        # Let's check columns again. 
        # In previously checked file 1100015.parquet, columns were:
        # ['SE', 'casos_est', ..., 'municipio_geocodigo', ..., 'pop', ...]
        
        # DuckDB can read a list of files.
        # We use filename as geocode if municipio_geocodigo is missing, 
        # but it should be there.
        
        # Since reading all 5500 files might be memory intensive even for duckdb 
        # if not careful, but let's try.
        # We only need the latest population.
        df_pop = con.execute(f"""
            SELECT municipio_geocodigo as geocode, arg_max(pop, data_iniSE) as population
            FROM read_parquet('data/[0-9]*.parquet')
            GROUP BY municipio_geocodigo
        """).df()
        print(f"Found population for {len(df_pop)} cities.")
    except Exception as e:
        print(f"Error reading Infodengue files with DuckDB: {e}")
        # Fallback to a slower method or handle error
        df_pop = pd.DataFrame(columns=['geocode', 'population'])

    # 3. Merge and Save to SQLite
    df_stats = pd.merge(df_epi, df_pop, on='geocode', how='outer')
    
    print(f"Total cities with stats: {len(df_stats)}")
    
    with sqlite3.connect(db_path) as conn:
        # Create table
        cursor = conn.cursor()
        cursor.execute("DROP TABLE IF EXISTS city_stats")
        cursor.execute("""
            CREATE TABLE city_stats (
                geocode INTEGER PRIMARY KEY,
                total_cases REAL,
                population REAL,
                incidence REAL
            )
        """)
        
        # Calculate incidence (cases per 100k)
        df_stats['incidence'] = (df_stats['total_cases'] / df_stats['population']) * 100000
        
        df_stats.to_sql("city_stats", conn, if_exists="append", index=False)
        conn.commit()
    
    print("Done! city_stats table updated.")

if __name__ == "__main__":
    aggregate_stats()
