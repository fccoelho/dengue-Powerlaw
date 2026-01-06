import asyncio
import sqlite3
import os
import sys
import pytest

# Add parent directory to path to allow importing fitpl
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fitpl import FitPL

@pytest.mark.anyio
async def test_fit_pl():
    print("Starting verification...")
    
    # Create specific test DB
    db_path = "test_powerlaw.db"
    if os.path.exists(db_path):
        os.remove(db_path)
        
    fpl = FitPL(db_path=db_path)
    
    # Select a few geocodes to test
    test_geocodes = {
        3304557: "Rio de Janeiro",
        3550308: "São Paulo",
        3106200: "Belo Horizonte"
    }
    
    try:
        print(f"Running scan for {len(test_geocodes)} cities with force_download=True...")
        await fpl.run_scan(test_geocodes, force_download=True)
        
        print("Scan complete. Checking database...")
        
        assert os.path.exists(db_path), "Database file not created"
            
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM powerlaw_fits")
            rows = cursor.fetchall()
            
            # Get column names
            cursor.execute("PRAGMA table_info(powerlaw_fits)")
            columns = [description[1] for description in cursor.fetchall()]
            print(f"Columns: {columns}")
            
            print(f"Found {len(rows)} rows in database.")
            for row in rows:
                print(row)
                
            assert len(rows) > 0, "Database created but empty"
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)
