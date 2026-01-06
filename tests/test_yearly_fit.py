import asyncio
import sqlite3
import os
import sys
import pytest

# Add parent directory to path to allow importing fitpl
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fitpl import FitPL

@pytest.mark.anyio
async def test_yearly_fit():
    db_path = "test_powerlaw_results_yearly.db"
    # Remove existing test DB if any
    if os.path.exists(db_path):
        os.remove(db_path)
    
    fit_pl = FitPL(db_path=db_path)
    
    # Test with a subset of cities
    test_geocodes = {
        3304557: "Rio de Janeiro",
        3550308: "São Paulo"
    }
    
    print("Starting yearly scan for test cities...")
    await fit_pl.run_yearly_scan(geocodes=test_geocodes, max_workers=2)
    print("Yearly scan completed.")
    
    # Verify results in DB
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM powerlaw_fits_yearly")
            rows = cursor.fetchall()
            
            print(f"\nFound {len(rows)} rows in powerlaw_fits_yearly:")
            for row in rows:
                print(row)
                
            assert len(rows) > 0, "No data found in powerlaw_fits_yearly"
    finally:
        # Clean up test database
        if os.path.exists(db_path):
            os.remove(db_path)
