import asyncio
import sqlite3
import os
from fitpl import FitPL

async def verify():
    print("Starting verification...")
    
    # Create specific test DB
    db_path = "test_powerlaw.db"
    if os.path.exists(db_path):
        os.remove(db_path)
        
    fpl = FitPL(db_path=db_path)
    
    # Select a few geocodes to test
    # 3304557 is Rio de Janeiro
    # 3550308 is São Paulo
    # 3106200 is Belo Horizonte
    test_geocodes = {
        3304557: "Rio de Janeiro",
        3550308: "São Paulo",
        3106200: "Belo Horizonte"
    }
    
    print(f"Running scan for {len(test_geocodes)} cities with force_download=True...")
    await fpl.run_scan(test_geocodes, force_download=True)
    
    print("Scan complete. Checking database...")
    
    if not os.path.exists(db_path):
        print("FAIL: Database file not created.")
        return
        
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
            
        if len(rows) > 0:
            print("SUCCESS: Database populated.")
        else:
            print("WARNING: Database created but empty (might be due to empty data or filter).")

if __name__ == "__main__":
    asyncio.run(verify())
