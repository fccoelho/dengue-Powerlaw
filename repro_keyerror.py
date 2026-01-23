
import pandas as pd
import os
from fitpl import fetch_infodengue

def test_fetch_infodengue():
    geocode = 3304557  # Rio de Janeiro city
    state = "RJ"
    
    print(f"--- Testing geocode {geocode} ---")
    # Fresh fetch (force download)
    df_fresh = fetch_infodengue(geocode, force_download=True)
    if df_fresh is not None:
        print(f"Fresh fetch columns: {df_fresh.columns.tolist()}")
        print(f"Fresh fetch index name: {df_fresh.index.name}")
    
    # Cached fetch
    df_cached = fetch_infodengue(geocode)
    if df_cached is not None:
        print(f"Cached fetch columns: {df_cached.columns.tolist()}")
        print(f"Cached fetch index name: {df_cached.index.name}")
        
    print(f"\n--- Testing state {state} ---")
    # Fresh fetch (force download)
    df_fresh_state = fetch_infodengue(state, force_download=True)
    if df_fresh_state is not None:
        print(f"Fresh state fetch columns: {df_fresh_state.columns.tolist()}")
        print(f"Fresh state fetch index name: {df_fresh_state.index.name}")
        
    # Cached fetch
    df_cached_state = fetch_infodengue(state)
    if df_cached_state is not None:
        print(f"Cached state fetch columns: {df_cached_state.columns.tolist()}")
        print(f"Cached state fetch index name: {df_cached_state.index.name}")

if __name__ == "__main__":
    # Ensure we don't accidentally download too much if the API is slow, 
    # but we need to see the behavior.
    test_fetch_infodengue()
