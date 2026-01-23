import pandas as pd
from fitpl import fetch_infodengue
import pytest
import os
import dotenv

dotenv.load_dotenv()

def test_fetch_infodengue_state_rj():
    """Test fetching state-level data for RJ."""
    df = fetch_infodengue("RJ")
    assert df is not None
    assert not df.empty
    assert 'casos_est' in df.columns
    # Verify it has some entries
    assert len(df) > 0
