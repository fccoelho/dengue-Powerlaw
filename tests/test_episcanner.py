import os
import pandas as pd
from fitpl import fetch_episcanner
import dotenv

dotenv.load_dotenv()

def test_fetch_episcanner_success():
    """Test that fetch_episcanner returns a valid DataFrame for RS 2024."""
    df = fetch_episcanner(state="RS", year=2024)
    assert df is not None
    assert not df.empty
    assert isinstance(df, pd.DataFrame)
    # Basic columns check if possible, or just head check
    assert len(df) > 0
