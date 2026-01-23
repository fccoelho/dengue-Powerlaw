import os
import pytest
import pandas as pd
from fitpl import fetch_episcanner
import dotenv

dotenv.load_dotenv()

@pytest.mark.parametrize("year", [2023, 2024])
def test_fetch_episcanner_years(year):
    """Test fetch_episcanner for multiple years in SP."""
    state = "SP"
    df = fetch_episcanner(state=state, year=year)
    # We expect data to be found for these years based on previous manual check
    assert df is not None
    assert not df.empty
