"""Regression tests for B1 (year storage class) and B6 (cache key/metadata).

B1: SQLite types per value, so a year bound as bytes lands as a BLOB even in
an INTEGER column and every downstream consumer crashes on sort_values.

B6: the parquet cache key must contain every parameter that changes the API
payload (geocode, disease, window); files carry their request embedded in the
Arrow schema so a mismatched request is warned about instead of silently
served.
"""
import json

import numpy as np
import pandas as pd
import pytest

import fitpl


# ---------------------------------------------------------------------------
# B1: year coercion / INTEGER storage class
# ---------------------------------------------------------------------------

def test_coerce_year_accepts_upstream_dtypes():
    assert fitpl._coerce_year(2023) == 2023
    assert fitpl._coerce_year(np.uint32(2023)) == 2023
    assert fitpl._coerce_year(np.int64(1990)) == 1990
    # the blob shape actually found in powerlaw_fits_yearly
    assert fitpl._coerce_year(b"\xe7\x07\x00\x00") == 2023
    assert fitpl._coerce_year(b"\xe8\x07\x00\x00") == 2024


def test_save_yearly_to_db_stores_integer_storage_class(tmp_path):
    """B1: no matter what dtype arrives, 'year' must be INTEGER storage."""
    pytest.importorskip("powerlaw")
    db_path = str(tmp_path / "test.db")
    fitter = fitpl.FitPL(db_path=db_path)
    results = (2.9, 9.0, None, -0.87, 0.37)  # alpha, xmin, xmax, R, p

    fitter.save_yearly_to_db(3117836, "Cônego Marinho", np.uint32(2023), results)
    fitter.save_yearly_to_db(3152131, "Ponto Chique", b"\xe8\x07\x00\x00", results)

    import sqlite3

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT geocode, typeof(year), year FROM powerlaw_fits_yearly"
        ).fetchall()

    assert rows == [
        (3117836, "integer", 2023),
        (3152131, "integer", 2024),
    ]


# ---------------------------------------------------------------------------
# B6: cache keys
# ---------------------------------------------------------------------------

def test_default_window_keeps_legacy_cache_name(tmp_path, monkeypatch):
    """The historical filename is kept for the historical request semantics."""
    monkeypatch.chdir(tmp_path)
    assert fitpl._cache_path(4300001, "dengue", "2010-01-01", "2026-11-03") == (
        "data/4300001.parquet"
    )


def test_non_default_window_or_disease_gets_parameterised_name(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    p1 = fitpl._cache_path(4300001, "dengue", "2019-01-01", "2020-12-31")
    p2 = fitpl._cache_path(4300001, "chikungunya", "2010-01-01", "2026-11-03")
    # must live outside data/ so the climate glob data/[0-9]*.parquet cannot
    # pick it up alongside the default-window file of the same city
    assert p1.startswith("data/cache/")
    assert "2019-01-01" in p1 and "2020-12-31" in p1
    assert p2.startswith("data/cache/") and "chikungunya" in p2
    assert p1 != p2


# ---------------------------------------------------------------------------
# B6: embedded request metadata
# ---------------------------------------------------------------------------

def _frame():
    return pd.DataFrame(
        {"casos_est": [1.0, 2.0]},
        index=pd.DatetimeIndex(["2020-01-05", "2020-01-12"], name="data_iniSE"),
    )


def test_metadata_roundtrip(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    request = {
        "geocode": 4300001,
        "disease": "dengue",
        "start_date": "2019-01-01",
        "end_date": "2020-12-31",
    }
    path = fitpl._cache_path(4300001, "dengue", "2019-01-01", "2020-12-31")
    fitpl._write_cache_with_metadata(path, _frame(), request)

    out = fitpl._read_cache(path, request)
    assert out is not None
    assert len(out) == 2
    assert str(out["data_iniSE"].iloc[0]).startswith("2020-01-05")


def test_metadata_mismatch_bypasses_cache_with_warning(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = fitpl._cache_path(4300001, "dengue", "2019-01-01", "2020-12-31")
    fitpl._write_cache_with_metadata(
        path,
        _frame(),
        {"geocode": 4300001, "disease": "dengue",
         "start_date": "2019-01-01", "end_date": "2020-12-31"},
    )

    # different window -> must NOT be served silently
    with pytest.warns(UserWarning, match="re-downloading"):
        out = fitpl._read_cache(
            path,
            {"geocode": 4300001, "disease": "dengue",
             "start_date": "2021-01-01", "end_date": "2022-12-31"},
        )
    assert out is None

    # different disease -> same
    with pytest.warns(UserWarning, match="disease"):
        out2 = fitpl._read_cache(
            path,
            {"geocode": 4300001, "disease": "zika",
             "start_date": "2019-01-01", "end_date": "2020-12-31"},
        )
    assert out2 is None


def test_legacy_cache_without_metadata_is_default_window_only(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    legacy = tmp_path / "data" / "4300001.parquet"
    legacy.parent.mkdir()
    _frame().reset_index().to_parquet(legacy)

    # default window: legacy semantics match, file accepted
    out = fitpl._read_cache(
        str(legacy),
        {"geocode": 4300001, "disease": "dengue",
         "start_date": "2010-01-01", "end_date": "2026-11-03"},
    )
    assert out is not None

    # non-default request: an unverifiable legacy file must not be served
    with pytest.warns(UserWarning, match="no request metadata"):
        out2 = fitpl._read_cache(
            str(legacy),
            {"geocode": 4300001, "disease": "chikungunya",
             "start_date": "2019-01-01", "end_date": "2020-12-31"},
        )
    assert out2 is None


def test_fetch_episcanner_nondefault_disease_is_isolated(tmp_path, monkeypatch):
    """A chikungunya request must never touch the dengue cache file."""
    monkeypatch.chdir(tmp_path)
    calls = []

    def fake_get_episcanner(disease, uf, year, api_key=None):
        calls.append(disease)
        return pd.DataFrame(
            {"geocode": [4300001], "total_cases": [100.0], "ep_dur": [8.0]}
        )

    monkeypatch.setattr(fitpl, "get_episcanner", fake_get_episcanner)

    df = fitpl.fetch_episcanner(disease="chikungunya", state="RS", year=2023)
    assert df is not None and len(calls) == 1

    # second call served from the isolated cache, no second API call
    df2 = fitpl.fetch_episcanner(disease="chikungunya", state="RS", year=2023)
    assert df2 is not None and len(calls) == 1
    cached = fitpl.CACHE_SUBDIR + "/episcanner_chikungunya_RS_2023.parquet"
    import os

    assert os.path.exists(cached)
    assert not os.path.exists("data/episcanner_RS_2023.parquet")

    # metadata records the disease
    import pyarrow.parquet as pq

    meta = pq.read_schema(cached).metadata or {}
    assert json.loads(meta[fitpl._REQUEST_META_KEY])["disease"] == "chikungunya"


if __name__ == "__main__":
    import sys

    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"PASS {name}")
        except Exception as exc:  # pragma: no cover - manual runner
            failures += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    sys.exit(1 if failures else 0)
