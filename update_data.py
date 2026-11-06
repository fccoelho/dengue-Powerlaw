"""
One-shot data update pipeline for the dashboard.

Refreshes every cached dataset from the Mosqlimate API and re-runs all
power law fits so that powerlaw_results.db is up to date.

Phases (sequential, idempotent - all DB writes are INSERT OR REPLACE):
  0. Download all Episcanner state-year parquets (2011..current year).
  1. run_scan(force_download=True)          - weekly-case fits; downloads all
     city-level Infodengue parquets (~5,570 municipalities x yearly chunks).
  2. run_yearly_scan(force_download=False)  - yearly fits, reuses the fresh cache.
  3. fit_episcanner_pl.main()               - state/national Episcanner metric fits.
  4. run_total_cases_scan()                 - total-cases fits over Episcanner data.

Usage:
    python update_data.py
"""

import asyncio
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime

sys.stdout.reconfigure(line_buffering=True)

import dotenv

dotenv.load_dotenv()

from fitpl import FitPL, fetch_episcanner, STATES
import fit_episcanner_pl

MAX_WORKERS = 8


def download_episcanner_all():
    """Phase 0: re-download every Episcanner state-year parquet."""
    years = list(range(2011, date.today().year + 1))
    tasks = [(state, year) for year in years for state in STATES]
    print(f"[episcanner] requesting {len(tasks)} state-year files...", flush=True)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        results = list(
            ex.map(lambda a: fetch_episcanner(state=a[0], year=a[1]), tasks)
        )
    ok = sum(1 for r in results if r is not None)
    print(
        f"[episcanner] {ok}/{len(tasks)} files retrieved "
        "(None = no data published for that year)", flush=True
    )


def report_stale_city_cache():
    """Print diagnostics: city parquets not refreshed during this run."""
    import os

    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
    stale = []
    for f in os.listdir("data"):
        if f.endswith(".parquet") and f[:1].isdigit() and len(f) == 15:
            if os.path.getmtime(os.path.join("data", f)) < today:
                stale.append(f)
    print(f"[check] stale city parquets (not refreshed today): {len(stale)}", flush=True)
    if stale:
        print("  examples:", stale[:10], flush=True)
    return stale


def stale_geocodes():
    """Geocodes whose Infodengue cache predates the refresh reference.

    Reference: the ``STALE_SINCE_EPOCH`` environment variable when set
    (e.g. the completion time of a previous, mostly-successful run, so only
    true stragglers are re-downloaded); otherwise midnight today, which
    matches the original "one full update per day" semantics.

    Cities already refreshed after the reference were downloaded AND fitted
    by the interrupted run (run_scan upserts fits as it goes), so they are
    skipped entirely; everything else is re-downloaded with force_download.
    """
    import os

    from fitpl import NAME_BY_GEOCODE

    ref = os.environ.get("STALE_SINCE_EPOCH")
    if ref:
        ref = float(ref)
    else:
        ref = datetime.now().replace(
            hour=0, minute=0, second=0, microsecond=0
        ).timestamp()
    stale = {}
    for geocode, name in NAME_BY_GEOCODE.items():
        path = os.path.join("data", f"{geocode}.parquet")
        if not os.path.exists(path) or os.path.getmtime(path) < ref:
            stale[geocode] = name
    print(f"[resume] {len(stale)} of {len(NAME_BY_GEOCODE)} municipalities "
          "have a stale (or missing) cache and will be re-downloaded", flush=True)
    return stale


def main():
    t0 = time.time()
    print(f"=== Mosqlimate data update (resumed) started {datetime.now()} ===",
          flush=True)

    print("\n=== Phase 0/4: Episcanner download ===", flush=True)
    download_episcanner_all()

    fitter = FitPL()

    print("\n=== Phase 1/4: run_scan (Infodengue download + weekly-case fits) ===",
          flush=True)
    # Resume: only re-download municipalities whose cache predates today;
    # cities refreshed earlier today were already fitted with fresh data.
    asyncio.run(fitter.run_scan(
        geocodes=stale_geocodes(), force_download=True, max_workers=MAX_WORKERS
    ))

    print("\n=== Phase 2/4: run_yearly_scan (yearly fits, full pass over fresh cache) ===",
          flush=True)
    asyncio.run(fitter.run_yearly_scan(force_download=False, max_workers=MAX_WORKERS))

    print("\n=== Phase 3/4: fit_episcanner_pl (state/national metric fits) ===", flush=True)
    fit_episcanner_pl.main()

    print("\n=== Phase 4/4: run_total_cases_scan (total-cases fits) ===", flush=True)
    fitter.run_total_cases_scan()

    report_stale_city_cache()

    print(
        f"\n=== DONE in {(time.time() - t0) / 3600:.2f} h ===", flush=True
    )


if __name__ == "__main__":
    main()
