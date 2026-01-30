import time
import pandas as pd
import duckdb
import cProfile
import pstats
import io
from dashboard import (
    plot_episcanner_region_timeseries, 
    plot_combined_episcanner_fit,
    plot_episcanner_dispersion_alpha,
    plot_episcanner_state_map,
    get_episcanner_fit_results
)
import plotly.graph_objects as go

def profile_update_episcanner(region="BR"):
    print(f"Profiling update_episcanner for region: {region}")
    
    pr = cProfile.Profile()
    pr.enable()
    
    start = time.time()
    
    # Simulate update_episcanner
    r1 = plot_combined_episcanner_fit(region, "total_cases")
    r2 = plot_combined_episcanner_fit(region, "ep_dur")
    r3 = plot_episcanner_dispersion_alpha(region, "total_cases")
    r4 = plot_episcanner_dispersion_alpha(region, "ep_dur")
    r5 = plot_episcanner_state_map("total_cases")
    r6 = plot_episcanner_state_map("ep_dur")
    r7 = get_episcanner_fit_results(region, "total_cases")
    r8 = get_episcanner_fit_results(region, "ep_dur")
    r9 = plot_episcanner_region_timeseries(region)
    
    end = time.time()
    pr.disable()
    
    print(f"Total time: {end - start:.4f}s")
    
    s = io.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats(20)
    print(s.getvalue())

if __name__ == "__main__":
    profile_update_episcanner("BR")
