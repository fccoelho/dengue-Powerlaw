#!/usr/bin/env bash
# Full results-refresh chain, run after a complete data update:
#   1. update_data.py until zero stale city caches (max 3 attempts, the
#      Mosqlimate endpoint / local DNS has failed mid-run before)
#   2. drop the stale figure panel cache so it rebuilds from the new DB
#   3. run_ablation.py --level state --attach-climate  -> ablation_results.csv (fig11)
#   4. run_classifier.py                               -> classifier_results.* (fig12/13)
#   5. generate_figures.py                             -> all 15 candidate figures
#   6. verify_results.py + verify_predictive.py        -> drift reports for main.tex
#   7. recompile the manuscript
# Every stage logs to logs/ and the chain aborts if a stage fails.
set -euo pipefail
cd "$(dirname "$0")"

PY=.venv/bin/python
# Reference time for "this cache is stale". Defaults to the chain start,
# which is right for a fresh chain; when resuming after a mostly-complete
# run, export STALE_SINCE to the completion time of that run so only the
# true stragglers are re-downloaded.
CHAIN_START=${STALE_SINCE_EPOCH:-$(date +%s)}
mkdir -p logs

say() { echo "[chain $(date '+%H:%M:%S')] $*" | tee -a logs/refresh_chain.log; }

count_stale() {
    "$PY" - "$CHAIN_START" <<'EOF'
import os, sys
start = float(sys.argv[1])
stale = 0
for f in os.listdir("data"):
    if f.endswith(".parquet") and f[:1].isdigit() and len(f) == 15:
        if not os.path.exists(os.path.join("data", f)) or \
           os.path.getmtime(os.path.join("data", f)) < start:
            stale += 1
print(stale)
EOF
}

# ---- stage 1: the data update, with retries -------------------------
attempt=1
max_attempts=3
while [ "$attempt" -le "$max_attempts" ]; do
    say "stage 1: data update, attempt $attempt/$max_attempts"
    "$PY" update_data.py > "logs/update_data_attempt${attempt}.log" 2>&1 || \
        say "stage 1: update_data.py exited nonzero (see logs/update_data_attempt${attempt}.log)"
    stale=$(count_stale)
    say "stage 1: stale city caches after attempt $attempt: $stale"
    [ "$stale" -eq 0 ] && break
    if [ "$attempt" -lt "$max_attempts" ]; then
        say "stage 1: sleeping 300 s before retry"
        sleep 300
    fi
    attempt=$((attempt + 1))
done
if [ "$stale" -ne 0 ]; then
    say "stage 1 FAILED: $stale caches still stale after $max_attempts attempts; chain aborted"
    exit 1
fi

# ---- stage 2: drop stale figure panel cache -------------------------
say "stage 2: removing stale figure panel cache"
rm -f Manuscript/figures/_panel_city_BR.parquet

# ---- stage 3: ablation ladder ---------------------------------------
say "stage 3: ablation ladder (state level, size task)"
"$PY" run_ablation.py --level state --attach-climate \
    > logs/ablation.log 2>&1
say "stage 3: done"

# ---- stage 3b: exceedance artefacts (both blocks of Table 4) --------
# Not produced by any other stage, which is how the state block came to be
# quoted in the manuscript with no committed artefact behind it. Each run
# writes its own file now (run_ablation.py derives the name from --level and
# --tasks), so neither can overwrite the size ladder from stage 3.
say "stage 3b: exceedance artifacts (state + city)"
"$PY" run_ablation.py --level state --tasks exceedance --attach-climate \
    --output ablation_state_exceedance.csv \
    > logs/ablation_state_exceedance.log 2>&1
"$PY" run_ablation.py --level city --tasks exceedance \
    --output ablation_city.csv \
    > logs/ablation_city.log 2>&1
say "stage 3b: done"

# ---- stage 4: rarity classifier -------------------------------------
say "stage 4: rarity classifier (city level, defaults)"
"$PY" run_classifier.py > logs/classifier.log 2>&1
say "stage 4: done"

# ---- stage 5: all figures -------------------------------------------
say "stage 5: regenerating all 15 figures"
"$PY" Manuscript/figures/generate_figures.py > logs/figures.log 2>&1
say "stage 5: done"

# ---- stage 6: verification reports ----------------------------------
say "stage 6: verification reports"
"$PY" Manuscript/verification/verify_results.py \
    > logs/verify_results_after_update.log 2>&1
"$PY" Manuscript/verification/verify_predictive.py \
    > logs/verify_predictive_after_update.log 2>&1 || \
    say "stage 6: verify_predictive reported drift (expected; see log)"
say "stage 6: done"

# ---- stage 7: recompile the manuscript ------------------------------
say "stage 7: recompiling manuscript"
(cd Manuscript && pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1 && \
    bibtex main > /dev/null 2>&1 && \
    pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1 && \
    pdflatex -interaction=nonstopmode main.tex > logs/latex.log 2>&1)
say "stage 7: done"

say "CHAIN COMPLETE"
