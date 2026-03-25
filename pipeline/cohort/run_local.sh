#!/usr/bin/env bash
# pipeline/cohort/run_local.sh
#
# Run the full cohort locally (sequential, one GPU).
# Grid is defined in cohort_def.sh — edit that, not this file.
#
# Usage:
#   cd pipeline/cohort
#   bash run_local.sh [--force] 2>&1 | tee run_local.log
#
# Tunables via env vars:
#   CONDA_ENV   conda environment name   (default: jax-env)
#   NMAX        entangle FIRE budget     (default from cohort_def.sh: 10000)
#   MAX_ITERS   relax FIRE budget        (default from cohort_def.sh: 1000000)

set -euo pipefail

COHORT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_DIR="$(dirname "$COHORT_DIR")"
RESULTS_DIR="$COHORT_DIR/results"

source "$COHORT_DIR/cohort_def.sh"

CONDA_ENV="${CONDA_ENV:-jax-env}"
FORCE_FLAG=""
for arg in "$@"; do [[ "$arg" == "--force" ]] && FORCE_FLAG="--force"; done

log() { echo "[$(date '+%H:%M:%S')] $*"; }

run_packing() {
    local N=$1 AR=$2
    local out_dir="$RESULTS_DIR/N${N}/AR${AR}"
    mkdir -p "$out_dir"

    log "━━━  N=$N  AR=$AR  ━━━"

    conda run -n "$CONDA_ENV" python "$PIPELINE_DIR/entangle.py" \
        --num-rods "$N" --AR "$AR" --Nmax "$NMAX" \
        --out-dir  "$out_dir" $FORCE_FLAG

    local q_path
    q_path=$(ls -td "$out_dir"/*/q_entangled.npy 2>/dev/null | head -1 || true)
    if [[ -z "$q_path" ]]; then
        log "  ERROR: q_entangled.npy not found — skipping relax."; return 1
    fi

    conda run -n "$CONDA_ENV" python "$PIPELINE_DIR/relax.py" \
        "$q_path" --AR-list "$AR" --max-iters "$MAX_ITERS" $FORCE_FLAG

    log "  done  N=$N  AR=$AR"
}

TOTAL=$(( ${#N_MAIN[@]} * ${#AR_MAIN[@]} + ${#N_LARGE[@]} * ${#AR_LARGE[@]} ))
COUNT=0
log "Local cohort: $TOTAL packings → $RESULTS_DIR"
log "Env: CONDA_ENV=$CONDA_ENV  NMAX=$NMAX  MAX_ITERS=$MAX_ITERS"

for N in "${N_MAIN[@]}"; do
    for AR in "${AR_MAIN[@]}"; do
        COUNT=$((COUNT+1)); log "[$COUNT/$TOTAL]"; run_packing "$N" "$AR"
    done
done
for N in "${N_LARGE[@]}"; do
    for AR in "${AR_LARGE[@]}"; do
        COUNT=$((COUNT+1)); log "[$COUNT/$TOTAL]"; run_packing "$N" "$AR"
    done
done

log "Done. Results in: $RESULTS_DIR"
