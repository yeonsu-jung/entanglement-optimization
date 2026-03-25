#!/usr/bin/env bash
# pipeline/cohort/run_cohort.sh
#
# Produce 68 independent (N, AR) entangled+relaxed packings.
#
# Grid
#   main  : N ∈ {10,20,50,100,200,300,500,1000} × AR ∈ {10,20,50,100,200,300,500,1000}  → 64
#   large : N ∈ {1500,2000}                      × AR ∈ {500,1000}                       →  4
#
# For each (N, AR):
#   1. entangle.py  --num-rods N --AR AR   →  cohort/results/N<N>/AR<AR>/<seeds>/q_entangled.npy
#   2. relax.py     q_entangled.npy --AR-list AR
#                                          →  …/<seeds>/<ts>_Relaxed-N<N>/AR<AR>/
#
# Packing is skipped automatically (no --force) when q_relaxed.npy already exists,
# so the script is safe to resume after interruption.
#
# Tunables (override via environment):
#   CONDA_ENV   jax conda environment name          (default: jax-env)
#   NMAX        max FIRE iters for entangle         (default: 10000)
#   MAX_ITERS   max FIRE iters for relax            (default: 1000000)
#
# Usage:
#   cd pipeline/cohort
#   bash run_cohort.sh [--force] 2>&1 | tee run_cohort.log

set -euo pipefail

COHORT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_DIR="$(dirname "$COHORT_DIR")"
RESULTS_DIR="$COHORT_DIR/results"

CONDA_ENV="${CONDA_ENV:-jax-env}"
NMAX="${NMAX:-10000}"
MAX_ITERS="${MAX_ITERS:-1000000}"
FORCE_FLAG=""

# Pass --force through if the user requests it
for arg in "$@"; do
    [[ "$arg" == "--force" ]] && FORCE_FLAG="--force"
done

# ── helpers ──────────────────────────────────────────────────────────────────

log() { echo "[$(date '+%H:%M:%S')] $*"; }

run_packing() {
    local N=$1 AR=$2
    local out_dir="$RESULTS_DIR/N${N}/AR${AR}"
    mkdir -p "$out_dir"

    log "━━━  N=$N  AR=$AR  ━━━"

    # ── 1. Entangle ──────────────────────────────────────────────────────────
    conda run -n "$CONDA_ENV" python "$PIPELINE_DIR/entangle.py" \
        --num-rods  "$N"    \
        --AR        "$AR"   \
        --Nmax      "$NMAX" \
        --out-dir   "$out_dir" \
        $FORCE_FLAG

    # ── 2. Find freshest q_entangled.npy ─────────────────────────────────────
    local q_path
    q_path=$(ls -td "$out_dir"/*/q_entangled.npy 2>/dev/null | head -1 || true)
    if [[ -z "$q_path" ]]; then
        log "  ERROR: q_entangled.npy not found under $out_dir — skipping relax."
        return 1
    fi

    # ── 3. Relax to exactly this AR ──────────────────────────────────────────
    conda run -n "$CONDA_ENV" python "$PIPELINE_DIR/relax.py" \
        "$q_path"           \
        --AR-list  "$AR"    \
        --max-iters "$MAX_ITERS" \
        $FORCE_FLAG

    log "  done  N=$N  AR=$AR"
}

# ── cohort definition ─────────────────────────────────────────────────────────

N_MAIN=(10 20 50 100 200 300 500 1000)
AR_MAIN=(10 20 50 100 200 300 500 1000)

N_LARGE=(1500 2000)
AR_LARGE=(500 1000)

# ── run ───────────────────────────────────────────────────────────────────────

TOTAL=$(( ${#N_MAIN[@]} * ${#AR_MAIN[@]} + ${#N_LARGE[@]} * ${#AR_LARGE[@]} ))
COUNT=0

log "Starting cohort: $TOTAL packings → $RESULTS_DIR"
log "Env: CONDA_ENV=$CONDA_ENV  NMAX=$NMAX  MAX_ITERS=$MAX_ITERS"

for N in "${N_MAIN[@]}"; do
    for AR in "${AR_MAIN[@]}"; do
        COUNT=$(( COUNT + 1 ))
        log "[$COUNT/$TOTAL]"
        run_packing "$N" "$AR"
    done
done

for N in "${N_LARGE[@]}"; do
    for AR in "${AR_LARGE[@]}"; do
        COUNT=$(( COUNT + 1 ))
        log "[$COUNT/$TOTAL]"
        run_packing "$N" "$AR"
    done
done

log "Cohort complete. Results in: $RESULTS_DIR"
