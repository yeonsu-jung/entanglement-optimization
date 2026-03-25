#!/usr/bin/env bash
# pipeline/cohort/run_local.sh
#
# Run the full cohort locally (sequential, one GPU) in two phases:
#   Phase 1: entangle all (N, AR) pairs  — JAX compiles once per N, reused for
#             all random seeds of that N.
#   Phase 2: relax every q_entangled.npy produced in Phase 1.
#
# Grid is defined in cohort_def.sh — edit that, not this file.
#
# Usage:
#   cd pipeline/cohort
#   bash run_local.sh [--force] [--n-packings K] 2>&1 | tee run_local.log
#
# Tunables via env vars:
#   MAMBA_ENV   mamba environment name   (default: simdata-analysis)
#   NMAX        entangle FIRE budget     (default from cohort_def.sh: 10000)
#   MAX_ITERS   relax FIRE budget        (default from cohort_def.sh: 1000000)

set -euo pipefail

COHORT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_DIR="$(dirname "$COHORT_DIR")"
RESULTS_DIR="$COHORT_DIR/results"

source "$COHORT_DIR/cohort_def.sh"

MAMBA_ENV="${MAMBA_ENV:-simdata-analysis}"
MAMBA_EXE="${MAMBA_EXE:-/n/sw/Miniforge3-25.3.1-0/bin/mamba}"
MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/.local/share/mamba}"

FORCE_FLAG=""
N_PACKINGS=1
while [[ $# -gt 0 ]]; do
    case "$1" in
        --force)      FORCE_FLAG="--force"; shift ;;
        --n-packings) N_PACKINGS="$2"; shift 2 ;;
        *) echo "ERROR: unknown option: $1"; exit 1 ;;
    esac
done

# Activate mamba without depending on module/lmod
eval "$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX")"
mamba activate "$MAMBA_ENV"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ── Phase 1: entangle ─────────────────────────────────────────────────────────
run_entangle() {
    local N=$1 AR=$2
    local out_dir="$RESULTS_DIR/N${N}/AR${AR}"
    mkdir -p "$out_dir"
    log "Entangle: N=$N  AR=$AR  N_PACKINGS=$N_PACKINGS"
    python "$PIPELINE_DIR/entangle.py" \
        --num-rods "$N" --AR "$AR" --Nmax "$NMAX" \
        --N-packings "$N_PACKINGS" \
        --out-dir  "$out_dir" $FORCE_FLAG
}

# ── Phase 2: relax ────────────────────────────────────────────────────────────
# Called once per N; all AR seeds collected together so that
# _fire_repulsion_fast traces only once per N (col_rad is a dynamic JAX arg).
run_relax_N() {
    local N="$1"; shift
    local ars=("$@")
    local q_paths=()
    local found=0

    for AR in "${ars[@]}"; do
        local out_dir="$RESULTS_DIR/N${N}/AR${AR}"
        [[ -d "$out_dir" ]] || continue
        while IFS= read -r -d '' q_path; do
            found=1
            q_paths+=("$q_path")
        done < <(find "$out_dir" -type f -name q_entangled.npy -print0 | sort -z)
    done

    if [[ "$found" -eq 0 ]]; then
        log "Relax: no q_entangled.npy found for N=$N"
        return
    fi

    log "Relax: N=$N  ARs=${ars[*]}  total_seeds=${#q_paths[@]}"
    python "$PIPELINE_DIR/relax.py" \
        "${q_paths[@]}" --AR-list auto --max-iters "$MAX_ITERS" $FORCE_FLAG
}

TOTAL=$(( ${#N_MAIN[@]} * ${#AR_MAIN[@]} + ${#N_LARGE[@]} * ${#AR_LARGE[@]} ))
log "Local cohort: $TOTAL (N,AR) pairs  N_PACKINGS=$N_PACKINGS  → $RESULTS_DIR"
log "Env: MAMBA_ENV=$MAMBA_ENV  NMAX=$NMAX  MAX_ITERS=$MAX_ITERS"

log "Phase 1/2: entangle all packings"
for N in "${N_MAIN[@]}"; do
    for AR in "${AR_MAIN[@]}"; do run_entangle "$N" "$AR"; done
done
for N in "${N_LARGE[@]}"; do
    for AR in "${AR_LARGE[@]}"; do run_entangle "$N" "$AR"; done
done

log "Phase 2/2: relax all generated packings"
for N in "${N_MAIN[@]}"; do
    run_relax_N "$N" "${AR_MAIN[@]}"
done
for N in "${N_LARGE[@]}"; do
    run_relax_N "$N" "${AR_LARGE[@]}"
done

log "Done. Results in: $RESULTS_DIR"
