#!/usr/bin/env bash
# run_sequential_gpu.sh
#
# Submit GPU jobs to create an entangled packing for each N, then relax
# sequentially through AR = 1000 → 500 → 300 → 200 → 150 → 100 → 50 → 25 → 10.
#
# One SLURM job is submitted per random-seed triple, per N.
#
# Usage:
#   bash run_sequential_gpu.sh            # submit all N values
#   bash run_sequential_gpu.sh --dry-run  # preview without submitting
#
# Adjust SLURM resource defaults below if needed.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT="$REPO_DIR/submit_sequential_protocol_gpu.py"

# ── Configuration ─────────────────────────────────────────────────────────
AR_LIST="1000,500,300,200,150,100,50,25,10"
COUNT=5                    # random-key triples per N
MAX_RELAX_ITERS=1000000
RELAX_DT=1e-4
NMAX_ENTANGLE=10000
N_OUTER_ENTANGLE=1
AMP=100.0
CLEARANCE=1.005

# SLURM
PARTITION="seas_gpu"
TIME="0-04:00"
MEM="16000"
GRES="gpu:1"
CUDA_MODULE="cuda/12.9"
MAIL_USER="jung@seas.harvard.edu"

DRY_RUN_FLAG=""
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN_FLAG="--dry-run"
    echo "=== DRY RUN: no jobs will be submitted ==="
fi

# ── N values ──────────────────────────────────────────────────────────────
N_VALUES=(1250 1500)

echo "Submitting sequential GPU packing jobs for N = ${N_VALUES[*]}"
echo "AR sequence: $AR_LIST"
echo "Seeds per N: $COUNT"
echo "Partition:   $PARTITION  |  Time: $TIME  |  GPU: $GRES"
echo "--------------------------------------------------------------"

for N in "${N_VALUES[@]}"; do
    echo ""
    echo ">>> N=$N"
    python3 "$SCRIPT" \
        --num-rods "$N" \
        --count "$COUNT" \
        --AR-list "$AR_LIST" \
        --max-relax-iters "$MAX_RELAX_ITERS" \
        --relax-dt "$RELAX_DT" \
        --Nmax-entangle "$NMAX_ENTANGLE" \
        --N-outer-entangle "$N_OUTER_ENTANGLE" \
        --amp "$AMP" \
        --clearance "$CLEARANCE" \
        --partition "$PARTITION" \
        --time "$TIME" \
        --mem "$MEM" \
        --gres "$GRES" \
        --cuda-module "$CUDA_MODULE" \
        --mail-user "$MAIL_USER" \
        $DRY_RUN_FLAG
done

echo ""
echo "Done."
