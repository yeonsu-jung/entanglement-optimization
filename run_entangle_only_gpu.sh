#!/usr/bin/env bash
# run_entangle_only_gpu.sh
#
# Submit GPU jobs to create an entangled packing for each N.
#
# One SLURM job is submitted per random-seed triple, per N.
#
# Usage:
#   bash run_entangle_only_gpu.sh            # submit all N values
#   bash run_entangle_only_gpu.sh --dry-run  # preview without submitting
#
# Adjust SLURM resource defaults below if needed.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT="$REPO_DIR/submit_entangle_only_gpu.py"

# ── Configuration ─────────────────────────────────────────────────────────
# For entanglement-only, an AR value is still typically provided or standard is used.
AR=1000
COUNT=1                    # random-key triples per N
NMAX_ENTANGLE=1000000
N_OUTER_ENTANGLE=1

# SLURM
PARTITION="seas_gpu"
TIME="0-04:00"
MEM="8000"
GRES="gpu:1"
CUDA_MODULE="cuda/12.9"
MAIL_USER="jung@seas.harvard.edu"

DRY_RUN_FLAG=""
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN_FLAG="--dry-run"
    echo "=== DRY RUN: no jobs will be submitted ==="
fi

# ── N values ──────────────────────────────────────────────────────────────
N_VALUES=(1000 1250 1500 2000)

echo "Submitting entangle-only GPU jobs for N = ${N_VALUES[*]}"
echo "Seeds per N: $COUNT"
echo "AR: $AR"
echo "Partition:   $PARTITION  |  Time: $TIME  |  GPU: $GRES"
echo "--------------------------------------------------------------"

for N in "${N_VALUES[@]}"; do
    echo ""
    echo ">>> N=$N"
    python3 "$SCRIPT" \
        --num-rods "$N" \
        --AR "$AR" \
        --count "$COUNT" \
        --Nmax "$NMAX_ENTANGLE" \
        --N-outer "$N_OUTER_ENTANGLE" \
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
