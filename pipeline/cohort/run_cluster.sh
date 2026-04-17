#!/usr/bin/env bash
# pipeline/cohort/run_cluster.sh
#
# Submit one SLURM job per (N, AR) packing to the Harvard RC cluster.
# Grid is defined in cohort_def.sh.
# Machine settings are read from cluster.env (gitignored; copy from cluster.env.template).
#
# Usage (from the cluster login node):
#   cd pipeline/cohort
#   bash run_cluster.sh [--dry-run] [--force] [--cohort-name NAME]
#
#   --dry-run   print sbatch commands without submitting
#   --force     pass --force to entangle.py and relax.py (recompute existing)
#   --cohort-name output subfolder under results/ (default: timestamp)

set -euo pipefail

COHORT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$COHORT_DIR/cohort_def.sh"

# ── load cluster.env (required) ───────────────────────────────────────────────
CLUSTER_ENV="$COHORT_DIR/cluster.env"
if [[ ! -f "$CLUSTER_ENV" ]]; then
    echo "ERROR: $CLUSTER_ENV not found."
    echo "  Copy cluster.env.template → cluster.env and fill in your machine values."
    exit 1
fi
source "$CLUSTER_ENV"

RESULTS_ROOT="$COHORT_DIR/results"
JOBS_DIR="$COHORT_DIR/jobs"          # generated sbatch scripts land here
mkdir -p "$RESULTS_ROOT" "$JOBS_DIR"

DRY_RUN=0
FORCE_FLAG=""
COHORT_NAME=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --force)
            FORCE_FLAG="--force"
            shift
            ;;
        --cohort-name)
            if [[ $# -lt 2 ]]; then
                echo "ERROR: --cohort-name requires a value"
                exit 1
            fi
            COHORT_NAME="$2"
            shift 2
            ;;
        *)
            echo "ERROR: unknown option: $1"
            echo "Usage: bash run_cluster.sh [--dry-run] [--force] [--cohort-name NAME]"
            exit 1
            ;;
    esac
done

if [[ -z "$COHORT_NAME" ]]; then
    COHORT_NAME="$(date '+%Y-%m-%d_%H-%M-%S')"
fi
if [[ ! "$COHORT_NAME" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "ERROR: --cohort-name may only contain letters, numbers, dot, underscore, and dash"
    exit 1
fi

RESULTS_DIR="$RESULTS_ROOT/$COHORT_NAME"
mkdir -p "$RESULTS_DIR"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

if [[ "$DRY_RUN" -eq 0 ]]; then
    SNAPSHOT_DIR="$RESULTS_DIR/cohort_submit_snapshot"
    mkdir -p "$SNAPSHOT_DIR"
    cp -f "$COHORT_DIR/run_cluster.sh" "$SNAPSHOT_DIR/"
    cp -f "$COHORT_DIR/cohort_def.sh" "$SNAPSHOT_DIR/"
    cp -f "$COHORT_DIR/cluster.env" "$SNAPSHOT_DIR/"
fi

# ── generate + submit one job ─────────────────────────────────────────────────
submit_packing() {
    local N=$1 AR=$2
    local job_name="ent_N${N}_AR${AR}"
    local out_dir="$RESULTS_DIR/N${N}/AR${AR}"
    local pipeline_dir="$REPO_ROOT/pipeline"
    local job_script="$JOBS_DIR/${job_name}.sh"

    mkdir -p "$out_dir"

    cat > "$job_script" <<SBATCH
#!/bin/bash
#SBATCH -J ${job_name}
#SBATCH -p ${PARTITION}
#SBATCH -t ${TIME_LIMIT}
#SBATCH --mem=${MEM_MB}
#SBATCH --gres=${GRES}
#SBATCH -n 1 -c 1 -N 1
#SBATCH -o ${JOBS_DIR}/${job_name}_%j.out
#SBATCH -e ${JOBS_DIR}/${job_name}_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=${MAIL_USER}

set -euo pipefail

MAMBA_ENV="${MAMBA_ENV}"
CUDA_MODULE="${CUDA_MODULE}"
PYTHON_MODULE="${PYTHON_MODULE}"
MAMBA_EXE="${MAMBA_EXE:-/n/sw/Miniforge3-25.3.1-0/bin/mamba}"
MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/.local/share/mamba}"

safe_module_load() {
    local module_name="\$1"
    if [[ -z "\$module_name" ]]; then
        return 0
    fi
    if ! command -v module >/dev/null 2>&1; then
        echo "WARNING: module command unavailable; skipping '\$module_name'" >&2
        return 0
    fi
    if ! module load "\$module_name"; then
        echo "WARNING: failed to load module '\$module_name'; continuing" >&2
    fi
}

eval "\$("\$MAMBA_EXE" shell hook --shell bash --root-prefix "\$MAMBA_ROOT_PREFIX")"

safe_module_load "\$CUDA_MODULE"
safe_module_load "\$PYTHON_MODULE"
mamba activate "\$MAMBA_ENV"

# ── 1. Entangle ───────────────────────────────────────────────────────────────
python "${pipeline_dir}/entangle.py" \
    --num-rods ${N} --AR ${AR} --Nmax ${NMAX} \\
    --out-dir  "${out_dir}" ${FORCE_FLAG}

# ── 2. Find q_entangled.npy ───────────────────────────────────────────────────
Q_PATH=\$(ls -td "${out_dir}"/*/q_entangled.npy 2>/dev/null | head -1 || true)
if [[ -z "\$Q_PATH" ]]; then
    echo "ERROR: q_entangled.npy not found under ${out_dir}"
    exit 1
fi

# ── 3. Relax ──────────────────────────────────────────────────────────────────
python "${pipeline_dir}/relax.py" \
    "\$Q_PATH" --AR-list ${AR} --max-iters ${MAX_ITERS} ${FORCE_FLAG}

echo "Done: N=${N}  AR=${AR}"
SBATCH

    if [[ "$DRY_RUN" -eq 1 ]]; then
        log "[dry-run] sbatch $job_script"
    else
        local job_id
        job_id=$(env -u BASH_ENV -u ENV sbatch --parsable "$job_script")
        log "Submitted $job_name → job $job_id"
    fi
}

# ── submit cohort ─────────────────────────────────────────────────────────────
TOTAL=$(( ${#N_MAIN[@]} * ${#AR_MAIN[@]} + ${#N_LARGE[@]} * ${#AR_LARGE[@]} ))
log "Cluster cohort: $TOTAL jobs  partition=$PARTITION  dry_run=$DRY_RUN"
log "Cohort name: $COHORT_NAME"
log "Results dir: $RESULTS_DIR"
if [[ "$DRY_RUN" -eq 0 ]]; then
    log "Snapshot dir: $SNAPSHOT_DIR"
fi

for N in "${N_MAIN[@]}"; do
    for AR in "${AR_MAIN[@]}"; do submit_packing "$N" "$AR"; done
done
for N in "${N_LARGE[@]}"; do
    for AR in "${AR_LARGE[@]}"; do submit_packing "$N" "$AR"; done
done

log "All jobs submitted. Scripts in: $JOBS_DIR"
log "Monitor with:  squeue -u \$USER  |  sacct -j <jobid>"
