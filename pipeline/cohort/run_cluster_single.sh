#!/usr/bin/env bash
# pipeline/cohort/run_cluster_single.sh
#
# Submit one SLURM job for the full cohort, executed sequentially in two phases:
#   1) entangle all (N, AR) combinations (optionally multiple random seeds each)
#   2) relax all generated q_entangled.npy files
#
# This avoids per-packing job startup overhead and allows JAX compile reuse inside
# each entangle.py invocation when N_PACKINGS > 1.
#
# Usage:
#   cd pipeline/cohort
#   bash run_cluster_single.sh [--dry-run] [--force] [--n-packings K]
#
#   --dry-run         print sbatch command without submitting
#   --force           recompute existing outputs
#   --n-packings K    number of random seeds per (N,AR), default 1

set -euo pipefail

COHORT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$COHORT_DIR/cohort_def.sh"

CLUSTER_ENV="$COHORT_DIR/cluster.env"
if [[ ! -f "$CLUSTER_ENV" ]]; then
    echo "ERROR: $CLUSTER_ENV not found."
    echo "  Copy cluster.env.template -> cluster.env and fill in your machine values."
    exit 1
fi
source "$CLUSTER_ENV"

RESULTS_DIR="$COHORT_DIR/results"
JOBS_DIR="$COHORT_DIR/jobs"
mkdir -p "$RESULTS_DIR" "$JOBS_DIR"

DRY_RUN=0
FORCE_FLAG=""
N_PACKINGS=1

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
        --n-packings)
            if [[ $# -lt 2 ]]; then
                echo "ERROR: --n-packings requires an integer value."
                exit 1
            fi
            N_PACKINGS="$2"
            shift 2
            ;;
        *)
            echo "ERROR: unknown option: $1"
            echo "Usage: bash run_cluster_single.sh [--dry-run] [--force] [--n-packings K]"
            exit 1
            ;;
    esac
done

if ! [[ "$N_PACKINGS" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: --n-packings must be a positive integer (got: $N_PACKINGS)"
    exit 1
fi

job_name="ent_cohort_seq"
pipeline_dir="$REPO_ROOT/pipeline"
job_script="$JOBS_DIR/${job_name}.sh"

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

RESULTS_DIR="${RESULTS_DIR}"
NMAX="${NMAX}"
MAX_ITERS="${MAX_ITERS}"
N_PACKINGS="${N_PACKINGS}"
FORCE_FLAG="${FORCE_FLAG}"

N_MAIN=(${N_MAIN[*]})
AR_MAIN=(${AR_MAIN[*]})
N_LARGE=(${N_LARGE[*]})
AR_LARGE=(${AR_LARGE[*]})

log() { echo "[\$(date '+%H:%M:%S')] \$*"; }

run_entangle() {
    local N="\$1" AR="\$2"
    local out_dir="\${RESULTS_DIR}/N\${N}/AR\${AR}"
    mkdir -p "\$out_dir"

    log "Entangle: N=\$N AR=\$AR N_PACKINGS=\$N_PACKINGS"
    python "${pipeline_dir}/entangle.py" \
        --num-rods "\$N" --AR "\$AR" --Nmax "\$NMAX" \
        --N-packings "\$N_PACKINGS" \
        --out-dir "\$out_dir" \$FORCE_FLAG
}

run_relax_N() {
    local N="\$1"; shift
    local ars=("\$@")
    local q_paths=()
    local found=0

    for AR in "\${ars[@]}"; do
        local out_dir="\${RESULTS_DIR}/N\${N}/AR\${AR}"
        [[ -d "\$out_dir" ]] || continue
        while IFS= read -r -d '' q_path; do
            found=1
            q_paths+=("\$q_path")
        done < <(find "\$out_dir" -type f -name q_entangled.npy -print0 | sort -z)
    done

    if [[ "\$found" -eq 0 ]]; then
        log "Relax: no q_entangled.npy found for N=\$N"
        return
    fi

    log "Relax: N=\$N  ARs=\${ars[*]}  total_seeds=\${#q_paths[@]}"
    python "${pipeline_dir}/relax.py" \
        "\${q_paths[@]}" --AR-list auto --max-iters "\$MAX_ITERS" \$FORCE_FLAG
}

log "Phase 1/2: entangle all packings"
for N in "\${N_MAIN[@]}"; do
    for AR in "\${AR_MAIN[@]}"; do
        run_entangle "\$N" "\$AR"
    done
done
for N in "\${N_LARGE[@]}"; do
    for AR in "\${AR_LARGE[@]}"; do
        run_entangle "\$N" "\$AR"
    done
done

log "Phase 2/2: relax all generated packings"
for N in "\${N_MAIN[@]}"; do
    run_relax_N "\$N" "\${AR_MAIN[@]}"
done
for N in "\${N_LARGE[@]}"; do
    run_relax_N "\$N" "\${AR_LARGE[@]}"
done

log "Done: full sequential cohort run complete"
SBATCH

if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[dry-run] sbatch $job_script"
else
    job_id=$(env -u BASH_ENV -u ENV sbatch --parsable "$job_script")
    echo "Submitted ${job_name} -> job $job_id"
    echo "Script: $job_script"
fi
