#!/bin/bash
#SBATCH -n 1
#SBATCH -c 1
#SBATCH -N 1
#SBATCH -t 0-08:00
#SBATCH -p seas_gpu
#SBATCH --mem=64000
#SBATCH --gres=gpu:1
#SBATCH -o standard_protocol_gpu_%j.out
#SBATCH -e standard_protocol_gpu_%j.err
#SBATCH -J std_proto_gpu

module load cuda/12.9
module load python
mamba activate simdata-analysis

echo "=== Node: $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null

K1=${1:-42}
K2=${2:-0}
K3=${3:-0}
NUM_RODS=${4:-500}
AR=${5:-1000}

echo ""
echo ">>> Standard Protocol GPU: N=${NUM_RODS}, AR=${AR}"
# Using refined parameters: 
# - Potential diam = 1.005 * d
# - Stop condition = strictly min_dist > d
# - Entanglement: 10k iters
# - Relaxation: dt=1e-4, amp=1000
python standard_protocol_gpu.py ${K1} ${K2} ${K3} \
    --AR ${AR} \
    --num-rods ${NUM_RODS} \
    --Nmax-entangle 10000 \
    --N-outer-entangle 1 \
    --max-relax-iters 1000000 \
    --relax-dt 1e-4 \
    --amp 1000 \
    --clearance 1.005 \
    --force \
    --no-save-trajectory

echo ""
echo "=== Complete ==="
