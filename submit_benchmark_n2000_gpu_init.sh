#!/bin/bash
#SBATCH -n 1
#SBATCH -c 1
#SBATCH -N 1
#SBATCH -t 0-01:00
#SBATCH -p gpu_test
#SBATCH --mem=32000
#SBATCH --gres=gpu:1
#SBATCH -o benchmark_n2000_gpu_init_%j.out
#SBATCH -e benchmark_n2000_gpu_init_%j.err
#SBATCH -J bench_gpu_init

module load cuda/12.9
module load python
mamba activate simdata-analysis

echo "=== Node: $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null

# Use entangle_only_gpu.py which has GPU init built in
echo ""
echo ">>> Full GPU pipeline: 200 rods, AR=1000, Nmax=100"
python entangle_only_gpu.py 42 0 0 --AR 1000 --num-rods 200 --Nmax 100 --N-outer 1 --force --no-save-trajectory

echo ""
echo ">>> Full GPU pipeline: 500 rods, AR=1000, Nmax=100"
python entangle_only_gpu.py 42 0 0 --AR 1000 --num-rods 500 --Nmax 100 --N-outer 1 --force --no-save-trajectory

echo ""
echo ">>> Full GPU pipeline: 1000 rods, AR=1000, Nmax=100"
python entangle_only_gpu.py 42 0 0 --AR 1000 --num-rods 1000 --Nmax 100 --N-outer 1 --force --no-save-trajectory

echo ""
echo "=== All GPU benchmarks complete ==="
