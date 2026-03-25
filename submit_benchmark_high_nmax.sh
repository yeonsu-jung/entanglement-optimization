#!/bin/bash
#SBATCH -n 1
#SBATCH -c 1
#SBATCH -N 1
#SBATCH -t 0-01:00
#SBATCH -p gpu_test
#SBATCH --mem=32000
#SBATCH --gres=gpu:1
#SBATCH -o benchmark_high_nmax_%j.out
#SBATCH -e benchmark_high_nmax_%j.err
#SBATCH -J bench_high_nmax

module load cuda/12.9
module load python
mamba activate simdata-analysis

echo "=== Node: $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null

echo ""
echo ">>> 200 rods, AR=1000, Nmax=1000"
python entangle_only_gpu.py 42 0 0 --AR 1000 --num-rods 200 --Nmax 1000 --N-outer 5 --force --no-save-trajectory

echo ""
echo ">>> 500 rods, AR=1000, Nmax=1000"
python entangle_only_gpu.py 42 0 0 --AR 1000 --num-rods 500 --Nmax 1000 --N-outer 5 --force --no-save-trajectory

echo ""
echo ">>> 1000 rods, AR=1000, Nmax=1000"
python entangle_only_gpu.py 42 0 0 --AR 1000 --num-rods 1000 --Nmax 1000 --N-outer 5 --force --no-save-trajectory

echo ""
echo ">>> 2000 rods, AR=1000, Nmax=1000"
python entangle_only_gpu.py 42 0 0 --AR 1000 --num-rods 2000 --Nmax 1000 --N-outer 5 --force --no-save-trajectory

echo ""
echo "=== All benchmarks complete ==="
