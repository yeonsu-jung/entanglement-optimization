#!/bin/bash
#SBATCH -n 1
#SBATCH -c 1
#SBATCH -N 1
#SBATCH -t 0-04:00
#SBATCH -p seas_compute
#SBATCH --mem=32000
#SBATCH -o benchmark_cpu_%j.out
#SBATCH -e benchmark_cpu_%j.err
#SBATCH -J cpu_comparison

module load python
mamba activate simdata-analysis

echo "=== CPU Comparison Run ==="
echo "=== Node: $(hostname) ==="

echo ""
echo ">>> CPU: 200 rods, AR=1000, Nmax=1000, N_outer=5"
python entangle_only.py 42 0 0 --AR 1000 --num-rods 200 --Nmax 1000 --N-outer 5 --force --no-save-trajectory

echo ""
echo ">>> CPU: 500 rods, AR=1000, Nmax=1000, N_outer=5"
python entangle_only.py 42 0 0 --AR 1000 --num-rods 500 --Nmax 1000 --N-outer 5 --force --no-save-trajectory

echo ""
echo ">>> CPU: 1000 rods, AR=1000, Nmax=1000, N_outer=5"
python entangle_only.py 42 0 0 --AR 1000 --num-rods 1000 --Nmax 1000 --N-outer 5 --force --no-save-trajectory

echo ""
echo "=== All CPU benchmarks complete ==="
