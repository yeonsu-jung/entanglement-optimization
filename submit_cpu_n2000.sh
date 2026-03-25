#!/bin/bash
#SBATCH -n 1
#SBATCH -c 1
#SBATCH -N 1
#SBATCH -t 0-04:00
#SBATCH -p seas_compute
#SBATCH --mem=32000
#SBATCH -o benchmark_cpu_n2000_%j.out
#SBATCH -e benchmark_cpu_n2000_%j.err
#SBATCH -J cpu_n2000

module load python
mamba activate simdata-analysis

# Force JAX to use CPU only (jax[cuda] is installed but no GPU on this partition)
export JAX_PLATFORMS=cpu

echo "=== CPU Run: N=2000, AR=1000, Nmax=1000, N_outer=5 ==="
echo "=== Node: $(hostname) ==="
echo ""

python entangle_only.py 42 0 0 --AR 1000 --num-rods 2000 --Nmax 1000 --N-outer 5 --force --no-save-trajectory

echo ""
echo "=== CPU run complete ==="
