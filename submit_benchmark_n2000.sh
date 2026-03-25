#!/bin/bash
#SBATCH -n 1
#SBATCH -c 1
#SBATCH -N 1
#SBATCH -t 0-01:00
#SBATCH -p gpu_test
#SBATCH --mem=32000
#SBATCH --gres=gpu:1
#SBATCH -o benchmark_n2000_%j.out
#SBATCH -e benchmark_n2000_%j.err
#SBATCH -J bench_n2000

module load cuda/12.9
module load python
mamba activate simdata-analysis

echo "=== Node: $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null

echo ""
echo ">>> Benchmark: 2000 rods, Nmax=100"
python benchmark_cpu_vs_gpu.py --num-rods 2000 --AR 10 --Nmax 100 --N-outer 1

echo ""
echo "=== Benchmark complete ==="
