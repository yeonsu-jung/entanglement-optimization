#!/bin/bash
#SBATCH -n 1
#SBATCH -c 1
#SBATCH -N 1
#SBATCH -t 0-01:00
#SBATCH -p gpu_test
#SBATCH --mem=8000
#SBATCH --gres=gpu:1
#SBATCH -o benchmark_%j.out
#SBATCH -e benchmark_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=jung@seas.harvard.edu
#SBATCH -J benchmark_cpu_gpu

module load cuda/12.9
module load python
mamba activate simdata-analysis

echo "=== Node: $(hostname) ==="
echo "=== GPU info ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "nvidia-smi not available"
echo ""

# Small benchmark (fast, ~2-5 min)
echo ">>> Benchmark 1: 20 rods, Nmax=100"
python benchmark_cpu_vs_gpu.py --num-rods 20 --AR 10 --Nmax 100 --N-outer 1

echo ""
echo ">>> Benchmark 2: 50 rods, Nmax=100"
python benchmark_cpu_vs_gpu.py --num-rods 50 --AR 10 --Nmax 100 --N-outer 1

echo ""
echo ">>> Benchmark 3: 100 rods, Nmax=100"
python benchmark_cpu_vs_gpu.py --num-rods 100 --AR 10 --Nmax 100 --N-outer 1

echo ""
echo ">>> Benchmark 4: 200 rods, Nmax=100"
python benchmark_cpu_vs_gpu.py --num-rods 200 --AR 10 --Nmax 100 --N-outer 1

echo ""
echo "=== All benchmarks complete ==="
