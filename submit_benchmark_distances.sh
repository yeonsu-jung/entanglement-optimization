#!/bin/bash
#SBATCH -n 1
#SBATCH -c 1
#SBATCH -N 1
#SBATCH -t 0-01:00
#SBATCH -p gpu_requeue
#SBATCH --mem=32000
#SBATCH --gres=gpu:1
#SBATCH -o benchmark_distances_%j.out
#SBATCH -e benchmark_distances_%j.err
#SBATCH -J bench_dist

module load cuda/12.9
module load python
mamba activate simdata-analysis

echo "=== Node: $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null
echo ""

python benchmark_distances.py --sizes 50 100 200 500 1000 2000 --repeats 5

echo ""
echo "=== Done ==="
