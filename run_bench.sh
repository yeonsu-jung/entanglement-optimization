#!/bin/bash
#SBATCH -n 1
#SBATCH -c 1
#SBATCH -N 1
#SBATCH -t 0-00:10
#SBATCH -p gpu_test
#SBATCH --mem=8000
#SBATCH --gres=gpu:1
#SBATCH -o output_bench.out
#SBATCH -e errors_bench.err

module load cuda/12.9
module load python
mamba activate simdata-analysis

python benchmark_cpu_vs_gpu.py --num-rods 200 --Nmax 1000000 --dt 0.0001
