#!/bin/bash
#SBATCH -n 1
#SBATCH -c 1
#SBATCH -N 1
#SBATCH -t 0-00:30
#SBATCH -p seas_gpu
#SBATCH --mem=8000
#SBATCH --gres=gpu:1
#SBATCH -o bench_gpu_n200_%j.out
#SBATCH -e bench_gpu_n200_%j.err

module load cuda/12.9
module load python
mamba activate simdata-analysis

python benchmark_cpu_vs_gpu.py --num-rods 200 --Nmax 300 --AR 1000
