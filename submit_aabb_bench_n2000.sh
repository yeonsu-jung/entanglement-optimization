#!/bin/bash
#SBATCH -p seas_gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -t 0-01:00:00
#SBATCH -o aabb_bench_n2000_%j.out
#SBATCH -e aabb_bench_n2000_%j.err
#SBATCH -J aabb_bench_n2k

module load cuda/12.9
module load python
mamba activate simdata-analysis

python benchmark_aabb.py --N 2000 --AR 1000 --iters 200
