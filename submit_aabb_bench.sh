#!/bin/bash
#SBATCH -p seas_gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH -t 0-00:30:00
#SBATCH -o aabb_bench_%j.out
#SBATCH -e aabb_bench_%j.err
#SBATCH -J aabb_bench

module load cuda/12.9
module load python
mamba activate simdata-analysis

python benchmark_aabb.py --N 500 --AR 1000 --iters 500
