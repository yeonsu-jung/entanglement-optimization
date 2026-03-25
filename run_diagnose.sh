#!/bin/bash
#SBATCH -p seas_gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH -t 0-00:10:00
#SBATCH -o diagnose_nan_%j.out
#SBATCH -e diagnose_nan_%j.err

module load cuda/12.2.0-fasrc01
module load cuda/12.2.0-fasrc01

/n/home01/yjung/.conda/envs/simdata-analysis/bin/python3 diagnose_nan.py
