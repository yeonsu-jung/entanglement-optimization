#!/usr/bin/env python3
"""Submit optimized GPU relaxation jobs for all entangled packings in a directory.

Usage:
    python submit_batch_relax_gpu.py --base-dir results/2026-03-01_BATCH_...
"""

import argparse
import os
import subprocess
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent

FILES_TO_COPY = [
    "batch_relax_gpu.py",
    "protocols.py",
    "optimization.py",
    "potentials.py",
    "transforms.py",
    "utils.py",
    "visualizations.py",
]

def parse_args():
    p = argparse.ArgumentParser(description="Submit Optimized GPU Relaxation Jobs")
    p.add_argument("--base-dir", type=str, required=True, 
                   help="Directory to search for q_entangled.npy files")
    p.add_argument("--AR-list", type=str, default="1000,500,300,100,50,25,10")
    p.add_argument("--Nmax", type=int, default=1000000)
    p.add_argument("--atol", type=float, default=1e-6)
    p.add_argument("--relax-dt", type=float, default=0.01)
    p.add_argument("--clearance", type=float, default=1.005)
    p.add_argument("--amp", type=float, default=100.0)

    # SLURM
    p.add_argument("--partition", type=str, default="gpu_requeue")
    p.add_argument("--time", type=str, default="0-12:00")
    p.add_argument("--mem", type=str, default="16000")
    p.add_argument("--gres", type=str, default="gpu:1")
    p.add_argument("--cuda-module", type=str, default="cuda/12.9")
    p.add_argument("--mail-user", type=str, default="jung@seas.harvard.edu")

    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true")
    return p.parse_args()

def generate_sbatch(args, run_py_name):
    return f"""#!/bin/bash
#SBATCH -n 1
#SBATCH -c 1
#SBATCH -N 1
#SBATCH -t {args.time}
#SBATCH -p {args.partition}
#SBATCH --mem={args.mem}
#SBATCH --gres={args.gres}
#SBATCH -o output_%j.out
#SBATCH -e errors_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user={args.mail_user}

module load {args.cuda_module}
module load python
mamba activate simdata-analysis

python {run_py_name}
"""

def main():
    args = parse_args()
    base_dir = Path(args.base_dir).resolve()
    
    # 1. Recursive search for q_entangled.npy
    print(f"Searching for entangled packings in {base_dir}...")
    target_dirs = []
    for root, dirs, files in os.walk(base_dir):
        if "q_entangled.npy" in files:
            target_dirs.append(Path(root))
            
    if not target_dirs:
        print("No q_entangled.npy files found.")
        return
        
    print(f"Found {len(target_dirs)} packings to relax.")

    for target_dir in target_dirs:
        print(f"\nPreparing job in: {target_dir}")
        
        # Copy files
        for f in FILES_TO_COPY:
            os.system(f"cp {REPO_DIR / f} {target_dir / f}")
            
        # Write runner
        run_py_name = "run_optimized_relax.py"
        run_py_content = f"""
import subprocess, sys
cmd = [
    sys.executable, 'batch_relax_gpu.py', 'q_entangled.npy',
    '--AR-list', '{args.AR_list}',
    '--Nmax', '{args.Nmax}',
    '--atol', '{args.atol}',
    '--dt', '{args.relax_dt}',
    '--clearance', '{args.clearance}',
    '--amp', '{args.amp}'
]
"""
        if args.force:
            run_py_content += "cmd.append('--force')\n"
        run_py_content += "print('Executing:', ' '.join(cmd))\nsubprocess.check_call(cmd)\n"
        (target_dir / run_py_name).write_text(run_py_content)
        
        # Write SBATCH
        sbatch_txt = generate_sbatch(args, run_py_name)
        (target_dir / "Sbatch_relax.sh").write_text(sbatch_txt)
        
        # Submit
        if not args.dry_run:
            print(f"Submitting job for {target_dir.name}...")
            os.chdir(target_dir)
            os.system("sbatch Sbatch_relax.sh")
            os.chdir(REPO_DIR)
        else:
            print(f"Dry run: folder prepared for {target_dir.name}")

    print("\nSubmission complete.")

if __name__ == "__main__":
    main()
