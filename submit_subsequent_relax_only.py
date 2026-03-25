#!/usr/bin/env python3

import argparse
import os
import subprocess
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent

FILES_TO_COPY = [
    "subsequent_relax_only.py",
    "protocols.py",
    "optimization.py",
    "potentials.py",
    "transforms.py",
    "utils.py",
    "visualizations.py",
]

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Submit GPU jobs to run sequential relaxation on existing entangled packings."
    )
    p.add_argument("--base-dir", type=str, default="runs",
                   help="Base directory containing the run subfolders (default: runs).")
    p.add_argument("--AR-list", type=str,
                   default="1000,500,300",
                   help="Comma-separated AR sequence (default: 1000,500,300).")

    p.add_argument("--max-relax-iters", type=int, default=1000000)
    p.add_argument("--relax-dt", type=float, default=1e-4)
    p.add_argument("--clearance", type=float, default=1.005)
    p.add_argument("--amp", type=float, default=100.0)

    # SLURM resources
    p.add_argument("--partition", type=str, default="gpu_requeue",
                   help="SLURM partition (default: gpu_requeue).")
    p.add_argument("--time", type=str, default="0-08:00",
                   help="Wall-clock time limit D-HH:MM (default: 0-08:00).")
    p.add_argument("--mem", type=str, default="16000",
                   help="Memory in MB (default: 16000).")
    p.add_argument("--gres", type=str, default="gpu:1",
                   help="GPU resource request (default: gpu:1).")
    p.add_argument("--cuda-module", type=str, default="cuda/12.9",
                   help="CUDA module to load (default: cuda/12.9).")
    p.add_argument("--mail-user", type=str, default="jung@seas.harvard.edu")

    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true")
    
    return p.parse_args()


def generate_sbatch(args: argparse.Namespace, target_dir: str) -> str:
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

python run_subsequent_relax.py
"""


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir).resolve()
    
    if not base_dir.exists():
        print(f"Directory {base_dir} does not exist.")
        return

    # Find all directories that contain q_entangled.npy
    target_dirs = []
    for root, dirs, files in os.walk(base_dir):
        if "q_entangled.npy" in files:
            target_dirs.append(Path(root))
            
    if not target_dirs:
        print(f"No directories with q_entangled.npy found inside {base_dir}")
        return

    print(f"Found {len(target_dirs)} packing(s) to run subsequent relaxation on.")

    for target_dir in target_dirs:
        print(f"Processing: {target_dir}")
        
        # Copy required source files
        for fname in FILES_TO_COPY:
            src = REPO_DIR / fname
            if src.exists():
                os.system(f"cp {src} {target_dir / fname}")
            else:
                print(f"WARNING: {src} not found, skipping copy.")
                
        # Write the runner python script localized to that dir
        run_py_name = "run_subsequent_relax.py"
        run_py_path = target_dir / run_py_name
        lines = [
            "import subprocess, sys",
            "cmd = [",
            "    sys.executable,",
            "    'subsequent_relax_only.py',",
            f"    '.',",
            f"    '--AR-list', '{args.AR_list}',",
            f"    '--max-relax-iters', '{args.max_relax_iters}',",
            f"    '--relax-dt', '{args.relax_dt}',",
            f"    '--clearance', '{args.clearance}',",
            f"    '--amp', '{args.amp}',",
            "]"
        ]
        if args.force:
            lines[-1:0] = ["    '--force',"]
            
        lines.extend([
            "print('Running:', ' '.join(cmd))",
            "subprocess.check_call(cmd)",
            ""
        ])
        run_py_path.write_text("\n".join(lines))

        # Write SBATCH
        sbatch_txt = generate_sbatch(args, str(target_dir))
        (target_dir / "Sbatch.sh").write_text(sbatch_txt)

    # Submit
    for target_dir in target_dirs:
        if args.dry_run:
            continue
        print(f"Submitting job in {target_dir}")
        os.chdir(target_dir)
        os.system("sbatch Sbatch.sh")
        os.chdir(REPO_DIR)

    if args.dry_run:
        print(f"\nDry run: {len(target_dirs)} run folders prepared, no jobs submitted.")
    else:
        print(f"\nSubmitted {len(target_dirs)} jobs.")


if __name__ == "__main__":
    main()
