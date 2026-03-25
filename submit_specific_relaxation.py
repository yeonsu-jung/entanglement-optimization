#!/usr/bin/env python3
import argparse
import os
import shutil
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent

FILES_TO_COPY = [
    "relax_specific_packing.py",
    "protocols.py",
    "optimization.py",
    "potentials.py",
    "transforms.py",
    "utils.py",
    "visualizations.py",
]

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Submit a specific packing file for GPU relaxation.")
    p.add_argument("input_file", type=str, help="Path to the packing file (e.g. x_entangled.txt)")
    p.add_argument("--AR-list", type=str, default="1000,500,300")
    p.add_argument("--relax-dt", type=float, default=1e-4)
    p.add_argument("--partition", type=str, default="gpu_requeue")
    p.add_argument("--time", type=str, default="0-08:00")
    p.add_argument("--gres", type=str, default="gpu:1")
    p.add_argument("--mem", type=str, default="16000")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()

def main():
    args = parse_args()
    input_file = Path(args.input_file).resolve()
    if not input_file.exists():
        raise SystemExit(f"File {input_file} not found.")

    # Create a job directory next to the input file
    job_dir = input_file.parent / f"job_relax_{input_file.stem}"
    job_dir.mkdir(parents=True, exist_ok=True)
    print(f"Preparing job in {job_dir}")

    # Copy dependencies
    for f in FILES_TO_COPY:
        shutil.copy2(REPO_DIR / f, job_dir / f)

    # Create Sbatch
    sbatch_content = f"""#!/bin/bash
#SBATCH -n 1
#SBATCH -c 1
#SBATCH -N 1
#SBATCH -t {args.time}
#SBATCH -p {args.partition}
#SBATCH --mem={args.mem}
#SBATCH --gres={args.gres}
#SBATCH -o output_%j.out
#SBATCH -e errors_%j.err

module load cuda/12.9
module load python
mamba activate simdata-analysis

python relax_specific_packing.py {input_file} --AR-list {args.AR_list} --relax-dt {args.relax_dt}
"""
    (job_dir / "Sbatch.sh").write_text(sbatch_content)

    if args.dry_run:
        print("Dry run. Sbatch created but not submitted.")
    else:
        os.chdir(job_dir)
        os.system("sbatch Sbatch.sh")
        print("Job submitted.")

if __name__ == "__main__":
    main()
