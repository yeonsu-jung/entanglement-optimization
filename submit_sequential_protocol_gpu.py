#!/usr/bin/env python3

"""Submit GPU jobs for sequential entangle + multi-AR relaxation protocol.

Each submitted job runs sequential_protocol_gpu.py for one random-key triple,
with the full AR sequence 1000→500→...→10.

Example:
    # 5 seeds, N=100, dry-run preview
    python submit_sequential_protocol_gpu.py --num-rods 100 --count 5 --dry-run

    # Actually submit for N=200
    python submit_sequential_protocol_gpu.py --num-rods 200 --count 5
"""

import argparse
import os
import random
import subprocess
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parent

FILES_TO_COPY = [
    "sequential_protocol_gpu.py",
    "protocols.py",
    "optimization.py",
    "potentials.py",
    "transforms.py",
    "utils.py",
    "visualizations.py",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Submit GPU sequential-protocol jobs."
    )
    p.add_argument("--num-rods", type=int, required=True,
                   help="Number of rods N.")
    p.add_argument("--count", type=int, default=1,
                   help="Number of random-key triples to submit (default: 5).")
    p.add_argument("--AR-list", type=str,
                   default="1000,500,300,200,150,100,50,25,10",
                   help="Comma-separated AR sequence (default: 1000,...,10).")

    # Physics parameters
    p.add_argument("--Nmax-entangle", type=int, default=10000)
    p.add_argument("--N-outer-entangle", type=int, default=1)
    p.add_argument("--max-relax-iters", type=int, default=1000000)
    p.add_argument("--relax-dt", type=float, default=1e-4)
    p.add_argument("--clearance", type=float, default=1.005)
    p.add_argument("--amp", type=float, default=100.0)
    p.add_argument("--dt", type=float, default=1e-2)
    p.add_argument("--initial-q", type=str, default="non-intersecting",
                   choices=["non-intersecting", "random", "aligned"])

    # SLURM resources
    p.add_argument("--partition", type=str, default="gpu_requeue",
                   help="SLURM partition (default: gpu_requeue).")
    p.add_argument("--time", type=str, default="0-04:00",
                   help="Wall-clock time limit D-HH:MM (default: 0-04:00).")
    p.add_argument("--mem", type=str, default="16000",
                   help="Memory in MB (default: 16000).")
    p.add_argument("--gres", type=str, default="gpu:1",
                   help="GPU resource request (default: gpu:1).")
    p.add_argument("--cuda-module", type=str, default="cuda/12.9",
                   help="CUDA module to load (default: cuda/12.9).")
    p.add_argument("--mail-user", type=str, default="jung@seas.harvard.edu")

    p.add_argument("--dry-run", action="store_true",
                   help="Create run folders but do not submit.")
    return p.parse_args()


def generate_sbatch(run_py: str, args: argparse.Namespace, num_rods: int) -> str:
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

python {run_py}
"""


def pick_unique_key_triples(count):
    rng = random.SystemRandom()
    keys: set[tuple[int, int, int]] = set()
    while len(keys) < count:
        keys.add((
            rng.randrange(0, 1000),
            rng.randrange(0, 1000),
            rng.randrange(0, 1000),
        ))
    return sorted(keys)


def main() -> None:
    args = parse_args()

    timestamp = os.popen('date +"%Y%m%d-%H%M"').read().strip()

    run_dirs = []
    for (k1, k2, k3) in pick_unique_key_triples(args.count):
        run_id = (
            f"{timestamp}_SEQUENTIAL_GPU_N{args.num_rods}"
            f"_randomkeys{k1},{k2},{k3}"
        )
        run_dir = REPO_DIR / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        run_dirs.append(run_dir)

        # Copy required source files
        for fname in FILES_TO_COPY:
            src = REPO_DIR / fname
            if src.exists():
                os.system(f"cp {src} {run_dir / fname}")
            else:
                print(f"WARNING: {src} not found, skipping copy.")

        # Write the runner script
        run_py_name = "run_sequential.py"
        run_py_path = run_dir / run_py_name
        lines = [
            "import subprocess, sys",
            "cmd = [",
            "    sys.executable,",
            "    'sequential_protocol_gpu.py',",
            f"    '{k1}', '{k2}', '{k3}',",
            f"    '--num-rods', '{args.num_rods}',",
            f"    '--AR-list', '{args.AR_list}',",
            f"    '--Nmax-entangle', '{args.Nmax_entangle}',",
            f"    '--N-outer-entangle', '{args.N_outer_entangle}',",
            f"    '--max-relax-iters', '{args.max_relax_iters}',",
            f"    '--relax-dt', '{args.relax_dt}',",
            f"    '--clearance', '{args.clearance}',",
            f"    '--amp', '{args.amp}',",
            f"    '--dt', '{args.dt}',",
            f"    '--initial-q', '{args.initial_q}',",
            "]",
            "print('Running:', ' '.join(cmd))",
            "subprocess.check_call(cmd)",
            "",
        ]
        run_py_path.write_text("\n".join(lines))

        sbatch_txt = generate_sbatch(run_py_name, args, args.num_rods)
        (run_dir / "Sbatch.sh").write_text(sbatch_txt)

    # Submit
    for run_dir in run_dirs:
        print(f"Created: {run_dir.name}")
        if args.dry_run:
            continue
        os.chdir(run_dir)
        os.system("sbatch Sbatch.sh")
        os.chdir(REPO_DIR)

    if args.dry_run:
        print(f"\nDry run: {len(run_dirs)} run folders created, no jobs submitted.")
    else:
        print(f"\nSubmitted {len(run_dirs)} jobs.")


if __name__ == "__main__":
    main()
