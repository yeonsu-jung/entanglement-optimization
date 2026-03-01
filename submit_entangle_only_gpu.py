#!/usr/bin/env python3

"""Submit GPU-accelerated entanglement-only jobs using the batched script.

Example:
    python submit_entangle_only_gpu.py --AR 10 --num-rods 200 --count 5
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Submit GPU batched entangle-only packings.")
    p.add_argument("--AR", type=int, required=True)
    p.add_argument("--num-rods", type=int, default=200)
    p.add_argument("--count", type=int, default=5, help="Number of packings to run in the single batch job")
    p.add_argument("--dt", type=float, default=1e-2)
    p.add_argument("--Nmax", type=int, default=1000000, help="Max iterations for FIRE optimizer")
    p.add_argument("--N-outer", type=int, default=5, dest="N_outer")
    p.add_argument("--atol", type=float, default=1e-5)
    p.add_argument("--initial-q", type=str, default="non-intersecting",
                   choices=["non-intersecting", "test", "aligned", "random"])
    p.add_argument("--snapshot-every", type=int, default=1)
    
    # GPU-specific SLURM defaults
    p.add_argument("--partition", type=str, default="gpu_requeue")
    p.add_argument("--time", type=str, default="0-04:00")
    p.add_argument("--mem", type=str, default="16000")
    p.add_argument("--gres", type=str, default="gpu:1",
                   help="GPU resource request (default: gpu:1)")
    p.add_argument("--cuda-module", type=str, default="cuda/12.9",
                   help="CUDA module to load")
    p.add_argument("--mail-user", type=str, default="jung@seas.harvard.edu")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def generate_sbatch(run_py: str, partition: str, time_limit: str,
                    mem: str, gres: str, cuda_module: str,
                    mail_user: str) -> str:
    return f"""#!/bin/bash
#SBATCH -n 1
#SBATCH -c 1
#SBATCH -N 1
#SBATCH -t {time_limit}
#SBATCH -p {partition}
#SBATCH --mem={mem}
#SBATCH --gres={gres}
#SBATCH -o output_%j.out
#SBATCH -e errors_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user={mail_user}

module load {cuda_module}
module load python
mamba activate simdata-analysis

python {run_py}
"""


def main() -> None:
    args = parse_args()

    root_dir = Path(__file__).parent.absolute()
    timestamp = os.popen('date +"%Y%m%d-%H%M"').read().strip()

    run_id = f"{timestamp}_BATCH_ENTANGLE_GPU_AR{args.AR}_N{args.num_rods}_packings{args.count}"
    run_dir = root_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    # Copy required files into run directory
    for fname in [
        "batch_entangle_gpu.py",
        "protocols.py",
        "optimization.py",
        "potentials.py",
        "transforms.py",
        "utils.py",
        "visualizations.py",
    ]:
        os.system(f"cp {root_dir / fname} {run_dir / fname}")

    # Create runner script
    run_py_name = "run_batch_entangle_gpu.py"
    run_py = run_dir / run_py_name
    
    cmd_lines = [
        "import subprocess",
        "import sys",
        "cmd = [",
        "    sys.executable,",
        "    'batch_entangle_gpu.py',",
        f"    '--N-packings', '{args.count}',",
        f"    '--AR', '{args.AR}',",
        f"    '--num-rods', '{args.num_rods}',",
        f"    '--dt', '{args.dt}',",
        f"    '--Nmax', '{args.Nmax}',",
        f"    '--N-outer', '{args.N_outer}',",
        f"    '--atol', '{args.atol}',",
        f"    '--initial-q', '{args.initial_q}',",
        f"    '--snapshot-every', '{args.snapshot_every}',"
    ]
    if args.force:
        cmd_lines.append("    '--force',")
    cmd_lines.extend([
        "]",
        "print('Running:', ' '.join(cmd))",
        "subprocess.check_call(cmd)",
        ""
    ])
    
    run_py.write_text("\n".join(cmd_lines))

    sbatch_txt = generate_sbatch(
        run_py_name, args.partition, args.time, args.mem,
        args.gres, args.cuda_module, args.mail_user,
    )
    (run_dir / "Sbatch.sh").write_text(sbatch_txt)

    # Submit
    print(f"Created run: {run_dir.name}")
    if not args.dry_run:
        os.chdir(run_dir)
        os.system("sbatch Sbatch.sh")
        os.chdir(root_dir)


if __name__ == "__main__":
    main()
