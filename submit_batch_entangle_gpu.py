#!/usr/bin/env python3

"""Submits a single GPU job that generates N_packings sequentially using a single compiled JAX function.

Example:
    python submit_batch_entangle_gpu.py --AR 1000 --num-rods 2000 --N-packings 5
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Submit a single batched GPU job for multiple packings.")
    p.add_argument("--AR", type=int, required=True)
    p.add_argument("--num-rods", type=int, default=2000)
    p.add_argument("--N-packings", type=int, default=5, help="Number of packings to generate in one compiled run")
    p.add_argument("--dt", type=float, default=1e-2)
    p.add_argument("--Nmax", type=int, default=10000)
    p.add_argument("--N-outer", type=int, default=1, dest="N_outer")
    p.add_argument("--atol", type=float, default=1e-8)
    p.add_argument("--initial-q", type=str, default="non-intersecting", choices=["non-intersecting", "test", "aligned", "random"],)
    p.add_argument("--partition", type=str, default="gpu_requeue")
    p.add_argument("--time", type=str, default="0-12:00")
    p.add_argument("--mem", type=str, default="32000")
    p.add_argument("--mail-user", type=str, default="jung@seas.harvard.edu")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true", help="Overwrite existing cached npy files")
    return p.parse_args()


def generate_sbatch(run_py: str, partition: str, time_limit: str, mem: str, mail_user: str) -> str:
    return f"""#!/bin/bash
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -N 1
#SBATCH -t {time_limit}
#SBATCH -p {partition}
#SBATCH --mem={mem}
#SBATCH --gres=gpu:1
#SBATCH -o output_%j.out
#SBATCH -e errors_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user={mail_user}

module load python
mamba activate simdata-analysis
module load cuda/12.2.0-fasrc01

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

python {run_py}
"""


def main() -> None:
    args = parse_args()
    root_dir = Path(os.getcwd())

    # Generate a master tracking ID for this batch submission
    import datetime
    now = datetime.datetime.now()
    dt_string = now.strftime("%Y%m%d-%H%M")
    
    batch_name = f"{dt_string}_BATCH_ENTANGLE_GPU_AR{args.AR}_N{args.num_rods}_packings{args.N_packings}"
    run_dir = root_dir / "runs" / batch_name
    
    if run_dir.exists():
        print(f"Run directory already exists: {run_dir}. Please wait a minute or delete it.")
        return
        
    run_dir.mkdir(parents=True, exist_ok=False)

    # Copy required files into run directory for reproducibility
    for fname in [
        "batch_entangle_gpu.py",
        "protocols.py",
        "optimization.py",
        "potentials.py",
        "transforms.py",
        "utils.py",
        "visualizations.py",
    ]:
        if (root_dir / fname).exists():
            os.system(f"cp {root_dir / fname} {run_dir / fname}")
        else:
            print(f"Warning: {fname} not found in {root_dir}. Ensure it is created before jobs run.")

    # Create the python runner command
    run_py_name = "run_batch_entangle_gpu.py"
    run_py = run_dir / run_py_name
    
    cmd_args = [
        "sys.executable",
        "'batch_entangle_gpu.py'",
        f"'--AR', '{args.AR}'",
        f"'--num-rods', '{args.num_rods}'",
        f"'--N-packings', '{args.N_packings}'",
        f"'--dt', '{args.dt}'",
        f"'--Nmax', '{args.Nmax}'",
        f"'--N-outer', '{args.N_outer}'",
        f"'--atol', '{args.atol}'",
        f"'--initial-q', '{args.initial_q}'",
    ]
    if args.force:
        cmd_args.append("'--force'")
        
    run_py.write_text(
        "\n".join(
            [
                "import subprocess",
                "import sys",
                "import os",
                "cmd = [",
                "    " + ",\n    ".join(cmd_args),
                "]",
                "print('Running:', ' '.join(cmd))",
                "subprocess.check_call(cmd)",
                "",
            ]
        )
    )

    sbatch_txt = generate_sbatch(run_py_name, args.partition, args.time, args.mem, args.mail_user)
    (run_dir / "Sbatch.sh").write_text(sbatch_txt)

    print(f"Created batch run: {run_dir.name}")
    if not args.dry_run:
        os.chdir(run_dir)
        os.system("sbatch Sbatch.sh")
        os.chdir(root_dir)


if __name__ == "__main__":
    main()
