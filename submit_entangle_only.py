#!/usr/bin/env python3

"""Submit 5 entanglement-only jobs (no relaxation).

Creates 5 run folders under `runs/`, each with its own random key triplet.
Each job runs `entangle_only.py` and saves:
- final `q_entangled.{npy,txt}` and `x_entangled.{npy,txt}` (N,6)
- trajectory `qq.npy` (T,N,5) and `xx.npy` (T,N,6)

Example:
    python submit_entangle_only.py --AR 10 --num-rods 200
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Submit 5 entangle-only packings.")
    p.add_argument("--AR", type=int, required=True)
    p.add_argument("--num-rods", type=int, default=200)
    p.add_argument("--count", type=int, default=5)
    p.add_argument("--dt", type=float, default=1e-2)
    p.add_argument("--Nmax", type=int, default=10000)
    p.add_argument("--N-outer", type=int, default=1, dest="N_outer")
    p.add_argument("--atol", type=float, default=1e-8)
    p.add_argument("--initial-q", type=str, default="non-intersecting", choices=["non-intersecting", "test", "aligned", "random"],)
    p.add_argument("--snapshot-every", type=int, default=1)
    p.add_argument("--partition", type=str, default="seas_compute")
    p.add_argument("--time", type=str, default="0-04:00")
    p.add_argument("--mem", type=str, default="1000")
    p.add_argument("--mail-user", type=str, default="jung@seas.harvard.edu")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def generate_sbatch(run_py: str, partition: str, time_limit: str, mem: str, mail_user: str) -> str:
    return f"""#!/bin/bash
#SBATCH -n 1
#SBATCH -c 1
#SBATCH -N 1
#SBATCH -t {time_limit}
#SBATCH -p {partition}
#SBATCH --cpus-per-task=8
#SBATCH --mem={mem}
#SBATCH -o output_%j.out
#SBATCH -e errors_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user={mail_user}

module load python
mamba activate simdata-analysis

export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=8"
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8

python {run_py}
"""


def pick_unique_key_triples(count: int) -> list[tuple[int, int, int]]:
    rng = random.SystemRandom()
    keys: set[tuple[int, int, int]] = set()
    while len(keys) < count:
        keys.add((rng.randrange(0, 1000), rng.randrange(0, 1000), rng.randrange(0, 1000)))
    return sorted(keys)


def main() -> None:
    args = parse_args()

    root_dir = Path(__file__).parent.absolute()
    timestamp = os.popen('date +"%Y%m%d-%H%M"').read().strip()

    run_dirs: list[Path] = []
    for (k1, k2, k3) in pick_unique_key_triples(args.count):
        run_id = f"{timestamp}_ENTANGLEONLY_AR{args.AR}_N{args.num_rods}_randomkeys{k1},{k2},{k3}"
        run_dir = root_dir / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        run_dirs.append(run_dir)

        # Copy required files into run directory for reproducibility
        for fname in [
            "entangle_only.py",
            "protocols.py",
            "optimization.py",
            "potentials.py",
            "transforms.py",
            "utils.py",
            "visualizations.py",
        ]:
            os.system(f"cp {root_dir / fname} {run_dir / fname}")

        # Create a tiny runner so the sbatch command is stable and self-contained
        run_py_name = "run_entangle_only.py"
        run_py = run_dir / run_py_name
        run_py.write_text(
            "\n".join(
                [
                    "import subprocess",
                    "import sys",
                    "import os",
                    "os.environ['JAX_PLATFORMS'] = 'cpu'",
                    "cmd = [",
                    "    sys.executable,",
                    "    'entangle_only.py',",
                    f"    '{k1}', '{k2}', '{k3}',",
                    f"    '--AR', '{args.AR}',",
                    f"    '--num-rods', '{args.num_rods}',",
                    f"    '--dt', '{args.dt}',",
                    f"    '--Nmax', '{args.Nmax}',",
                    f"    '--N-outer', '{args.N_outer}',",
                    f"    '--atol', '{args.atol}',",
                    f"    '--initial-q', '{args.initial_q}',",
                    f"    '--snapshot-every', '{args.snapshot_every}',",
                    "    '--force',",
                    "]",
                    "print('Running:', ' '.join(cmd))",
                    "subprocess.check_call(cmd)",
                    "",
                ]
            )
        )

        sbatch_txt = generate_sbatch(run_py_name, args.partition, args.time, args.mem, args.mail_user)
        (run_dir / "Sbatch.sh").write_text(sbatch_txt)

    # Submit
    for run_dir in run_dirs:
        print(f"Created run: {run_dir.name}")
        if args.dry_run:
            continue
        os.chdir(run_dir)
        os.system("sbatch Sbatch.sh")
        os.chdir(root_dir)


if __name__ == "__main__":
    main()
