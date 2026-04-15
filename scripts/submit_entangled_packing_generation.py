#!/usr/bin/env python3
"""submit_entangled_packing_generation.py

Submit SLURM jobs to generate entangled packings, one job per `num_rods`.

Each job runs:
  python scripts/entangled_packing_generation.py --num-rods <N> --random-keys a,b,c ...

Run folders are created under:
  <runs-root>/<job-name>/<timestamp>_N####_keysA,B,C/

Notes
- DEFAULT_RUNS_ROOT is a *base* path; by default we create runs under DEFAULT_RUNS_ROOT/runs.
- This script does not copy the repo; it `cd`s to the repo root and runs the generator in-place.
"""

import argparse
import os
import re
import shutil
import stat
import subprocess

from datetime import datetime
from pathlib import Path
from typing import List, Optional


# User-requested base path
DEFAULT_RUNS_ROOT = Path(
    "/n/holylabs/LABS/mahadevan_lab/Users/yjung/entanglement-optimization"
)


def find_root_dir(start: Optional[Path] = None, target_name: str = "entanglement-optimization") -> Path:
    p = (Path.cwd() if start is None else start).resolve()
    for ancestor in [p, *p.parents]:
        if ancestor.name == target_name:
            return ancestor
    raise SystemExit(f"Could not find repository root named '{target_name}' starting from {p}")


def now_ts() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._\-]+", "_", s)


def parse_int_list(csv: str) -> List[int]:
    out: List[int] = []
    for part in (csv or "").split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def shlex_quote(s: str) -> str:
    if s == "":
        return "''"
    if re.fullmatch(r"[A-Za-z0-9_@%+=:,./\-]+", s):
        return s
    return "'" + s.replace("'", "'\\''") + "'"


def ensure_executable(path: Path) -> None:
    if not path.exists():
        raise SystemExit(f"File not found: {path}")
    if not os.access(path, os.X_OK):
        os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR)


class SlurmCfg:
    def __init__(
        self,
        partition: str = "seas_compute",
        time: str = "0-04:00",
        mem_gb: int = 8,
        ntasks: int = 1,
        cpus: int = 1,
        nodes: int = 1,
        mail_user: str = os.environ.get("USER_EMAIL", ""),
        mail_type: str = "END",
        module_line: str = "module load python",
    ):
        self.partition = partition
        self.time = time
        self.mem_gb = mem_gb
        self.ntasks = ntasks
        self.cpus = cpus
        self.nodes = nodes
        self.mail_user = mail_user
        self.mail_type = mail_type
        self.module_line = module_line


def main() -> None:
    ap = argparse.ArgumentParser(description="Submit SLURM jobs to generate entangled packings (one job per N).")
    ap.add_argument("--job-name", type=str, default="gen_entangled_packings", help="Runs subfolder name and SLURM job-name prefix.")
    ap.add_argument(
        "--runs-root",
        type=Path,
        default=None,
        help="Where to create run folders (default: DEFAULT_RUNS_ROOT/runs).",
    )
    ap.add_argument(
        "--python",
        type=str,
        default="python3",
        help="Python executable to use in the job (e.g. a conda env python path).",
    )
    ap.add_argument(
        "--num-rods-list",
        type=str,
        default="10,20,50,100,200,500",
        help="Comma-separated N values. One job will be submitted per N.",
    )
    ap.add_argument(
        "--random-keys",
        type=str,
        default="56,321,194",
        help="Comma-separated PRNG keys (exactly 3 ints).",
    )
    ap.add_argument("--Nmax", type=int, default=300)
    ap.add_argument("--N-outer", type=int, default=5)
    ap.add_argument("--dt", type=float, default=1e-2)
    ap.add_argument("--atol", type=float, default=1e-8)
    ap.add_argument("--initial-q", type=str, default="gathered")
    ap.add_argument("--rod-diameter", type=float, default=-1.0)
    ap.add_argument("--scale-factor", type=int, default=1)

    ap.add_argument("--partition", type=str, default="seas_compute")
    ap.add_argument("--time", type=str, default="0-04:00")
    ap.add_argument("--mem-gb", type=int, default=8)
    ap.add_argument("--cpus", type=int, default=1)
    ap.add_argument("--nodes", type=int, default=1)
    ap.add_argument("--module-line", type=str, default="module load python")

    ap.add_argument("--dry-run", action="store_true", help="Create folders/scripts but do not call sbatch.")
    ap.add_argument("--limit", type=int, default=0, help="If >0, only submit the first N jobs.")

    args = ap.parse_args()

    root_dir = find_root_dir()
    gen_script = root_dir / "scripts" / "entangled_packing_generation.py"
    if not gen_script.exists():
        raise SystemExit(f"Generator script not found: {gen_script}")

    num_rods_list = parse_int_list(args.num_rods_list)
    if not num_rods_list:
        raise SystemExit("--num-rods-list produced an empty list")

    rk = parse_int_list(args.random_keys)
    if len(rk) != 3:
        raise SystemExit("--random-keys must contain exactly 3 integers, e.g. '56,321,194'")

    runs_root = args.runs_root if args.runs_root is not None else (DEFAULT_RUNS_ROOT / "runs")
    runs_root = runs_root / args.job_name
    runs_root.mkdir(parents=True, exist_ok=True)

    shutil.copy2(Path(__file__), runs_root / Path(__file__).name)

    timestamp = now_ts()
    if args.limit > 0:
        num_rods_list = num_rods_list[: args.limit]

    slurm = SlurmCfg(
        partition=args.partition,
        time=args.time,
        mem_gb=int(args.mem_gb),
        ntasks=1,
        cpus=int(args.cpus),
        nodes=int(args.nodes),
        module_line=args.module_line,
    )

    submitted = 0
    print(f"Repo root: {root_dir}")
    print(f"Runs root: {runs_root}")
    print(f"Submitting {len(num_rods_list)} jobs (one per N)")

    for num_rods in num_rods_list:
        run_name = safe_name(f"{timestamp}_N{num_rods:04d}_keys{rk[0]},{rk[1]},{rk[2]}")
        run_dir = runs_root / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        cmd = " ".join(
            [
                shlex_quote(args.python),
                shlex_quote(str(gen_script)),
                f"--num-rods {int(num_rods)}",
                f"--random-keys {rk[0]},{rk[1]},{rk[2]}",
                f"--Nmax {int(args.Nmax)}",
                f"--N-outer {int(args.N_outer)}",
                f"--dt {float(args.dt)}",
                f"--atol {float(args.atol)}",
                f"--initial-q {shlex_quote(str(args.initial_q))}",
                f"--rod-diameter {float(args.rod_diameter)}",
                f"--scale-factor {int(args.scale_factor)}",
            ]
        )

        sb = f"""#!/bin/bash
#SBATCH -n {slurm.ntasks}
#SBATCH -c {slurm.cpus}
#SBATCH -N {slurm.nodes}
#SBATCH -t {slurm.time}
#SBATCH -p {slurm.partition}
#SBATCH --mem={slurm.mem_gb}G
#SBATCH -o output_%j.out
#SBATCH -e errors_%j.err
#SBATCH --mail-type={slurm.mail_type}
{f"#SBATCH --mail-user={slurm.mail_user}" if slurm.mail_user else ""}
#SBATCH --job-name={safe_name(args.job_name)}_N{num_rods:04d}

set -euo pipefail
{slurm.module_line}

cd {shlex_quote(str(root_dir))}

echo "======================================"
echo "Entangled packing generation"
echo "N: {num_rods}"
echo "random_keys: {rk[0]},{rk[1]},{rk[2]}"
echo "PWD: $(pwd)"
echo "======================================"

echo "Running..."
echo "{cmd}"
{cmd}

echo "Job complete."
"""

        sbatch_path = run_dir / "Sbatch.sh"
        sbatch_path.write_text(sb)
        ensure_executable(sbatch_path)

        if not args.dry_run:
            print(f"Submitting {run_name}...")
            subprocess.run(["sbatch", "Sbatch.sh"], cwd=run_dir, check=True)
            submitted += 1
        else:
            print(f"Dry run: Created {run_dir}")

    print(f"Submitted {submitted} jobs.")


if __name__ == "__main__":
    main()
