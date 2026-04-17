#!/usr/bin/env python3
"""Full pipeline: entangle → relax → perturb → standalone web viewer.

Steps
-----
1. eo-entangle   – maximise linking number at thin AR, save trajectory
2. eo-relax      – inflate rods step by step, save trajectory snapshots
3. eo-perturb    – identify tightest/loosest rods, run 3 dynamics cases
4. web_export    – embed all trajectories into a self-contained index.html

Usage
-----
# Full run from scratch
eo-make-video --num-rods 200 --out-dir results/my_run \\
    --dynamics-binary /path/to/rigidbody_viewer_3d

# Skip entanglement (use an existing q_entangled.npy)
eo-make-video --q-path results/my_run/entangled/.../q_entangled.npy \\
    --dynamics-binary /path/to/rigidbody_viewer_3d

# Skip all computation, only re-export the web viewer
eo-make-video --q-path .../q_entangled.npy --skip-perturb

Stride knobs (all adjustable here)
------------------------------------
  --entangle-stride   iterations between snapshots during entanglement
  --relax-stride      iterations between snapshots during relaxation
  --perturb-frames    number of trajectory frames saved per dynamics case
  --every             frame decimation applied to ALL phases at export time
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


# ── Helpers ────────────────────────────────────────────────────────────────

def run_step(label: str, cmd: list) -> None:
    """Print and execute a command, exit on failure."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print("$", " ".join(str(c) for c in cmd))
    print()
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(f"\nStep '{label}' failed (exit code {result.returncode})")


def latest(parent: Path, glob: str) -> Path:
    """Return the most recently modified path matching glob under parent."""
    matches = sorted(parent.glob(glob), key=lambda p: p.stat().st_mtime)
    if not matches:
        raise FileNotFoundError(f"No match for '{glob}' under {parent}")
    return matches[-1]


# ── CLI ────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Entangle → Relax → Perturb → Web viewer full pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Geometry ──────────────────────────────────────────────────────────
    p.add_argument("--num-rods", type=int, default=200)
    p.add_argument("--AR-list", type=str, default="1000,500,300,200,100",
                   help="AR values for relaxation, largest→smallest")

    # ── Output ────────────────────────────────────────────────────────────
    p.add_argument("--out-dir", type=str, default="results/pipeline",
                   help="Root output directory")
    p.add_argument("--force", action="store_true",
                   help="Recompute all steps even if outputs already exist")

    # ── Entanglement ──────────────────────────────────────────────────────
    p.add_argument("--q-path", type=str, default=None,
                   help="Skip entanglement, use this existing q_entangled.npy")
    p.add_argument("--entangle-Nmax", type=int, default=10_000,
                   help="Max FIRE iterations for entanglement")
    p.add_argument("--entangle-stride", type=int, default=500,
                   help="Iterations between snapshots during entanglement")

    # ── Relaxation ────────────────────────────────────────────────────────
    p.add_argument("--relax-max-iters", type=int, default=1_000_000,
                   help="Max FIRE iterations per AR step")
    p.add_argument("--relax-dt", type=float, default=1e-4,
                   help="FIRE time step for relaxation")
    p.add_argument("--relax-stride", type=int, default=10_000,
                   help="Iterations between snapshots during relaxation")

    # ── Perturbation dynamics ──────────────────────────────────────────────
    p.add_argument("--skip-perturb", action="store_true",
                   help="Skip the dynamics perturbation step")
    p.add_argument("--dynamics-binary", type=str, default=None,
                   help="Path to rigidbody_viewer_3d binary (auto-detected if omitted)")
    p.add_argument("--perturb-steps", type=int, default=200_000,
                   help="Total simulation time-steps per dynamics case")
    p.add_argument("--perturb-frames", type=int, default=300,
                   help="Trajectory frames saved per dynamics case "
                        "(stride = steps / frames)")
    p.add_argument("--perturb-dt", type=float, default=0.0005,
                   help="Simulation time-step for dynamics")
    p.add_argument("--perturb-lin-vel", type=float, default=0.1,
                   help="Initial linear velocity sigma for perturbed rods")
    p.add_argument("--perturb-ang-vel", type=float, default=0.2,
                   help="Initial angular speed for perturbed rods")
    p.add_argument("--perturb-seed", type=int, default=42)
    p.add_argument("--perturb-frictions", type=str, default="0,0.1,0.4",
                   help="Comma-separated friction coefficients to sweep")
    p.add_argument("--cases", nargs="+", choices=["tight", "loose", "all"],
                   default=["tight", "loose", "all"],
                   help="Which perturbation cases to run")

    # ── Web export ────────────────────────────────────────────────────────
    p.add_argument("--every", type=int, default=1,
                   help="Keep every Nth frame across relax+perturb phases at export time")
    p.add_argument("--entangle-frames", type=int, default=300,
                   help="Use only the first N frames of the entanglement trajectory "
                        "(stride=1, independent of --every)")

    return p.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Entangle ──────────────────────────────────────────────────
    entangle_traj: Path | None = None   # passed explicitly to export_web

    if args.q_path:
        q_path = Path(args.q_path).resolve()
        print(f"Using existing q: {q_path}")
        # Reuse the trajectory saved alongside the q file (if present)
        candidate = q_path.parent / "trajectory.npy"
        if candidate.exists() and not args.force:
            entangle_traj = candidate
            print(f"  Entangle trajectory: {entangle_traj}")
        else:
            # No trajectory yet — run entangle just to get the visualisation.
            # The resulting q is discarded; only trajectory.npy is kept.
            entangle_dir = out / "entangled"
            cmd = [
                sys.executable, "-m", "entanglement_optimization.cli.entangle",
                "--num-rods",  str(args.num_rods),
                "--AR",        "1000",
                "--Nmax",      str(args.entangle_Nmax),
                "--out-dir",   str(entangle_dir),
                "--save-traj",
                "--stride",    str(args.entangle_stride),
            ]
            if args.force:
                cmd.append("--force")
            run_step("Entangle (trajectory only)", cmd)
            entangle_traj = latest(entangle_dir, "*/trajectory.npy")
            print(f"  Generated entangle trajectory: {entangle_traj}")
    else:
        entangle_dir = out / "entangled"
        cmd = [
            sys.executable, here / "entangle.py",
            "--num-rods",  str(args.num_rods),
            "--AR",        "1000",
            "--Nmax",      str(args.entangle_Nmax),
            "--out-dir",   str(entangle_dir),
            "--save-traj",
            "--stride",    str(args.entangle_stride),
        ]
        if args.force:
            cmd.append("--force")
        run_step("Entangle", cmd)
        q_path = latest(entangle_dir, "*/q_entangled.npy")
        entangle_traj = q_path.parent / "trajectory.npy"
        print(f"\nEntangled config: {q_path}")

    # ── Step 2: Relax ─────────────────────────────────────────────────────
    relax_cmd = [
        sys.executable, "-m", "entanglement_optimization.cli.relax",
        str(q_path),
        "--AR-list",   args.AR_list,
        "--max-iters", str(args.relax_max_iters),
        "--relax-dt",  str(args.relax_dt),
        "--save-traj",
        "--stride",    str(args.relax_stride),
    ]
    if args.force:
        relax_cmd.append("--force")
    run_step("Relax", relax_cmd)
    run_dir = latest(q_path.parent, "*_Relaxed-N*")
    print(f"\nRelaxation dir: {run_dir}")

    # ── Step 3: Perturb ───────────────────────────────────────────────────
    if not args.skip_perturb:
        perturb_cmd = [
            sys.executable, "-m", "entanglement_optimization.cli.perturb",
            str(run_dir),
            "--steps",   str(args.perturb_steps),
            "--dt",      str(args.perturb_dt),
            "--lin-vel", str(args.perturb_lin_vel),
            "--ang-vel", str(args.perturb_ang_vel),
            "--seed",      str(args.perturb_seed),
            "--frictions", args.perturb_frictions,
            "--frames",    str(args.perturb_frames),
            "--cases",     *args.cases,
        ]
        if args.dynamics_binary:
            perturb_cmd.extend(["--binary", str(args.dynamics_binary)])
        if args.force:
            perturb_cmd.append("--force")
        run_step("Perturb dynamics (tight / loose / all) × friction sweep", perturb_cmd)
    else:
        print("\nSkipping perturbation step (--skip-perturb).")

    # ── Step 4: Export web viewer ─────────────────────────────────────────
    export_cmd = [
        sys.executable, "-m", "entanglement_optimization.viz.web_export",
        str(run_dir),
        "--every",           str(args.every),
        "--entangle-frames", str(args.entangle_frames),
    ]
    if entangle_traj and entangle_traj.exists():
        export_cmd.extend(["--entangle-traj", str(entangle_traj)])
    run_step("Export web viewer", export_cmd)

    web_html = run_dir / "web" / "index.html"
    print(f"\n{'='*60}")
    print(f"  Pipeline complete!")
    print(f"  Viewer : {web_html}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
