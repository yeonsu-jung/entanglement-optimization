#!/usr/bin/env python3
"""Dynamics perturbation step for web visualization pipeline.

For a given relaxed packing, this script:
  1. Computes per-rod minimum gap (proxy for Free Solid Angle) using JAX.
  2. Identifies the tightest rod (smallest gap) and loosest rod (largest gap).
  3. Runs three C++ rigid-body dynamics simulations:
       tight  — only the tightest rod is given an initial velocity kick
       loose  — only the loosest rod is kicked
       all    — all rods are kicked simultaneously
  4. Converts perrod.csv output to (T, N, 6) trajectory.npy files.

Requires:
  rigidbody_viewer_3d  (rod-dynamics-3d C++ binary, build_wsl recommended)

Usage
-----
python pipeline/perturb.py <run_dir> [--binary /path/to/rigidbody_viewer_3d]

Output layout
-------------
<run_dir>/dynamics/
    metadata.json          tight/loose rod info, simulation params
    tight/
        trajectory.npy     (T, N, 6) float32 endpoints
        perrod.csv         raw per-rod output from simulator
        scene.json
        stdout.log / stderr.log
    loose/
        ...
    all/
        ...
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


# ── Default binary search paths ───────────────────────────────────────────
_DYN_ROOT = Path("/mnt/c/Users/yjung/Documents/GitHub/rod-dynamics-3d")
DEFAULT_BINARY_CANDIDATES = [
    _DYN_ROOT / "build_wsl"        / "rigidbody_viewer_3d",
    _DYN_ROOT / "build_wsl_gl"     / "rigidbody_viewer_3d",
    _DYN_ROOT / "build_wsl_cuda"   / "rigidbody_viewer_3d",
    _DYN_ROOT / "build_wsl_gl_cuda"/ "rigidbody_viewer_3d",
    _DYN_ROOT / "build"            / "rigidbody_viewer_3d",
]


# ── CLI ────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Find tight/loose rods and run dynamics perturbation simulations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("run_dir",
                   help="Relax run directory (contains AR*/ subdirs)")
    p.add_argument("--binary", default=None,
                   help="Path to rigidbody_viewer_3d binary (auto-detected if omitted)")
    p.add_argument("--output-dir", default=None,
                   help="Output directory (default: <run_dir>/dynamics/)")
    p.add_argument("--steps", type=int, default=200_000,
                   help="Total simulation time-steps")
    p.add_argument("--dt", type=float, default=0.0005,
                   help="Simulation time-step")
    p.add_argument("--lin-vel", type=float, default=0.1,
                   help="Initial linear velocity sigma for perturbed rods")
    p.add_argument("--ang-vel", type=float, default=0.2,
                   help="Initial angular speed for perturbed rods")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--frictions", type=str, default="0,0.1,0.4",
                   help="Comma-separated friction coefficients to sweep")
    p.add_argument("--frames", type=int, default=300,
                   help="Number of trajectory frames to save per case")
    p.add_argument("--cases", nargs="+", choices=["tight", "loose", "all"],
                   default=["tight", "loose", "all"])
    p.add_argument("--force", action="store_true",
                   help="Recompute cases even if trajectory.npy already exists")
    return p.parse_args()


# ── Helpers ────────────────────────────────────────────────────────────────

def resolve_binary(explicit: str | None) -> Path:
    if explicit:
        b = Path(explicit)
        if not b.exists():
            raise FileNotFoundError(f"Binary not found: {b}")
        return b
    for c in DEFAULT_BINARY_CANDIDATES:
        if c.exists():
            return c
    tried = "\n  ".join(str(c) for c in DEFAULT_BINARY_CANDIDATES)
    raise FileNotFoundError(
        f"No dynamics binary found. Tried:\n  {tried}\n"
        "Pass --binary explicitly."
    )


def find_final_ar_dir(run_dir: Path) -> Path:
    """Return the AR dir with smallest AR number (thickest rods = final relaxed state)."""
    ar_dirs = sorted(run_dir.glob("AR*/"), key=lambda p: int(p.stem[2:]))
    if not ar_dirs:
        raise FileNotFoundError(f"No AR*/ directories found under {run_dir}")
    return ar_dirs[0]   # smallest AR = thickest rods


def compute_per_rod_min_gap(x_relaxed: np.ndarray, rod_diameter: float) -> np.ndarray:
    """Return per-rod minimum gap (min distance to nearest neighbor − rod_diameter)."""
    sys.path.insert(0, str(Path(__file__).parent))
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from jax import vmap, jit
    from physics import dist_lin_seg

    starts = jnp.array(x_relaxed[:, :3])
    ends   = jnp.array(x_relaxed[:, 3:])
    N = starts.shape[0]

    @jit
    def _min_dist(i):
        dists = vmap(lambda s, e: dist_lin_seg(starts[i], ends[i], s, e))(starts, ends)
        return jnp.min(jnp.where(jnp.arange(N) != i, dists, jnp.inf))

    print("  Computing per-rod minimum gaps (JAX)...")
    min_dists = np.array([float(_min_dist(i)) for i in range(N)])
    return min_dists - rod_diameter


def make_scene_json(N: int, rod_radius: float,
                    lin_vel: float, ang_vel: float,
                    seed: int, dt: float, friction: float) -> dict:
    return {
        "scene": {
            "populate": {
                "count": int(N),
                "length": 1.0,
                "radius": float(rod_radius),
            },
            "randomInit": {
                "enabled": True,
                "vSigma": float(lin_vel),
                "wSpeed": float(ang_vel),
                "seed": int(seed),
            },
            "randomForce": {"enabled": False},
            "bodies": [{
                "length": 1.0,
                "diameter": float(2.0 * rod_radius),
                "density": 1000.0,
                "restitution": 1.0,
                "friction": float(friction),
                "friction_s": float(friction),
                "friction_d": float(friction),
            }],
        },
        "physics": {
            "dt": float(dt),
            "gravity": [0.0, 0.0, 0.0],
            "lin_damp": 0.0,
            "ang_damp": 0.0,
            "substeps": 1,
            "soft_contact": {
                "enabled": True,
                "delta": 0.0,
                "k_scaler": 100.0,
                "mu": float(friction),
                "mu_static": float(friction),
                "nu": 1e-9,
                "enable_friction": True,
                "use_aabb": True,
            },
        },
    }


def perrod_csv_to_traj(perrod_csv: Path, N: int, rod_length: float = 1.0) -> np.ndarray:
    """Convert perrod.csv → (T, N, 6) float32 endpoint trajectory (fully vectorised).

    Rod axis convention (confirmed from convert_perrod_endpoints.cpp):
      axis = rotate_by_quat(qw, qx, qy, qz)  applied to  (0, 1, 0)
      endpoint1 = center − axis * (length / 2)
      endpoint2 = center + axis * (length / 2)
    """
    import pandas as pd
    df = pd.read_csv(perrod_csv, comment='#')
    df = df.sort_values(['frame', 'rod'])
    T = df['frame'].nunique()

    qw = df['qw'].to_numpy().reshape(T, N)
    qx = df['qx'].to_numpy().reshape(T, N)
    qy = df['qy'].to_numpy().reshape(T, N)
    qz = df['qz'].to_numpy().reshape(T, N)
    centers = df[['px', 'py', 'pz']].to_numpy().reshape(T, N, 3)

    # Rotate (0, 1, 0) by unit quaternion (qw, qx, qy, qz):
    #   t  = 2 * cross([qx,qy,qz], [0,1,0]) = [-2qz, 0, 2qx]
    #   v' = [0,1,0] + qw*t + cross([qx,qy,qz], t)
    ax = 2 * (-qw * qz + qy * qx)
    ay = 1  - 2 * (qx**2 + qz**2)
    az = 2 * ( qw * qx + qy * qz)
    axis = np.stack([ax, ay, az], axis=2)   # (T, N, 3)

    half = rod_length / 2.0
    p1 = (centers - half * axis).astype(np.float32)
    p2 = (centers + half * axis).astype(np.float32)
    return np.concatenate([p1, p2], axis=2)  # (T, N, 6)


def run_case(
    case_name: str,
    rod_index: int | None,
    endpoints_csv: Path,
    N: int,
    rod_radius: float,
    binary: Path,
    case_dir: Path,
    steps: int,
    dt: float,
    lin_vel: float,
    ang_vel: float,
    seed: int,
    friction: float,
    frames: int,
) -> Path:
    case_dir.mkdir(parents=True, exist_ok=True)

    scene_path = case_dir / "scene.json"
    scene_path.write_text(
        json.dumps(make_scene_json(N, rod_radius, lin_vel, ang_vel, seed, dt, friction), indent=2)
    )

    stride     = max(1, steps // frames)
    perrod_path = case_dir / "perrod.csv"

    cmd = [
        str(binary), "--headless",
        "--scene",          str(scene_path),
        "--init-csv",       str(endpoints_csv),
        "--output",         str(case_dir / "output.csv"),
        "--output-stride",  str(stride),
        "--output-max",     str(frames),
        "--perrod",         str(perrod_path),
        "--perrod-stride",  str(stride),
        "--perrod-max",     str(frames),
        "--steps",          str(steps),
        "--dt",             str(dt),
        "--seed",           str(seed),
        "--no-network", "--no-csv",
    ]
    if rod_index is not None:
        cmd.extend(["--perturb-rod", str(rod_index)])

    print(f"  [{case_name}] running {steps:,} steps → {frames} frames ...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    (case_dir / "stdout.log").write_text(result.stdout)
    (case_dir / "stderr.log").write_text(result.stderr)

    if result.returncode != 0:
        raise RuntimeError(
            f"Dynamics failed for case '{case_name}' (exit {result.returncode}).\n"
            f"See {case_dir / 'stderr.log'}"
        )

    traj = perrod_csv_to_traj(perrod_path, N)
    traj_path = case_dir / "trajectory.npy"
    np.save(traj_path, traj)
    print(f"  [{case_name}] → {traj_path}  shape={traj.shape}")
    return traj_path


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    args     = parse_args()
    run_dir  = Path(args.run_dir).resolve()
    out_dir  = Path(args.output_dir).resolve() if args.output_dir else run_dir / "dynamics"
    binary   = resolve_binary(args.binary)

    ar_dir      = find_final_ar_dir(run_dir)
    ar          = int(ar_dir.stem[2:])
    rod_diameter = 1.0 / ar
    rod_radius   = rod_diameter / 2.0
    endpoints_csv = ar_dir / "endpoints.csv"

    x_relaxed = np.loadtxt(endpoints_csv, delimiter=',', comments='#').astype(np.float64)
    N = x_relaxed.shape[0]

    print(f"Run dir    : {run_dir}")
    print(f"Final AR   : AR{ar}  radius={rod_radius:.6f}  N={N}")
    print(f"Binary     : {binary}")
    print(f"Output     : {out_dir}")
    print(f"Cases      : {args.cases}")
    print()

    # ── Identify tight and loose rods ─────────────────────────────────────
    min_gaps   = compute_per_rod_min_gap(x_relaxed, rod_diameter)
    tight_idx  = int(np.argmin(min_gaps))
    loose_idx  = int(np.argmax(min_gaps))
    print(f"  Tight rod: idx={tight_idx:>3d}  gap={min_gaps[tight_idx]:.6f}")
    print(f"  Loose rod: idx={loose_idx:>3d}  gap={min_gaps[loose_idx]:.6f}")
    print()

    # ── Write metadata ────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "AR":         ar,
        "rod_radius": rod_radius,
        "N":          N,
        "tight_rod":  {"index": tight_idx, "min_gap": float(min_gaps[tight_idx])},
        "loose_rod":  {"index": loose_idx, "min_gap": float(min_gaps[loose_idx])},
        "steps":      args.steps,
        "dt":         args.dt,
        "frames":     args.frames,
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

    # ── Run each perturbation case × friction sweep ───────────────────────
    frictions = [float(f) for f in args.frictions.split(",")]
    case_rod  = {"tight": tight_idx, "loose": loose_idx, "all": None}
    for case in args.cases:
        for mu in frictions:
            mu_str    = f"mu{mu:.4g}"
            case_dir  = out_dir / case / mu_str
            traj_existing = case_dir / "trajectory.npy"
            if traj_existing.exists() and not args.force:
                print(f"  [{case}/{mu_str}] cached → {traj_existing}  (use --force to recompute)")
                continue
            run_case(
                case_name    = f"{case}/{mu_str}",
                rod_index    = case_rod[case],
                endpoints_csv= endpoints_csv,
                N            = N,
                rod_radius   = rod_radius,
                binary       = binary,
                case_dir     = case_dir,
                steps        = args.steps,
                dt           = args.dt,
                lin_vel      = args.lin_vel,
                ang_vel      = args.ang_vel,
                seed         = args.seed,
                friction     = mu,
                frames       = args.frames,
            )

    print(f"\nPerturb done → {out_dir}")


if __name__ == "__main__":
    main()
