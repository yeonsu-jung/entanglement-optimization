#!/usr/bin/env python3
import argparse
import datetime
import os
import shutil
import time
from pathlib import Path

# ── JAX GPU setup ────────────────────────────────────────────────────────
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from jax import grad

import potentials as pt
from potentials import (
    create_pairs,
    total_effective_potential,
    all_pairwise_distances,
)
from transforms import q_to_x, x_to_q
import protocols as pr

def _print_device_info() -> None:
    """Print JAX device info and confirm GPU is available."""
    try:
        devices = jax.devices()
        print("=" * 60)
        print("JAX device info")
        print(f"  Default backend : {jax.default_backend()}")
        print(f"  Devices         : {devices}")
        for d in devices:
            print(f"    {d.device_kind} | {d.platform} | id={d.id}")
        if jax.default_backend() != "gpu":
            print("  ⚠  WARNING: JAX is NOT using a GPU.")
        else:
            print("  ✓  GPU detected — computations will run on GPU.")
        print("=" * 60)
    except RuntimeError as e:
        print("=" * 60)
        print("JAX device info")
        print(f"  ⚠  WARNING: Could not initialize JAX devices: {e}")
        print("  Falling back to CPU backend.")
        jax.config.update("jax_platform_name", "cpu")
        print(f"  Default backend : {jax.default_backend()}")
        print("=" * 60)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run sequential relaxation on a specific packing file."
    )
    p.add_argument("input_file", type=str, help="Path to the .txt or .npy packing file.")
    p.add_argument(
        "--AR-list",
        type=str,
        default="1000,500,300,200,100,50",
        help="Comma-separated AR values. Default: 1000,500,300,200,100,50",
    )
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument(
        "--max-relax-iters", type=int, default=1000000,
        help="Max iterations per relaxation step (default: 1M)",
    )
    p.add_argument("--relax-dt", type=float, default=1e-4)
    p.add_argument("--clearance", type=float, default=1.005)
    p.add_argument("--amp", type=float, default=100.0)
    p.add_argument("--force", action="store_true", help="Overwrite existing directory.")

    return p.parse_args()


def load_packing(path: Path) -> jnp.ndarray:
    """Load packing from .txt or .npy and return flattened q config."""
    print(f"Loading packing from {path} ...")
    if path.suffix == ".npy":
        data = np.load(path)
    else:
        data = np.loadtxt(path)

    # Determine if it's x-format (N, 6) or q-format (N, 5) or flattened
    if data.ndim == 1:
        # Assume it's already a flattened q config
        return jnp.asarray(data, dtype=jnp.float64)
    
    rows, cols = data.shape
    if cols == 6:
        print("Detected x-format (6 columns). Converting to q-format...")
        q_shaped = x_to_q(data)
        return jnp.asarray(q_shaped, dtype=jnp.float64).flatten()
    elif cols == 5:
        print("Detected q-format (5 columns).")
        return jnp.asarray(data, dtype=jnp.float64).flatten()
    else:
        raise ValueError(f"Unexpected data shape: {data.shape}")


def _save_ar_result(
    q_relaxed: jnp.ndarray,
    ar_dir: Path,
    col_rad: float,
    AR: int,
    scale_factor: float,
    t_relax: float,
    q_initial: jnp.ndarray,
) -> None:
    ar_dir.mkdir(parents=True, exist_ok=True)
    rod_diameter = 2.0 * col_rad

    q_np = np.asarray(q_relaxed)
    np.savetxt(ar_dir / "q_relaxed.txt", q_np)
    np.save(ar_dir / "q_relaxed.npy", q_np)

    x_relaxed = np.asarray(q_to_x(jnp.asarray(q_relaxed)))
    center = np.mean((x_relaxed[:, :3] + x_relaxed[:, 3:]) / 2, axis=0)
    x_centered = x_relaxed - np.concatenate([center, center])
    x_scaled = scale_factor * x_centered
    np.savetxt(ar_dir / "x_relaxed.txt", x_scaled)
    np.save(ar_dir / "x_relaxed.npy", x_scaled)

    with open(ar_dir / "endpoints_formatted.csv", "w") as f:
        f.write(f"# rod_radius={col_rad}\n")
        f.write(f"# rod_length=1\n")
        for row in x_scaled:
            f.write(",".join(f"{v:.10f}" for v in row) + "\n")

    q_pairs = create_pairs(jnp.reshape(jnp.asarray(q_relaxed), (-1, 5)))
    d = all_pairwise_distances(q_pairs)
    initial_energy = float(total_effective_potential(jnp.asarray(q_initial)))
    final_energy = float(total_effective_potential(jnp.asarray(q_relaxed)))

    log_lines = [
        f"AR: {AR}",
        f"rod_diameter: {rod_diameter}",
        f"Initial potential: {initial_energy}",
        f"Final potential: {final_energy}",
        f"Minimum pairwise distance: {float(jnp.min(d)):.8f}",
        f"Relaxation time: {t_relax:.2f}s",
        f"Backend: {jax.default_backend()}",
    ]
    (ar_dir / "log.txt").write_text("\n".join(log_lines) + "\n")
    print(f"  AR={AR} Complete. min_d={float(jnp.min(d)):.6e}")


def main() -> None:
    _print_device_info()
    args = parse_args()

    input_path = Path(args.input_file).resolve()
    if not input_path.exists():
        raise SystemExit(f"Input file {input_path} not found.")

    ar_list = sorted(
        [int(x.strip()) for x in args.AR_list.split(",") if x.strip()],
        reverse=True,
    )

    # 1. Load data
    q0 = load_packing(input_path)
    num_rods = q0.size // 5

    # 2. Setup output directory in the same place as input
    now = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    out_dir = input_path.parent / f"{now}_RelaxationResults_{input_path.stem}"
    if out_dir.exists() and not args.force:
        print(f"Output directory {out_dir} already exists. Use --force to overwrite.")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be saved to: {out_dir}")

    # 3. Preserve original file
    shutil.copy2(input_path, out_dir / f"ORIGINAL_{input_path.name}")
    print(f"Original file copied to {out_dir / f'ORIGINAL_{input_path.name}'}")

    # 4. Warm up JIT
    print("Warming up JIT...")
    _ = total_effective_potential(jnp.zeros_like(q0))
    jax.block_until_ready(_)

    # 5. Iterative Relaxation
    q_current = q0
    for AR in ar_list:
        rod_diameter = 1.0 / float(AR)
        params = {"col_rad": rod_diameter / 2.0, "amp": args.amp, "sigma": 0.025}
        
        t0 = time.time()
        q_relaxed = pr.gpu_relax_collision(
            q_current,
            args.relax_dt,
            params,
            max_iters=args.max_relax_iters,
            effective_diameter_factor=args.clearance,
        )
        jax.block_until_ready(q_relaxed)
        dt = time.time() - t0

        _save_ar_result(
            q_relaxed=q_relaxed,
            ar_dir=out_dir / f"AR{AR}",
            col_rad=params["col_rad"],
            AR=AR,
            scale_factor=args.scale,
            t_relax=dt,
            q_initial=q_current
        )
        q_current = q_relaxed

    print(f"\nAll done. Outputs in {out_dir}")

if __name__ == "__main__":
    main()
