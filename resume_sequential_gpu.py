#!/usr/bin/env python3
import argparse
import datetime
import os
import time
from pathlib import Path

# ── JAX GPU setup ────────────────────────────────────────────────────────
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from jax import grad

from potentials import (
    create_pairs,
    total_effective_potential,
    all_pairwise_distances,
)
from transforms import q_to_x
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
        description="Resume sequential multi-AR relaxation (GPU) from a pre-entangled packing."
    )
    p.add_argument("entangled_path", type=str, help="Path to the directory containing q_entangled.npy")
    p.add_argument(
        "--AR-list",
        type=str,
        default="1000,500,300",
        help="Comma-separated AR values ordered largest→smallest. Default: 1000,500,300",
    )
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument(
        "--max-relax-iters", type=int, default=1000000,
        help="Max gradient-descent iterations per relaxation step (default: 1M)",
    )
    p.add_argument("--relax-dt", type=float, default=1e-4)
    p.add_argument("--clearance", type=float, default=1.005)
    p.add_argument("--amp", type=float, default=100.0)
    p.add_argument("--force", action="store_true",
                   help="Recompute even if cached files exist")

    return p.parse_args()


def _save_ar_result(
    q_relaxed: jnp.ndarray,
    ar_dir: Path,
    col_rad: float,
    AR: int,
    scale_factor: float,
    t_relax: float,
    q_entangled: jnp.ndarray,
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
    ent_initial = float(total_effective_potential(jnp.asarray(q_entangled)))
    ent_final = float(total_effective_potential(jnp.asarray(q_relaxed)))

    log_lines = [
        f"AR: {AR}",
        f"rod_diameter: {rod_diameter}",
        f"rod_radius: {col_rad}",
        f"Initial entanglement potential: {ent_initial}",
        f"Final entanglement potential: {ent_final}",
        f"Minimum pairwise distance: {float(jnp.min(d)):.8f}",
        f"Median pairwise distance: {float(jnp.median(d)):.8f}",
        f"Rod pairs in contact (d < diameter): {int(jnp.count_nonzero(d < rod_diameter))}",
        f"Total rod pairs: {q_pairs.shape[0]}",
        f"Relaxation time: {t_relax:.2f}s",
        f"Backend: {jax.default_backend()}",
    ]
    log_text = "\n".join(log_lines) + "\n"
    with open(ar_dir / "log_standard_protocol.txt", "w") as f:
        f.write(log_text)
    print(log_text, end="")


def main() -> None:
    _print_device_info()
    args = parse_args()

    ar_list = sorted(
        [int(x.strip()) for x in args.AR_list.split(",") if x.strip()],
        reverse=True,
    )
    if not ar_list:
        raise SystemExit("--AR-list is empty.")
    print(f"AR sequence: {ar_list}")

    entangled_dir = Path(args.entangled_path).resolve()
    if not entangled_dir.is_dir():
        raise SystemExit(f"Directory {entangled_dir} not found.")
        
    q_file = entangled_dir / "q_entangled.npy"
    if not q_file.exists():
        raise SystemExit(f"File {q_file} not found inside {entangled_dir}.")

    print(f"Loading entangled configuration from {q_file}")
    q_entangled = np.load(q_file)
    q_entangled = jnp.asarray(q_entangled, dtype=jnp.float64).flatten()
    
    num_rods = q_entangled.size // 5
    scale_factor = args.scale

    now = datetime.datetime.now()
    dt_string = now.strftime("%Y-%m-%d_%H")
    packing_id = f"{dt_string}_ResumedSequentialRelaxedPacking-N{num_rods}-Scale{scale_factor:g}"
    
    packing_dir = entangled_dir / packing_id
    packing_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {packing_dir}\n")

    print("Warming up JIT compilation...")
    _warmup_q = jnp.zeros(num_rods * 5, dtype=jnp.float64)
    _ = total_effective_potential(_warmup_q)
    jax.block_until_ready(_)
    print("JIT warm-up complete.\n")

    print("=" * 60)
    print("  STEP 2: Sequential relaxation")
    print(f"  AR sequence: {' -> '.join(str(a) for a in ar_list)}")
    print("=" * 60 + "\n")

    q_current = q_entangled

    for idx, AR in enumerate(ar_list):
        rod_diameter = 1.0 / float(AR)
        col_rad = rod_diameter / 2.0
        ar_dir = packing_dir / f"AR{AR}"
        result_path = ar_dir / "q_relaxed.npy"

        print(f"[{idx+1}/{len(ar_list)}] AR={AR}  "
              f"diameter={rod_diameter:.6f} ...")

        if result_path.exists() and not args.force:
            print(f"  Cached result found, loading {result_path}")
            q_current = jnp.asarray(
                np.load(result_path), dtype=jnp.float64
            ).flatten()
            continue

        params = {"col_rad": col_rad, "amp": args.amp, "sigma": 0.025}
        t_start = time.time()

        q_relaxed = pr.gpu_relax_collision(
            q_current,
            args.relax_dt,
            params,
            max_iters=args.max_relax_iters,
            effective_diameter_factor=args.clearance,
        )
        jax.block_until_ready(q_relaxed)
        t_relax = time.time() - t_start

        print(f"  Done in {t_relax:.2f}s")

        _save_ar_result(
            q_relaxed=q_relaxed,
            ar_dir=ar_dir,
            col_rad=col_rad,
            AR=AR,
            scale_factor=scale_factor,
            t_relax=t_relax,
            q_entangled=q_entangled,
        )

        q_current = q_relaxed

    print(f"\nAll done. Outputs: {packing_dir}")

if __name__ == "__main__":
    main()
