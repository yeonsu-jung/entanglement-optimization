#!/usr/bin/env python3

"""GPU-accelerated entanglement + relaxation protocol.

Mirrors `standard_protocol` from protocols.py but with:
  - GPU-accelerated initialization (vmap distance checks)
  - vmap-vectorized potentials for GPU parallelism
  - Timing instrumentation
  - Clean CLI interface

Usage (on a GPU node):
    python standard_protocol_gpu.py <k1> <k2> <k3> --AR 1000 --num-rods 200
"""

from __future__ import annotations

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
from jax import grad, jit

import potentials as pt
from potentials import (
    create_pairs,
    total_effective_potential,
    total_harmonic_line,
    all_pairwise_distances,
)
from transforms import q_to_x
import protocols as pr


def _print_device_info() -> None:
    """Print JAX device info and confirm GPU is available."""
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run entanglement + relaxation protocol (GPU)."
    )
    p.add_argument("k1", type=int)
    p.add_argument("k2", type=int)
    p.add_argument("k3", type=int)

    p.add_argument("--AR", type=int, default=None,
                   help="Aspect ratio (length/diameter).")
    p.add_argument("--rod-diameter", type=float, default=None,
                   help="Rod diameter. Overrides --AR if provided.")

    p.add_argument("--num-rods", type=int, default=200)
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--dt", type=float, default=1e-2)
    p.add_argument("--Nmax-entangle", type=int, default=10000,
                   help="Max FIRE iters for entanglement (default: 10000)")
    p.add_argument("--N-outer-entangle", type=int, default=1,
                   help="Outer iters for entanglement (default: 1)")
    p.add_argument("--max-relax-iters", type=int, default=1000000,
                   help="Max gradient descent iters for relaxation (default: 1M)")
    p.add_argument("--relax-dt", type=float, default=1e-4,
                   help="Step size for relaxation GD (default: 1e-4)")
    p.add_argument("--clearance", type=float, default=1.005,
                   help="Effective diameter factor for relaxation potential (default: 1.005)")
    p.add_argument("--amp", type=float, default=100.0,
                   help="Harmonic potential amplitude (default: 100)")
    p.add_argument("--atol", type=float, default=1e-8)
    p.add_argument("--initial-q", type=str, default="non-intersecting",
                   choices=["non-intersecting", "test", "aligned", "random"])
    p.add_argument("--force", action="store_true",
                   help="Recompute even if cached files exist")
    p.add_argument("--save-trajectory", action=argparse.BooleanOptionalAction,
                   default=True)
    p.add_argument("--snapshot-every", type=int, default=1,
                   help="Keep every k-th callback snapshot.")

    return p.parse_args()


def main() -> None:
    _print_device_info()
    args = parse_args()

    random_keys = [args.k1, args.k2, args.k3]
    num_rods = args.num_rods
    scale_factor = args.scale

    # Rod diameter / AR logic
    if args.rod_diameter is not None:
        rod_diameter = float(args.rod_diameter)
        AR = None
    elif args.AR is not None:
        AR = int(args.AR)
        rod_diameter = 1.0 / float(AR)
    else:
        rod_diameter = 0.1
        AR = None

    col_rad = rod_diameter / 2.0

    results_per_random_keys = Path("results") / f"{args.k1},{args.k2},{args.k3}"
    results_per_random_keys.mkdir(parents=True, exist_ok=True)

    now = datetime.datetime.now()
    dt_string = now.strftime("%Y-%m-%d_%H")
    if AR is not None:
        size_tag = f"AR{AR:04d}"
    else:
        size_tag = f"D{int(round(rod_diameter * 1e4)):04d}"

    packing_id = (f"{dt_string}_EntangledRelaxedPacking-N{num_rods:04d}-"
                  f"{size_tag}-Scale{scale_factor:g}")

    packing_dir = results_per_random_keys / packing_id
    packing_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = results_per_random_keys / f"N{num_rods}"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # ── Callback for trajectory saving ───────────────────────────────
    q_snapshots: list[np.ndarray] = []

    def _callback(q_state, callback_params):
        if args.save_trajectory:
            numbering = int(callback_params.get("numbering", 0))
            if args.snapshot_every <= 1 or (numbering % args.snapshot_every) == 0:
                q_snapshots.append(np.asarray(q_state))

        min_d = callback_params.get("min_distance", None)
        if min_d is None:
            return False
        return (jnp.abs(min_d - rod_diameter) < 1e-6 * rod_diameter) | (min_d > rod_diameter)

    # ── JIT warm-up ──────────────────────────────────────────────────
    print("\nWarming up JIT compilation...")
    _warmup_q = jnp.zeros(num_rods * 5, dtype=jnp.float64)
    _ = total_effective_potential(_warmup_q)
    jax.block_until_ready(_)
    print("JIT warm-up complete.\n")

    # ══════════════════════════════════════════════════════════════════
    # STEP 1: Entanglement
    # ══════════════════════════════════════════════════════════════════
    cache_q_path = cache_dir / "q_entangled.npy"

    if cache_q_path.exists() and not args.force:
        q_entangled = np.load(cache_q_path)
        q_entangled = jnp.asarray(q_entangled, dtype=jnp.float64).flatten()
        print(f"Loaded cached q_entangled from {cache_q_path}")
    else:
        # GPU-accelerated initialization
        if args.initial_q == "non-intersecting" and jax.default_backend() == "gpu":
            print("Using GPU-accelerated rod placement...")
            t_init = time.time()
            q0 = pr.create_nonintersecting_random_rods_gpu(num_rods, rod_diameter)
            q0 = jnp.array(q0, dtype=jnp.float64).flatten()
            print(f"GPU rod placement completed in {time.time() - t_init:.2f}s\n")
        elif args.initial_q == "non-intersecting":
            q0 = pr.create_nonintersecting_random_rods(num_rods, rod_diameter)
            q0 = jnp.array(q0, dtype=jnp.float64).flatten()
        elif args.initial_q == "random":
            q0 = pr.create_random_rods(num_rods, random_keys)
        elif args.initial_q == "test":
            q0 = pr.create_intersecting_rods(num_rods)
        elif args.initial_q == "aligned":
            q0 = pr.create_aligned_rods(num_rods)

        # Entanglement optimization
        print("=" * 60)
        print("  STEP 1: Entanglement optimisation")
        print("=" * 60)
        t_start = time.time()

        f_ent = total_effective_potential
        df_ent = grad(f_ent)
        df0 = df_ent(q0)
        print(f"Initial error: {jnp.max(jnp.abs(df0))}")
        atol_ent = args.atol * jnp.max(jnp.abs(df0))

        from optimization import optimize_fire_nonjax_individual
        q = q0
        for k in range(args.N_outer_entangle):
            q, f_val, num_iterations, error = optimize_fire_nonjax_individual(
                q, f_ent, df_ent, args.Nmax_entangle, atol_ent, args.dt,
                callback=_callback
            )
            atol_ent = atol_ent / 2

        jax.block_until_ready(q)
        t_ent = time.time() - t_start

        fval0 = float(f_ent(q0))
        fval_final = float(f_ent(q))
        print(f"f_val, initial: {fval0:.2f}")
        print(f"f_val: {fval_final:.2f}")
        print(f"error: {error}")

        q_entangled = q

        print(f"\n{'='*60}")
        print(f"  Entanglement completed in {t_ent:.2f}s")
        print(f"{'='*60}\n")

        np.save(cache_q_path, np.asarray(q_entangled))

    # Save entangled result
    np.savetxt(packing_dir / "q_entangled.txt", np.asarray(q_entangled))
    np.save(packing_dir / "q_entangled.npy", np.asarray(q_entangled))

    # ══════════════════════════════════════════════════════════════════
    # STEP 2: Relaxation (collision resolution)
    # ══════════════════════════════════════════════════════════════════
    print("=" * 60)
    print("  STEP 2: Collision relaxation")
    print("=" * 60)

    params = {"col_rad": col_rad, "amp": args.amp, "sigma": 0.025}
    relax_dt = args.relax_dt if args.relax_dt is not None else args.dt
    t_start = time.time()

    q_relaxed = pr.gpu_relax_collision(
        jnp.asarray(q_entangled, dtype=jnp.float64).flatten(),
        relax_dt,
        params,
        max_iters=args.max_relax_iters,
        effective_diameter_factor=args.clearance,
    )
    jax.block_until_ready(q_relaxed)
    t_relax = time.time() - t_start

    print(f"\n{'='*60}")
    print(f"  Relaxation completed in {t_relax:.2f}s")
    print(f"{'='*60}\n")

    # ── Save results ─────────────────────────────────────────────────
    q_relaxed_np = np.asarray(q_relaxed)
    np.savetxt(packing_dir / "q_relaxed.txt", q_relaxed_np)
    np.save(packing_dir / "q_relaxed.npy", q_relaxed_np)

    x_relaxed = np.asarray(q_to_x(jnp.asarray(q_relaxed)))
    center = np.mean((x_relaxed[:, :3] + x_relaxed[:, 3:]) / 2, axis=0)
    x_centered = x_relaxed - np.concatenate([center, center])
    x_scaled = scale_factor * x_centered
    np.savetxt(packing_dir / "x_relaxed.txt", x_scaled)
    np.save(packing_dir / "x_relaxed.npy", x_scaled)

    # Endpoint format with metadata
    with open(packing_dir / "endpoints_formatted.csv", "w") as f:
        f.write(f"# rod_radius={col_rad}\n")
        f.write(f"# rod_length=1\n")
        for row in x_scaled:
            f.write(",".join(f"{v:.10f}" for v in row) + "\n")

    # Save trajectory
    if args.save_trajectory and q_snapshots:
        qq = np.stack([np.asarray(s).reshape((-1, 5)) for s in q_snapshots], axis=0)
        np.save(packing_dir / "qq.npy", qq)
        xx = np.stack(
            [np.asarray(q_to_x(jnp.asarray(s))).reshape((-1, 6))
             for s in q_snapshots],
            axis=0,
        )
        np.save(packing_dir / "xx.npy", xx)

    # ── Stats ────────────────────────────────────────────────────────
    q_pairs = create_pairs(jnp.reshape(jnp.asarray(q_relaxed), (-1, 5)))
    d = all_pairwise_distances(q_pairs)
    ent_initial = float(total_effective_potential(jnp.asarray(q_entangled)))
    ent_final = float(total_effective_potential(jnp.asarray(q_relaxed)))

    log_output = ""
    log_output += f"rod_length: 1\n"
    if AR is not None:
        log_output += f"AR: {AR}\n"
    log_output += f"rod_diameter: {rod_diameter}\n"
    log_output += f"rod_radius: {col_rad}\n"
    log_output += f"Initial entanglement: {ent_initial}\n"
    log_output += f"Final entanglement: {ent_final}\n"
    log_output += f"Minimum distance: {jnp.min(d)}\n"
    log_output += f"Distance median: {jnp.median(d)}\n"
    log_output += f"Number of rod pairs in contact: {jnp.count_nonzero(d < rod_diameter)}\n"
    log_output += f"Total number of rod pairs: {q_pairs.shape[0]}\n"
    log_output += f"Entanglement time: {t_ent:.2f}s\n" if 't_ent' in dir() else ""
    log_output += f"Relaxation time: {t_relax:.2f}s\n"
    log_output += f"Backend: {jax.default_backend()}\n"

    with open(packing_dir / "log_standard_protocol.txt", "w") as f:
        f.write(log_output)

    print(log_output, end="")
    print(f"Wrote outputs under {packing_dir}")


if __name__ == "__main__":
    main()
