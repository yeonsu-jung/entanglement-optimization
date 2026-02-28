#!/usr/bin/env python3

"""GPU-accelerated sequential entangle + multi-AR relaxation protocol.

Workflow per job:
  1. Entangle at the *smallest* rod diameter (largest AR in the list).
  2. Relax sequentially through each AR (largest → smallest, i.e.
     progressively fatter rods), feeding each result into the next step.

Usage (on a GPU node):
    python sequential_protocol_gpu.py <k1> <k2> <k3> \\
        --num-rods 100 \\
        --AR-list 1000,500,300,200,150,100,50,25,10

Output layout:
    results/{k1},{k2},{k3}/
        N{N}/
            q_entangled.npy            (cache – reused on re-run)
        {timestamp}_SequentialRelaxedPacking-N{N}-Scale{scale}/
            q_entangled.txt / .npy
            AR1000/
                q_relaxed.txt / .npy
                x_relaxed.txt / .npy
                endpoints_formatted.csv
                log.txt
            AR500/ ...
            AR10/  ...
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
        description="Run entanglement + sequential multi-AR relaxation (GPU)."
    )
    p.add_argument("k1", type=int)
    p.add_argument("k2", type=int)
    p.add_argument("k3", type=int)

    p.add_argument(
        "--AR-list",
        type=str,
        default="1000,500,300,200,150,100,50,25,10",
        help="Comma-separated AR values ordered largest→smallest "
             "(thinnest→fattest rods). Entanglement uses the first (largest) AR. "
             "Default: 1000,500,300,200,150,100,50,25,10",
    )
    p.add_argument("--num-rods", type=int, default=200)
    p.add_argument("--scale", type=float, default=1.0,
                   help="Scale factor applied to output coordinates.")
    p.add_argument("--dt", type=float, default=1e-2,
                   help="Step size for the entanglement FIRE optimizer.")
    p.add_argument("--Nmax-entangle", type=int, default=10000,
                   help="Max FIRE iterations for entanglement (default: 10000)")
    p.add_argument("--N-outer-entangle", type=int, default=1,
                   help="Outer iterations for entanglement (default: 1)")
    p.add_argument(
        "--max-relax-iters", type=int, default=1000000,
        help="Max gradient-descent iterations per relaxation step (default: 1M)",
    )
    p.add_argument("--relax-dt", type=float, default=1e-4,
                   help="Step size for relaxation GD (default: 1e-4)")
    p.add_argument(
        "--clearance", type=float, default=1.005,
        help="Effective diameter factor for relaxation potential (default: 1.005)",
    )
    p.add_argument("--amp", type=float, default=100.0,
                   help="Harmonic potential amplitude (default: 100)")
    p.add_argument("--atol", type=float, default=1e-8)
    p.add_argument(
        "--initial-q", type=str, default="non-intersecting",
        choices=["non-intersecting", "test", "aligned", "random"],
    )
    p.add_argument("--force", action="store_true",
                   help="Recompute even if cached files exist")
    p.add_argument(
        "--save-trajectory", action=argparse.BooleanOptionalAction, default=False,
        help="Save trajectory snapshots per AR step (disabled by default for large N).",
    )
    p.add_argument("--snapshot-every", type=int, default=1,
                   help="Keep every k-th callback snapshot.")

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
    """Save all outputs for a single AR relaxation step.

    Each AR directory contains the same file set as standard_protocol_gpu.py:
      q_entangled.txt / .npy
      q_relaxed.txt   / .npy
      x_relaxed.txt   / .npy
      endpoints_formatted.csv
      log_standard_protocol.txt
    """
    ar_dir.mkdir(parents=True, exist_ok=True)

    rod_diameter = 2.0 * col_rad

    # ── q_entangled ───────────────────────────────────────────────────
    q_ent_np = np.asarray(q_entangled)
    np.savetxt(ar_dir / "q_entangled.txt", q_ent_np)
    np.save(ar_dir / "q_entangled.npy", q_ent_np)

    # ── q_relaxed ────────────────────────────────────────────────────
    q_np = np.asarray(q_relaxed)
    np.savetxt(ar_dir / "q_relaxed.txt", q_np)
    np.save(ar_dir / "q_relaxed.npy", q_np)

    # ── x_relaxed (scaled, centred) ──────────────────────────────────
    x_relaxed = np.asarray(q_to_x(jnp.asarray(q_relaxed)))
    center = np.mean((x_relaxed[:, :3] + x_relaxed[:, 3:]) / 2, axis=0)
    x_centered = x_relaxed - np.concatenate([center, center])
    x_scaled = scale_factor * x_centered
    np.savetxt(ar_dir / "x_relaxed.txt", x_scaled)
    np.save(ar_dir / "x_relaxed.npy", x_scaled)

    # ── endpoints_formatted.csv ───────────────────────────────────────
    with open(ar_dir / "endpoints_formatted.csv", "w") as f:
        f.write(f"# rod_radius={col_rad}\n")
        f.write(f"# rod_length=1\n")
        for row in x_scaled:
            f.write(",".join(f"{v:.10f}" for v in row) + "\n")

    # ── stats / log ───────────────────────────────────────────────────
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

    # Parse and sort AR list: largest first (thinnest → fattest)
    ar_list = sorted(
        [int(x.strip()) for x in args.AR_list.split(",") if x.strip()],
        reverse=True,
    )
    if not ar_list:
        raise SystemExit("--AR-list is empty.")
    print(f"AR sequence: {ar_list}")

    random_keys = [args.k1, args.k2, args.k3]
    num_rods = args.num_rods
    scale_factor = args.scale

    # Entanglement uses smallest rod diameter = largest AR
    entangle_AR = ar_list[0]
    entangle_rod_diameter = 1.0 / float(entangle_AR)

    results_per_random_keys = Path("results") / f"{args.k1},{args.k2},{args.k3}"
    results_per_random_keys.mkdir(parents=True, exist_ok=True)

    now = datetime.datetime.now()
    dt_string = now.strftime("%Y-%m-%d_%H")
    packing_id = (
        f"{dt_string}_SequentialRelaxedPacking-N{num_rods}-Scale{scale_factor:g}"
    )
    packing_dir = results_per_random_keys / packing_id
    packing_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {packing_dir}\n")

    cache_dir = results_per_random_keys / f"N{num_rods}"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # ── Callback (optional trajectory for entanglement step) ─────────
    q_snapshots: list[np.ndarray] = []

    def _callback(q_state, callback_params):
        if args.save_trajectory:
            numbering = int(callback_params.get("numbering", 0))
            if args.snapshot_every <= 1 or (numbering % args.snapshot_every) == 0:
                q_snapshots.append(np.asarray(q_state))
        return False

    # ── JIT warm-up ──────────────────────────────────────────────────
    print("Warming up JIT compilation...")
    _warmup_q = jnp.zeros(num_rods * 5, dtype=jnp.float64)
    _ = total_effective_potential(_warmup_q)
    jax.block_until_ready(_)
    print("JIT warm-up complete.\n")

    # ══════════════════════════════════════════════════════════════════
    # STEP 1: Entanglement  (at smallest diameter = AR_max)
    # ══════════════════════════════════════════════════════════════════
    print("=" * 60)
    print(f"  STEP 1: Entanglement  (AR={entangle_AR}, "
          f"diameter={entangle_rod_diameter:.6f})")
    print("=" * 60)

    cache_q_path = cache_dir / "q_entangled.npy"

    if cache_q_path.exists() and not args.force:
        q_entangled = np.load(cache_q_path)
        q_entangled = jnp.asarray(q_entangled, dtype=jnp.float64).flatten()
        print(f"Loaded cached q_entangled from {cache_q_path}")
    else:
        if args.initial_q == "non-intersecting" and jax.default_backend() == "gpu":
            print("Using GPU-accelerated rod placement...")
            t_init = time.time()
            q0 = pr.create_nonintersecting_random_rods_gpu(
                num_rods, entangle_rod_diameter
            )
            q0 = jnp.array(q0, dtype=jnp.float64).flatten()
            print(f"GPU rod placement done in {time.time() - t_init:.2f}s\n")
        elif args.initial_q == "non-intersecting":
            q0 = pr.create_nonintersecting_random_rods(
                num_rods, entangle_rod_diameter
            )
            q0 = jnp.array(q0, dtype=jnp.float64).flatten()
        elif args.initial_q == "random":
            q0 = pr.create_random_rods(num_rods, random_keys)
        elif args.initial_q == "test":
            q0 = pr.create_intersecting_rods(num_rods)
        elif args.initial_q == "aligned":
            q0 = pr.create_aligned_rods(num_rods)

        from jax import jit
        f_ent = total_effective_potential
        df_ent = jit(grad(f_ent))
        df0 = df_ent(q0)
        atol_ent = args.atol * jnp.max(jnp.abs(df0))

        from optimization import optimize_fire2
        t_start = time.time()
        q = q0
        
        @jit
        def _gpu_optimize_entangle(q_in, atol_in):
            return optimize_fire2(q_in, f_ent, df_ent, Nmax=args.Nmax_entangle, atol=atol_in, dt=args.dt, logoutput=False)

        for k in range(args.N_outer_entangle):
            print(f"Outer iteration {k+1}/{args.N_outer_entangle}, atol_ent = {atol_ent:.2e} ...")
            q, f_val, num_iterations, error = _gpu_optimize_entangle(q, atol_ent)
            jax.block_until_ready(q)
            atol_ent = atol_ent / 2

        t_ent = time.time() - t_start
        q_entangled = q

        print(f"f_val initial:  {float(f_ent(q0)):.4f}")
        print(f"f_val final:    {float(f_ent(q_entangled)):.4f}")
        print(f"error (max grad): {error:.2e}")
        print(f"num_iterations: {num_iterations}")
        print(f"\n{'='*60}")
        print(f"  Entanglement completed in {t_ent:.2f}s")
        print(f"{'='*60}\n")

        np.save(cache_q_path, np.asarray(q_entangled))

    np.savetxt(packing_dir / "q_entangled.txt", np.asarray(q_entangled))
    np.save(packing_dir / "q_entangled.npy", np.asarray(q_entangled))

    # ══════════════════════════════════════════════════════════════════
    # STEP 2: Sequential relaxation over all ARs
    # ══════════════════════════════════════════════════════════════════
    print("=" * 60)
    print("  STEP 2: Sequential relaxation")
    print(f"  AR sequence: {' -> '.join(str(a) for a in ar_list)}")
    print("=" * 60 + "\n")

    q_current = jnp.asarray(q_entangled, dtype=jnp.float64).flatten()

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
