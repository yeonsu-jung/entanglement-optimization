#!/usr/bin/env python3

"""GPU-accelerated entanglement-only protocol runner (no relaxation).

Identical to `entangle_only.py` but with:
  - Explicit JAX GPU detection & logging
  - Timing instrumentation
  - Data kept on GPU device throughout computation

Usage (on a GPU node):
    python entangle_only_gpu.py <k1> <k2> <k3> [--AR AR | --rod-diameter D] [--num-rods N] ...
"""

from __future__ import annotations

import argparse
import datetime
import os
import time
from pathlib import Path

# ── JAX GPU setup (must come before any JAX import) ──────────────────────
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

import potentials as pt
from potentials import create_pairs, total_effective_potential
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
        print("  ⚠  WARNING: JAX is NOT using a GPU. "
              "Check your jax[cuda] installation.")
    else:
        print("  ✓  GPU detected — computations will run on GPU.")
    print("=" * 60)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run entanglement step only (GPU).")
    p.add_argument("k1", type=int)
    p.add_argument("k2", type=int)
    p.add_argument("k3", type=int)

    p.add_argument("--AR", type=int, default=None,
                   help="Aspect ratio (length/diameter). If provided, diameter = 1/AR.")
    p.add_argument("--rod-diameter", type=float, default=None,
                   help="Rod diameter. Overrides --AR if provided.")

    p.add_argument("--num-rods", type=int, default=200)
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--dt", type=float, default=1e-2)
    p.add_argument("--Nmax", type=int, default=300)
    p.add_argument("--N-outer", type=int, default=1, dest="N_outer")
    p.add_argument("--atol", type=float, default=1e-8)
    p.add_argument("--initial-q", type=str, default="non-intersecting",
                   choices=["non-intersecting", "test", "aligned", "random"])
    p.add_argument("--force", action="store_true",
                   help="Recompute even if cached q_entangled.npy exists")
    p.add_argument("--save-trajectory", action=argparse.BooleanOptionalAction,
                   default=True)
    p.add_argument("--snapshot-every", type=int, default=1,
                   help="Keep every k-th callback snapshot.")

    return p.parse_args()


def main() -> None:
    # ── Device check ─────────────────────────────────────────────────────
    _print_device_info()

    args = parse_args()

    random_keys = [args.k1, args.k2, args.k3]
    num_rods = args.num_rods
    scale_factor = args.scale

    results_per_random_keys = Path("results") / f"{args.k1},{args.k2},{args.k3}"
    results_per_random_keys.mkdir(parents=True, exist_ok=True)

    # Rod diameter / AR logic (identical to entangle_only.py)
    if args.rod_diameter is not None:
        rod_diameter = float(args.rod_diameter)
        AR = None
    elif args.AR is not None:
        AR = int(args.AR)
        rod_diameter = 1.0 / float(AR)
    else:
        rod_diameter = 0.1
        AR = None

    now = datetime.datetime.now()
    dt_string = now.strftime("%Y-%m-%d_%H")
    if AR is not None:
        size_tag = f"AR{AR:04d}"
    else:
        size_tag = f"D{int(round(rod_diameter * 1e4)):04d}"
    packing_id = (f"{dt_string}_EntangledPacking-N{num_rods:04d}-"
                  f"{size_tag}-Scale{scale_factor:g}")

    packing_dir = results_per_random_keys / packing_id
    packing_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = results_per_random_keys / f"N{num_rods}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_q_path = cache_dir / "q_entangled.npy"

    if cache_q_path.exists() and not args.force:
        q_entangled = np.load(cache_q_path)
        print(f"Loaded cached q_entangled from {cache_q_path}")
    else:
        initial_q = None if args.initial_q == "random" else args.initial_q

        q_snapshots: list[np.ndarray] = []

        def _callback(q_state, callback_params):
            if not args.save_trajectory:
                return False
            numbering = int(callback_params.get("numbering", 0))
            if args.snapshot_every > 1 and (numbering % args.snapshot_every) != 0:
                return False
            q_snapshots.append(np.asarray(q_state))
            return False

        # ── JIT warm-up: trigger compilation before timing ───────────
        print("\nWarming up JIT compilation...")
        _warmup_q = jnp.zeros(num_rods * 5, dtype=jnp.float64)
        _ = total_effective_potential(_warmup_q)
        jax.block_until_ready(_)
        print("JIT warm-up complete.\n")

        # ── GPU-accelerated initialization ───────────────────────────
        # Use the GPU version for non-intersecting placement when GPU is available
        if initial_q == "non-intersecting" and jax.default_backend() == "gpu":
            print("Using GPU-accelerated rod placement...")
            t_init = time.time()
            q0 = pr.create_nonintersecting_random_rods_gpu(num_rods, rod_diameter)
            q0 = jnp.array(q0, dtype=jnp.float64).flatten()
            print(f"GPU rod placement completed in {time.time() - t_init:.2f}s\n")
            # Pass pre-computed initial config directly (not as a string)
            _initial_q_arg = q0
        else:
            _initial_q_arg = initial_q

        # ── Main optimisation (timed) ────────────────────────────────
        t_start = time.time()

        if isinstance(_initial_q_arg, str) or _initial_q_arg is None:
            # Let create_entangled_rods handle initialization
            q_entangled = pr.create_entangled_rods(
                num_rods,
                total_effective_potential,
                random_keys,
                rod_diameter=rod_diameter,
                Nmax=args.Nmax,
                N_outer=args.N_outer,
                atol=args.atol,
                dt=args.dt,
                initial_q=_initial_q_arg,
                callback=_callback,
            )
        else:
            # Use pre-computed GPU initial config
            from jax import grad, jit
            f = total_effective_potential
            df = jit(grad(f))
            df0 = df(_initial_q_arg)
            print(f"Initial error: {jnp.max(jnp.abs(df0))}")
            atol_eff = args.atol * jnp.max(jnp.abs(df0))

            from optimization import optimize_fire2
            q = _initial_q_arg
            
            # Create a localized JIT wrapper so the while_loop compiles natively
            @jit
            def _gpu_optimize(q_in, atol_in):
                return optimize_fire2(q_in, f, df, Nmax=args.Nmax, atol=atol_in, dt=args.dt)

            t_opt_start = time.time()
            for k in range(args.N_outer):
                print(f"Outer iteration {k+1}/{args.N_outer}, atol_eff = {atol_eff:.2e} ...")
                q, f_val, num_iterations, error = _gpu_optimize(q, atol_eff)
                # Ensure device execution finishes before next check/timing
                jax.block_until_ready(q)
                
                atol_eff = atol_eff / 2

            fval0 = f(_initial_q_arg)
            print(f"\nf_val (initial): {fval0:.2f}")
            print(f"f_val (final): {f_val:.2f}")
            print(f"error (max gradient): {error:.2e}")
            print(f"num_iterations (inner loop steps total): {num_iterations}")
            print(f"Pure GPU optimization part took {time.time() - t_opt_start:.2f}s")
            
            q_entangled = q

        # Make sure GPU work is actually done before measuring time
        jax.block_until_ready(q_entangled)
        t_end = time.time()

        print(f"\n{'='*60}")
        print(f"  Entanglement optimisation completed in {t_end - t_start:.2f}s")
        print(f"  Backend: {jax.default_backend()}")
        print(f"{'='*60}\n")

        np.save(cache_q_path, np.asarray(q_entangled))
        print(f"Saved cached q_entangled to {cache_q_path}")

        if args.save_trajectory and q_snapshots:
            qq = np.stack([np.asarray(s).reshape((-1, 5)) for s in q_snapshots],
                          axis=0)
            np.save(packing_dir / "qq.npy", qq)
            xx = np.stack(
                [np.asarray(q_to_x(jnp.asarray(s))).reshape((-1, 6))
                 for s in q_snapshots],
                axis=0,
            )
            np.save(packing_dir / "xx.npy", xx)

    # ── Save to packing directory ────────────────────────────────────
    q_entangled = np.asarray(q_entangled)
    np.savetxt(packing_dir / "q_entangled.txt", q_entangled)
    np.save(packing_dir / "q_entangled.npy", q_entangled)

    x_entangled = np.asarray(q_to_x(jnp.asarray(q_entangled)))
    np.savetxt(packing_dir / "x_entangled.txt", x_entangled)
    np.save(packing_dir / "x_entangled.npy", x_entangled)

    # ── Basic stats ──────────────────────────────────────────────────
    q_pairs = create_pairs(jnp.reshape(jnp.asarray(q_entangled), (-1, 5)))
    d = pt.all_pairwise_distances(q_pairs)
    ent = total_effective_potential(jnp.asarray(q_entangled))

    log_output = ""
    log_output += f"rod_length: 1\n"
    if AR is not None:
        log_output += f"AR: {AR}\n"
    log_output += f"rod_diameter: {rod_diameter}\n"
    log_output += f"Minimum distance: {jnp.min(d)}\n"
    log_output += f"Distance median: {jnp.median(d)}\n"
    log_output += f"Entanglement (energy): {ent}\n"
    log_output += f"Backend: {jax.default_backend()}\n"

    with open(packing_dir / "log_entangle_only.txt", "w") as f:
        f.write(log_output)

    print(log_output, end="")
    print(f"Wrote outputs under {packing_dir}")

    # ── Auto-Export to Structured Directory ──────────────────────────
    # To bypass manual export steps, format the output for relaxation here directly
    base_out_dir = Path("/n/home01/yjung/Github/entanglement-optimization/relaxation_4th_gpu")
    out_folder = base_out_dir / f"N{num_rods}" / f"{args.k1},{args.k2},{args.k3}"
    out_folder.mkdir(parents=True, exist_ok=True)
    
    if AR is not None:
        out_file = out_folder / f"x_relaxed_AR{AR}.txt"
        rod_radius = 0.5 / AR
        
        with open(packing_dir / "x_entangled.txt", 'r') as f_in, open(out_file, 'w') as f_out:
            f_out.write(f"# rod_radius = {rod_radius:g}\n")
            f_out.write(f_in.read())
            
        print(f"Auto-exported formatted output to: {out_file}")


if __name__ == "__main__":
    main()
