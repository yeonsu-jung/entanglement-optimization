#!/usr/bin/env python3
"""Entangle N rods by minimising the total linking-number potential.

JIT compilation happens once; subsequent packings reuse the compiled graph.

Usage
-----
python entangle.py \\
    --num-rods 2000 \\
    --AR 1000 \\
    --N-packings 5 \\
    --Nmax 10000 \\
    --out-dir results/entangled

Output layout (one subdirectory per packing)::

    {out_dir}/{k1},{k2},{k3}/
        q_entangled.npy     # (N*5,) rod state
        x_entangled.npy     # (N,6) endpoints
        log.txt
"""
from __future__ import annotations

import argparse
import datetime
import random
import sys
import time
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from jax import grad, jit

import physics
import fire as fire_mod
import init as init_mod


# ── Device info ────────────────────────────────────────────────────────────

def _print_device_info():
    print("=" * 60)
    backend = jax.default_backend()
    print(f"  Backend  : {backend}")
    for d in jax.devices():
        print(f"  Device   : {d.device_kind} | id={d.id}")
    if backend != "gpu":
        print("  WARNING: not running on GPU")
    print("=" * 60)


# ── CLI ────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Entangle rods by maximising pairwise linking numbers."
    )
    p.add_argument("--num-rods",   type=int,   default=200)
    p.add_argument("--AR",         type=int,   default=100000,
                   help="Aspect ratio (rod length / rod diameter). "
                        "Sets initial-placement diameter.")
    p.add_argument("--N-packings", type=int,   default=1,
                   help="Number of independent packings to produce.")
    p.add_argument("--Nmax",       type=int,   default=3000,
                   help="Max FIRE iterations per outer loop.")
    p.add_argument("--N-outer",    type=int,   default=1,
                   help="Outer iterations (halves atol each time).")
    p.add_argument("--atol",       type=float, default=0,
                   help="Relative force tolerance (scaled by initial force).")
    p.add_argument("--dt",            type=float, default=5e-3)
    p.add_argument("--dtmax-factor",  type=float, default=1.0,
                   help="Max dt = dtmax_factor * dt. Default 1.0 (fixed dt, matches "
                        "benchmark). Use 10.0 to enable adaptive dt growth.")
    p.add_argument("--out-dir",    type=str,   default="results/entangled")
    p.add_argument("--force",      action="store_true",
                   help="Overwrite existing results.")
    p.add_argument("--save-traj",  action="store_true",
                   help="Export endpoint snapshots during entanglement optimisation.")
    p.add_argument("--stride",     type=int,   default=500,
                   help="FIRE iterations between snapshots (--save-traj only).")
    p.add_argument("--initial-spread",     type=float,   default=0.1,
                   help="Initial spread.")
    return p.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    _print_device_info()
    args = parse_args()

    num_rods     = args.num_rods
    rod_diameter = 1.0 / float(args.AR)
    out_dir      = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Build potential and gradient — JIT once ──────────────────────────
    f_ent  = physics.make_entangle_potential(num_rods)
    df_ent = jit(grad(f_ent))

    @jit
    def _optimize(q, atol):
        return fire_mod.optimize_fire(
            q, f_ent, df_ent,
            Nmax=args.Nmax, atol=atol, dt=args.dt,
            dtmax_factor=args.dtmax_factor,
        )

    def _optimize_with_traj(q0, atol, stride, Nmax):
        """Chunked FIRE — captures endpoint snapshots every `stride` iters."""
        run_chunk = fire_mod.make_fire_runner(f_ent, df_ent, args.dt,
                                              dtmax_factor=args.dtmax_factor)
        carry     = fire_mod.fire_init_carry(q0, args.dt)
        snapshots: list[np.ndarray] = []
        total = 0

        while total < Nmax:
            chunk = min(stride, Nmax - total)
            carry = run_chunk(carry, chunk)
            jax.block_until_ready(carry[0])
            total = int(carry[5])
            error = float(carry[6])
            snapshots.append(np.asarray(carry[0]))

            if error <= atol:
                break

        q = carry[0]
        return q, f_ent(q), carry[5], carry[6], snapshots

    print("\nCompiling JIT graph (once)…")
    dummy = jnp.zeros(num_rods * 5, dtype=jnp.float64)
    t0 = time.time()
    _ = f_ent(dummy);  _ = df_ent(dummy);  _ = _optimize(dummy, 1.0)
    jax.block_until_ready(_[0])
    print(f"  compiled in {time.time()-t0:.1f}s\n")

    # ── Generate packings ────────────────────────────────────────────────
    used_seeds: set = set()

    for packing_idx in range(args.N_packings):
        print(f"\n{'='*60}")
        print(f"  PACKING {packing_idx+1}/{args.N_packings}")
        print(f"{'='*60}")

        # Unique random key triple
        while True:
            k1, k2, k3 = (random.randint(1, 999) for _ in range(3))
            if (k1, k2, k3) not in used_seeds:
                used_seeds.add((k1, k2, k3))
                break
        print(f"  seeds: {k1},{k2},{k3}")

        packing_dir = out_dir / f"{k1},{k2},{k3}"
        q_path = packing_dir / "q_entangled.npy"
        if q_path.exists() and not args.force:
            print(f"  Already exists, skipping ({q_path})")
            continue
        packing_dir.mkdir(parents=True, exist_ok=True)

        # Initialise
        t_init = time.time()
        q0_np = init_mod.create_nonintersecting_rods_gpu(num_rods, rod_diameter, args.initial_spread)
        q0 = jnp.array(q0_np, dtype=jnp.float64).flatten()
        print(f"  init: {time.time()-t_init:.1f}s")

        # Compute initial atol
        df0   = df_ent(q0)
        atol  = float(args.atol * jnp.max(jnp.abs(df0)))

        # Optimise
        t_opt = time.time()
        q = q0
        all_snapshots: list[np.ndarray] = [np.asarray(q0)] if args.save_traj else []
        for k in range(args.N_outer):
            print(f"  outer {k+1}/{args.N_outer}  atol={atol:.2e}")
            if args.save_traj:
                q, f_val, n_iters, error, snapshots = _optimize_with_traj(
                    q, atol, args.stride, args.Nmax)
                all_snapshots.extend(snapshots)
            else:
                q, f_val, n_iters, error = _optimize(q, atol)
            jax.block_until_ready(q)
            atol /= 2.0
        t_opt = time.time() - t_opt

        print(f"  f_initial={float(f_ent(q0)):.4f}  f_final={float(f_val):.4f}")
        print(f"  error={float(error):.2e}  iters={int(n_iters)}  time={t_opt:.1f}s")
        
        # ball of centroids
        
        x = physics.q_to_x(q)
        centroids = (x[:,:3] + x[:,3:])/2 # (N,3)
        packing_center = jnp.mean(centroids,axis=0) # (3,)
        center_spread = jnp.sqrt( jnp.mean( jnp.sum( (centroids - packing_center)**2, axis = 1) ) )
        
        bounding_box_size = jnp.max(x,axis=0) - jnp.min(x,axis=0) # (3,)
        

        # Save
        q_np  = np.asarray(q)
        x_np  = np.asarray(physics.q_to_x(jnp.asarray(q_np)))
        np.save(packing_dir / "q_entangled.npy", q_np)
        np.save(packing_dir / "x_entangled.npy", x_np)
        
        # save txt
        
        np.savetxt(packing_dir / "x_entangled.txt", x_np)
        

        if args.save_traj and all_snapshots:
            traj = np.stack([
                np.asarray(physics.q_to_x(jnp.asarray(snap)))
                for snap in all_snapshots
            ])
            np.save(packing_dir / "trajectory.npy", traj)
            print(f"  trajectory saved: shape={traj.shape}")

        d_all = physics.all_pairwise_distances(
            physics.create_pairs(jnp.reshape(jnp.asarray(q_np), (-1, 5)))
        )
        log = (
            f"num_rods: {num_rods}\n"
            f"AR: {args.AR}\n"
            f"rod_diameter: {rod_diameter}\n"
            f"f_initial: {float(f_ent(q0)):.6f}\n"
            f"f_final: {float(f_val):.6f}\n"
            f"f_initial_normalized: {float(f_ent(q0))/(num_rods*(num_rods-1)/2):.6f}\n"
            f"f_final_normalized: {float(f_val)/(num_rods*(num_rods-1)/2):.6f}\n"
            f"error: {float(error):.4e}\n"
            f"iterations: {int(n_iters)}\n"
            f"time_s: {t_opt:.2f}\n"
            f"min_dist: {float(jnp.min(d_all)):.8f}\n"
            f"median_dist: {float(jnp.median(d_all)):.8f}\n"
            f"center_spread: {float(center_spread):.8f}\n"
            f"bounding_box_size: {float(bounding_box_size[0]):.4f}, {float(bounding_box_size[1]):.4f}, {float(bounding_box_size[2]):.4f}\n"
            f"backend: {jax.default_backend()}\n"
        )
        (packing_dir / "log.txt").write_text(log)
        print(log, end="")
        print(f"  saved → {packing_dir}")

    print("\nDone.")


if __name__ == "__main__":
    main()
