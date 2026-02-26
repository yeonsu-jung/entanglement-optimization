#!/usr/bin/env python3

"""Entanglement-only protocol runner (no relaxation).

Designed to be used in the same way as `protocols.py` in cluster runs:

    python entangle_only.py <k1> <k2> <k3> [--AR AR | --rod-diameter D] [--num-rods N] [--dt DT] ...

Outputs are written under:
    results/<k1,k2,k3>/<dt_string>_EntangledPacking-N####-AR####-ScaleS/

and a cache copy under:
    results/<k1,k2,k3>/N####/q_entangled.npy
"""

from __future__ import annotations

import argparse
import datetime
import os
import time
from pathlib import Path

import numpy as np

import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)

import potentials as pt
from potentials import create_pairs, total_effective_potential
from transforms import q_to_x

import protocols as pr


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run entanglement step only (no relaxation).")
    p.add_argument("k1", type=int)
    p.add_argument("k2", type=int)
    p.add_argument("k3", type=int)

    p.add_argument("--AR", type=int, default=None, help="Aspect ratio (length/diameter). If provided, diameter is set to 1/AR.")
    p.add_argument("--rod-diameter", type=float, default=None, help="Rod diameter. If provided, overrides --AR.")

    p.add_argument("--num-rods", type=int, default=200)
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--dt", type=float, default=1e-2)
    p.add_argument("--Nmax", type=int, default=300)
    p.add_argument("--N-outer", type=int, default=5, dest="N_outer")
    p.add_argument("--atol", type=float, default=1e-8)
    p.add_argument("--initial-q", type=str, default="non-intersecting", choices=["non-intersecting", "test", "aligned", "random"],)
    p.add_argument("--force", action="store_true", help="Recompute even if cached q_entangled.npy exists")
    p.add_argument("--save-trajectory", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--snapshot-every", type=int, default=1, help="Keep every k-th callback snapshot (callback happens every ~100 FIRE iters).")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    random_keys = [args.k1, args.k2, args.k3]
    num_rods = args.num_rods
    scale_factor = args.scale

    results_per_random_keys = Path("results") / f"{args.k1},{args.k2},{args.k3}"
    results_per_random_keys.mkdir(parents=True, exist_ok=True)

    # Rod length is fixed to 1 (by construction in q_to_x / sph2cart). For bookkeeping
    # we still track rod diameter, used for generating a non-intersecting initial state.
    if args.rod_diameter is not None:
        rod_diameter = float(args.rod_diameter)
        AR = None
    elif args.AR is not None:
        AR = int(args.AR)
        rod_diameter = 1.0 / float(AR)
    else:
        # Default diameter used only for non-intersecting initial placement.
        rod_diameter = 0.1
        AR = None

    now = datetime.datetime.now()
    dt_string = now.strftime("%Y-%m-%d_%H")
    if AR is not None:
        size_tag = f"AR{AR:04d}"
    else:
        # Encode diameter with 1e4 precision (e.g., D0100 for 0.0100)
        size_tag = f"D{int(round(rod_diameter * 1e4)):04d}"
    packing_id = f"{dt_string}_EntangledPacking-N{num_rods:04d}-{size_tag}-Scale{scale_factor:g}"

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

        t_start = time.time()

        q_entangled = pr.create_entangled_rods(
            num_rods,
            total_effective_potential,
            random_keys,
            rod_diameter=rod_diameter,
            Nmax=args.Nmax,
            N_outer=args.N_outer,
            atol=args.atol,
            dt=args.dt,
            initial_q=initial_q,
            callback=_callback,
        )

        t_end = time.time()
        print(f"\n{'='*60}")
        print(f"  Entanglement optimisation completed in {t_end - t_start:.2f}s")
        print(f"  Backend: {jax.default_backend()}")
        print(f"{'='*60}\n")

        np.save(cache_q_path, np.asarray(q_entangled))
        print(f"Saved cached q_entangled to {cache_q_path}")

        if args.save_trajectory and q_snapshots:
            qq = np.stack([np.asarray(s).reshape((-1, 5)) for s in q_snapshots], axis=0)
            np.save(packing_dir / "qq.npy", qq)
            # Save endpoints trajectory too (T, N, 6)
            xx = np.stack([np.asarray(q_to_x(jnp.asarray(s))).reshape((-1, 6)) for s in q_snapshots], axis=0)
            np.save(packing_dir / "xx.npy", xx)

    # Save to packing directory
    q_entangled = np.asarray(q_entangled)
    np.savetxt(packing_dir / "q_entangled.txt", q_entangled)
    np.save(packing_dir / "q_entangled.npy", q_entangled)

    # Also save x endpoints for convenience
    x_entangled = np.asarray(q_to_x(jnp.asarray(q_entangled)))
    np.savetxt(packing_dir / "x_entangled.txt", x_entangled)
    np.save(packing_dir / "x_entangled.npy", x_entangled)

    # Basic stats
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

    with open(packing_dir / "log_entangle_only.txt", "w") as f:
        f.write(log_output)

    print(log_output, end="")
    print(f"Wrote outputs under {packing_dir}")


if __name__ == "__main__":
    main()
