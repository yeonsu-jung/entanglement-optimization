#!/usr/bin/env python3
"""Sequential multi-AR relaxation from a pre-entangled configuration.

Takes a q_entangled.npy file and inflates rod diameters step by step,
relaxing contacts at each AR via FIRE gradient descent.

Usage
-----
python relax.py /path/to/q_entangled.npy \\
    --AR-list 1000,500,300,200,150,100,50,25,10 \\
    --max-iters 1000000 \\
    --relax-dt 1e-4 \\
    --amp 100

Trajectory export (slower — sacrifices speed for snapshots)::

    python relax.py /path/to/q_entangled.npy \\
        --AR-list 1000,500,300 \\
        --save-traj \\
        --stride 5000

Output layout (inside the same directory as q_entangled.npy)::

    {timestamp}_Relaxed-N{N}/
        AR1000/
            q_relaxed.npy
            x_relaxed.npy          # (N,6) centred + scaled endpoints
            endpoints.csv          # plain CSV, radius in header comment
            log.txt
            trajectory.npy         # (T,N,6) only if --save-traj
        AR500/
            ...
"""
from __future__ import annotations

import argparse
import datetime
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
        description="Sequential multi-AR relaxation from a pre-entangled state."
    )
    p.add_argument("q_paths", nargs="+", type=str,
                   help="One or more paths to q_entangled.npy (same N; JIT shared).")
    p.add_argument("--AR-list", type=str,
                   default="1000,500,300,200,150,100,50,25,10",
                   help="Comma-separated AR values largest→smallest.")
    p.add_argument("--max-iters",  type=int,   default=1_000_000)
    p.add_argument("--relax-dt",   type=float, default=1e-4)
    p.add_argument("--clearance",  type=float, default=1.005,
                   help="Effective diameter = clearance * rod_diameter.")
    p.add_argument("--amp",        type=float, default=100.0,
                   help="Harmonic repulsion amplitude.")
    p.add_argument("--scale",      type=float, default=1.0,
                   help="Output coordinate scale factor.")
    p.add_argument("--force",      action="store_true",
                   help="Recompute AR steps even if q_relaxed.npy exists.")
    # Trajectory options
    p.add_argument("--save-traj",  action="store_true",
                   help="Export endpoint snapshots during relaxation.")
    p.add_argument("--stride",     type=int,   default=10_000,
                   help="FIRE iterations between snapshots (--save-traj only).")
    return p.parse_args()


# ── Save helpers ───────────────────────────────────────────────────────────

def _save_ar_result(q_relaxed, q_entangled, ar_dir: Path,
                    col_rad: float, AR: int, scale: float, t_relax: float):
    ar_dir.mkdir(parents=True, exist_ok=True)

    rod_diameter = 2.0 * col_rad

    q_np = np.asarray(q_relaxed)
    np.save(ar_dir / "q_relaxed.npy", q_np)

    x_raw    = np.asarray(physics.q_to_x(jnp.asarray(q_np)))
    centre   = np.mean((x_raw[:, :3] + x_raw[:, 3:]) / 2, axis=0)
    x_scaled = scale * (x_raw - np.concatenate([centre, centre]))
    np.save(ar_dir / "x_relaxed.npy", x_scaled)

    with open(ar_dir / "endpoints.csv", "w") as f:
        f.write(f"# rod_radius={col_rad}\n# rod_length=1\n")
        for row in x_scaled:
            f.write(",".join(f"{v:.10f}" for v in row) + "\n")

    pairs  = physics.create_pairs(jnp.reshape(jnp.asarray(q_np), (-1, 5)))
    d      = physics.all_pairwise_distances(pairs)
    f_ent0 = float(jnp.sum(physics.all_pairwise_distances(
        physics.create_pairs(jnp.reshape(jnp.asarray(q_entangled), (-1, 5)))
    )))  # placeholder — same structure

    log = (
        f"AR: {AR}\n"
        f"rod_diameter: {rod_diameter:.8f}\n"
        f"rod_radius: {col_rad:.8f}\n"
        f"min_dist: {float(jnp.min(d)):.8f}\n"
        f"median_dist: {float(jnp.median(d)):.8f}\n"
        f"pairs_in_contact: {int(jnp.count_nonzero(d < rod_diameter))}\n"
        f"total_pairs: {pairs.shape[0]}\n"
        f"time_s: {t_relax:.2f}\n"
        f"backend: {jax.default_backend()}\n"
    )
    (ar_dir / "log.txt").write_text(log)
    print(log, end="")


# ── Relax one AR step ──────────────────────────────────────────────────────

def _relax_fast(q, f, df, dt, target_min_dist, max_iters):
    """Single lax.while_loop relax — fastest, no snapshot.

    Uses an AABB-pruned min_dist function as dist_fn so the stopping
    criterion (min_dist >= target_min_dist) is checked every FIRE step
    with floating-point precision, without the O(N^2) full-distance cost.
    """
    min_d_fn = physics.make_min_dist_fn(target_min_dist)

    q_out, _, n, _ = fire_mod.optimize_fire(
        q, f, df,
        Nmax=max_iters, atol=1e-8, dt=dt,
        dist_fn=min_d_fn, target_dist=target_min_dist,
    )
    jax.block_until_ready(q_out)
    return q_out


def _relax_with_traj(q, f, df, dt, target_min_dist, max_iters, stride):
    """Chunked FIRE relax — captures (N*5,) snapshots every `stride` iters."""
    min_d_fn  = physics.make_min_dist_fn(target_min_dist)
    run_chunk = fire_mod.make_fire_runner(f, df, dt,
                                          dist_fn=min_d_fn,
                                          target_dist=target_min_dist)
    carry     = fire_mod.fire_init_carry(q, dt)
    carry     = carry[:7] + (min_d_fn(q),)
    snapshots: list[np.ndarray] = []
    total     = 0

    print(f"  chunked relax: stride={stride}, max_iters={max_iters}")
    t0 = time.time()

    while total < max_iters:
        chunk = min(stride, max_iters - total)
        carry = run_chunk(carry, chunk)
        jax.block_until_ready(carry[0])
        total = int(carry[5])

        error    = float(carry[6])
        min_dist = float(carry[7])
        snapshots.append(np.asarray(carry[0]))

        if total % (stride * 10) == 0:
            print(f"  iter={total:>9d}  error={error:.3e}  min_dist={min_dist:.6e}"
                  f"  snaps={len(snapshots)}")

        if min_dist >= target_min_dist:
            print(f"  converged: min_dist={min_dist:.6e} >= target={target_min_dist:.6e}")
            break
        if error <= 1e-8:
            print(f"  converged: force={error:.2e}")
            break

    print(f"  chunked relax done: {total} iters, {len(snapshots)} snaps, "
          f"{time.time()-t0:.1f}s")
    return carry[0], snapshots


# ── JIT cache (keyed by eff_col_rad; reused across seeds with same N) ─────

_jit_fn_cache: dict[float, tuple] = {}

def _get_fns(eff_col_rad: float, amp: float):
    """Return (f, df) for this eff_col_rad, compiling df only once per value."""
    if eff_col_rad not in _jit_fn_cache:
        f  = physics.make_repulsion_potential(eff_col_rad, amp)
        df = jit(grad(f))
        _jit_fn_cache[eff_col_rad] = (f, df)
    return _jit_fn_cache[eff_col_rad]


# ── Per-file relaxation ────────────────────────────────────────────────────

def _relax_file(q_path: Path, ar_list: list[int], args,
                warmed_up_n: int | None) -> int:
    """Relax one q_entangled.npy through the full AR sequence.

    Returns num_rods so the caller knows whether the JIT warm-up is still valid
    for the next file.
    """
    if not q_path.exists():
        print(f"WARNING: file not found, skipping: {q_path}")
        return warmed_up_n  # type: ignore[return-value]

    q_entangled = jnp.asarray(np.load(q_path), dtype=jnp.float64).flatten()
    num_rods    = q_entangled.size // 5
    print(f"\nLoaded {q_path}  ({num_rods} rods)")

    ts      = datetime.datetime.now().strftime("%Y-%m-%d_%H")
    run_dir = q_path.parent / f"{ts}_Relaxed-N{num_rods}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {run_dir}\n")

    # Warm-up min_dist only when N changes (first file or new N)
    if warmed_up_n != num_rods:
        print("JIT warm-up…")
        _dummy     = jnp.zeros(num_rods * 5, dtype=jnp.float64)
        _dummy_min = physics.make_min_dist_fn(1.0)
        _r = _dummy_min(_dummy)
        jax.block_until_ready(_r)
        print("  done\n")

    q_current = q_entangled

    for idx, AR in enumerate(ar_list):
        rod_diameter = 1.0 / float(AR)
        col_rad      = rod_diameter / 2.0
        eff_col_rad  = (rod_diameter * args.clearance) / 2.0
        target_d     = rod_diameter

        ar_dir     = run_dir / f"AR{AR}"
        cache_path = ar_dir / "q_relaxed.npy"

        print(f"[{idx+1}/{len(ar_list)}] AR={AR}  diameter={rod_diameter:.6f}")

        traj_missing = args.save_traj and not (ar_dir / "trajectory.npy").exists()
        if cache_path.exists() and not args.force and not traj_missing:
            print(f"  cached → loading {cache_path}")
            q_current = jnp.asarray(np.load(cache_path), dtype=jnp.float64).flatten()
            continue

        # (f, df) cached by eff_col_rad — compiled once, reused for all seeds
        needs_compile = eff_col_rad not in _jit_fn_cache
        f, df = _get_fns(eff_col_rad, args.amp)
        if needs_compile:
            # Trigger JIT compilation before the timer starts
            _ = f(q_current); _ = df(q_current)
            jax.block_until_ready(_)

        t_start = time.time()

        if args.save_traj:
            q_relaxed, snapshots = _relax_with_traj(
                q_current, f, df, args.relax_dt,
                target_d, args.max_iters, args.stride,
            )
        else:
            q_relaxed  = _relax_fast(
                q_current, f, df, args.relax_dt,
                target_d, args.max_iters,
            )
            snapshots = None

        t_relax = time.time() - t_start
        print(f"  done in {t_relax:.1f}s")

        _save_ar_result(q_relaxed, q_entangled, ar_dir,
                        col_rad, AR, args.scale, t_relax)

        if snapshots is not None:
            traj = np.stack([
                np.asarray(physics.q_to_x(jnp.asarray(snap)))
                for snap in snapshots
            ])
            traj_path = ar_dir / "trajectory.npy"
            np.save(traj_path, traj)
            print(f"  trajectory saved: {traj_path}  shape={traj.shape}")

        q_current = q_relaxed

    print(f"\nAll done. Outputs: {run_dir}")
    return num_rods


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    _print_device_info()
    args = parse_args()

    ar_list = sorted(
        [int(x.strip()) for x in args.AR_list.split(",") if x.strip()],
        reverse=True,
    )
    print(f"AR sequence: {ar_list}")
    print(f"Processing {len(args.q_paths)} file(s)…\n")

    warmed_up_n: int | None = None
    for q_path_str in args.q_paths:
        warmed_up_n = _relax_file(Path(q_path_str).resolve(), ar_list, args, warmed_up_n)


if __name__ == "__main__":
    main()
