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
import re
import time
from functools import partial
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from jax import grad, jit, lax

from entanglement_optimization.core import physics
from entanglement_optimization.core import fire as fire_mod


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
                   default="auto",
                   help='Comma-separated AR values largest→smallest, '
                        'or "auto" to infer AR from the /AR{N}/ path component.')
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

# FIRE hyper-parameters (duplicated from fire.py to avoid exporting private names)
_FIRE_NDELAY = 10
_FIRE_FINC   = 1.1
_FIRE_FDEC   = 0.5
_FIRE_FA     = 0.99
_FIRE_ALPHA0 = 0.1


@partial(jit, static_argnames=["dt", "dtmax_factor"])
def _fire_repulsion_fast(q0, col_rad, amp, Nmax, dt=1e-4,
                         dtmax_factor=10.0, atol=1e-8, target_dist=-1.0):
    """FIRE optimizer for harmonic repulsion with dynamic col_rad/amp.

    Traces once per (q shape, dt, dtmax_factor) — the same XLA program is
    reused for every AR value and every seed that shares the same N.
    """
    dtmax = jnp.float64(dtmax_factor * dt)
    dtmin = jnp.float64(0.02 * dt)

    def df(q):
        return physics.repulsion_gradient(q, col_rad, amp)

    def body(carry):
        q, V, alpha, dt_c, Npos, step, _err, _md = carry
        F      = -df(q)
        P      = jnp.sum(F * V)
        P_pos  = P > 0
        V      = jnp.where(P_pos, V, jnp.zeros_like(V))
        dt_c   = jnp.where(P_pos,
                     jnp.where(Npos > _FIRE_NDELAY,
                               jnp.minimum(dt_c * _FIRE_FINC, dtmax), dt_c),
                     jnp.maximum(dt_c * _FIRE_FDEC, dtmin))
        alpha  = jnp.where(P_pos,
                     jnp.where(Npos > _FIRE_NDELAY, alpha * _FIRE_FA, alpha),
                     jnp.float64(_FIRE_ALPHA0))
        Npos   = jnp.where(P_pos, Npos + 1, jnp.int32(0))
        V_half = V + 0.5 * dt_c * F
        nV     = jnp.linalg.norm(V_half)
        nF     = jnp.linalg.norm(F)
        V_mix  = jnp.where(nF > 1e-12,
                            (1.0 - alpha) * V_half + alpha * F * (nV / nF),
                            V_half)
        q      = q + dt_c * V_mix
        F2     = -df(q)
        V      = V_mix + 0.5 * dt_c * F2
        error  = jnp.max(jnp.abs(F2))
        md     = physics.min_pairwise_distance(q)
        return q, V, alpha, dt_c, Npos, step + 1, error, md

    def cond(carry):
        _, _, _, _, _, step, error, min_dist = carry
        return ((error > atol) & (step < Nmax) &
                jnp.where(target_dist > 0.0, min_dist < target_dist, True))

    carry = (
        q0,
        jnp.zeros_like(q0),
        jnp.float64(_FIRE_ALPHA0),
        jnp.float64(dt),
        jnp.int32(0),
        jnp.int32(0),
        jnp.float64(1.0),
        physics.min_pairwise_distance(q0),
    )
    carry = lax.while_loop(cond, body, carry)
    return carry[0]


def _relax_fast(q, col_rad, amp, dt, target_min_dist, max_iters):
    """Thin wrapper: run FIRE with dynamic col_rad/amp, no snapshots."""
    q_out = _fire_repulsion_fast(
        q, col_rad, amp, max_iters,
        dt=dt, target_dist=target_min_dist,
    )
    jax.block_until_ready(q_out)
    return q_out


# ── Trajectory-mode JIT cache (compiled per eff_col_rad; rare path) ──────────
_traj_fn_cache: dict[tuple, tuple] = {}

def _get_traj_fns(eff_col_rad: float, amp: float):
    """Return (f, df) for trajectory mode, compiled once per (eff_col_rad, amp)."""
    key = (float(eff_col_rad), float(amp))
    if key not in _traj_fn_cache:
        _cr = jnp.float64(eff_col_rad)
        _a  = jnp.float64(amp)
        _traj_fn_cache[key] = (
            jit(lambda q: physics.repulsion_potential(q, _cr, _a)),
            jit(lambda q: physics.repulsion_gradient(q, _cr, _a)),
        )
    return _traj_fn_cache[key]


def _relax_with_traj(q, col_rad, amp, dt, target_min_dist, max_iters, stride):
    """Chunked FIRE relax — captures (N*5,) snapshots every `stride` iters."""
    f, df     = _get_traj_fns(float(col_rad), float(amp))
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


# ── Per-file relaxation ────────────────────────────────────────────────────

def _relax_file(q_path: Path, ar_list: list[int] | None, args) -> None:
    """Relax one q_entangled.npy through the AR sequence.

    If ar_list is None (--AR-list auto), the AR is inferred from the path
    component /AR{N}/.  col_rad is passed as a dynamic JAX scalar so that
    _fire_repulsion_fast reuses the same XLA program for all AR values
    sharing the same N.
    """
    if not q_path.exists():
        print(f"WARNING: file not found, skipping: {q_path}")
        return

    # Infer AR from path when --AR-list auto
    if ar_list is None:
        m = re.search(r"/AR(\d+)/", str(q_path))
        if m is None:
            print(f"WARNING: cannot infer AR from path {q_path}; skipping")
            return
        ar_list = [int(m.group(1))]

    q_entangled = jnp.asarray(np.load(q_path), dtype=jnp.float64).flatten()
    num_rods    = q_entangled.size // 5
    print(f"\nLoaded {q_path}  ({num_rods} rods)")

    ts      = datetime.datetime.now().strftime("%Y-%m-%d_%H")
    run_dir = q_path.parent / f"{ts}_Relaxed-N{num_rods}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {run_dir}\n")

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

        col_rad_jnp = jnp.float64(eff_col_rad)
        amp_jnp     = jnp.float64(args.amp)

        t_start = time.time()

        if args.save_traj:
            q_relaxed, snapshots = _relax_with_traj(
                q_current, col_rad_jnp, amp_jnp, args.relax_dt,
                target_d, args.max_iters, args.stride,
            )
        else:
            q_relaxed = _relax_fast(
                q_current, col_rad_jnp, amp_jnp, args.relax_dt,
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


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    _print_device_info()
    args = parse_args()

    if args.AR_list.strip().lower() == "auto":
        ar_list = None
        print("AR sequence: auto (inferred per file from /AR{N}/ path component)")
    else:
        ar_list = sorted(
            [int(x.strip()) for x in args.AR_list.split(",") if x.strip()],
            reverse=True,
        )
        print(f"AR sequence: {ar_list}")
    print(f"Processing {len(args.q_paths)} file(s)…\n")

    for q_path_str in args.q_paths:
        _relax_file(Path(q_path_str).resolve(), ar_list, args)


if __name__ == "__main__":
    main()
