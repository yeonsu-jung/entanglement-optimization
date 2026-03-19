#!/usr/bin/env python3
"""Benchmark: effect of relax-dt on FIRE relaxation quality (N=20).

For each dt in DT_LIST, runs one full entangle+relax cycle (AR=1000->500->200)
and reports: iterations, wall time, final min distance, and whether converged.

Usage:
    python benchmark_relax_dt.py
    python benchmark_relax_dt.py --AR-list 1000,500 --num-rods 20
"""

import argparse
import time

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from jax import grad, jit

import protocols as pr
from potentials import create_pairs, total_effective_potential, all_pairwise_distances
from optimization import optimize_fire2

# ── dt values to sweep ───────────────────────────────────────────────────────
DT_LIST = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 1e-5]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--num-rods", type=int, default=20)
    p.add_argument("--AR-list", type=str, default="1000,500,200",
                   help="Comma-separated AR sequence (largest→smallest)")
    p.add_argument("--max-relax-iters", type=int, default=200_000)
    p.add_argument("--amp", type=float, default=100.0)
    p.add_argument("--clearance", type=float, default=1.005)
    p.add_argument("--Nmax-entangle", type=int, default=5000)
    p.add_argument("--dt-entangle", type=float, default=1e-2)
    p.add_argument("--atol", type=float, default=1e-8)
    return p.parse_args()


def min_dist(q):
    q_mat = jnp.reshape(q, (-1, 5))
    q_pairs = create_pairs(q_mat)
    return float(jnp.min(all_pairwise_distances(q_pairs)))


def entangle(num_rods, AR, dt_ent, Nmax_ent, atol):
    rod_diameter = 1.0 / float(AR)
    print(f"  Placing {num_rods} rods (diameter={rod_diameter:.4f})...")
    q0 = pr.create_nonintersecting_random_rods_gpu(num_rods, rod_diameter)
    q0 = jnp.array(q0, dtype=jnp.float64).flatten()

    f_ent = total_effective_potential
    df_ent = jit(grad(f_ent))
    df0 = df_ent(q0)
    atol_ent = atol * jnp.max(jnp.abs(df0))

    @jit
    def _fire(q_in, atol_in):
        return optimize_fire2(q_in, f_ent, df_ent, Nmax=Nmax_ent, atol=atol_in, dt=dt_ent)

    t0 = time.time()
    q, _, iters, _ = _fire(q0, atol_ent)
    jax.block_until_ready(q)
    print(f"  Entangled in {time.time()-t0:.1f}s ({int(iters)} iters), "
          f"f={float(f_ent(q)):.2f}")
    return q


def run_relax_sweep(q_entangled, ar_list, dt_list, max_iters, amp, clearance, atol):
    """For each dt, relax through ar_list from q_entangled and collect stats."""
    results = []  # list of dicts

    for dt in dt_list:
        print(f"\n{'─'*50}")
        print(f"  dt = {dt:.1e}")
        print(f"{'─'*50}")
        row = {"dt": dt, "ARs": []}
        q_cur = jnp.asarray(q_entangled, dtype=jnp.float64).flatten()

        for AR in ar_list:
            rod_diameter = 1.0 / float(AR)
            col_rad = rod_diameter / 2.0
            params = {"col_rad": col_rad, "amp": amp, "sigma": 0.025}

            t0 = time.time()
            q_relaxed = pr.gpu_relax_collision(
                q_cur, dt, params,
                max_iters=max_iters,
                effective_diameter_factor=clearance,
            )
            wall = time.time() - t0

            d_final = min_dist(q_relaxed)
            converged = d_final >= rod_diameter

            row["ARs"].append({
                "AR": AR,
                "rod_diameter": rod_diameter,
                "final_min_dist": d_final,
                "converged": converged,
                "wall_s": wall,
            })
            q_cur = q_relaxed

        results.append(row)

    return results


def print_table(results, ar_list):
    print(f"\n{'='*70}")
    print("BENCHMARK RESULTS")
    print(f"{'='*70}")

    # Header
    header = f"{'dt':>10}"
    for AR in ar_list:
        header += f"  AR{AR:>5} (cvg? min_d/diam)  iters"
    print(header)
    print("-" * 70)

    for row in results:
        line = f"{row['dt']:>10.1e}"
        for ar_row in row["ARs"]:
            ratio = ar_row["final_min_dist"] / ar_row["rod_diameter"]
            cvg = "YES" if ar_row["converged"] else "NO "
            line += f"    {cvg}  {ratio:.4f}  {ar_row['wall_s']:.1f}s"
        print(line)

    print(f"\n{'='*70}")
    print("DETAIL TABLE (one row per dt × AR)")
    print(f"{'='*70}")
    print(f"{'dt':>10}  {'AR':>6}  {'diam':>10}  {'min_dist':>12}  {'ratio':>6}  {'cvg':>5}  {'wall_s':>7}")
    print("-" * 70)
    for row in results:
        for ar_row in row["ARs"]:
            ratio = ar_row["final_min_dist"] / ar_row["rod_diameter"]
            print(
                f"{row['dt']:>10.1e}  "
                f"{ar_row['AR']:>6}  "
                f"{ar_row['rod_diameter']:>10.6f}  "
                f"{ar_row['final_min_dist']:>12.8f}  "
                f"{ratio:>6.4f}  "
                f"{'YES' if ar_row['converged'] else 'NO ':>5}  "
                f"{ar_row['wall_s']:>7.1f}s"
            )
    print(f"{'='*70}")


def main():
    args = parse_args()
    ar_list = [int(x) for x in args.AR_list.split(",")]

    devices = jax.devices()
    print(f"JAX backend: {jax.default_backend()}  devices: {devices}")
    print(f"N={args.num_rods}  AR sequence: {ar_list}")
    print(f"dt sweep: {DT_LIST}")
    print(f"max_relax_iters={args.max_relax_iters}\n")

    # Entangle once at largest AR; reuse for all dt trials
    print("=== Entanglement (shared across all dt trials) ===")
    q_entangled = entangle(
        args.num_rods, ar_list[0],
        args.dt_entangle, args.Nmax_entangle, args.atol,
    )

    print(f"\n=== Relaxation sweep (max_iters={args.max_relax_iters}) ===")
    results = run_relax_sweep(
        q_entangled, ar_list, DT_LIST,
        args.max_relax_iters, args.amp, args.clearance, args.atol,
    )

    print_table(results, ar_list)


if __name__ == "__main__":
    main()
