#!/usr/bin/env python3

"""Benchmark: pairwise distance & potential computation — CPU vs GPU.

Microbenchmark that times the core computations at different problem sizes
on both CPU and GPU, isolating the vmap-parallelized distance and potential
calculations from the FIRE optimizer loop.

Usage (on a GPU node):
    python benchmark_distances.py [--sizes 100 500 1000 2000] [--repeats 5]
"""

from __future__ import annotations

import argparse
import time
import json
from pathlib import Path

import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import grad, jit, vmap

import potentials as pt
from potentials import (
    create_pairs,
    total_effective_potential,
    all_pairwise_distances,
    collision_penalized_entanglement_potential,
)


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark distance calculations.")
    p.add_argument("--sizes", type=int, nargs="+", default=[50, 100, 200, 500, 1000, 2000],
                   help="Rod counts to benchmark")
    p.add_argument("--repeats", type=int, default=5,
                   help="Number of timed repeats per benchmark (default: 5)")
    return p.parse_args()


def make_random_q(n_rods, key=42):
    """Generate a random rod configuration (no collision avoidance)."""
    rng = np.random.RandomState(key)
    q = np.zeros((n_rods, 5))
    q[:, 0] = rng.uniform(-1, 1, n_rods)  # x
    q[:, 1] = rng.uniform(-1, 1, n_rods)  # y
    q[:, 2] = rng.uniform(-1, 1, n_rods)  # z
    q[:, 3] = np.arccos(rng.uniform(-1, 1, n_rods))  # phi
    q[:, 4] = rng.uniform(0, 2 * np.pi, n_rods)  # theta
    return q.flatten()


def time_fn(fn, *args, repeats=5, warmup=2):
    """Time a function, returning (mean_time, std_time) in seconds."""
    # Warmup
    for _ in range(warmup):
        out = fn(*args)
        jax.block_until_ready(out)

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn(*args)
        jax.block_until_ready(out)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    return np.mean(times), np.std(times)


def run_benchmark(device, device_label, q_np, repeats):
    """Run all benchmarks on a given device, return dict of results."""
    n_rods = len(q_np) // 5
    n_pairs = n_rods * (n_rods - 1) // 2
    results = {"device": device_label, "n_rods": n_rods, "n_pairs": n_pairs}
    values = {}  # store computed values for accuracy comparison

    with jax.default_device(device):
        q = jnp.array(q_np, dtype=jnp.float64)

        # 1) create_pairs
        q_mat = jnp.reshape(q, (-1, 5))
        t_mean, t_std = time_fn(create_pairs, q_mat, repeats=repeats)
        results["create_pairs"] = {"mean": t_mean, "std": t_std}

        # 2) all_pairwise_distances (vmap)
        q_pairs = create_pairs(q_mat)
        jax.block_until_ready(q_pairs)
        t_mean, t_std = time_fn(all_pairwise_distances, q_pairs, repeats=repeats)
        results["all_pairwise_distances"] = {"mean": t_mean, "std": t_std}
        values["distances"] = np.asarray(all_pairwise_distances(q_pairs))

        # 3) total_effective_potential (vmap entanglement)
        t_mean, t_std = time_fn(total_effective_potential, q, repeats=repeats)
        results["total_effective_potential"] = {"mean": t_mean, "std": t_std}
        values["potential"] = float(total_effective_potential(q))

        # 4) grad(total_effective_potential) — this is the expensive one in FIRE
        grad_fn = jit(grad(total_effective_potential))
        t_mean, t_std = time_fn(grad_fn, q, repeats=repeats)
        results["grad_total_effective_potential"] = {"mean": t_mean, "std": t_std}
        values["gradient"] = np.asarray(grad_fn(q))

    return results, values


def main():
    args = parse_args()

    print("=" * 70)
    print("  Distance & Potential Benchmark: CPU vs GPU")
    print("=" * 70)
    print(f"  Default backend: {jax.default_backend()}")
    print(f"  Devices: {jax.devices()}")
    print(f"  Rod counts: {args.sizes}")
    print(f"  Repeats per bench: {args.repeats}")
    print("=" * 70)

    cpu_device = jax.devices("cpu")[0]
    try:
        gpu_devices = jax.devices("gpu")
        gpu_device = gpu_devices[0]
    except RuntimeError:
        gpu_device = None
        print("⚠  No GPU found! Only CPU benchmarks will run.")

    all_results = []
    all_accuracy = []

    for n_rods in args.sizes:
        n_pairs = n_rods * (n_rods - 1) // 2
        print(f"\n{'─'*70}")
        print(f"  N = {n_rods} rods  ({n_pairs:,} pairs)")
        print(f"{'─'*70}")

        q_np = make_random_q(n_rods)

        # CPU
        print(f"  [CPU] Running benchmarks...")
        cpu_res, cpu_vals = run_benchmark(cpu_device, "cpu", q_np, args.repeats)

        # GPU
        if gpu_device:
            print(f"  [GPU] Running benchmarks...")
            gpu_res, gpu_vals = run_benchmark(gpu_device, "gpu", q_np, args.repeats)
        else:
            gpu_res, gpu_vals = None, None

        # Print timing results
        benchmarks = [
            ("create_pairs", "Create pairs"),
            ("all_pairwise_distances", "All pairwise distances"),
            ("total_effective_potential", "Total effective potential"),
            ("grad_total_effective_potential", "Gradient of potential"),
        ]

        print(f"\n  {'Benchmark':<30s} {'CPU (ms)':>12s} {'GPU (ms)':>12s} {'Speedup':>10s}")
        print(f"  {'─'*30} {'─'*12} {'─'*12} {'─'*10}")

        row_results = {"n_rods": n_rods, "n_pairs": n_pairs}
        for key, label in benchmarks:
            cpu_ms = cpu_res[key]["mean"] * 1000
            cpu_std = cpu_res[key]["std"] * 1000
            if gpu_res and key in gpu_res:
                gpu_ms = gpu_res[key]["mean"] * 1000
                gpu_std = gpu_res[key]["std"] * 1000
                speedup = cpu_res[key]["mean"] / gpu_res[key]["mean"]
                print(f"  {label:<30s} {cpu_ms:>8.2f}±{cpu_std:<3.1f} {gpu_ms:>8.2f}±{gpu_std:<3.1f} {speedup:>8.1f}x")
                row_results[key] = {
                    "cpu_ms": cpu_ms, "cpu_std_ms": cpu_std,
                    "gpu_ms": gpu_ms, "gpu_std_ms": gpu_std,
                    "speedup": speedup,
                }
            else:
                print(f"  {label:<30s} {cpu_ms:>8.2f}±{cpu_std:<3.1f} {'N/A':>12s} {'N/A':>10s}")
                row_results[key] = {"cpu_ms": cpu_ms, "cpu_std_ms": cpu_std}

        # Accuracy comparison
        if gpu_vals:
            acc = {"n_rods": n_rods, "n_pairs": n_pairs}

            print(f"\n  ACCURACY (CPU vs GPU):")
            print(f"  {'Quantity':<30s} {'Max |diff|':>12s} {'Rel err':>12s} {'Status':>10s}")
            print(f"  {'─'*30} {'─'*12} {'─'*12} {'─'*10}")

            # Distances
            d_cpu, d_gpu = cpu_vals["distances"], gpu_vals["distances"]
            d_abs = np.max(np.abs(d_cpu - d_gpu))
            d_scale = np.max(np.abs(d_cpu))
            d_rel = d_abs / d_scale if d_scale > 0 else 0.0
            d_ok = "✓" if d_rel < 1e-10 else ("~" if d_rel < 1e-6 else "✗")
            print(f"  {'Pairwise distances':<30s} {d_abs:>12.2e} {d_rel:>12.2e} {d_ok:>10s}")
            acc["distances"] = {"max_abs": float(d_abs), "rel_err": float(d_rel)}

            # Potential (scalar)
            p_cpu, p_gpu = cpu_vals["potential"], gpu_vals["potential"]
            p_abs = abs(p_cpu - p_gpu)
            p_scale = abs(p_cpu) if abs(p_cpu) > 0 else 1.0
            p_rel = p_abs / p_scale
            p_ok = "✓" if p_rel < 1e-10 else ("~" if p_rel < 1e-6 else "✗")
            print(f"  {'Potential value':<30s} {p_abs:>12.2e} {p_rel:>12.2e} {p_ok:>10s}")
            print(f"    CPU: {p_cpu:.12f}")
            print(f"    GPU: {p_gpu:.12f}")
            acc["potential"] = {"cpu": p_cpu, "gpu": p_gpu, "max_abs": p_abs, "rel_err": p_rel}

            # Gradient
            g_cpu, g_gpu = cpu_vals["gradient"], gpu_vals["gradient"]
            g_abs = np.max(np.abs(g_cpu - g_gpu))
            g_scale = np.max(np.abs(g_cpu))
            g_rel = g_abs / g_scale if g_scale > 0 else 0.0
            g_ok = "✓" if g_rel < 1e-10 else ("~" if g_rel < 1e-6 else "✗")
            print(f"  {'Gradient':<30s} {g_abs:>12.2e} {g_rel:>12.2e} {g_ok:>10s}")
            acc["gradient"] = {"max_abs": float(g_abs), "rel_err": float(g_rel)}

            # Additional gradient stats
            g_corr = np.corrcoef(g_cpu.flatten(), g_gpu.flatten())[0, 1]
            print(f"    Gradient correlation: {g_corr:.15f}")
            print(f"    (✓ = rel err < 1e-10,  ~ = < 1e-6,  ✗ = > 1e-6)")
            acc["gradient_correlation"] = float(g_corr)

            all_accuracy.append(acc)

        all_results.append(row_results)

    # Save results
    out_dir = Path("benchmark_results")
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / "benchmark_distances.json"
    with open(out_file, "w") as f:
        json.dump({"timing": all_results, "accuracy": all_accuracy}, f, indent=2)
    print(f"\n\nSaved results to {out_file}")

    # Print summary tables
    if gpu_device:
        print(f"\n{'='*70}")
        print(f"  SPEEDUP SUMMARY (GPU / CPU)")
        print(f"{'='*70}")
        print(f"  {'N':>6s} {'Pairs':>10s} │ {'Distances':>10s} {'Potential':>10s} {'Gradient':>10s}")
        print(f"  {'─'*6} {'─'*10} ┼ {'─'*10} {'─'*10} {'─'*10}")
        for r in all_results:
            n = r["n_rods"]
            p = r["n_pairs"]
            d = r.get("all_pairwise_distances", {}).get("speedup", 0)
            e = r.get("total_effective_potential", {}).get("speedup", 0)
            g = r.get("grad_total_effective_potential", {}).get("speedup", 0)
            print(f"  {n:>6d} {p:>10,d} │ {d:>9.1f}x {e:>9.1f}x {g:>9.1f}x")
        print(f"{'='*70}")

        print(f"\n{'='*70}")
        print(f"  ACCURACY SUMMARY (CPU vs GPU)")
        print(f"{'='*70}")
        print(f"  {'N':>6s} │ {'Dist rel err':>14s} {'Pot rel err':>14s} {'Grad rel err':>14s} {'Grad corr':>16s}")
        print(f"  {'─'*6} ┼ {'─'*14} {'─'*14} {'─'*14} {'─'*16}")
        for a in all_accuracy:
            n = a["n_rods"]
            dr = a["distances"]["rel_err"]
            pr = a["potential"]["rel_err"]
            gr = a["gradient"]["rel_err"]
            gc = a["gradient_correlation"]
            print(f"  {n:>6d} │ {dr:>14.2e} {pr:>14.2e} {gr:>14.2e} {gc:>16.14f}")
        print(f"{'='*70}")


if __name__ == "__main__":
    main()

