#!/usr/bin/env python3

"""Benchmark: JAX CPU vs GPU for entanglement optimization.

Runs the same entanglement computation on both CPU and GPU backends,
reports timing, and verifies that results are numerically close.

Usage (on a GPU node):
    python benchmark_cpu_vs_gpu.py [--num-rods N] [--AR AR] [--Nmax NMAX]
"""

from __future__ import annotations

import argparse
import time
import sys
from pathlib import Path

import numpy as np

# Must set float64 before any other JAX import
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

# Project imports
import potentials as pt
from potentials import create_pairs, total_effective_potential
from transforms import q_to_x
import protocols as pr


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark CPU vs GPU entanglement.")
    p.add_argument("--num-rods", type=int, default=50,
                   help="Number of rods (default: 50)")
    p.add_argument("--AR", type=int, default=10,
                   help="Aspect ratio (default: 10)")
    p.add_argument("--Nmax", type=int, default=100,
                   help="Max FIRE iterations (default: 100)")
    p.add_argument("--N-outer", type=int, default=1, dest="N_outer",
                   help="Outer iterations (default: 1)")
    p.add_argument("--dt", type=float, default=1e-2)
    p.add_argument("--atol", type=float, default=1e-8)
    p.add_argument("--random-keys", type=int, nargs=3, default=[42, 0, 0],
                   help="Random key triplet (default: 42 0 0)")
    return p.parse_args()


def run_entanglement(device_label: str, device, args):
    """Run entanglement optimization on a specific device, return (q, time)."""
    rod_diameter = 1.0 / float(args.AR)

    with jax.default_device(device):
        # JIT warm-up
        print(f"\n{'='*60}")
        print(f"  [{device_label}] Warming up JIT on {device}...")
        warmup_q = jnp.zeros(args.num_rods * 5, dtype=jnp.float64)
        _ = total_effective_potential(warmup_q)
        jax.block_until_ready(_)
        print(f"  [{device_label}] JIT warm-up complete.")

        # Run
        print(f"  [{device_label}] Running entanglement optimisation "
              f"({args.num_rods} rods, Nmax={args.Nmax})...")
        t0 = time.time()

        q_result = pr.create_entangled_rods(
            args.num_rods,
            total_effective_potential,
            args.random_keys,
            rod_diameter=rod_diameter,
            Nmax=args.Nmax,
            N_outer=args.N_outer,
            atol=args.atol,
            dt=args.dt,
            initial_q="non-intersecting",
            callback=None,
        )
        jax.block_until_ready(q_result)
        elapsed = time.time() - t0

        # Compute final energy
        energy = float(total_effective_potential(jnp.asarray(q_result)))
        jax.block_until_ready(energy)

        print(f"  [{device_label}] Done in {elapsed:.2f}s  |  energy = {energy:.6f}")
        print(f"{'='*60}")

    return np.asarray(q_result), elapsed, energy


def main() -> None:
    args = parse_args()

    # ── Device detection ─────────────────────────────────────────────
    print("JAX benchmark: CPU vs GPU")
    print(f"  Default backend: {jax.default_backend()}")
    print(f"  All devices: {jax.devices()}")

    cpu_devices = jax.devices("cpu")
    try:
        gpu_devices = jax.devices("gpu")
    except RuntimeError:
        gpu_devices = []

    if not gpu_devices:
        print("\n⚠  No GPU found! Only CPU benchmark will run.")
        print("   Make sure you loaded CUDA and have jax[cuda] installed.")

    cpu_device = cpu_devices[0]

    # ── CPU benchmark ────────────────────────────────────────────────
    # Clear any cached compilations by using the specific device
    q_cpu, t_cpu, e_cpu = run_entanglement("CPU", cpu_device, args)

    # ── GPU benchmark ────────────────────────────────────────────────
    if gpu_devices:
        gpu_device = gpu_devices[0]
        q_gpu, t_gpu, e_gpu = run_entanglement("GPU", gpu_device, args)
    else:
        q_gpu, t_gpu, e_gpu = None, None, None

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"  Problem size : {args.num_rods} rods, AR={args.AR}, "
          f"Nmax={args.Nmax}, N_outer={args.N_outer}")
    print(f"  Random keys  : {args.random_keys}")
    print(f"")
    print(f"  CPU time     : {t_cpu:8.2f}s  |  energy = {e_cpu:.6f}")

    if t_gpu is not None:
        print(f"  GPU time     : {t_gpu:8.2f}s  |  energy = {e_gpu:.6f}")
        speedup = t_cpu / t_gpu if t_gpu > 0 else float("inf")
        print(f"  Speedup      : {speedup:.2f}x")

        # Check numerical agreement (shapes may differ due to random placement)
        if q_cpu.size == q_gpu.size:
            diff = np.max(np.abs(q_cpu.flatten() - q_gpu.flatten()))
            print(f"  Max |q diff| : {diff:.2e}")
            if diff < 1e-6:
                print("  ✓  Results match (< 1e-6)")
            else:
                print(f"  ⚠  Results differ (max diff = {diff:.2e})")
                print("     (Small differences are expected due to floating-point "
                      "order of operations on different hardware.)")
        else:
            print(f"  q shapes differ: CPU={q_cpu.shape}, GPU={q_gpu.shape}")
            print("     (Random rod placement produces different configurations.)")
    else:
        print(f"  GPU time     : N/A (no GPU)")

    print("=" * 60)

    # ── Save results ─────────────────────────────────────────────────
    out_dir = Path("benchmark_results")
    out_dir.mkdir(exist_ok=True)

    summary = {
        "num_rods": args.num_rods,
        "AR": args.AR,
        "Nmax": args.Nmax,
        "N_outer": args.N_outer,
        "random_keys": args.random_keys,
        "cpu_time_s": t_cpu,
        "cpu_energy": e_cpu,
    }
    if t_gpu is not None:
        summary["gpu_time_s"] = t_gpu
        summary["gpu_energy"] = e_gpu
        summary["speedup"] = t_cpu / t_gpu if t_gpu > 0 else None

    import json
    fname = (f"benchmark_N{args.num_rods}_AR{args.AR}_Nmax{args.Nmax}.json")
    with open(out_dir / fname, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved benchmark results to {out_dir / fname}")


if __name__ == "__main__":
    main()
