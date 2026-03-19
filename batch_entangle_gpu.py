"""Run GPU optimization iteratively over multiple packings without recompiling.

This script randomly generates N packings, uses a single JIT compiled 
entanglement optimization function `optimize_fire2`, and processes each packing sequentially 
to bypass the 450s compilation penalty per random seed.
"""

import argparse
import datetime
import os
import sys
import time
from pathlib import Path
import random

import numpy as np

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

# Project imports
import potentials as pt
import transforms
from protocols import create_nonintersecting_random_rods_gpu
import optimization as optim
from transforms import q_to_x
from potentials import create_pairs


def _print_device_info():
    print("=" * 60)
    print("JAX device info")
    print(f"  Default backend : {jax.default_backend()}")
    print(f"  Devices         : {jax.devices()}")
    
    # Optional nice print if GPU
    if jax.default_backend() == "gpu":
        for d in jax.devices():
            print(f"    {d.device_kind} | {d.platform} | id={d.id}")
        print("  ✓ GPU detected — computations will run on GPU.")
    else:
        print("  ! WARNING: No GPU detected. Running on CPU.")
    print("=" * 60)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--N-packings", type=int, default=5, help="Number of distinct random initial packings to generate and optimize")
    parser.add_argument("--num-rods", type=int, default=2000)
    parser.add_argument("--scale", type=float, default=1.0)
    # Give either rod_diameter or AR
    parser.add_argument("--rod-diameter", type=float, default=None)
    parser.add_argument("--AR", type=int, default=None)

    # Optimization args
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--Nmax", type=int, default=10000)
    parser.add_argument("--N-outer", type=int, default=1)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--initial-q", type=str, default="non-intersecting")
    
    parser.add_argument("--save-trajectory", action="store_true")
    parser.add_argument("--snapshot-every", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    _print_device_info()
    args = parse_args()
    
    if args.rod_diameter is not None:
        rod_diameter = float(args.rod_diameter)
        AR = None
    elif args.AR is not None:
        AR = int(args.AR)
        rod_diameter = 1.0 / float(AR)
    else:
        rod_diameter = 0.1
        AR = None

    num_rods = args.num_rods
    N_packings = args.N_packings
    
    # Build wrapper functions strictly once for JIT
    from jax import grad, jit
    
    # ── Define scalar potential and its gradient ────────
    def total_effective_potential(_q_flat: jnp.ndarray) -> jnp.ndarray:
        """Computes objective = -SUM(L_ij^2) for N unit vectors utilizing fully nested vmap."""
        _q_state = jnp.reshape(_q_flat, (num_rods, 5))

        def compute_energy_ij(rod_i, rod_j):
            linking_number = pt.compute_linking_number_fast(rod_i, rod_j, args.scale)
            # Use linear objective to match legacy behavior and depth (-0.5 per pair)
            return linking_number

        # Vectorize over j
        def _rod_i_to_all(rod_i):
            return jax.vmap(lambda rj: compute_energy_ij(rod_i, rj))(_q_state)
        
        # Vectorize over i
        pot_matrix = jax.vmap(_rod_i_to_all)(_q_state)
        
        # Sum upper triangle
        mask = jnp.triu(jnp.ones((num_rods, num_rods)), k=1)
        return jnp.sum(pot_matrix * mask)

    # JIT compile the objective and its gradient
    f_jitted = jit(total_effective_potential)
    df_jitted = jit(grad(total_effective_potential))
    
    # Expose optimize_fire2 as a JIT-able function
    @jit
    def optimize_jitted(q_in, atol_in):
        return optim.optimize_fire_jax_individual(
            q_in, f_jitted, df_jitted,
            Nmax=args.Nmax,
            atol=atol_in,
            dt=args.dt
        )

    print("\n============================================================")
    print(f"JIT COMPILING THE OPTIMIZATION LOOP ({num_rods} rods)")
    print("This will take ~450s but happens exactly ONCE...")
    
    # Dummy pass to force compilation
    # Add a small random offset so gradients aren't exactly 0, preventing early termination
    dummy_q = jax.random.uniform(jax.random.PRNGKey(0), (num_rods * 5,), dtype=jnp.float64)
    t_compile_start = time.time()
    _ = f_jitted(dummy_q)
    _ = df_jitted(dummy_q)
    _ = optimize_jitted(dummy_q, args.atol)
    print(f"Compilation finished in {time.time() - t_compile_start:.2f}s!")
    print("============================================================\n")

    # Storage for random seeds we've used
    used_seeds = set()
    
    # Generate and optimize N_packings sequentially
    for packing_idx in range(N_packings):
        print(f"\n>>>> PROCESSING PACKING {packing_idx + 1} / {N_packings} <<<<")
        
        # 1. Generate unique random seeds for this packing
        while True:
            k1 = random.randint(1, 1000)
            k2 = random.randint(1, 1000)
            k3 = random.randint(1, 1000)
            seed_tuple = (k1, k2, k3)
            if seed_tuple not in used_seeds:
                used_seeds.add(seed_tuple)
                break
                
        print(f"Random keys chosen: {k1}, {k2}, {k3}")
        
        # 2. Setup directory
        results_dir = Path("results") / f"{k1},{k2},{k3}"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        now = datetime.datetime.now()
        dt_string = now.strftime("%Y-%m-%d_%H")
        size_tag = f"AR{AR:04d}" if AR is not None else f"D{int(round(rod_diameter * 1e4)):04d}"
        packing_id = f"{dt_string}_EntangledPacking-N{num_rods:04d}-{size_tag}-Scale{args.scale:g}"
        packing_dir = results_dir / packing_id
        packing_dir.mkdir(parents=True, exist_ok=True)
        
        # 3. Initialization
        t_init_start = time.time()
        print(f"Initializing packing...")
        if args.initial_q == 'non-intersecting':
            try:
                # Assuming jax gpu nonintersecting works, note this is host-dispatched
                q0 = create_nonintersecting_random_rods_gpu(num_rods, rod_diameter)
                q0 = jnp.array(q0, dtype=jnp.float64).flatten()
            except Exception as e:
                print(f"Failed to generate nonintersecting packing: {e}")
                continue
        else:
            raise ValueError(f"initial-q={args.initial_q} not fully supported in batch script.")
        print(f"Initialization took {time.time() - t_init_start:.2f}s")

        # 4. Fire Optimization (Already JITTED!)
        df0 = df_jitted(q0)
        initial_error = jnp.max(jnp.abs(df0))
        print(f"Initial error (max grad): {initial_error:.2f}")

        # Scale atol per the old script logic
        atol_eff = args.atol * initial_error

        t_opt_start = time.time()
        
        q_entangled = q0
        total_iterations = 0
        
        for k in range(args.N_outer):
            print(f"Outer iteration {k+1}/{args.N_outer}, atol_eff = {atol_eff:.2e} ...")
            
            # The actual blazing fast GPU call, parameterized with decaying atol
            q_entangled, f_val, num_iterations, final_error = optimize_jitted(q_entangled, atol_eff)
            
            # Wait for GPU buffer sync
            jax.block_until_ready(q_entangled)
            
            total_iterations += num_iterations
            atol_eff /= 2.0

        t_opt_end = time.time()
        
        print(f"\nOptimization Finished:")
        print(f"f_val (initial): {f_jitted(q0):.2f}")
        print(f"f_val (final): {f_val:.2f}")
        print(f"error (max gradient): {final_error:.2e}")
        print(f"inner loop iterations: {total_iterations}")
        print(f"Pure GPU optimization took {t_opt_end - t_opt_start:.2f}s")
        
        # 5. Disk Output (NPY / TXT files)
        q_entangled_np = np.asarray(q_entangled)
        np.savetxt(packing_dir / "q_entangled.txt", q_entangled_np)
        np.save(packing_dir / "q_entangled.npy", q_entangled_np)

        # We also need x_entangled
        x_entangled_np = np.asarray(q_to_x(jnp.asarray(q_entangled_np)))
        np.savetxt(packing_dir / "x_entangled.txt", x_entangled_np)
        np.save(packing_dir / "x_entangled.npy", x_entangled_np)
        
        # Save a duplicate at the base level to match old structures if needed
        cache_dir = results_dir / f"N{num_rods}"
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(cache_dir / "q_entangled.npy", q_entangled_np)
        
        # Metrics Log
        q_pairs = create_pairs(jnp.reshape(jnp.asarray(q_entangled_np), (-1, 5)))
        d = pt.all_pairwise_distances(q_pairs)
        ent = f_jitted(jnp.asarray(q_entangled_np))

        log_output = (
            f"rod_length: 1\n"
            f"{f'AR: {AR}' if AR is not None else ''}\n"
            f"rod_diameter: {rod_diameter}\n"
            f"Minimum distance: {jnp.min(d)}\n"
            f"Distance median: {jnp.median(d)}\n"
            f"Entanglement (energy): {ent}\n"
            f"Backend: {jax.default_backend()}\n"
        )
        
        with open(packing_dir / "log_entangle_only.txt", "w") as f_log:
            f_log.write(log_output)
            
        print("Wrote outputs to:", packing_dir)


if __name__ == "__main__":
    main()
