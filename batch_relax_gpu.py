#!/usr/bin/env python3
"""Highly optimized GPU relaxation processing for entangled rod packings.

This script takes one or more entangled configurations (q_entangled.npy) 
and relaxes them sequentially through a series of decreasing Aspect Ratios (AR).
It uses a JITTED FIRE optimizer and memory-efficient pairwise potentials.
"""

import argparse
import datetime
import os
import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap

# Configuration
jax.config.update("jax_enable_x64", True)

# Project imports
import potentials as pt
import optimization as optim
from transforms import q_to_x

def _print_device_info():
    print("=" * 60)
    print("JAX device info")
    print(f"  Default backend : {jax.default_backend()}")
    print(f"  Devices         : {jax.devices()}")
    if jax.default_backend() == "gpu":
        for d in jax.devices():
            print(f"    {d.device_kind} | {d.platform} | id={d.id}")
        print("  ✓ GPU detected — computations will run on GPU.")
    else:
        print("  ! WARNING: No GPU detected. Running on CPU.")
    print("=" * 60)

def parse_args():
    parser = argparse.ArgumentParser(description="Optimized Batched GPU Relaxation")
    parser.add_argument("input_path", type=str, help="Path to entangled .npy file or directory containing it")
    parser.add_argument("--AR-list", type=str, default="1000,500,300,100,50,25,10", 
                        help="Comma separated list of AR values to relax through")
    parser.add_argument("--Nmax", type=int, default=1000000, help="Max iterations per AR step")
    parser.add_argument("--atol", type=float, default=1e-6, help="Convergence tolerance for relaxation")
    parser.add_argument("--dt", type=float, default=0.01, help="Initial time step for FIRE")
    parser.add_argument("--clearance", type=float, default=1.005, help="Effective diameter multiplier")
    parser.add_argument("--amp", type=float, default=100.0, help="Repulsion amplitude")
    parser.add_argument("--force", action="store_true", help="Overwrite existing results")
    return parser.parse_args()

def setup_relaxation_potential(num_rods, col_rad_effective, amp):
    """Creates a memory-efficient repulsion potential using nested vmap."""
    
    threshold = col_rad_effective * 2.0
    
    def harmonic_repulsion(rod_i, rod_j):
        p_i = rod_i[:3]
        phi_i, theta_i = rod_i[3], rod_i[4]
        u_i = jnp.array([jnp.sin(phi_i)*jnp.cos(theta_i), jnp.sin(phi_i)*jnp.sin(theta_i), jnp.cos(phi_i)])
        p_ii = p_i + u_i # internal l=1 assumed

        p_j = rod_j[:3]
        phi_j, theta_j = rod_j[3], rod_j[4]
        u_j = jnp.array([jnp.sin(phi_j)*jnp.cos(theta_j), jnp.sin(phi_j)*jnp.sin(theta_j), jnp.cos(phi_j)])
        p_jj = p_j + u_j

        # AABB check for speed
        def _compute_full_dist():
            dist = pt.dist_lin_seg(p_i, p_ii, p_j, p_jj)
            return jax.lax.cond(dist < threshold,
                                lambda _: amp * (dist - threshold)**2,
                                lambda _: 0.0,
                                None)

        return jax.lax.cond(pt.aabb_overlap_capsule(p_i, p_ii, p_j, p_jj, threshold),
                            lambda _: _compute_full_dist(),
                            lambda _: 0.0,
                            None)

    def total_potential(q_flat):
        q_mat = jnp.reshape(q_flat, (num_rods, 5))
        
        def _sum_repulsion_i(rod_i):
            return jnp.sum(vmap(lambda rj: harmonic_repulsion(rod_i, rj))(q_mat))
        
        # This is a full N^2 check but uses nested vmap for memory efficiency
        pot_vec = vmap(_sum_repulsion_i)(q_mat)
        return jnp.sum(pot_vec) / 2.0 # double counted pairs

    return total_potential

def main():
    _print_device_info()
    args = parse_args()
    
    # Parse AR list
    ar_list = [int(x.strip()) for x in args.AR_list.split(",") if x.strip()]
    ar_list.sort(reverse=True)
    
    # Resolve input path
    input_path = Path(args.input_path).resolve()
    if input_path.is_dir():
        q_file = input_path / "q_entangled.npy"
    else:
        q_file = input_path
        input_path = q_file.parent
        
    if not q_file.exists():
        print(f"Error: {q_file} not found.")
        return
        
    # Load configuration
    q_init = jnp.asarray(np.load(q_file), dtype=jnp.float64).flatten()
    num_rods = q_init.size // 5
    
    # Setup results directory
    now = datetime.datetime.now()
    dt_string = now.strftime("%Y-%m-%d_%H")
    session_id = f"{dt_string}_BatchRelaxed-N{num_rods}"
    base_results_dir = input_path / session_id
    base_results_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Starting sequential relaxation for {num_rods} rods.")
    print(f"AR sequence: {ar_list}")
    print(f"Output directory: {base_results_dir}\n")
    
    q_current = q_init
    
    for AR in ar_list:
        rod_diameter = 1.0 / float(AR)
        col_rad_eff = (rod_diameter * args.clearance) / 2.0
        
        ar_dir = base_results_dir / f"AR{AR}"
        ar_dir.mkdir(exist_ok=True)
        
        if (ar_dir / "q_relaxed.npy").exists() and not args.force:
            print(f"Skipping AR={AR} (already exists)")
            q_current = jnp.asarray(np.load(ar_dir / "q_relaxed.npy"), dtype=jnp.float64).flatten()
            continue
            
        print(f"\n>>>> RELAXING AR {AR} (diameter={rod_diameter:.6f}) <<<<")
        
        # Setup specific potential for this AR
        f_relax = setup_relaxation_potential(num_rods, col_rad_eff, args.amp)
        df_relax = jit(grad(f_relax))
        f_relax_jit = jit(f_relax)
        
        # Warmup and dummy check for overlap
        min_d_init = pt.dist_lin_seg_over_ij(
            jnp.reshape(q_current, (num_rods, 5))[:,:3],
            jnp.reshape(q_current, (num_rods, 5))[:,:3] + jnp.array([jnp.sin(jnp.reshape(q_current,(num_rods,5))[:,3])*jnp.cos(jnp.reshape(q_current,(num_rods,5))[:,4]), jnp.sin(jnp.reshape(q_current,(num_rods,5))[:,3])*jnp.sin(jnp.reshape(q_current,(num_rods,5))[:,4]), jnp.cos(jnp.reshape(q_current,(num_rods,5))[:,3])]).T,
            *jnp.triu_indices(num_rods, k=1)
        )
        # Wait, that was complex. Let's just use the JITTED potential.
        init_energy = f_relax_jit(q_current)
        print(f"Initial Relaxation Energy: {init_energy:.2e}")
        
        t0 = time.time()
        # Use our high-performance per-DOF FIRE
        q_relaxed, f_val, iters, final_err = optim.optimize_fire_jax_individual(
            q_current, f_relax_jit, df_relax,
            Nmax=args.Nmax,
            atol=args.atol,
            dt=args.dt
        )
        jax.block_until_ready(q_relaxed)
        t1 = time.time()
        
        print(f"Relaxation Finished in {t1-t0:.2f}s ({iters} iterations)")
        print(f"Final Energy: {f_val:.2e}, Max Grad: {final_err:.2e}")
        
        # Save results
        q_np = np.asarray(q_relaxed)
        np.save(ar_dir / "q_relaxed.npy", q_np)
        np.savetxt(ar_dir / "q_relaxed.txt", q_np)
        
        x_relaxed = q_to_x(jnp.asarray(q_np))
        np.save(ar_dir / "x_relaxed.npy", np.asarray(x_relaxed))
        
        # Update current for next AR
        q_current = q_relaxed

    print(f"\nAll sequential relaxations completed in {base_results_dir}")

if __name__ == "__main__":
    main()
