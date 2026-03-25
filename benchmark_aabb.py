#!/usr/bin/env python3
import time
import jax
import jax.numpy as jnp
from jax import jit, vmap, grad
import potentials as pt
import protocols as pr
import numpy as onp
import argparse

# Enable Float64
jax.config.update("jax_enable_x64", True)

def simple_harmonic_line_jump_no_aabb(qp, params):
    col_rad = params["col_rad"]
    amp = params["amp"]
    # Unpack manually because we are in a JAX context
    x_i, y_i, z_i, phi_i, theta_i = qp[0], qp[1], qp[2], qp[3], qp[4]
    x_j, y_j, z_j, phi_j, theta_j = qp[5], qp[6], qp[7], qp[8], qp[9]
    
    p_i = jnp.array([x_i, y_i, z_i])
    p_j = jnp.array([x_j, y_j, z_j])
    u_i = jnp.array([jnp.sin(phi_i)*jnp.cos(theta_i), jnp.sin(phi_i)*jnp.sin(theta_i), jnp.cos(phi_i)])
    u_j = jnp.array([jnp.sin(phi_j)*jnp.cos(theta_j), jnp.sin(phi_j)*jnp.sin(theta_j), jnp.cos(phi_j)])
    l = 1
    p_ii = p_i + l*u_i
    p_jj = p_j + l*u_j
    
    dist = pt.dist_lin_seg(p_i, p_ii, p_j, p_jj)
    threshold = col_rad * 2.0
    
    return jax.lax.cond(dist < threshold,
                        lambda _: amp*(dist-threshold)**2,
                        lambda _: 0.,
                        None)

@jit
def total_harmonic_line_no_aabb(q, params):
    q_mat = jnp.reshape(q, (-1, 5))
    q_pairs = pt.create_pairs(q_mat)
    return jnp.sum(vmap(lambda qp: simple_harmonic_line_jump_no_aabb(qp, params))(q_pairs))

def run_relax_bench(q_init, params, dt, max_iters, potential_fn, label):
    @jit
    def _pot(q_flat):
        return potential_fn(q_flat, params)
    
    _grad = jit(grad(_pot))
    
    @jit
    def step(q, _):
        g = _grad(q)
        q_new = q - dt * g
        return q_new, None

    # Warmup
    print(f"[{label}] Warming up...")
    q_warm, _ = jax.lax.scan(step, q_init, jnp.arange(10))
    jax.block_until_ready(q_warm)
    
    print(f"[{label}] Benchmarking {max_iters} iterations...")
    t0 = time.time()
    q_final, _ = jax.lax.scan(step, q_init, jnp.arange(max_iters))
    jax.block_until_ready(q_final)
    t1 = time.time()
    
    elapsed = t1 - t0
    print(f"[{label}] Time: {elapsed:.4f}s ({elapsed/max_iters*1000:.4f} ms/iter)")
    return elapsed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=500)
    parser.add_argument("--AR", type=int, default=1000)
    parser.add_argument("--iters", type=int, default=1000)
    args = parser.parse_args()
    
    num_rods = args.N
    AR = args.AR
    rod_diameter = 1.0 / AR
    col_rad = rod_diameter / 2.0
    params = {"col_rad": col_rad, "amp": 1000.0}
    dt = 1e-4
    
    print(f"Benchmarking AABB vs Exhaustive: N={num_rods}, AR={AR}, {args.iters} iters")
    
    # Generate a truly overlapping initial state for relaxation benchmarking
    print("Generating initial configuration (highly overlapping state)...")
    # create_random_rods places N rods in a 1x1x1 volume centered at 0, 
    # which will have many overlaps for N=500/2000 and rod_diameter=1e-3.
    q_init = pr.create_random_rods(num_rods, [42, 0, 0])
    q_init = jnp.array(q_init, dtype=jnp.float64).flatten()
    
    # Verify we actually have overlaps
    init_energy = pt.total_harmonic_line(q_init, params)
    print(f"Initial energy (AABB version): {init_energy:.2e}")
    if init_energy == 0:
        print("WARNING: Initial state has 0 overlaps! Benchmark will not be meaningful.")
    
    # 1. Bench AABB (Current implementation)
    t_aabb = run_relax_bench(q_init, params, dt, args.iters, pt.total_harmonic_line, "AABB (Current)")
    
    # 2. Bench Exhaustive (Manual non-AABB)
    t_no_aabb = run_relax_bench(q_init, params, dt, args.iters, total_harmonic_line_no_aabb, "Exhaustive")
    
    print("\n" + "="*40)
    print(f"RESULTS for N={num_rods}, AR={AR}")
    print(f"AABB        : {t_aabb:.4f}s")
    print(f"Exhaustive  : {t_no_aabb:.4f}s")
    speedup = t_no_aabb / t_aabb if t_aabb > 0 else 0
    print(f"Speedup     : {speedup:.2f}x")
    print("="*40)

if __name__ == "__main__":
    main()
