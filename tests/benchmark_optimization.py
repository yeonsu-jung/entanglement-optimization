
import os
import sys
import timeit
import jax.numpy as jnp
from jax import random, grad, jit
import numpy as onp

# Add parent directory to path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Also add core to path so that internal imports in core work (e.g. 'import potentials' from optimization.py)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'core')))

try:
    import potentials
    from potentials import total_effective_potential
    from optimization import optimize_fire_jax_individual
    from transforms import q_to_x
    from protocols import create_random_rods
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

def main():
    print("Benchmarking JAX Optimization Protocol (FIRE)")
    
    # Parameters matching C++ benchmark
    NUM_RODS = 200
    MAX_ITER = 300
    DT = 0.01
    
    # 1. Initialize random rods
    # We use random keys as in standard_protocol
    random_keys = jnp.array([42, 42, 42]) # Deterministic for benchmark
    q0 = create_random_rods(NUM_RODS, random_keys)
    
    # 2. Define function and gradient
    f = total_effective_potential
    # grad_fn matches signature of f
    # f takes q (1D array)
    grad_fn = jit(grad(f))

    # Define scan-based FIRE optimizer for JAX
    from jax import lax
    from functools import partial
    
    @partial(jit, static_argnames=['Nmax'])
    def optimize_fire(q_init, Nmax=MAX_ITER, dt=0.01):
        dtmax = 10 * dt
        dtmin = 0.02 * dt
        alpha0 = 0.1
        finc = 1.1
        fdec = 0.5
        fa = 0.99
        
        # State: q, v, alpha, dt, Npos
        v_init = jnp.zeros_like(q_init)
        init_state = (q_init, v_init, alpha0, dt, 0)
        
        def body_fun(carry, i):
            q, v, alpha, dt, Npos = carry
            
            # Gradient
            g = grad_fn(q)
            F = -g
            
            # Power
            P = jnp.dot(F, v)
            
            # Update hyperparameters
            dt = lax.cond(P > 0, 
                          lambda _: jnp.minimum(dt * finc, dtmax),
                          lambda _: jnp.maximum(dt * fdec, dtmin),
                          None)
            alpha = lax.cond(P > 0, lambda _: alpha * fa, lambda _: alpha0, None)
            Npos = lax.cond(P > 0, lambda _: Npos + 1, lambda _: 0, None)
            
            # Velocity update
            norm_v = jnp.linalg.norm(v)
            norm_F = jnp.linalg.norm(F)
            
            v = (1.0 - alpha) * v + alpha * F * norm_v / (norm_F + 1e-8)
            
            # Integration
            v = v + dt * F
            q = q + dt * v
            
            return (q, v, alpha, dt, Npos), None
            
        final_state, _ = lax.scan(body_fun, init_state, jnp.arange(Nmax))
        return final_state[0], f(final_state[0])

    # Warmup
    print("Warming up JAX JIT...")
    optimize_fire(q0, Nmax=10, dt=DT)
    
    print(f"Running Optimization (N={NUM_RODS}, Iters={MAX_ITER})...")
    
    start_time = timeit.default_timer()
    
    q_final, f_val = optimize_fire(q0, Nmax=MAX_ITER, dt=DT)
    q_final.block_until_ready()
    
    end_time = timeit.default_timer()
    elapsed = end_time - start_time
    
    print("Benchmark Complete.")
    print(f"Final Energy: {f_val:.4f}")
    print(f"Total Time: {elapsed:.4f} seconds")
    print(f"Time per Iter: {elapsed/MAX_ITER:.4f} seconds")

if __name__ == "__main__":
    main()
