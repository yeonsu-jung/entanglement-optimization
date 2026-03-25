import jax
import jax.numpy as jnp
from jax import grad, jit
import time
import os
import sys

# Assume the backend is set by the environment or jax defaults to gpu if available
print("=" * 60)
print(f"JAX backend : {jax.default_backend()}")
print(f"Devices     : {jax.devices()}")
print("=" * 60)

import potentials as pt
from optimization import optimize_fire2
from protocols import create_nonintersecting_random_rods_gpu
import numpy as np

num_rods = 200
scale = 1.0

# Same potential as batch_entangle_gpu.py
def total_effective_potential(_q_flat: jnp.ndarray) -> jnp.ndarray:
    _q_state = jnp.reshape(_q_flat, (num_rods, 5))
    def compute_energy_ij(rod_i, rod_j):
        linking_number = pt.compute_linking_number_fast(rod_i, rod_j, scale)
        return -(linking_number ** 2)

    def _rod_i_to_all(rod_i):
        return jax.vmap(lambda rj: compute_energy_ij(rod_i, rj))(_q_state)
    
    pot_matrix = jax.vmap(_rod_i_to_all)(_q_state)
    mask = jnp.triu(jnp.ones((num_rods, num_rods)), k=1)
    return jnp.sum(pot_matrix * mask)

f_jitted = jit(total_effective_potential)
df_jitted = jit(grad(total_effective_potential))

try:
    np.random.seed(42)
    q0 = create_nonintersecting_random_rods_gpu(num_rods, 0.1)
    q0 = jnp.array(q0, dtype=jnp.float64).flatten()
except Exception as e:
    print("Failed to initialize:", e)
    sys.exit(1)

# Compilation timing
print("\n--- Compiling single force evaluation ---")
t0 = time.time()
_ = df_jitted(q0)
jax.block_until_ready(_)
print(f"Force compilation + first eval took: {time.time() - t0:.2f} s")

print("\n--- Profiling 10 force evaluations ---")
t0 = time.time()
for _ in range(10):
    _ = df_jitted(q0)
    jax.block_until_ready(_)
print(f"10 force evaluations took: {time.time() - t0:.4f} s")
print(f"Average time per eval: {(time.time() - t0) / 10:.4f} s")


optimize_jitted = jit(
    lambda q: optimize_fire2(
        q, f_jitted, df_jitted,
        Nmax=100,  # Small Nmax just for profiling
        atol=1e-5,
        dt=0.01
    )
)

print("\n--- Compiling FIRE optimizer loop ---")
t0 = time.time()
dummy_q = jax.random.uniform(jax.random.PRNGKey(0), (num_rods * 5,), dtype=jnp.float64)
_ = optimize_jitted(dummy_q)
jax.block_until_ready(_)
print(f"Optimizer loop compilation (dummy) took: {time.time() - t0:.2f} s")

print("\n--- Running FIRE optimizer on real data (100 steps) ---")
t0 = time.time()
out = optimize_jitted(q0)
jax.block_until_ready(out)
print(f"FIRE optimization took: {time.time() - t0:.2f} s")
