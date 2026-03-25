import jax
import jax.numpy as jnp
import numpy as np
from protocols import create_nonintersecting_random_rods_gpu
from potentials import all_pairwise_distances, create_pairs

def diagnose():
    num_rods = 500
    rod_diameter = 0.001
    
    print(f"Generating {num_rods} rods...")
    q = create_nonintersecting_random_rods_gpu(num_rods, rod_diameter)
    
    if np.any(np.isnan(q)):
        print("CRITICAL: NaNs found in q after placement!")
    else:
        print("q is clean.")
        
    q_jax = jnp.asarray(q)
    q_pairs = create_pairs(q_jax)
    print(f"Computing {len(q_pairs)} pairwise distances...")
    
    dists = all_pairwise_distances(q_pairs)
    
    nan_count = jnp.sum(jnp.isnan(dists))
    if nan_count > 0:
        print(f"CRITICAL: Found {nan_count} NaNs in pairwise distances!")
    else:
        print(f"Distances are clean. Min: {jnp.min(dists):.6f}, Mean: {jnp.mean(dists):.6f}")

if __name__ == "__main__":
    diagnose()
