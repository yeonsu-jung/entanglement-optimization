
import subprocess
import timeit
import os
import sys
import numpy as np

# Add parent directory to path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    import jax
    import jax.numpy as jnp
    from jax import jit, vmap, grad
    from core.potentials import dist_lin_seg, compute_linking_number_cartesian
except ImportError:
    print("Error: JAX not found. Please run in jax-env environment.")
    sys.exit(1)

def run_cpp_benchmark(n_pairs):
    try:
        result = subprocess.run(
            ['./tests/benchmark_runner', str(n_pairs)],
            capture_output=True, text=True, check=True
        )
        for line in result.stdout.splitlines():
            if "Average Time per Pair" in line:
                return float(line.split(":")[1].strip().split()[0])
    except Exception as e:
        print(f"C++ Benchmark failed: {e}")
        return None
    return None

def setup_jax_benchmark(n_pairs):
    key = jax.random.PRNGKey(0)
    
    # Generate data
    coords = jax.random.uniform(key, (n_pairs, 2, 3), minval=-5.0, maxval=5.0)
    # Endpoints logic: center +/- direction * 0.5
    # For benchmark, just generating 4 points is enough to test the core math functions
    # p1s, p1e, p2s, p2e
    
    # Let's generate 4 points directly to savesetup overhead
    p1s = jax.random.uniform(key, (n_pairs, 3))
    p1e = jax.random.uniform(key, (n_pairs, 3))
    p2s = jax.random.uniform(key, (n_pairs, 3))
    p2e = jax.random.uniform(key, (n_pairs, 3))
    
    return p1s, p1e, p2s, p2e

@jit
def jax_workload(p1s, p1e, p2s, p2e):
    # Vectorized computation of distance and linking number
    # Note: vmap is automatic in JAX for elementwise ops on arrays, 
    # but our potential functions are written for single inputs.
    # We need to vmap them.
    
    v_dist = vmap(dist_lin_seg)
    v_lk = vmap(compute_linking_number_cartesian)
    
    dists = v_dist(p1s, p1e, p2s, p2e)
    lks = v_lk(p1s, p1e, p2s, p2e)
    
    # Gradients?
    # Usually we take grad of a scalar loss.
    # Let's simulate a total potential energy gradient.
    # L = sum(dists) + sum(lks)
    
    return jnp.sum(dists) + jnp.sum(lks)

# We need the gradient function compiled
grad_fn = jit(grad(jax_workload, argnums=(0,1,2,3)))

def run_jax_benchmark(n_pairs, p1s, p1e, p2s, p2e):
    # grad_fn returns a tuple of gradients
    grads = grad_fn(p1s, p1e, p2s, p2e)
    # Block on the first element to ensure computation is finished
    grads[0].block_until_ready()
    
    start_time = timeit.default_timer()
    
    # Run once (it's vectorized over n_pairs)
    grads = grad_fn(p1s, p1e, p2s, p2e)
    grads[0].block_until_ready()
    
    end_time = timeit.default_timer()
    
    total_time = end_time - start_time
    avg_us = (total_time * 1e6) / n_pairs
    return avg_us

def main():
    print(f"JAX Platform: {jax.devices()[0]}")
    
    # Compile C++ first? Assuming it's done by task runner
    
    N_PAIRS_LIST = [1000, 10000, 100000]
    
    print(f"{'N_Pairs':<10} | {'C++ (us/pair)':<15} | {'JAX (us/pair)':<15} | {'Speedup (C++/JAX)':<15}")
    print("-" * 65)
    
    for n in N_PAIRS_LIST:
        # C++
        cpp_time = run_cpp_benchmark(n)
        
        # JAX
        p1s, p1e, p2s, p2e = setup_jax_benchmark(n)
        jax_time = run_jax_benchmark(n, p1s, p1e, p2s, p2e)
        
        ratio = f"{cpp_time/jax_time:.2f}x" if cpp_time and jax_time else "N/A"
        c_str = f"{cpp_time:.3f}" if cpp_time else "Err"
        j_str = f"{jax_time:.3f}" if jax_time else "Err"
        
        print(f"{n:<10} | {c_str:<15} | {j_str:<15} | {ratio:<15}")

if __name__ == "__main__":
    main()
