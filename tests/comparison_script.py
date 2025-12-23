
import numpy as np
import subprocess
import json
import os
import sys

# Add parent directory to path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import JAX modules
# We need to run this in jax-env
try:
    import jax
    import jax.numpy as jnp
    from jax import grad
    from core.potentials import dist_lin_seg, compute_linking_number_cartesian, compute_linking_number_arai
except ImportError:
    print("Error: JAX not found. Please run in jax-env environment.")
    sys.exit(1)

def sph2cart(theta, phi):
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.array([x, y, z])

def run_cpp_benchmark(rods_data):
    # Prepare input string
    input_str = ""
    for rod_pair in rods_data:
        # Format: x1 y1 z1 phi1 theta1 x2 y2 z2 phi2 theta2 length
        line = " ".join(map(str, rod_pair))
        input_str += line + "\n"
        
    # Run C++ executable
    process = subprocess.Popen(
        ['./tests/cpp_runner'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    stdout, stderr = process.communicate(input=input_str)
    
    if process.returncode != 0:
        print(f"C++ Error: {stderr}")
        return []
        
    try:
        return json.loads(stdout)
    except json.JSONDecodeError as e:
        print(f"JSON Parse Error: {e}")
        print(f"Output: {stdout}")
        return []

def main():
    print("Generating test cases...")
    np.random.seed(42)
    
    # Generate N random rod pairs
    N_TESTS = 100
    L = 1.0
    
    test_cases = []
    
    # 5 params per rod (x,y,z, phi, theta)
    # Range: box [-5, 5], angles [0, 2pi]
    for _ in range(N_TESTS):
        # r1 = np.random.uniform(-1, 1, 3) # Close range for interaction
        # r2 = r1 + np.random.uniform(-0.5, 0.5, 3) 
        r1 = np.random.uniform(-2, 2, 3)
        r2 = np.random.uniform(-2, 2, 3)
        
        # Angles
        phi1 = np.random.uniform(0, 2*np.pi)
        theta1 = np.random.uniform(0, np.pi)
        
        phi2 = np.random.uniform(0, 2*np.pi)
        theta2 = np.random.uniform(0, np.pi)
        
        params = [r1[0], r1[1], r1[2], phi1, theta1, 
                  r2[0], r2[1], r2[2], phi2, theta2, L]
        test_cases.append(params)

    # 1. Run C++
    print(f"Running C++ implementation on {N_TESTS} cases...")
    cpp_results = run_cpp_benchmark(test_cases)
    
    if len(cpp_results) != N_TESTS:
        print(f"Error: Expected {N_TESTS} results, got {len(cpp_results)}")
        return

    # 2. Run JAX
    print("Running JAX implementation...")
    
    # JIT compile functions
    dist_fn = jax.jit(dist_lin_seg)
    lk_fn = jax.jit(compute_linking_number_cartesian)
    
    errors_dist = []
    errors_lk_gauss = []
    
    for i, params in enumerate(test_cases):
        # Parse params
        c1 = np.array(params[0:3])
        phi1, theta1 = params[3], params[4]
        c2 = np.array(params[5:8])
        phi2, theta2 = params[8], params[9]
        lc = params[10]
        
        # Convert to endpoints for JAX
        half_l = lc * 0.5
        dir1 = sph2cart(theta1, phi1) # JAX code uses (theta, phi) for sph2cart?
        # WAIT: C++ code used:
        # x = sin(phi) * cos(theta)
        # y = sin(phi) * sin(theta)
        # z = cos(phi)
        
        # Let's check Python code convention
        # In core/potentials.py:
        # u_i = jnp.array([jnp.sin(phi_i)*jnp.cos(theta_i), jnp.sin(phi_i)*jnp.sin(theta_i), jnp.cos(phi_i)])
        # This matches standard spherical coordinates where phi is polar angle from z-axis? 
        # Actually usually theta is polar angle (from z) and phi is azimuthal. 
        # But here: z = cos(phi), so phi is the angle from z-axis (polar).
        # And x,y depend on theta, so theta is azimuthal.
        # So it matches C++ convention: sin(phi)cos(theta)...
        
        dir1 = np.array([np.sin(phi1)*np.cos(theta1), np.sin(phi1)*np.sin(theta1), np.cos(phi1)])
        dir2 = np.array([np.sin(phi2)*np.cos(theta2), np.sin(phi2)*np.sin(theta2), np.cos(phi2)])
        
        p1s = c1 - half_l * dir1
        p1e = c1 + half_l * dir1
        
        p2s = c2 - half_l * dir2
        p2e = c2 + half_l * dir2
        
        # JAX computations
        jax_dist = float(dist_fn(p1s, p1e, p2s, p2e))
        
        # Linking number logic in potentials.py
        # compute_linking_number_cartesian(p_i, p_ii, p_j, p_jj)
        # It takes endpoints: start1, end1, start2, end2
        jax_lk = float(lk_fn(p1s, p1e, p2s, p2e))
        
        # Compare
        cpp_dist = cpp_results[i]['distance']
        cpp_lk = cpp_results[i]['lk_gauss']
        
        errors_dist.append(abs(cpp_dist - jax_dist))
        errors_lk_gauss.append(abs(cpp_lk - jax_lk))
        
    print("\nVerification Results:")
    print("-" * 30)
    print(f"Max Distance Error:       {max(errors_dist):.4e}")
    print(f"Mean Distance Error:      {np.mean(errors_dist):.4e}")
    print(f"Max Linking (Gauss) Err:  {max(errors_lk_gauss):.4e}")
    print(f"Mean Linking (Gauss) Err: {np.mean(errors_lk_gauss):.4e}")
    
    if max(errors_dist) < 1e-6 and max(errors_lk_gauss) < 1e-5:
        print("\n✅ SUCCESS: C++ and JAX implementations match!")
    else:
        print("\n❌ FAILURE: Mismatch detected.")
        # Find worst case
        idx = np.argmax(errors_dist)
        print(f"\nWorst Distance match (index {idx}):")
        print(f"  Params: {test_cases[idx]}")
        print(f"  C++: {cpp_results[idx]['distance']}")
        print(f"  JAX: {errors_dist[idx] + cpp_results[idx]['distance']} (diff: {errors_dist[idx]})")

if __name__ == "__main__":
    main()
