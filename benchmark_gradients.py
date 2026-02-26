import numpy as np
import jax.numpy as jnp
from jax import grad, jit
import subprocess
import os
import sys

# Add the directory containing potentials.py to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'entanglement-optimization/core'))
try:
    from potentials import dist_lin_seg, compute_linking_number_arai
except ImportError:
    # Fallback to local import if structure is different
    # Try absolute path
    sys.path.append("/Users/yeonsu/GitHub/entanglement-optimization-combined/entanglement-optimization/core")
    from potentials import dist_lin_seg, compute_linking_number_arai

# Enable double precision
import jax
jax.config.update("jax_enable_x64", True)


# ---------------------------
# JAX wrappers
# ---------------------------

def get_endpoints(q):
    """
    q: [x, y, z, phi, theta]
    l: length (assumed 1.0 for now, or passed separately)
    Returns: p_s, p_e
    """
    x, y, z, phi, theta = q
    # Note: The C++ code uses:
    # x = sin(phi) * cos(theta), y = sin(phi) * sin(theta), z = cos(phi)
    # potentials.py (JAX) uses the same convention.
    
    u = jnp.array([
        jnp.sin(phi)*jnp.cos(theta),
        jnp.sin(phi)*jnp.sin(theta),
        jnp.cos(phi)
    ])
    
    # We need to match the rod length definition.
    # C++ Rod: center +/- 0.5 * length * direction
    # Python potentials.py: often assumes length=1 implicit in some funcs, but let's see.
    # dist_lin_seg takes explicit endpoints.
    
    return u

def dist_func_q(q1, q2, l1=1.0, l2=1.0):
    p1 = jnp.array(q1[:3])
    u1 = get_endpoints(q1)
    p1s = p1 - 0.5 * l1 * u1
    p1e = p1 + 0.5 * l1 * u1
    
    p2 = jnp.array(q2[:3])
    u2 = get_endpoints(q2)
    p2s = p2 - 0.5 * l2 * u2
    p2e = p2 + 0.5 * l2 * u2
    
    return dist_lin_seg(p1s, p1e, p2s, p2e)

def linking_func_q(q1, q2, l1=1.0, l2=1.0):
    # compute_linking_number_arai signature:
    # x_i, y_i, z_i, phi_i, theta_i, x_j, y_j, z_j, phi_j, theta_j, l
    # Wait, the JAX function `compute_linking_number_arai` takes `l`.
    # But it calculates endpoints as p_i + l*u_i.
    # This implies p_i is the START point and p_ii = p_i + l*u_i is the END point?
    # Or is p_i the center?
    
    # Let's check potentials.py again.
    # def compute_linking_number(..., l):
    #     p_i = jnp.array([x_i, y_i, z_i])
    #     ...
    #     p_ii = p_i + l*u_i
    #
    # This suggests p_i is one endpoint and p_i + l*u i is the other.
    # So the rod is defined by (x,y,z) as start point and (l, phi, theta) vector.
    
    # HOWEVER, the C++ code defines Rod as center and orientation.
    # "Rod(const Vec3& center ..., double length ...)"
    # endpoints() returns {center - half, center + half}.
    
    # So we MUST bridge this gap. 
    # To compare correctly, we must pass the C++ CENTER-based rod to the JAX function
    # but the JAX function seems to expect START-point based input?
    
    # Actually, looking at `dist_lin_seg`, it takes `point1s, point1e`.
    # `compute_linking_number_arai` takes q params and `l`.
    # Let's see how `compute_linking_number_arai` is implemented.
    # p_i = [x,y,z], p_ii = p_i + l*u.
    # So yes, it treats (x,y,z) as start point.
    
    # So if our input q (from generation) represents CENTER,
    # we must convert it to START point for the JAX linking function.
    
    # q_center = [cx, cy, cz, phi, theta]
    # u = ...
    # start = center - 0.5 * l * u
    # end = center + 0.5 * l * u
    
    # But wait! `compute_linking_number_arai` takes (x,y,z) arguments.
    # So we should pass the computed START point coordinates to it?
    # NO. `compute_linking_number_arai` takes `x_i, ...` which are usually the generalized coordinates.
    # If the convention in Python code (which I need to verify) is that q represents particle position (center?),
    # then `compute_linking_number_arai` might be WRONG or using a different convention.
    
    # Let's look at `potentials.py` lines 351+:
    # p_i = jnp.array([x_i, y_i, z_i])
    # ...
    # p_ii = p_i + l*u_i
    
    # This confirms p_i is the start point in that implementation.
    
    # So for the benchmark wrapper:
    # We will generate test cases in terms of CENTER (for C++ compatibility).
    # Then for JAX linking call:
    #   Calculate start point `ps`.
    #   Pass `ps` components as `x_i, y_i, z_i`.
    #   Pass `phi, theta`.
    #   Pass `length`.
    #   BUT! The gradient will be w.r.t the arguments we pass (start point).
    #   The C++ gradient is w.r.t CENTER.
    
    #   WE NEED TO TRANSFORM THE GRADIENT.
    #   q_center = q_start + 0.5 * l * u(phi, theta)
    #   Or rather: q_start = q_center - 0.5 * l * u
    
    #   df/dq_center_x = df/dq_start_x * dx_start/dx_center = df/dq_start_x * 1
    #   df/dq_center_phi = df/dq_start * dq_start/dphi + df/dphi_direct
    
    # ALTERNATIVELY, and EASIER:
    # define a JAX wrapper `linking_func_center(q_center, ...)`
    # that computes start point inside, calls the original function,
    # and we take grad of THAT wrapper.
    # JAX will handle the chain rule automatically!
    
    pass

# JAX Gradients
grad_dist = jit(grad(dist_func_q, argnums=(0, 1)))

def linking_wrapper_center(q1_center, q2_center, l1, l2):
    # Unpack
    cx1, cy1, cz1, phi1, theta1 = q1_center
    cx2, cy2, cz2, phi2, theta2 = q2_center
    
    u1 = jnp.array([jnp.sin(phi1)*jnp.cos(theta1), jnp.sin(phi1)*jnp.sin(theta1), jnp.cos(phi1)])
    u2 = jnp.array([jnp.sin(phi2)*jnp.cos(theta2), jnp.sin(phi2)*jnp.sin(theta2), jnp.cos(phi2)])
    
    p1s = jnp.array([cx1, cy1, cz1]) - 0.5 * l1 * u1
    p2s = jnp.array([cx2, cy2, cz2]) - 0.5 * l2 * u2
    
    # Call the original function with start points
    return compute_linking_number_arai(
        p1s[0], p1s[1], p1s[2], phi1, theta1,
        p2s[0], p2s[1], p2s[2], phi2, theta2,
        l1 # Assuming second rod length is handled or function only takes one l?
           # Checked: compute_linking_number_arai takes 'l', seems to apply to both or just first?
           # Line 451: takes 'l'. Line 457: p_ii = p_i + l*u_i. Line 458: p_jj = p_j + l*u_j.
           # So it assumes EQUAL LENGTHS 'l'.
    )
    # Note: If we want different lengths, we need to modify the JAX function or wrap it differently.
    # For this benchmark, let's stick to l1=l2=length to match the JAX limitation if it exists.
    
grad_lk = jit(grad(linking_wrapper_center, argnums=(0, 1)))


# ---------------------------
# Test Case Generation
# ---------------------------

def generate_test_cases():
    cases = []
    
    # 1. Random valid cases
    for _ in range(5):
        q1 = np.random.rand(5)
        q2 = np.random.rand(5)
        l = 1.0
        cases.append((q1, q2, l, "random"))
        
    # 2. Parallel rods (small angle)
    q1 = np.array([0,0,0, 1.57, 0])      # Along x
    q2 = np.array([0,0.5,0, 1.57, 0.01]) # Along x slightly tilted
    cases.append((q1, q2, 1.0, "parallel_offset"))

    # 3. Crossing very close (small distance)
    q1 = np.array([0,0,0, 1.57, 0])     # Along x
    q2 = np.array([0,0,0.001, 1.57, 1.57]) # Along y, shifted z
    cases.append((q1, q2, 1.0, "crossing_close"))
    
    # 4. Touching
    q1 = np.array([0,0,0, 1.57, 0])
    q2 = np.array([0.5, 0.5, 0, 1.57, 1.57]) # T shape touching?
    # rod1: x in [-0.5, 0.5]
    # rod2: y at 0.5? No.
    # rod1 center 0,0,0, len 1, dir x. spans x=[-0.5, 0.5]
    # rod2 center 0.5, 0.5, 0. len 1, dir y. spans y=[0, 1] at x=0.5.
    # Touches at (0.5, 0, 0)? rod2 starts at (0.5, 0, 0).
    cases.append((q1, q2, 1.0, "touching_endpoint"))

    return cases

def run_cpp_benchmark(cases):
    # Prepare input string
    input_str = ""
    for c in cases:
        q1, q2, l, _ = c
        # cx1 cy1 cz1 phi1 theta1 len1 ...
        line = f"{q1[0]} {q1[1]} {q1[2]} {q1[3]} {q1[4]} {l} "
        line += f"{q2[0]} {q2[1]} {q2[2]} {q2[3]} {q2[4]} {l}\n"
        input_str += line
        
    # Run executable
    exe_path = "/Users/yeonsu/GitHub/entanglement-optimization-combined/entanglement-optimization-cpp/build/benchmark_gradients"
    if not os.path.exists(exe_path):
        # try without build/ if not found (fallback)
        exe_path = "/Users/yeonsu/GitHub/entanglement-optimization-combined/entanglement-optimization-cpp/benchmark_gradients"
    
    print(f"DEBUG: Using executable at {exe_path}")
    if not os.path.exists(exe_path):
        print("ERROR: Executable not found!")
        return []

    process = subprocess.Popen([exe_path], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate(input=input_str)
    
    print("DEBUG: C++ stdout len:", len(stdout))
    print("DEBUG: C++ stderr:", stderr)
        
    # Parse output
    results = []
    lines = stdout.strip().split('\n')
    for line in lines:
        parts = list(map(float, line.split()))
        # Format: dist(1) g1(5) g2(5) lk(1) gl1(5) gl2(5) -> 1+5+5+1+5+5 = 22 floats
        if len(parts) >= 22:
            dist = parts[0]
            g_dist_1 = np.array(parts[1:6])
            g_dist_2 = np.array(parts[6:11])
            lk = parts[11]
            g_lk_1 = np.array(parts[12:17])
            g_lk_2 = np.array(parts[17:22])
            results.append({
                'dist': dist, 'grad_dist': (g_dist_1, g_dist_2),
                'lk': lk, 'grad_lk': (g_lk_1, g_lk_2)
            })
    return results

def main():
    cases = generate_test_cases()
    cpp_results = run_cpp_benchmark(cases)
    
    print(f"{'Case':<20} | {'Type':<10} | {'Rel Err Dist':<12} | {'Rel Err Lk':<12}")
    print("-" * 65)
    
    for i, (q1, q2, l, desc) in enumerate(cases):
        if i >= len(cpp_results): break
        
        # JAX
        # Distance
        j_dist = dist_func_q(q1, q2, l, l)
        j_grad_dist = grad_dist(q1, q2, l, l)
        
        # Linking
        j_lk = linking_wrapper_center(q1, q2, l, l)
        j_grad_lk = grad_lk(q1, q2, l, l)
        
        # C++
        c_res = cpp_results[i]
        
        # Compare Distance
        err_dist = abs(c_res['dist'] - j_dist)
        norm_g_dist = np.linalg.norm(np.concatenate(j_grad_dist))
        err_g_dist = np.linalg.norm(np.concatenate(j_grad_dist) - np.concatenate(c_res['grad_dist']))
        rel_err_dist = err_g_dist / (norm_g_dist + 1e-9)
        
        # Compare Linking
        # Note: Linking number sign might differ or definition factor.
        # C++ Linking uses Arai, JAX uses Arai. Should match.
        # Check values
        err_lk = abs(c_res['lk'] - j_lk)
        norm_g_lk = np.linalg.norm(np.concatenate(j_grad_lk))
        err_g_lk = np.linalg.norm(np.concatenate(j_grad_lk) - np.concatenate(c_res['grad_lk']))
        rel_err_lk = err_g_lk / (norm_g_lk + 1e-9)
        
        print(f"{desc:<20} | Dist       | {rel_err_dist:.2e}     | -")
        print(f"{'':<20} | Link       | -            | {rel_err_lk:.2e}")
        
        # specific details for extreme errors
        if rel_err_dist > 1e-3 or rel_err_lk > 1e-3:
            print("  -> High Error Details:")
            print("  JAX Lk Grad:", j_grad_lk)
            print("  C++ Lk Grad:", c_res['grad_lk'])

if __name__ == "__main__":
    main()
