"""
This script provides functions and protocols for generating, optimizing,
and analyzing 3D rod packing configurations.
"""
# ==============================================================================
# IMPORTS
# ==============================================================================
import os
from pathlib import Path
from datetime import datetime
from typing import Callable, Tuple, Optional, Dict, Any

# Third-party libraries
import jax
import jax.numpy as jnp
import numpy as np
from jax import grad, random, jit
from matplotlib import pyplot as plt
from scipy.io import savemat

# --- Local Module Imports ---
# Note: Ensure these modules are in your Python path.
# These are assumed to exist based on the provided script.
from optimization import optimize_fire_nonjax_individual
from potentials import total_harmonic_line, create_pairs, all_pairwise_distances
from transforms import q_to_x
from visualizations import set_3d_plot, plot_many_rods
# from voxelize import voxelize_rods # Assumed to exist for one of the protocols.

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# Enable 64-bit precision for JAX for numerical stability.
jax.config.update("jax_enable_x64", True)

# Define and create common data paths for better portability.
HOME_DIR = Path.home()
DATA_DIR = HOME_DIR / "Data"
FIGURES_DIR = HOME_DIR / "Figures"
EXPORT_DIR = DATA_DIR / "export"

for directory in [DATA_DIR, FIGURES_DIR, EXPORT_DIR]:
    directory.mkdir(exist_ok=True)

# ==============================================================================
# CORE GEOMETRY & UTILITY FUNCTIONS
# ==============================================================================
def sph2cart(phi: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
    """Converts spherical coordinates to Cartesian unit vectors."""
    x = jnp.sin(phi) * jnp.cos(theta)
    y = jnp.sin(phi) * jnp.sin(theta)
    z = jnp.cos(phi)
    return jnp.array([x, y, z]).transpose()

# ==============================================================================
# ROD CONFIGURATION GENERATORS
# ==============================================================================
def create_random_rods(num_rods: int, key: jax.random.PRNGKey) -> jnp.ndarray:
    """
    Creates a random, centered configuration of rods.

    Args:
        num_rods: The number of rods to create.
        key: JAX random key for reproducibility.

    Returns:
        A flattened array representing the rod configurations (q-coordinates).
    """
    k1, k2, k3 = random.split(key, 3)
    p1s = random.uniform(k1, (num_rods, 3), minval=-0.5, maxval=0.5)
    phi1 = random.uniform(k2, (num_rods, 1), minval=0., maxval=jnp.pi)
    theta1 = random.uniform(k3, (num_rods, 1), minval=0., maxval=2 * jnp.pi)

    q0 = jnp.concatenate([p1s, phi1, theta1], axis=1)

    # Center the configuration's starting points for consistency.
    x0 = q_to_x(q0)
    center = jnp.mean(x0[:, :3], axis=0)
    q0 = q0.at[:, :3].add(-center)

    return q0.flatten().astype(jnp.float64)


def create_nonintersecting_rods(
    num_rods: int,
    rod_diameter: float,
    container_dims: Tuple[float, float, float],
    max_attempts_per_rod: int = 10000
) -> Optional[np.ndarray]:
    """
    Generates a random configuration of non-intersecting rods inside a box
    using a Random Sequential Addition (RSA) method.

    Args:
        num_rods: The target number of rods to place.
        rod_diameter: The minimum allowed distance between any two rods.
        container_dims: A tuple (lx, ly, lz) defining the box dimensions.
        max_attempts_per_rod: The max number of tries to place a single rod.

    Returns:
        A NumPy array of rod q-coordinates (shape N x 5), or None if it fails.
    """
    print(f"Attempting to place {num_rods} non-intersecting rods...")
    q_coords = np.zeros((num_rods, 5), dtype=np.float64)
    endpoints = np.zeros((num_rods, 2, 3), dtype=np.float64)
    half_dims = np.array(container_dims) / 2.0

    for i in range(num_rods):
        is_placed = False
        for _ in range(max_attempts_per_rod):
            # 1. Generate a candidate rod
            pos = np.random.uniform(-half_dims, half_dims)
            phi = np.arccos(2 * np.random.rand() - 1)
            theta = np.random.uniform(0, 2 * np.pi)
            orientation = np.array([np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)])
            
            p_start, p_end = pos, pos + orientation

            # 2. Check if it's within the container boundaries
            if np.any(np.abs(p_start) > half_dims) or np.any(np.abs(p_end) > half_dims):
                continue

            # 3. Check for intersections with previously placed rods
            has_intersection = False
            for j in range(i):
                p_j_start, p_j_end = endpoints[j]
                # Re-using the pure numpy distance function
                if dist_lin_seg(p_start, p_end, p_j_start, p_j_end) < rod_diameter:
                    has_intersection = True
                    break
            
            if not has_intersection:
                q_coords[i] = np.array([*pos, phi, theta])
                endpoints[i] = p_start, p_end
                is_placed = True
                break

        if not is_placed:
            print(f"🛑 Failed to place rod {i + 1} after {max_attempts_per_rod} attempts.")
            print(f"Returning {i} successfully placed rods.")
            return q_coords[:i] if i > 0 else None

        if (i + 1) % 100 == 0:
            print(f"✅ Rod {i + 1}/{num_rods} placed successfully.")

    return q_coords

def dist_lin_seg(p1s, p1e, p2s, p2e):
    """NumPy-based distance between two line segments."""
    d1, d2, d12 = p1e - p1s, p2e - p2s, p2s - p1s
    d1_len_sq, d2_len_sq = np.dot(d1, d1), np.dot(d2, d2)
    
    if d1_len_sq == 0 and d2_len_sq == 0: return np.linalg.norm(d12)
    if d1_len_sq == 0: return np.linalg.norm(p2s + np.clip(np.dot(-d12, d2)/d2_len_sq, 0, 1)*d2 - p1s)
    if d2_len_sq == 0: return np.linalg.norm(p1s + np.clip(np.dot(d12, d1)/d1_len_sq, 0, 1)*d1 - p2s)
        
    d12_dot_d1, d12_dot_d2 = np.dot(d12, d1), np.dot(d12, d2)
    d1_dot_d2 = np.dot(d1, d2)
    
    denom = d1_len_sq * d2_len_sq - d1_dot_d2**2
    t = np.clip((d12_dot_d1*d2_len_sq - d12_dot_d2*d1_dot_d2) / denom if denom != 0 else 0, 0, 1)
    u = np.clip((t*d1_dot_d2 - d12_dot_d2) / d2_len_sq, 0, 1)
    t = np.clip((u*d1_dot_d2 + d12_dot_d1) / d1_len_sq, 0, 1) # Re-clamp t based on clamped u

    return np.linalg.norm(p1s + t*d1 - (p2s + u*d2))

# ==============================================================================
# SIMULATION PROTOCOLS
# ==============================================================================
def run_energy_minimization(
    q0: jnp.ndarray,
    energy_function: Callable,
    params: Dict[str, Any],
    n_outer: int,
    n_max: int,
    dt: float = 1e-3,
    atol: float = 1e-7,
    callback: Optional[Callable] = None
) -> jnp.ndarray:
    """Runs an energy minimization protocol on a configuration of rods."""
    q = q0
    f = lambda q_vec: energy_function(q_vec, params)
    df = jit(grad(f))

    print(f"Initial max gradient: {jnp.max(jnp.abs(df(q))):.4e}")

    for k in range(n_outer):
        print(f"--- Outer relaxation loop {k + 1}/{n_outer} ---")
        q, _, iters, error = optimize_fire_nonjax_individual(q, f, df, Nmax=n_max, atol=atol, dt=dt, callback=callback)
        
        distances = all_pairwise_distances(create_pairs(q.reshape(-1, 5)))
        min_dist = jnp.min(distances)
        
        print(f"Loop finished: iters={iters}, final_error={error:.4e}, min_dist={min_dist:.4f}")

        if (min_dist - 2 * params.get("col_rad", 0)) > -1e-6:
             print("Convergence criteria met: Minimum distance is respected.")
             break

    return q

def run_nonintersecting_rod_protocol(
    num_rods: int,
    aspect_ratio: float,
    container_dims: Tuple[float, float, float]
):
    """
    Generates, analyzes, visualizes, and exports non-intersecting rods.
    """
    print("--- Starting Non-Intersecting Rod Generation Protocol ---")
    rod_diameter = 1.0 / aspect_ratio
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # 1. Generate coordinates
    q_coords = create_nonintersecting_rods(
        num_rods=num_rods,
        rod_diameter=rod_diameter,
        container_dims=container_dims,
        max_attempts_per_rod=50000
    )

    if q_coords is None:
        print("Protocol aborted: Failed to generate any rods.")
        return

    # 2. Verify distances
    final_num_rods = q_coords.shape[0]
    q_pairs = create_pairs(jnp.array(q_coords))
    min_dist = jnp.min(all_pairwise_distances(q_pairs))
    print(f"\nVerification: Smallest distance between rods is {min_dist:.6f}")
    print(f"Required minimum distance (diameter) was {rod_diameter:.6f}")
    
    # 3. Export data
    base_filename = f"NonIntersecting_N{final_num_rods}_AR{aspect_ratio}_{timestamp}"
    txt_path = EXPORT_DIR / f"{base_filename}.txt"
    mat_path = EXPORT_DIR / f"{base_filename}.mat"
    
    np.savetxt(txt_path, q_coords)
    savemat(mat_path, {"q_coords": q_coords, "rod_diameter": rod_diameter})
    print(f"📁 Rod data saved to {txt_path} and {mat_path}")
    
    # 4. Visualize the final configuration
    print("Visualizing rod packing...")
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    plot_many_rods(q_coords, ax=ax)
    ax.set_title(f"Non-Intersecting Rods (N={final_num_rods})")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.axis('equal')
    plt.tight_layout()
    plt.show()

# ==============================================================================
# SCRIPT EXECUTION
# ==============================================================================
if __name__ == "__main__":
    
    # --- Protocol Parameters ---
    NUM_RODS_TO_GENERATE = 150
    ASPECT_RATIO = 50
    CONTAINER_DIMS = (3.0, 3.0, 3.0) # (lx, ly, lz)
    
    # --- Run the Protocol ---
    run_nonintersecting_rod_protocol(
        num_rods=NUM_RODS_TO_GENERATE,
        aspect_ratio=ASPECT_RATIO,
        container_dims=CONTAINER_DIMS,
    )