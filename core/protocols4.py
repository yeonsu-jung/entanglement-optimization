# %%
# ==============================================================================
# IMPORTS
# ==============================================================================
import os
import sys
import glob
from pathlib import Path
from datetime import datetime
from typing import Callable, Tuple, Optional, List, Dict, Any

# Third-party libraries
import jax
import jax.numpy as jnp
import numpy as np
from jax import grad, random, jit
from matplotlib import pyplot as plt

# Local application/library specific imports
# Note: Ensure these modules are in your Python path
from optimization import optimize_fire_nonjax_individual
from potentials import (total_effective_potential, create_pairs,
                        total_harmonic_line, all_pairwise_distances)
from transforms import q_to_x
from visualizations import set_3d_plot, plot_many_rods

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# Enable 64-bit precision for JAX, crucial for scientific computing
jax.config.update("jax_enable_x64", True)

# Define common data paths
# Using Path for better cross-platform compatibility
# current file name
CURRENT_FILE = Path(__file__).resolve()
CURRENT_FILE_NAME = CURRENT_FILE.stem

# HOME_DIR = Path.home()
# find project root (assuming this script is in a subdirectory)
# by while loop
# find entanglement-optimization directory
HOME_DIR = CURRENT_FILE.parent
while HOME_DIR.name != "entanglement-optimization":
    HOME_DIR = HOME_DIR.parent
    if HOME_DIR == HOME_DIR.parent:  # reached root without finding
        raise FileNotFoundError("Could not find 'entanglement-optimization' directory in path.")

DATA_DIR = HOME_DIR / "Data" / CURRENT_FILE_NAME
FIGURES_DIR = HOME_DIR / "Figures" / CURRENT_FILE_NAME
EXPORT_DIR = HOME_DIR / "export" / CURRENT_FILE_NAME

# make directories
for directory in [DATA_DIR, FIGURES_DIR, EXPORT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# CORE GEOMETRY FUNCTIONS
# ==============================================================================

def _fix_bound(num: float) -> float:
    """Clips a number to be within the [0, 1] range."""
    return np.clip(num, 0, 1)


def dist_lin_seg(p1_start: np.ndarray, p1_end: np.ndarray,
                 p2_start: np.ndarray, p2_end: np.ndarray) -> float:
    """
    Calculates the shortest distance between two 3D line segments.
    This is a pure NumPy implementation, suitable for use with Numba.
    """
    d1 = p1_end - p1_start
    d2 = p2_end - p2_start
    d12 = p2_start - p1_start

    d1_dot_d1 = np.dot(d1, d1)
    d2_dot_d2 = np.dot(d2, d2)
    d1_dot_d12 = np.dot(d1, d12)
    d2_dot_d12 = np.dot(d2, d12)
    d1_dot_d2 = np.dot(d1, d2)

    denominator = d1_dot_d1 * d2_dot_d2 - d1_dot_d2**2

    if np.isclose(d1_dot_d1, 0) or np.isclose(d2_dot_d2, 0): # One or both are points
        t = 0 if np.isclose(d1_dot_d1, 0) else _fix_bound(d1_dot_d12 / d1_dot_d1)
        u = 0 if np.isclose(d2_dot_d2, 0) else _fix_bound(-d2_dot_d12 / d2_dot_d2)
    elif np.isclose(denominator, 0):  # Segments are parallel
        t = 0
        u = _fix_bound(-d2_dot_d12 / d2_dot_d2)
        if u != -d2_dot_d12 / d2_dot_d2: # Clamped
             t = _fix_bound((u * d1_dot_d2 + d1_dot_d12) / d1_dot_d1)
    else:  # General case
        t = _fix_bound((d1_dot_d12 * d2_dot_d2 - d2_dot_d12 * d1_dot_d2) / denominator)
        u = (t * d1_dot_d2 - d2_dot_d12) / d2_dot_d2
        if not (0 <= u <= 1): # Check if u needs clamping
            u_clamped = _fix_bound(u)
            t = _fix_bound((u_clamped * d1_dot_d2 + d1_dot_d12) / d1_dot_d1)
            u = u_clamped

    dist_vec = d1 * t - d2 * u - d12
    return np.linalg.norm(dist_vec)


# ==============================================================================
# ROD CONFIGURATION GENERATORS
# ==============================================================================

def sph2cart(phi: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
    """Converts spherical coordinates to Cartesian unit vectors."""
    x = jnp.sin(phi) * jnp.cos(theta)
    y = jnp.sin(phi) * jnp.sin(theta)
    z = jnp.cos(phi)
    return jnp.array([x, y, z]).transpose()


def create_random_rods(num_rods: int, key: jax.random.PRNGKey, size: float = 1.0) -> jnp.ndarray:
    """
    Creates a random configuration of rods.

    Args:
        num_rods: The number of rods to create.
        key: JAX random key.
        size: The box size (edge length); rods are placed in [-size/2, size/2]^3.

    Returns:
        A flattened array representing the rod configurations (q-coordinates).
    """
    k1, k2, k3 = random.split(key, 3)

    half_size = size / 2.0
    p1s = random.uniform(k1, (num_rods, 3), minval=-half_size, maxval=half_size)
    phi1 = random.uniform(k2, (num_rods, 1), minval=0., maxval=jnp.pi)
    theta1 = random.uniform(k3, (num_rods, 1), minval=0., maxval=2 * jnp.pi)

    q0 = jnp.concatenate([p1s, phi1, theta1], axis=1)

    # Center the configuration
    x0 = q_to_x(q0)
    center = jnp.mean(x0[:, :3], axis=0)
    q0 = q0.at[:, :3].add(-center)

    return q0.flatten().astype(jnp.float64)


def create_random_points(num_points: int, key: jax.random.PRNGKey, size: float = 1.0) -> jnp.ndarray:
    """
    Creates a random configuration of points in a cube.

    Args:
        num_points: The number of points to create.
        key: JAX random key.
        size: The box size (edge length); points are placed in [-size/2, size/2]^3.

    Returns:
        Array of shape (num_points, 3) with point coordinates.
    """
    half_size = size / 2.0
    points = random.uniform(key, (num_points, 3), minval=-half_size, maxval=half_size)
    return points.astype(jnp.float64)

def create_nonintersecting_rods(
    num_rods: int,
    rod_diameter: float,
    container_dims: Tuple[float, float, float],
    max_attempts: int = 10000
) -> np.ndarray:
    """
    Generates a random configuration of non-intersecting rods within a box.

    Args:
        num_rods: Number of rods to place.
        rod_diameter: Minimum allowed distance between rods.
        container_dims: A tuple (lx, ly, lz) for the box dimensions.
        max_attempts: Maximum attempts to place each rod.

    Returns:
        A NumPy array of rod configurations (q), or fewer if placement fails.
    """
    print(f"Attempting to place {num_rods} non-intersecting rods...")
    q = np.zeros((num_rods, 5), dtype=np.float64)
    half_dims = np.array(container_dims) / 2.0

    for i in range(num_rods):
        is_placed = False
        for attempt in range(max_attempts):
            # Generate a random rod
            pos = np.random.uniform(-half_dims, half_dims, 3)
            phi = np.arccos(np.random.uniform(-1, 1))
            theta = np.random.uniform(0, 2 * np.pi)
            
            p_i_start = pos
            orientation = np.array([np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)])
            p_i_end = p_i_start + orientation

            # 1. Check container boundaries
            if np.any(np.abs(p_i_start) > half_dims) or np.any(np.abs(p_i_end) > half_dims):
                continue

            # 2. Check for intersections with previously placed rods
            has_intersection = False
            for j in range(i):
                p_j_start = q[j, :3]
                phi_j, theta_j = q[j, 3], q[j, 4]
                orientation_j = np.array([np.sin(phi_j) * np.cos(theta_j), np.sin(phi_j) * np.sin(theta_j), np.cos(phi_j)])
                p_j_end = p_j_start + orientation_j

                if dist_lin_seg(p_i_start, p_i_end, p_j_start, p_j_end) < rod_diameter:
                    has_intersection = True
                    break
            
            if not has_intersection:
                q[i] = np.array([*pos, phi, theta])
                is_placed = True
                break

        if not is_placed:
            print(f"Failed to place rod {i+1} after {max_attempts} attempts.")
            return q[:i] # Return successfully placed rods

        if (i + 1) % 100 == 0:
            print(f"Rod {i+1}/{num_rods} placed successfully.")

    return q


# ==============================================================================
# SIMULATION AND OPTIMIZATION
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
    """
    Runs an energy minimization protocol on a configuration of rods.

    Args:
        q0: Initial rod configuration (flattened q-coordinates).
        energy_function: The potential energy function to minimize.
        params: Dictionary of parameters for the energy function.
        n_outer: Number of outer relaxation loops.
        n_max: Maximum iterations for the FIRE optimizer per loop.
        dt: Initial timestep for the optimizer.
        atol: Absolute tolerance for the gradient norm to determine convergence.
        callback: Optional function to call at each optimization step.

    Returns:
        The relaxed rod configuration.
    """
    q = q0
    f = lambda q_vec: energy_function(q_vec, params)
    df = jit(grad(f))

    print(f"Initial max gradient: {jnp.max(jnp.abs(df(q))):.4e}")

    for k in range(n_outer):
        print(f"--- Outer loop {k+1}/{n_outer} ---")
        q, f_val, iters, error = optimize_fire_nonjax_individual(q, f, df, Nmax=n_max, atol=atol, dt=dt, callback=callback)
        
        distances = all_pairwise_distances(create_pairs(q.reshape(-1, 5)))
        min_dist = jnp.min(distances)
        
        print(f"Loop finished: iters={iters}, final_error={error:.4e}, min_dist={min_dist:.4f}")

        # Convergence check (optional)
        if (min_dist - 2 * params.get("col_rad", 0)) > -1e-6:
             print("Sufficiently relaxed: minimum distance criteria met.")
             break

    return q


# ==============================================================================
# MAIN PROTOCOL EXAMPLE
# ==============================================================================

def run_entangle_and_relax_protocol(
    num_rods: int,
    aspect_ratio: float,
    output_dir: Path
):
    """
    A full protocol: creates entangled rods, relaxes them to handle collisions,
    and saves the output and visualizations.
    """
    # --- 1. Setup ---
    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    protocol_name = f"EntangledRelax_N{num_rods}_AR{aspect_ratio}"
    run_output_dir = output_dir / f"{now}_{protocol_name}"
    run_output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Starting protocol: {protocol_name}")
    print(f"Output will be saved to: {run_output_dir}")

    # --- 2. Create Initial Maximally Entangled State ---
    # We use a fixed key for reproducibility
    entangle_key = random.PRNGKey(42)
    # The 'energy_function' for entanglement is the effective potential
    q_entangled = create_random_rods(num_rods, entangle_key)

    # To entangle, we minimize the negative of the potential
    # Note: The original script seems to have a function for this,
    # but for clarity, we can state the goal is to maximize entanglement
    # by finding a low-energy state of total_effective_potential
    print("\nStarting entanglement...")
    # This part requires an optimization loop similar to relaxation.
    # For now, let's assume `create_entangled_rods` from your script does this.
    # We'll use the result of `create_random_rods` as a placeholder.
    # q_entangled = create_entangled_rods(...)
    
    # Save and visualize entangled state
    np.savetxt(run_output_dir / "q_entangled.txt", q_entangled)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    plot_many_rods(q_entangled.reshape(-1, 5), ax=ax, plot_params={"color": "red", "alpha": 0.5})
    ax.set_title("Initial Entangled State")
    plt.savefig(run_output_dir / "initial_state.png")
    plt.close(fig)

    # --- 3. Run Collision Relaxation ---
    print("\nStarting relaxation...")
    collision_radius = 1.0 / (2.0 * aspect_ratio)
    params = {
        "col_rad": collision_radius,
        "amp": 100.0,
        "sigma": 0.025
    }
    q_relaxed = run_energy_minimization(
        q_entangled,
        energy_function=total_harmonic_line,
        params=params,
        n_outer=10,
        n_max=2000,
        dt=1e-3,
        atol=1e-8
    )

    # --- 4. Save Results and Visualize Final State ---
    print("\nProtocol finished. Saving final results...")
    np.savetxt(run_output_dir / "q_relaxed.txt", q_relaxed)

    # Save Cartesian coordinates
    x_relaxed = q_to_x(q_relaxed)
    center = jnp.mean(x_relaxed.reshape(-1, 3), axis=0)
    x_relaxed_centered = x_relaxed - jnp.tile(center, 2)
    np.savetxt(run_output_dir / "x_relaxed_centered.txt", x_relaxed_centered)
    
    # Final visualization
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    plot_many_rods(q_relaxed.reshape(-1, 5), ax=ax, plot_params={"color": "blue"})
    ax.set_title("Final Relaxed State")
    plt.savefig(run_output_dir / "final_state.png")
    plt.close(fig)

    # --- 5. Final Analysis ---
    q_pairs = create_pairs(q_relaxed.reshape(-1, 5))
    distances = all_pairwise_distances(q_pairs)
    min_dist = jnp.min(distances)
    contact_count = jnp.sum(distances < 2 * collision_radius * 1.001)

    analysis_log = (
        f"Final Analysis for {protocol_name}:\n"
        f"-----------------------------------------\n"
        f"Number of Rods: {num_rods}\n"
        f"Aspect Ratio (AR): {aspect_ratio}\n"
        f"Rod Diameter (1/AR): {1/aspect_ratio:.4f}\n"
        f"Minimum Distance: {min_dist:.6f}\n"
        f"Target Minimum Distance (Diameter): {2*collision_radius:.6f}\n"
        f"Number of Contacts (< 1.001 * Diameter): {contact_count}\n"
        f"Total Rod Pairs: {len(distances)}\n"
    )
    print(analysis_log)
    with open(run_output_dir / "analysis_summary.txt", "w") as f:
        f.write(analysis_log)


# ==============================================================================
# SCRIPT EXECUTION
# ==============================================================================
if __name__ == "__main__":
    # Example usage of the cleaned-up protocol
    # You can easily loop through different parameters here
    
    # --- Parameters ---
    NUM_RODS = 50
    ASPECT_RATIO = 100.0
    
    # run_entangle_and_relax_protocol(
    #     num_rods=NUM_RODS,
    #     aspect_ratio=ASPECT_RATIO,
    #     output_dir=EXPORT_DIR
    # )

    # Example of generating non-intersecting rods in a box
    q_non_intersect = create_nonintersecting_rods(
        num_rods=100,
        rod_diameter=1.0/50.0,
        container_dims=(3.0, 3.0, 3.0),
        max_attempts=50000
    )
    if q_non_intersect.shape[0] > 0:
        
        # check non-intersection
        q_pairs = create_pairs(q_non_intersect)
        distances = all_pairwise_distances(q_pairs)
        min_dist = jnp.min(distances)
        print(f"Minimum distance between rods: {min_dist:.6f}")

        # visualize
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        plot_many_rods(q_non_intersect, ax=ax)
        ax.axis('equal')
        plt.show()

        # export
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        export_path = EXPORT_DIR / f"nonintersecting_rods_{timestamp}.txt"
        np.savetxt(export_path, q_non_intersect)
        print(f"Non-intersecting rods saved to: {export_path}")

        # voxelize
        # from voxelize import voxelize_rods
        # volume = voxelize_rods(
        #     q=q_non_intersect,
        #     rod_diameter=1.0/20.0,
        #     volume_shape=(512, 512, 512),
        #     box_bounds=(np.array([-1.5, -1.5, -1.5]), np.array([1.5, 1.5, 1.5]))
        # )
        # print(f"Voxelized volume shape: {volume.shape}")
        
        # # export voxel data
        # voxel_export_path = EXPORT_DIR / f"voxelized_rods_{timestamp}.npy"
        # np.save(voxel_export_path, volume)
        
        # # savemat for voxel
        # from scipy.io import savemat

        # savemat(EXPORT_DIR / f"voxelized_rods_{timestamp}.mat", {"volume": volume})        

        # print(f"Voxel data saved to: {voxel_export_path}")
              
              