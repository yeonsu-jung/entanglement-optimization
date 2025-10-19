import jax
import jax.numpy as jnp
import numpy as onp
from jax import grad, jit, lax, vmap
from typing import Callable, Dict, Optional, Tuple
from functools import partial
import matplotlib.pyplot as plt

# ==============================================================================
# Constants
# ==============================================================================
DOFS_PER_ROD = 5
ROD_LENGTH = 1.0

# ==============================================================================
# Core Geometry Functions
# ==============================================================================

def _fix_bound(num: jnp.ndarray) -> jnp.ndarray:
    """Clips a number to the interval [0, 1]."""
    return jnp.clip(num, 0, 1)

@partial(jit, static_argnames=('rod_length',))
def _unpack_rod_pair(
    rod_pair: jnp.ndarray, rod_length: float = ROD_LENGTH
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Unpacks a concatenated 1D array for two rods into their start/end points.

    Args:
        rod_pair: A 1D array of shape (10,) containing state vectors for two rods.
        rod_length: The length of the rods.

    Returns:
        A tuple: (p_i_start, p_i_end, p_j_start, p_j_end).
    """
    p_i_start, phi_i, theta_i = rod_pair[0:3], rod_pair[3], rod_pair[4]
    p_j_start, phi_j, theta_j = rod_pair[5:8], rod_pair[8], rod_pair[9]

    # Calculate direction vectors from spherical coordinates
    u_i = jnp.array([
        jnp.sin(phi_i) * jnp.cos(theta_i),
        jnp.sin(phi_i) * jnp.sin(theta_i),
        jnp.cos(phi_i)
    ])
    u_j = jnp.array([
        jnp.sin(phi_j) * jnp.cos(theta_j),
        jnp.sin(phi_j) * jnp.sin(theta_j),
        jnp.cos(phi_j)
    ])

    p_i_end = p_i_start + rod_length * u_i
    p_j_end = p_j_start + rod_length * u_j

    return p_i_start, p_i_end, p_j_start, p_j_end

@jit
def dist_lin_seg(p1_start: jnp.ndarray, p1_end: jnp.ndarray, p2_start: jnp.ndarray, p2_end: jnp.ndarray) -> jnp.ndarray:
    """
    Calculates the shortest distance between two 3D line segments.

    This implementation is based on the algorithm for finding the closest points
    of two segments, where 't' and 'u' are the normalized positions of those
    points on the respective segments.

    Args:
        p1_start: Start point of the first segment.
        p1_end: End point of the first segment.
        p2_start: Start point of the second segment.
        p2_end: End point of the second segment.

    Returns:
        The minimum Euclidean distance between the two segments.
    """
    d1, d2 = p1_end - p1_start, p2_end - p2_start
    d12 = p2_start - p1_start

    D1, D2 = jnp.dot(d1, d1), jnp.dot(d2, d2)
    S1, S2, R = jnp.dot(d1, d12), jnp.dot(d2, d12), jnp.dot(d1, d2)
    den = D1 * D2 - R**2

    def general_case():
        """Handles the general case where segments are not parallel."""
        t = _fix_bound((S1 * D2 - S2 * R) / den)
        u_unclamped = (t * R - S2) / D2
        u = _fix_bound(u_unclamped)
        # Recalculate t if u was clamped to ensure correctness
        t_recalc = _fix_bound((u * R + S1) / D1)
        t_final = jnp.where(u != u_unclamped, t_recalc, t)
        return t_final, u

    def parallel_case():
        """Handles cases where segments are parallel or one/both are points."""
        t = jnp.where(D1 > 1e-9, _fix_bound(S1 / D1), 0.0)
        u = jnp.where(D2 > 1e-9, _fix_bound(-S2 / D2), 0.0)
        return t, u

    # A small epsilon prevents division by zero for nearly parallel lines.
    t, u = lax.cond(den > 1e-9, general_case, parallel_case)

    closest_p1 = p1_start + t * d1
    closest_p2 = p2_start + u * d2
    return jnp.linalg.norm(closest_p1 - closest_p2)

# ==============================================================================
# System Configuration and Potential
# ==============================================================================

def create_rod_pairs(q_all: jnp.ndarray) -> jnp.ndarray:
    """
    Creates all unique pairs of rods from a system state array.

    Args:
        q_all: A flattened array of shape (N * DOFS_PER_ROD,) representing the system state.

    Returns:
        An array of shape (N*(N-1)/2, 2 * DOFS_PER_ROD) where each row is a
        concatenated pair of rod state vectors.
    """
    rods = jnp.reshape(q_all, (-1, DOFS_PER_ROD))
    num_rods, _ = rods.shape
    i, j = jnp.triu_indices(num_rods, k=1)
    return jnp.concatenate([rods[i], rods[j]], axis=1)

@jit
def _harmonic_repulsion_potential(
    q_pair: jnp.ndarray, collision_radius: float, amplitude: float
) -> jnp.ndarray:
    """Calculates a harmonic repulsion potential for a single pair of rods."""
    p1s, p1e, p2s, p2e = _unpack_rod_pair(q_pair, ROD_LENGTH)
    dist = dist_lin_seg(p1s, p1e, p2s, p2e)
    contact_dist = 2 * collision_radius
    
    return lax.cond(
        dist < contact_dist,
        lambda: amplitude * (dist - contact_dist)**2,
        lambda: 0.0,
    )

@jit
def total_system_potential(q_all: jnp.ndarray, params: Dict[str, float]) -> jnp.ndarray:
    """
    Calculates the total potential energy of the system from pairwise repulsions.
    """
    rod_pairs = create_rod_pairs(q_all)
    # Vmap is often more readable than lax.scan and highly performant.
    potential_vmap = vmap(_harmonic_repulsion_potential, in_axes=(0, None, None))
    return jnp.sum(potential_vmap(rod_pairs, params["col_rad"], params["amp"]))

# ==============================================================================
# Optimization Algorithm (FIRE)
# ==============================================================================

def optimize_fire_individual(
    q0: jnp.ndarray,
    f: Callable[[jnp.ndarray], jnp.ndarray],
    df: Callable[[jnp.ndarray], jnp.ndarray],
    max_steps: int,
    atol: float = 1e-4,
    dt_start: float = 0.002,
    dt_max_factor: float = 10.0,
    alpha_start: float = 0.1,
    n_delay: int = 10,
    f_inc: float = 1.1,
    f_dec: float = 0.5,
    f_alpha: float = 0.99,
    log_interval: int = 100,
    callback: Optional[Callable] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Minimizes a function using the Fast Inertial Relaxation Engine (FIRE) algorithm.
    (Docstring remains the same)
    """
    q, V, F = q0.copy(), jnp.zeros_like(q0), -df(q0)
    dt = jnp.full_like(q, dt_start)
    dt_max = dt_start * dt_max_factor
    alpha = jnp.full_like(q, alpha_start)
    n_pos = jnp.zeros_like(q, dtype=jnp.int32)

    # Re-create the function that computes distance from a q_pair for vmapping
    # This is cleaner than unpacking inside the loop
    pairwise_distance_from_q = jit(lambda q_pair: dist_lin_seg(*_unpack_rod_pair(q_pair)))

    print("--- Starting FIRE Optimization ---")
    for i in range(max_steps):
        P = F * V
        V_norm = jnp.linalg.norm(V)
        F_norm = jnp.linalg.norm(F)
        V = (1 - alpha) * V + alpha * F * V_norm / (F_norm + 1e-9)

        is_positive_power = P > 0
        n_pos = jnp.where(is_positive_power, n_pos + 1, 0)
        
        accel_condition = is_positive_power & (n_pos > n_delay)
        dt = jnp.where(accel_condition, jnp.minimum(dt * f_inc, dt_max), dt)
        alpha = jnp.where(accel_condition, alpha * f_alpha, alpha)
        
        decel_condition = ~is_positive_power
        dt = jnp.where(decel_condition, dt * f_dec, dt)
        alpha = jnp.where(decel_condition, alpha_start, alpha)
        V = jnp.where(decel_condition, 0.0, V)

        q += 0.5 * dt * V
        F = -df(q)
        q += 0.5 * dt * V

        error = jnp.max(jnp.abs(F))
        if onp.mod(i, log_interval) == 0:
            f_val = f(q)
            # --- THIS IS THE CORRECTED PART ---
            # Create pairs first, then map the distance function over them.
            rod_pairs = create_rod_pairs(q)
            # Check if any pairs were created before calculating min distance
            min_dist = jnp.inf
            if rod_pairs.shape[0] > 0:
                all_distances = vmap(pairwise_distance_from_q)(rod_pairs)
                min_dist = jnp.min(all_distances)
            # --- END CORRECTION ---
            print(
                f"Step: {i:5d} | Energy: {f_val:12.6f} | "
                f"Error (Max Force): {error:12.6f} | Min Distance: {min_dist:12.6f}"
            )
            if callback and callback(q, {"step": i, "min_dist": min_dist}):
                print("Callback requested to stop optimization.")
                break
        
        if error < atol:
            print(f"Converged in {i} steps. Final Error: {error:.2e}")
            break

    return q, f(q), error

# ==============================================================================
# Main Relaxation Protocol
# ==============================================================================

def collision_relaxation_protocol(
    q0: jnp.ndarray,
    potential_func: Callable,
    params: Dict[str, float],
    n_outer_loops: int,
    fire_max_steps: int,
    fire_atol: float,
    fire_dt: float,
    callback: Optional[Callable] = None,
) -> jnp.ndarray:
    """Runs a multi-step relaxation protocol to resolve collisions."""
    q = q0.copy()
    initial_contact_dist = 2 * params["col_rad"]
    energy_fn = lambda q_state: potential_func(q_state, params)
    force_fn = jit(grad(energy_fn))

    for k in range(n_outer_loops):
        print(f"\n=== Outer Relaxation Loop: {k + 1}/{n_outer_loops} ===")
        
        q, _, _ = optimize_fire_individual(
            q, energy_fn, force_fn, fire_max_steps, fire_atol, fire_dt,
            callback=callback
        )
        
        min_dist = jnp.min(vmap(dist_lin_seg)(*_unpack_rod_pair(create_rod_pairs(q).T)))
        if min_dist > initial_contact_dist:
            print(f"System relaxed: Min distance {min_dist:.2e} > Contact distance {initial_contact_dist:.2e}.")
            break
            
    return q

# ==============================================================================
# Example Usage
# ==============================================================================

if __name__ == "__main__":
    # These imports are assumed to be in helper files.
    # They are kept here so the core logic can be imported without graphics dependencies.
    from protocols import create_intersecting_rods
    from visualizations import plot_many_rods

    # 1. Setup Initial Configuration
    num_rods = 50  # Reduced for faster testing
    alpha = 100
    rod_length = 1.0 # fixed
    rod_radius = rod_length/alpha

    key = jax.random.PRNGKey(42)
    initial_q = create_intersecting_rods(num_rods)

    # check
    print("Initial configuration:")
    ax = plot_many_rods(initial_q.reshape(-1, DOFS_PER_ROD))
    ax.set_title("Initial Rod Configuration")
    ax.axis('equal')
    plt.show()

    # Add a small random nudge to break symmetries
    centers = initial_q.reshape(-1, DOFS_PER_ROD)[:, 0:3]
    noise = rod_radius*0.1 * jax.random.normal(key, shape=centers.shape)
    nudged_q = initial_q.reshape(-1, DOFS_PER_ROD).at[:, 0:3].add(noise).flatten()
    
    # 2. Define Simulation Parameters
    simulation_params = {
        "col_rad": 1.0e-3,  # Collision radius of each rod
        "amp": 1.0,         # Amplitude of the harmonic repulsion
    }

    # 3. Run the Relaxation Protocol
    final_q = collision_relaxation_protocol(
        q0=nudged_q,
        potential_func=total_system_potential,
        params=simulation_params,
        n_outer_loops=5,
        fire_max_steps=10000,
        fire_atol=1e-4,
        fire_dt=1e-1,
    )

    # 4. Visualize the final, relaxed configuration
    print("\nPlotting final configuration...")
    final_rods = final_q.reshape(-1, DOFS_PER_ROD)
    ax = plot_many_rods(final_rods)
    ax.set_title("Rod Configuration After Relaxation")
    ax.axis('equal')
    # Uncomment the following lines if running as a script to display the plot    
    plt.show()