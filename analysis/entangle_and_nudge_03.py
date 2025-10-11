import sys
import os
from pathlib import Path

# Use regular numpy for operations outside of JAX's scope, like saving files.
import numpy as np 
import jax
import jax.numpy as jnp
from jax import jit, vmap, lax
# import polyscope as ps
from jax.lax import cond


# --- User-Defined Module Imports ---
# Make sure the 'core' directory is in the Python path
sys.path.append('../core') 
from protocols import create_nonintersecting_random_rods_contained_pbc
from protocols import create_nonintersecting_random_rods_contained_centroids
from transforms import q_to_x
# from visualizations import prep_for_polyscope
from potentials import total_effective_potential, total_harmonic_line

# --- Core Geometric & Utility Functions ---


@jit
def dist_lin_seg(p1_start, p1_end, p2_start, p2_end):
    """
    Calculates the shortest distance between two line segments using a JIT-compiled
    and numerically stable approach.
    """
    # Epsilon for numerical stability to avoid division by zero
    EPS = 1e-8

    # Vectors for the line segments and their initial separation
    d1 = p1_end - p1_start
    d2 = p2_end - p2_start
    d12 = p2_start - p1_start

    # Squared lengths of the segments
    D1 = jnp.dot(d1, d1)
    D2 = jnp.dot(d2, d2)

    # Dot products for solving the system of linear equations
    S1 = jnp.dot(d1, d12)
    S2 = jnp.dot(d2, d12)
    R = jnp.dot(d1, d2)

    # Denominator for the system solution. If near zero, lines are parallel.
    den = D1 * D2 - R**2

    def non_parallel_case(_):
        """Computes the closest points for non-parallel lines."""
        t_unc = (S1 * D2 - S2 * R) / (den + EPS)
        u_unc = (S1 * R - S2 * D1) / (-den - EPS)
        
        # Clamp parameters to the segment range [0, 1]
        t_clamped = jnp.clip(t_unc, 0., 1.)
        u_clamped = jnp.clip(u_unc, 0., 1.)

        # If one parameter was clamped, we must re-calculate the other.
        # This handles cases where the closest point is at an endpoint.
        u_for_clamped_t = jnp.clip((t_clamped * R - S2) / (D2 + EPS), 0., 1.)
        t_for_clamped_u = jnp.clip((u_clamped * R + S1) / (D1 + EPS), 0., 1.)
        
        # Choose the final parameters based on which original value was in [0,1]
        t_final = jnp.where((t_unc >= 0) & (t_unc <= 1), t_clamped, t_for_clamped_u)
        u_final = jnp.where((u_unc >= 0) & (u_unc <= 1), u_clamped, u_for_clamped_t)

        return t_final, u_final

    def parallel_case(_):
        """Computes the closest points for parallel lines (or zero-length segments)."""
        t = jnp.clip(S1 / (D1 + EPS), 0., 1.)
        u = jnp.clip(-S2 / (D2 + EPS), 0., 1.)
        return t, u

    # Use lax.cond to choose between the parallel and non-parallel computation paths
    t_final, u_final = cond(
        den < EPS,
        parallel_case,
        non_parallel_case,
        operand=None
    )
    
    # Calculate final distance using the determined segment parameters
    dist_vec = d1 * t_final - d2 * u_final - d12
    return jnp.linalg.norm(dist_vec)



@jit
def dist_lin_seg_pairwise(starts1, ends1, starts2, ends2):
    """Computes distances between corresponding pairs of line segments via vmap."""
    return vmap(dist_lin_seg)(starts1, ends1, starts2, ends2)

def get_rod_endpoints(q):
    """Helper to get rod start and end points from generalized coordinates q."""
    x = q_to_x(q)
    starts = x.reshape(-1, 6)[:, :3]
    ends = x.reshape(-1, 6)[:, 3:]
    return starts, ends

# --- Simulation Logic ---

def create_simulation_step_fn(grad_fn_potential, grad_fn_repulsion, rod_indices, rod_diameter, max_projection_steps):
    """
    Creates a JIT-compiled function for performing one full simulation step.
    This factory pattern helps manage function arguments and JIT compilation.
    """
    i_indices, j_indices = rod_indices

    def calculate_min_distance(q):
        """Calculates the minimum distance between any pair of rods."""
        starts, ends = get_rod_endpoints(q)
        # Gather the endpoints for the specified pairs
        starts1, ends1 = starts[i_indices], ends[i_indices]
        starts2, ends2 = starts[j_indices], ends[j_indices]
        dist_mat = dist_lin_seg_pairwise(starts1, ends1, starts2, ends2)
        return jnp.min(dist_mat)

    @jit
    def simulation_step(q, step_size):
        """Performs one step of gradient descent followed by collision projection."""
        # 1. Apply the main potential gradient
        grad_potential = grad_fn_potential(q)
        q = q - 0.1*step_size * grad_potential
        
        # 2. Use a while_loop for efficient collision projection
        def projection_cond(state):
            q_proj, step_num = state
            min_dist = calculate_min_distance(q_proj)
            return (min_dist < rod_diameter) & (step_num < max_projection_steps)

        def projection_body(state):
            q_proj, step_num = state
            grad_repulsion = grad_fn_repulsion(q_proj)
            q_proj = q_proj - 2 * step_size * grad_repulsion
            return q_proj, step_num + 1

        q_final, num_steps = lax.while_loop(projection_cond, projection_body, (q, 0))
        
        return q_final, num_steps

    return simulation_step, calculate_min_distance

# --- Main Execution ---

def setup_directories(script_path):
    """Creates output directories for simulation results and movies."""
    script_name = Path(script_path).stem
    output_dir = Path(script_name)
    movie_dir = output_dir / "movie"
    output_dir.mkdir(exist_ok=True)
    movie_dir.mkdir(exist_ok=True)
    print(f"✅ Outputting to: {output_dir.resolve()}")
    return output_dir, movie_dir

def main():
    # --- Configuration ---
    # Simulation Parameters
    # NUM_RODS = 200
    ROD_DIAMETER = 1 / ASPECT_RATIO


    CONTAINER_SIZE = 0.52
    # RANDOM_SEED = 11
    
    # Gradient Descent Parameters
    TOTAL_STEPS = 10000
    STEP_SIZE = 1e-3
    MAX_PROJECTION_STEPS = 1000
    
    # Repulsion Potential Parameters
    REPULSION_AMP = 100
    REPULSION_RADIUS_FACTOR = 1.05 # Multiplier for collision radius
    
    # Visualization & Saving
    SAVE_EVERY_N_STEPS = 1
    
    # --- Setup ---
    output_dir, movie_dir = setup_directories(__file__)
    key = jax.random.PRNGKey(RANDOM_SEED)

    # q0 = create_nonintersecting_random_rods_contained_pbc(NUM_RODS, ROD_DIAMETER, CONTAINER_SIZE)
    q0 = create_nonintersecting_random_rods_contained_centroids(NUM_RODS,ROD_DIAMETER,CONTAINER_SIZE,random_seed=RANDOM_SEED,max_attempts=10000)
    
    # Define potential functions and their JIT-compiled gradients
    grad_potential_fn = jit(jax.grad(total_effective_potential))
    
    repulsion_params = {
        'col_rad': (ROD_DIAMETER / 2) * REPULSION_RADIUS_FACTOR,
        'amp': REPULSION_AMP
    }
    grad_repulsion_fn = jit(jax.grad(lambda q: total_harmonic_line(q, repulsion_params)))

    rod_indices = jnp.triu_indices(NUM_RODS, k=1)

    # Create the JIT-compiled simulation step function and a helper for distance checks
    step_fn, min_dist_fn = create_simulation_step_fn(
        grad_potential_fn,
        grad_repulsion_fn,
        rod_indices,
        ROD_DIAMETER,
        MAX_PROJECTION_STEPS
    )

    # --- Polyscope Visualization Setup ---
    import platform

    system_name = platform.system()
    if system_name == "Darwin":
        print("Running on macOS")
    elif system_name == "Linux":
        print("Running on Linux")
    else:
        print(f"Other OS: {system_name}")

    # only run polyscope if not on linux
    if system_name == "Darwin":
        import polyscope as ps
        from visualizations import prep_for_polyscope
        ps.init()
        ps.set_autoscale_structures(False)
        ps.set_automatically_compute_scene_extents(False)
        ps.set_ground_plane_mode("none")
        ps.set_length_scale(2.)
        ps.set_bounding_box((-2., -2., -2.), (2., 2., 2.))
        ps.set_up_dir("z_up")

        initial_curves = q_to_x(q0).reshape(NUM_RODS, -1, 3)
        nodes, edges, _ = prep_for_polyscope(initial_curves, NUM_RODS)
        ps_curves = ps.register_curve_network("filaments", nodes, edges)
        ps_curves.set_radius(ROD_DIAMETER / 2, relative=False)

    # --- Simulation Loop ---
    print("🚀 Starting simulation...")
    q = q0
    # Pre-allocate array for history instead of appending to a list
    q_history = np.zeros((TOTAL_STEPS // SAVE_EVERY_N_STEPS, *q0.shape))

    for step_idx in range(TOTAL_STEPS):
        q, num_proj_steps = step_fn(q, STEP_SIZE)
        
        # --- Visualization and Logging ---
        if step_idx % SAVE_EVERY_N_STEPS == 0:
            history_idx = step_idx // SAVE_EVERY_N_STEPS
            q_history[history_idx] = q
            
            min_dist = min_dist_fn(q)
            print(f"Step: {step_idx:5d} | Min Distance: {min_dist:.4f} | Projection Steps: {num_proj_steps}")


            # if os is linux, ignore below            
            if system_name == "Darwin":
                # ps.set_window_title(f"Step: {step_idx:5d} | Min Dist: {min_dist:.4f} | Proj Steps: {num_proj_steps}")

                # Update Polyscope view
                updated_curves = q_to_x(q).reshape(NUM_RODS, -1, 3)
                ps_curves.update_node_positions(updated_curves.reshape(-1, 3))
                
                # Save screenshot
                screenshot_path = movie_dir / f"step-{history_idx:04d}.png"
                ps.screenshot(str(screenshot_path), transparent_bg=True)

    # --- Save Final Results ---
    final_q_path = output_dir / "q_history.npy"
    np.save(final_q_path, q_history)
    print(f"✅ Simulation finished. Saved history to {final_q_path}")

if __name__ == "__main__":
    import sys
    assert(len(sys.argv) == 4)
    NUM_RODS = int(sys.argv[1])
    ASPECT_RATIO = int(sys.argv[2])
    RANDOM_SEED = int(sys.argv[3])

    print(f"NUM_RODS: {NUM_RODS}, ASPECT_RATIO: {ASPECT_RATIO}, RANDOM_SEED: {RANDOM_SEED}")

    main()