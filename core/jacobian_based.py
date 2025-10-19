import os
from pathlib import Path
from functools import partial
import jax
import jax.numpy as jnp
from jax import jit, vmap, grad
from jax.lax import cond, scan
import numpy as onp
import polyscope as ps

# --- Assumed Core Imports ---
# These functions are from your project's 'core' directory.
# Ensure they are accessible from where you run this script.
from protocols import create_nonintersecting_random_rods_contained_pbc
from potentials import total_effective_potential
from transforms import q_to_x
from visualizations import prep_for_polyscope


## ----------------------------------------------------------------------------
## Core Physics & Projection Functions
## ----------------------------------------------------------------------------

@jit
def dist_lin_seg(p1_start, p1_end, p2_start, p2_end):
    """
    Calculates the shortest distance between two line segments using a JIT-compiled
    and numerically stable approach.
    """
    EPS = 1e-8
    d1 = p1_end - p1_start
    d2 = p2_end - p2_start
    d12 = p2_start - p1_start
    D1 = jnp.dot(d1, d1)
    D2 = jnp.dot(d2, d2)
    S1 = jnp.dot(d1, d12)
    S2 = jnp.dot(d2, d12)
    R = jnp.dot(d1, d2)
    den = D1 * D2 - R**2

    def non_parallel_case(_):
        t_unc = (S1 * D2 - S2 * R) / (den + EPS)
        u_unc = (S1 * R - S2 * D1) / (-den - EPS)
        t_clamped, u_clamped = jnp.clip(t_unc, 0., 1.), jnp.clip(u_unc, 0., 1.)
        u_for_clamped_t = jnp.clip((t_clamped * R - S2) / (D2 + EPS), 0., 1.)
        t_for_clamped_u = jnp.clip((u_clamped * R + S1) / (D1 + EPS), 0., 1.)
        t_final = jnp.where((t_unc >= 0) & (t_unc <= 1), t_clamped, t_for_clamped_u)
        u_final = jnp.where((u_unc >= 0) & (u_unc <= 1), u_clamped, u_for_clamped_t)
        return t_final, u_final

    def parallel_case(_):
        t = jnp.clip(S1 / (D1 + EPS), 0., 1.)
        u = jnp.clip(-S2 / (D2 + EPS), 0., 1.)
        return t, u

    t_final, u_final = cond(den < EPS, parallel_case, non_parallel_case, operand=None)
    dist_vec = d1 * t_final - d2 * u_final - d12
    return jnp.linalg.norm(dist_vec)


@partial(jit, static_argnums=(1,))
def get_all_pairwise_distances(q, num_rods):
    """
    Calculates and returns a flat vector of distances for all unique
    rod pairs. This function is designed to be used with jax.jacobian.
    """
    x = q_to_x(q).reshape(num_rods, 2, 3)
    p_starts, p_ends = x[:, 0, :], x[:, 1, :]
    i_indices, j_indices = jnp.triu_indices(num_rods, k=1)
    dist_func_vmap = vmap(dist_lin_seg, in_axes=(0, 0, 0, 0))
    all_dists = dist_func_vmap(
        p_starts[i_indices], p_ends[i_indices],
        p_starts[j_indices], p_ends[j_indices]
    )
    return all_dists


@partial(jit, static_argnums=(1, 2))
def project_with_jacobian(q, num_rods, rod_diameter):
    """
    Projects rods to satisfy non-penetration using a single Jacobian-based step.
    """
    # Step 1 & 2: Identify violations and get penetration depths (C)
    all_dists = get_all_pairwise_distances(q, num_rods)
    violations = all_dists - rod_diameter
    is_violating_mask = violations < 0.0
    C = jnp.where(is_violating_mask, violations, 0.0)

    # Step 3: Compute the Jacobian (J) for violating pairs
    J_all = jax.jacobian(get_all_pairwise_distances, argnums=0)(q, num_rods)
    J_tensor = jnp.where(is_violating_mask[:, None, None], J_all, 0.0)

    # --- THIS IS THE FIX ---
    # Reshape the Jacobian from a 3D tensor to a 2D matrix for linear algebra.
    # The shape changes from (190, 20, 5) to (190, 100).
    num_constraints = J_tensor.shape[0]
    J_flat = J_tensor.reshape((num_constraints, -1))

    # Step 4: Solve the linear system with the flattened 2D matrix
    JJT = J_flat @ J_flat.T
    
    num_pairs = JJT.shape[0]
    lambda_reg = 1e-6
    JJT_reg = JJT + lambda_reg * jnp.eye(num_pairs)
    lambda_vec = jnp.linalg.solve(JJT_reg, -C)
    
    # The result is a flattened correction vector of shape (100,)
    delta_q_flat = J_flat.T @ lambda_vec

    # Step 5: Apply the correction
    # Reshape the correction back to the original shape of q (20, 5) before adding.
    delta_q = delta_q_flat.reshape(q.shape)
    q_new = q + delta_q
    
    return q_new

## ----------------------------------------------------------------------------
## Simulation Runner
## ----------------------------------------------------------------------------

from functools import partial # Make sure this is imported at the top

def get_jacobian_simulation_runner(num_rods, main_grad_fn, rod_diameter):
    """
    Returns a JIT-compiled function that runs the simulation using the
    Jacobian projection method.
    """
    def simulation_step(q_carry, _):
        """One full step: main gradient update + Jacobian projection."""
        grad_main = main_grad_fn(q_carry)
        q_after_main_step = q_carry - 1e-3 * grad_main # Main optimization step
        q_final = project_with_jacobian(q_after_main_step, num_rods, rod_diameter)
        return q_final, q_final

    # --- THIS DECORATOR IS THE FIX ---
    # We tell JAX that `num_steps` is a static constant for this compilation.
    @partial(jit, static_argnames='num_steps')
    def run_simulation(q_initial, num_steps):
        """Runs the full simulation for a set number of steps."""
        _, q_trajectory = scan(simulation_step, q_initial, None, length=num_steps)
        return q_trajectory

    return run_simulation

## ----------------------------------------------------------------------------
## Main Execution
## ----------------------------------------------------------------------------

def main():
    # --- Configuration ---
    NUM_RODS = 20
    ALPHA = 100
    ROD_DIAMETER = 1.0 / ALPHA
    CONTAINER_SIZE = 1.0
    SIMULATION_STEPS = 200
    SAVE_EVERY_N_STEPS = 5

    # --- Output Setup ---
    output_folder = Path(f"{Path(__file__).stem}_output")
    movie_dir = output_folder / "movie"
    movie_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output will be saved to: {output_folder.resolve()}")

    # --- Initialization ---
    key = jax.random.PRNGKey(11)
    q0 = create_nonintersecting_random_rods_contained_pbc(NUM_RODS, ROD_DIAMETER, CONTAINER_SIZE)

    # --- Define Potentials and Gradients ---
    grad_main_fn = jit(grad(total_effective_potential))

    # --- Run Simulation ---
    print("Compiling and running simulation... (This may take a moment)")
    simulation_runner = get_jacobian_simulation_runner(NUM_RODS, grad_main_fn, ROD_DIAMETER)
    qq_trajectory = simulation_runner(q0, SIMULATION_STEPS)
    qq_trajectory[-1].block_until_ready()
    print("Simulation finished.")

    # --- Save Trajectory ---
    onp.save(output_folder / "qq_trajectory.npy", onp.array(qq_trajectory))
    print("Trajectory saved.")

    # --- Visualization with Polyscope ---
    print("Preparing visualization...")
    ps.init()
    ps.set_autoscale_structures(False)
    ps.set_automatically_compute_scene_extents(False)
    ps.set_ground_plane_mode("none")
    ps.set_view_projection_mode("orthographic")

    # Initial structure
    a_list_of_curves = q_to_x(q0).reshape(NUM_RODS, -1, 3)
    nodes, edges, _ = prep_for_polyscope(a_list_of_curves, NUM_RODS)
    ps_curves = ps.register_curve_network("filaments", nodes, edges)
    ps_curves.set_radius(ROD_DIAMETER / 2, relative=False)
    sz = CONTAINER_SIZE * 1.2
    ps.set_bounding_box((-sz/2, -sz/2, -sz/2), (sz/2, sz/2, sz/2))

    # Loop through saved results and generate screenshots
    num_frames = SIMULATION_STEPS // SAVE_EVERY_N_STEPS
    for i in range(num_frames):
        q_current = qq_trajectory[i * SAVE_EVERY_N_STEPS]
        curves = q_to_x(q_current).reshape(NUM_RODS, -1, 3)
        ps_curves.update_node_positions(curves.reshape(-1, 3))
        pth = movie_dir / f"step-{i:04d}.png"
        ps.screenshot(str(pth), transparent_bg=True)
        print(f"Saved frame {i+1} of {num_frames}", end='\r')

    print("\nVisualization complete. Screenshots saved.")

if __name__ == "__main__":
    main()