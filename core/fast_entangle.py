# clean_opt_constrained_rods.py
import os
from pathlib import Path

import numpy as onp
import jax
import jax.numpy as jnp
from jax import jit, vmap, lax

# local imports (keep minimal; avoid repeated path hacking)
import sys
sys.path.append('../core')

from transforms import q_to_x
from potentials import total_effective_potential, total_harmonic_line
from protocols import create_nonintersecting_random_rods_contained_pbc
# from visualizations import prep_for_polyscope  # only needed if rendering

# -----------------------------
# Config
# -----------------------------
NUM_RODS        = 2
ALPHA           = 100.0
ROD_DIAMETER    = 1.0 / ALPHA
CONTAINER_SIZE  = 1.0

LR_MAIN         = 1e-3         # main descent step
LR_PROJECT      = 1e-3         # projection descent step
MAX_STEPS       = 10_000
PROJECT_MAXIT   = 50           # much smaller than 1000; we JIT it
CONTACT_FACTOR  = 0.99         # require min_dist > CONTACT_FACTOR * diameter

SEED            = 11

DO_RENDER       = False        # set True if using polyscope
PRINT_EVERY     = 100           # stats interval
SNAPSHOT_EVERY  = 10           # image interval

# -----------------------------
# Geometry helpers
# -----------------------------
@jit
def _clip01(x):
    return jnp.clip(x, 0.0, 1.0)

@jit
def dist_lin_seg(p1, q1, p2, q2):
    """
    Fast, robust distance between two line segments (p1->q1) and (p2->q2).
    JIT-safe; uses clamped closest-point parameters.
    """
    u = q1 - p1
    v = q2 - p2
    w0 = p1 - p2

    a = jnp.dot(u, u)
    b = jnp.dot(u, v)
    c = jnp.dot(v, v)
    d = jnp.dot(u, w0)
    e = jnp.dot(v, w0)

    denom = a * c - b * b

    # default s,t for parallel/degenerate
    s = 0.0
    t = _clip01(-e / jnp.where(c > 0.0, c, 1.0))

    # non-parallel case
    def _nonparallel(_):
        s_np = _clip01((b * e - c * d) / denom)
        t_np = _clip01((a * e - b * d) / denom)
        return s_np, t_np

    def _parallel(_):
        return s, t

    s, t = lax.cond(denom > 0.0, _nonparallel, _parallel, operand=None)

    # one more clamp pass to handle clamping interactions
    # (project t then recompute s if t clamped, and vice versa)
    t_clamped = _clip01((b * s + e) / jnp.where(c > 0.0, c, 1.0))
    s_clamped = _clip01((b * t + d) / jnp.where(a > 0.0, a, 1.0))

    # if clamping changed one variable significantly, update the other
    s = jnp.where(jnp.abs(t - t_clamped) > 0.0, s_clamped, s)
    t = jnp.where(jnp.abs(s - s_clamped) > 0.0, t_clamped, t)

    dP = w0 + u * s - v * t
    return jnp.linalg.norm(dP)

# vectorized over (i,j) index pairs
@jit
def dist_lin_seg_over_ij(r1, r2, i_idx, j_idx):
    return vmap(lambda i, j: dist_lin_seg(r1[i], r2[i], r1[j], r2[j]))(i_idx, j_idx)

# -----------------------------
# Packing helpers
# -----------------------------
@jit
def flatten_endpoints(q):
    """
    q: (..., 6) rods param as concatenated endpoints or
       (...,) raw params consumable by q_to_x.
    We accept q from user pipeline and convert once.
    """
    x = q_to_x(q)               # (N*2, 3) positions or (N, 2, 3)
    x = x.reshape(-1, 6)        # (N, 6)
    r1 = x[:, :3]
    r2 = x[:, 3:]
    return r1, r2

@jit
def min_pair_distance(q, i_idx, j_idx):
    r1, r2 = flatten_endpoints(q)
    dists = dist_lin_seg_over_ij(r1, r2, i_idx, j_idx)
    return jnp.min(dists)

# -----------------------------
# Energies & grads
# -----------------------------
# main energy grad
grad_main = jit(jax.grad(total_effective_potential))

# small harmonic line “nudge” energy for projection
def _harmonic_energy(q, col_rad, amp):
    params = {'col_rad': col_rad, 'amp': amp}
    return total_harmonic_line(q, params)

grad_project = jit(jax.grad(_harmonic_energy))

# -----------------------------
# Projection step (JITed while_loop)
# -----------------------------
def _project_body(carry):
    q, it, col_rad, amp, i_idx, j_idx = carry
    g = grad_project(q, col_rad, amp)
    q_new = q - LR_PROJECT * g
    return (q_new, it + 1, col_rad, amp, i_idx, j_idx)

def _project_cond(carry):
    q, it, col_rad, amp, i_idx, j_idx = carry
    md = min_pair_distance(q, i_idx, j_idx)
    good = md > (CONTACT_FACTOR * 2.0 * col_rad)  # diameter = 2 * col_rad
    return jnp.logical_and(jnp.logical_not(good), it < PROJECT_MAXIT)

project_loop = jit(lambda q, col_rad, amp, i_idx, j_idx:
                   lax.while_loop(_project_cond, _project_body, (q, 0, col_rad, amp, i_idx, j_idx))[0])

# -----------------------------
# Main single iteration (JIT)
# -----------------------------
@jit
def opt_step(q, i_idx, j_idx, col_rad, amp):
    # descent on main energy
    g = grad_main(q)
    q = q - LR_MAIN * g
    # light projection to keep separation
    q = project_loop(q, col_rad, amp, i_idx, j_idx)
    # diagnostics
    md = min_pair_distance(q, i_idx, j_idx)
    return q, md

# -----------------------------
# I/O helpers
# -----------------------------
def prepare_output_dirs(script_name: str):
    base = Path(script_name).with_suffix('').name
    out_dir = Path(base)
    out_dir.mkdir(exist_ok=True)
    movie_dir = out_dir / "movie"
    movie_dir.mkdir(exist_ok=True)
    return out_dir, movie_dir

# -----------------------------
# Main
# -----------------------------
def main():
    # output dirs
    this_file = __file__
    out_dir, movie_dir = prepare_output_dirs(os.path.basename(this_file))

    # random init
    key = jax.random.PRNGKey(SEED)

    # rod params
    col_rad = ROD_DIAMETER * 0.5
    amp     = 100.0

    # initial non-intersecting rods (PBC)
    q = create_nonintersecting_random_rods_contained_pbc(
        NUM_RODS, ROD_DIAMETER, CONTAINER_SIZE
    )

    # pair indices
    i_idx, j_idx = jnp.triu_indices(NUM_RODS, k=1)

    # optional rendering (off by default for speed)
    if DO_RENDER:
        import polyscope as ps
        from visualizations import prep_for_polyscope

        ps.init()
        ps.set_autoscale_structures(False)
        ps.set_automatically_compute_scene_extents(False)
        ps.set_ground_plane_mode("none")
        ps.set_length_scale(2.0)
        sz = 2.0
        low = onp.array((-sz, -sz, -sz))
        high = onp.array((sz, sz, sz))
        ps.set_bounding_box(low, high)
        ps.set_up_dir("z_up")

        curves = q_to_x(q).reshape(NUM_RODS, -1, 3)
        nodes, edges, edge_colors = prep_for_polyscope(curves, NUM_RODS)
        ps_curves = ps.register_curve_network("filaments", nodes, edges)
        ps_curves.add_color_quantity("edge_colors", edge_colors, defined_on='edges', enabled=True)
        ps_curves.set_radius(ROD_DIAMETER / 2, relative=False)

    # optimization loop
    frames = []
    for step in range(MAX_STEPS):
        q, md = opt_step(q, i_idx, j_idx, col_rad, amp)

        if (step % PRINT_EVERY) == 0:
            print(f"step {step:5d} | min dist: {float(md):.6e} | lr_main={LR_MAIN:.1e} lr_proj={LR_PROJECT:.1e}")

        if DO_RENDER and (step % SNAPSHOT_EVERY) == 0:
            curves = q_to_x(q).reshape(NUM_RODS, -1, 3)
            # update nodes in place (fast)
            ps.get_curve_network("filaments").update_node_positions(curves.reshape(-1, 3))
            img_path = movie_dir / f"step-{step:04d}.png"
            import polyscope as ps
            ps.screenshot(str(img_path))

        # keep a light trail of states (optional)
        if (step % 50) == 0:
            frames.append(onp.array(q))

    if frames:
        onp.save(out_dir / "qq.npy", onp.stack(frames, axis=0))
    else:
        # at least save the final
        onp.save(out_dir / "qq.npy", onp.array([onp.array(q)]))

if __name__ == "__main__":
    # main()  

    p0 = jnp.array([0.0, 0.0, 0.0])
    p1 = jnp.array([1.0, 0.0, 0.0])
    q0 = jnp.array([0.0, 1.0, 1.0])
    q1 = jnp.array([0.0, 1.0, 1.0])
    dist = dist_lin_seg(p0, p1, q0, q1)
    print(f"Distance between parallel segments: {dist:.6e} (should be 1.0)")
    