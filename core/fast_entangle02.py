# constrained_linesearch_rods_history.py
import os
from pathlib import Path
import numpy as onp
import jax
import jax.numpy as jnp
from jax import jit, vmap, lax

import sys
sys.path.append("../core")

from transforms import q_to_x
from potentials import total_effective_potential
from protocols import create_nonintersecting_random_rods_contained_pbc
from potentials import dist_lin_seg

# ================== CONFIG ==================
NUM_RODS        = 20
ALPHA           = 200.0
ROD_DIAMETER    = 1.0 / ALPHA
CONTAINER_SIZE  = 1.0

MAX_ITERS       = 10_000
PRINT_EVERY     = 10
SAVE_EVERY      = 100        # save q every N steps

LR_INIT         = 1e-2
LS_C1           = 1e-4
LS_SHRINK       = 0.5
LS_MAX_STEPS    = 20

CONTACT_FACTOR  = 0.99
SEED            = 11

# ================== GEOMETRY ==================
@jit
def _clip01(x): return jnp.clip(x, 0.0, 1.0)

@jit
def segseg_dist(p1, q1, p2, q2):
    u = q1 - p1
    v = q2 - p2
    w0 = p1 - p2
    a = jnp.dot(u,u); b = jnp.dot(u,v); c = jnp.dot(v,v)
    d = jnp.dot(u,w0); e = jnp.dot(v,w0)
    denom = a*c - b*b

    s = 0.0
    t = _clip01(-e / jnp.where(c > 0.0, c, 1.0))

    def _nonparallel(_):
        s_np = _clip01((b*e - c*d) / denom)
        t_np = _clip01((a*e - b*d) / denom)
        return s_np, t_np
    s, t = lax.cond(denom > 0.0, _nonparallel, lambda _: (s,t), operand=None)

    t_cl = _clip01((b*s + e) / jnp.where(c > 0.0, c, 1.0))
    s_cl = _clip01((b*t + d) / jnp.where(a > 0.0, a, 1.0))
    s = jnp.where(jnp.abs(t - t_cl) > 0.0, s_cl, s)
    t = jnp.where(jnp.abs(s - s_cl) > 0.0, t_cl, t)

    dP = w0 + u*s - v*t
    return jnp.linalg.norm(dP)

@jit
def flatten_endpoints(q):
    x = q_to_x(q)
    x = x.reshape(-1, 6)
    return x[:, :3], x[:, 3:]

@jit
def min_pair_distance(q, i_idx, j_idx):
    r1, r2 = flatten_endpoints(q)
    # dists = vmap(lambda i,j: segseg_dist(r1[i], r2[i], r1[j], r2[j]))(i_idx, j_idx)
    dists = vmap(lambda i, j: dist_lin_seg(r1[i], r2[i], r1[j], r2[j]))(i_idx, j_idx)

    return jnp.min(dists)

# ================== ENERGY ==================
@jit
def energy(q):
    return total_effective_potential(q)

grad_energy = jit(jax.grad(energy))

# ================== LINE SEARCH ==================
@jit
def line_search_step(q, f, g, p, step0, c1, shrink, ls_max_steps, min_required, i_idx, j_idx):
    f0 = f
    gTp = jnp.vdot(g, p)

    def cond_fun(state):
        qk, fk, ak, it, md = state
        not_armijo = fk > f0 + c1 * ak * gTp
        not_feasible = md < min_required
        try_more = it < ls_max_steps
        return jnp.logical_and(jnp.logical_or(not_armijo, not_feasible), try_more)

    def body_fun(state):
        qk, fk, ak, it, _ = state
        ak_new = ak * shrink
        q_try  = q + ak_new * p
        fk_new = energy(q_try)
        md_new = min_pair_distance(q_try, i_idx, j_idx)
        return (q_try, fk_new, ak_new, it + 1, md_new)

    a0   = step0
    q1   = q + a0 * p
    f1   = energy(q1)
    md1  = min_pair_distance(q1, i_idx, j_idx)
    init = (q1, f1, a0, 0, md1)

    qk, fk, ak, it, md = lax.while_loop(cond_fun, body_fun, init)
    return qk, fk, ak, md

# ================== OPT STEP ==================
@jit
def opt_iter(q, i_idx, j_idx, step0, c1, shrink, ls_max_steps, min_required):
    f = energy(q)
    g = grad_energy(q)
    p = -g
    q_new, f_new, step_used, md = line_search_step(
        q, f, g, p, step0, c1, shrink, ls_max_steps, min_required, i_idx, j_idx
    )
    return q_new, f_new, step_used, md, jnp.linalg.norm(g)

# ================== MAIN ==================
def main():
    out_dir = Path("constrained_linesearch_rods_history")
    out_dir.mkdir(exist_ok=True)

    key = jax.random.PRNGKey(SEED)
    q = create_nonintersecting_random_rods_contained_pbc(
        NUM_RODS, ROD_DIAMETER, CONTAINER_SIZE
    )

    i_idx, j_idx = jnp.triu_indices(NUM_RODS, k=1)
    min_required = CONTACT_FACTOR * ROD_DIAMETER

    step0 = LR_INIT
    q_hist = [onp.array(q)]  # store initial state

    for it in range(MAX_ITERS):
        q, f, step_used, md, gnorm = opt_iter(
            q, i_idx, j_idx, step0, LS_C1, LS_SHRINK, LS_MAX_STEPS, min_required
        )
        step0 = jnp.where(step_used >= LR_INIT * (LS_SHRINK ** 2),
                          step_used * 1.1, step_used)

        if it % PRINT_EVERY == 0:
            print(f"iter {it:5d}  f={float(f):.6e}  md={float(md):.6e}  "
                  f"step={float(step_used):.2e}  ||g||={float(gnorm):.3e}")

        # Save history every N steps (convert to NumPy once)
        if it % SAVE_EVERY == 0:
            q_hist.append(onp.array(q))

    # Final save
    q_hist.append(onp.array(q))
    q_hist = onp.stack(q_hist)
    onp.save(out_dir / "q_history.npy", q_hist)
    print(f"Saved full history with {len(q_hist)} frames to {out_dir/'q_history.npy'}")

if __name__ == "__main__":
    # segseg_dist
    # from potentials import dist_lin_seg

    # qq = jnp.load("/Users/yeonsu/GitHub/entanglement-optimization/core/fast_entangle/qq.npy")    
    # q = qq[-1]

    # x = q_to_x(q)
    # r1 = x.reshape(-1, 6)[:,:3]
    # r2 = x.reshape(-1, 6)[:,3:]
    # ii,jj = jnp.triu_indices(len(r1), k=1)

    # d1 = vmap(lambda i,j: segseg_dist(r1[i], r2[i], r1[j], r2[j]))(ii, jj)
    # d2 = vmap(lambda i,j: dist_lin_seg(r1[i], r2[i], r1[j], r2[j]))(ii, jj)

    # # return vmap(lambda i, j: dist_lin_seg(r1[i], r2[i], r1[j], r2[j]))(i_indices, j_indices)

    # from potentials import dist_lin_seg_over_ij

    # # compare d1 and d2

    # delta_d = jnp.linalg.norm(d1-d2)
    # print(delta_d)
        





    # main()

    # movie
    qq = jnp.load("/Users/yeonsu/GitHub/entanglement-optimization/core/fast_entangle/qq.npy")    

    import polyscope as ps
    from visualizations import prep_for_polyscope
    num_rods = 2
    rod_diameter = 1.0 / 100.0

    ps.init()
    ps.set_autoscale_structures(False)
    ps.set_automatically_compute_scene_extents(False)
    ps.set_ground_plane_mode("none")

    q = qq[0]
    a_list_of_curves = q_to_x(q).reshape(num_rods, -1, 3)
    nodes, edges, edge_colors = prep_for_polyscope(a_list_of_curves, num_rods)
    min_z = onp.min(nodes[:, 2])
    ps_curves = ps.register_curve_network( "filaments", nodes, edges )
    ps_curves.add_color_quantity( "edge_colors", edge_colors, defined_on='edges', enabled=True )
    ps_curves.set_radius( rod_diameter / 2, relative=False )

    ps.set_length_scale(2.)
    sz = 2.
    low = onp.array((-sz, -sz, -sz))
    high = onp.array((sz, sz, sz))
    ps.set_bounding_box(low, high)
    ps.set_up_dir("z_up")

    nodes, edges, edge_colors = prep_for_polyscope(a_list_of_curves, num_rods)

    from potentials import dist_lin_seg_vector

    i_indices, j_indices = jnp.triu_indices(num_rods, k=1)

    MOVIE_DIR = "/Users/yeonsu/GitHub/entanglement-optimization/core/constrained_linesearch_rods_history/movie"
    k = 0
    step_size = 1e-3
    
    for q in qq:
        x = q_to_x(q)
        r1 = x.reshape(-1, 6)[:,:3]
        r2 = x.reshape(-1, 6)[:,3:]        

        a_list_of_curves = q_to_x(q).reshape(num_rods, -1, 3)
        ps_curves.update_node_positions(a_list_of_curves.reshape(-1,3))
        # ps_curves.get_color_quantity("edge_colors").update_values(edge_colors)
        pth = f"{MOVIE_DIR}/step-{k:04d}.png"
        ps.screenshot(str(pth))
        
        k += 1    
