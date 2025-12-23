import jax
import jax.numpy as jnp
from jax import jit, grad, vmap, config, lax
import time
import numpy as np
import polyscope as ps
import optax

import sys
sys.path.append('core')
from utils import setup_directories

output_dir,movie_dir = setup_directories(__file__)

config.update("jax_enable_x64", True)

TIME_HORIZON = 3001

def fixbound(num):
    return jnp.clip(num, 0.0, 1.0)

# -----------------------
# Segment-segment distance
# -----------------------
def dist_lin_seg_flat(points):
    d1 = points[3:6] - points[0:3]
    d2 = points[9:12] - points[6:9]
    d12 = points[6:9] - points[0:3]

    D1 = jnp.dot(d1, d1); D2 = jnp.dot(d2, d2)
    S1 = jnp.dot(d1, d12); S2 = jnp.dot(d2, d12)
    R  = jnp.dot(d1, d2)
    den = D1 * D2 - R**2

    def case1():
        (t,u) = lax.cond(D1 != 0.,
            lambda _: (fixbound(S1/D1), 0.),
            lambda _: lax.cond(D2 != 0., lambda _: (0., fixbound(-S2/D2)), lambda _: (0.,0.), None),
            None)
        return (t,u)

    def case2_1():
        t = 0.; u = -S2/D2; uf = fixbound(u)
        (t,u) = lax.cond(uf != u,
            lambda _: (fixbound((uf*R + S1)/D1), uf),
            lambda _: (t,u),
            None)
        return (t,u)

    def case2_2():
        t = fixbound((S1*D2 - S2*R)/den)
        u = (t*R - S2)/D2; uf = fixbound(u)
        (t,u) = lax.cond(uf != u,
            lambda _: (fixbound((uf*R + S1)/D1), uf),
            lambda _: (t,u),
            None)
        return (t,u)

    def case2():
        return lax.cond(den == 0., lambda _: case2_1(), lambda _: case2_2(), None)

    (t,u) = lax.cond((D1 == 0.) & (D2 == 0.), lambda _: case1(), lambda _: case2(), None)
    dist = jnp.linalg.norm(d1 * t - d2 * u - d12)
    return dist

@jit
def dist_lin_seg(point1s, point1e, point2s, point2e):
    d1 = point1e - point1s
    d2 = point2e - point2s
    d12 = point2s - point1s

    D1 = jnp.dot(d1, d1); D2 = jnp.dot(d2, d2)
    S1 = jnp.dot(d1, d12); S2 = jnp.dot(d2, d12)
    R  = jnp.dot(d1, d2)
    den = D1 * D2 - R**2

    def case1():
        (t,u) = lax.cond(D1 != 0.,
            lambda _: (fixbound(S1/D1), 0.),
            lambda _: lax.cond(D2 != 0., lambda _: (0., fixbound(-S2/D2)), lambda _: (0.,0.), None),
            None)
        return (t,u)

    def case2_1():
        t = 0.; u = -S2/D2; uf = fixbound(u)
        (t,u) = lax.cond(uf != u,
            lambda _: (fixbound((uf*R + S1)/D1), uf),
            lambda _: (t,u),
            None)
        return (t,u)

    def case2_2():
        t = fixbound((S1*D2 - S2*R)/den)
        u = (t*R - S2)/D2; uf = fixbound(u)
        (t,u) = lax.cond(uf != u,
            lambda _: (fixbound((uf*R + S1)/D1), uf),
            lambda _: (t,u),
            None)
        return (t,u)

    def case2():
        return lax.cond(den == 0., lambda _: case2_1(), lambda _: case2_2(), None)

    (t,u) = lax.cond((D1 == 0.) & (D2 == 0.), lambda _: case1(), lambda _: case2(), None)
    dist = jnp.linalg.norm(d1 * t - d2 * u - d12)
    return dist

@jit
def dist_lin_seg_over_ij(r1, r2, i_indices, j_indices):
    return vmap(lambda i, j: dist_lin_seg(r1[i], r2[i], r1[j], r2[j]))(i_indices, j_indices)

def min_distance_between_polylines(A0, A1, B0, B1):
    def d_pair(a0,a1,b0,b1): return dist_lin_seg(a0,a1,b0,b1)
    D = jax.vmap(lambda a0,a1: jax.vmap(lambda b0,b1: d_pair(a0,a1,b0,b1))(B0,B1))(A0,A1)
    return jnp.min(D)

# --------------------------------
# Curvature-based kinematics
# --------------------------------
@jit
def rotate(u_i, kappa):
    kappa_norm_sq = jnp.dot(kappa, kappa)
    numerator = (4 - kappa_norm_sq) * u_i + 4 * jnp.cross(kappa, u_i) + 2 * kappa * jnp.dot(kappa, u_i)
    return numerator / (4 + kappa_norm_sq)

@jit
def reconstruct_curve_from_curvature_and_length(first_point, first_edge, curvatures, reference_lengths):
    num_segments = len(curvatures) + 1
    nodes = jnp.zeros((num_segments + 1, 3))
    nodes = nodes.at[0].set(first_point)
    tangents = jnp.zeros((num_segments, 3))
    tangents = tangents.at[0].set(first_edge / jnp.linalg.norm(first_edge))

    def tangent_body(i, tangents):
        next_tangent = rotate(tangents[i], curvatures[i])
        tangents = tangents.at[i+1].set(next_tangent)
        return tangents

    tangents = jax.lax.fori_loop(0, num_segments - 1, tangent_body, tangents)

    def node_body(i, nodes):
        return nodes.at[i+1].set(nodes[i] + reference_lengths[i] * tangents[i])

    nodes = jax.lax.fori_loop(0, num_segments, node_body, nodes)
    return nodes, tangents

# === soft-min distances for smooth active-set gradients ===
from jax.flatten_util import ravel_pytree

def softmin_pairwise_distance(A0,A1,B0,B1,beta=80.0):
    # distances matrix (NA, NB)
    pair = jax.vmap(lambda a0,a1: jax.vmap(lambda b0,b1: dist_lin_seg(a0,a1,b0,b1))(B0,B1))(A0,A1)
    # numerical stability: subtract min before exp
    m = jnp.min(pair)
    return -(1.0/beta) * (jnp.log(jnp.sum(jnp.exp(-beta*(pair - m)))) - (-beta*m))

@jit
def softmin_post_distance_from_kappa(kappa, first_point, first_edge, lengths, post, beta=80.0):
    q, _ = reconstruct_curve_from_curvature_and_length(first_point, first_edge, kappa, lengths)
    A0,A1 = q[:-1], q[1:]
    B0,B1 = post[:-1], post[1:]
    return softmin_pairwise_distance(A0,A1,B0,B1,beta=beta)

@jit
def softmin_self_distance_from_kappa(kappa, first_point, first_edge, lengths, i_idx, j_idx, beta=80.0):
    q, _ = reconstruct_curve_from_curvature_and_length(first_point, first_edge, kappa, lengths)
    r1, r2 = q[:-1], q[1:]
    D = vmap(lambda i,j: dist_lin_seg(r1[i], r2[i], r1[j], r2[j]))(i_idx, j_idx)  # (P,)
    m = jnp.min(D)
    return -(1.0/beta) * (jnp.log(jnp.sum(jnp.exp(-beta*(D - m)))) - (-beta*m))

def project_update_against_constraint(update_pytree, gradc_pytree, eps=1e-12):
    u_flat, unflatten = ravel_pytree(update_pytree)
    g_flat, _         = ravel_pytree(gradc_pytree)
    dot   = jnp.vdot(u_flat, g_flat)
    denom = jnp.vdot(g_flat, g_flat) + eps
    u_proj_flat = u_flat - (dot/denom) * g_flat
    return unflatten(u_proj_flat)

# -------------------------
# Utilities
# -------------------------
def get_edge_lengths(positions):
    tangents = positions[1:] - positions[:-1]
    return jnp.linalg.norm(tangents, axis=-1)

def get_der_curvature(x):
    tangents = x[1:] - x[:-1]
    tangents = tangents / jnp.linalg.norm(tangents, axis=-1, keepdims=True)
    kappa = 2 * jnp.cross(tangents[:-1], tangents[1:]) / (1 + jnp.sum(tangents[:-1] * tangents[1:], axis=-1, keepdims=True))
    return kappa

# -----------------------------------------------
# Per-segment linking-angle (yours)
# -----------------------------------------------
def compute_linking_number_cartesian(p_i, p_ii, p_j, p_jj):
    r_ij   = p_i  - p_j
    r_ijj  = p_i  - p_jj
    r_iij  = p_ii - p_j
    r_iijj = p_ii - p_jj

    tol = 1e-6
    n1 = jnp.cross(r_ij,  r_ijj);  n1 = n1 / (jnp.linalg.norm(n1)  + tol)
    n2 = jnp.cross(r_ijj, r_iijj); n2 = n2 / (jnp.linalg.norm(n2) + tol)
    n3 = jnp.cross(r_iijj, r_iij); n3 = n3 / (jnp.linalg.norm(n3) + tol)
    n4 = jnp.cross(r_iij,  r_ij);  n4 = n4 / (jnp.linalg.norm(n4)  + tol)

    tol = 0.0
    return -1.0/(4.0*jnp.pi) * jnp.abs(
        jnp.arcsin(jnp.clip(jnp.dot(n1,n2), -1.0+tol, 1.0-tol)) +
        jnp.arcsin(jnp.clip(jnp.dot(n2,n3), -1.0+tol, 1.0-tol)) +
        jnp.arcsin(jnp.clip(jnp.dot(n3,n4), -1.0+tol, 1.0-tol)) +
        jnp.arcsin(jnp.clip(jnp.dot(n4,n1), -1.0+tol, 1.0-tol))
    )

def _segments(P): return P[:-1], P[1:]

def _safe_seg_contrib(p_i, p_ii, p_j, p_jj, eps=1e-12):
    ok_i = jnp.linalg.norm(p_ii - p_i)  > eps
    ok_j = jnp.linalg.norm(p_jj - p_j)  > eps
    def _do(): return compute_linking_number_cartesian(p_i, p_ii, p_j, p_jj)
    return lax.cond(ok_i & ok_j, _do, lambda: 0.0)

_pair_vmap = vmap(vmap(_safe_seg_contrib, in_axes=(None, None, 0, 0)), in_axes=(0, 0, None, None))

def total_linking_number_polyline(A: jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:
    A0, A1 = _segments(A)
    B0, B1 = _segments(B)
    contribs = _pair_vmap(A0, A1, B0, B1)
    contribs = jnp.nan_to_num(contribs, nan=0.0, posinf=0.0, neginf=0.0)
    return jnp.sum(jnp.abs(contribs))

# -----------------------------
# Hard-constraint helpers (jitted)
# -----------------------------
@jit
def min_self_distance_from_kappa(kappa, first_point, first_edge, lengths, i_idx, j_idx):
    q, _ = reconstruct_curve_from_curvature_and_length(first_point, first_edge, kappa, lengths)
    r1, r2 = q[:-1], q[1:]
    pdist = dist_lin_seg_over_ij(r1, r2, i_idx, j_idx)
    return jnp.min(pdist)

@jit
def min_post_distance_from_kappa(kappa, first_point, first_edge, lengths, post):
    q, _ = reconstruct_curve_from_curvature_and_length(first_point, first_edge, kappa, lengths)
    A0, A1 = q[:-1], q[1:]
    B0, B1 = post[:-1], post[1:]
    return min_distance_between_polylines(A0, A1, B0, B1)

# -----------------------------
# Optax SGD with feasible scaling
# -----------------------------

def optimize_curvature_hard_constraints_optax(initial_curvature,
                                              first_point, first_edge, edge_lengths,
                                              post, i_indices, j_indices,
                                              d_self_min, d_post_min,
                                              lr=2e-4, momentum=0.9, nesterov=True,
                                              w_kappa=1e-3,
                                              max_scale_halvings=80,
                                              beta_softmin=80.0):
    """
    SGD (Optax) with feasible step-scaling AND projection of the update
    onto the tangent cone of active constraints (post and self).
    """

    def loss_fn(kappa):
        q, _ = reconstruct_curve_from_curvature_and_length(first_point, first_edge, kappa, edge_lengths)
        lk  = total_linking_number_polyline(q, post)   # keep using your post as-is for LK
        reg = w_kappa * jnp.sum(kappa * kappa)
        return -(lk) + reg, (lk, reg)

    loss_grad = jit(grad(lambda k: loss_fn(k)[0]))
    loss_only = jit(lambda k: loss_fn(k)[0])
    loss_aux  = jit(lambda k: loss_fn(k)[1])

    opt = optax.sgd(learning_rate=lr, momentum=momentum, nesterov=nesterov)
    kappa = initial_curvature
    opt_state = opt.init(kappa)

    snapshots = []
    start_time = time.time()

    for it in range(TIME_HORIZON):
        # plain SGD update
        g = loss_grad(kappa)
        updates, opt_state = opt.update(g, opt_state, params=kappa)

        # --- Active-set projection of the update ---
        # POST constraint gradient (c_post = d_post_min - softmin_post_distance)
        dsoft_post = softmin_post_distance_from_kappa(kappa, first_point, first_edge, edge_lengths, post, beta=beta_softmin)
        post_active = (dsoft_post <= d_post_min + 1e-4)

        def proj_post(u):
            grad_c_post = grad(lambda kk: d_post_min - softmin_post_distance_from_kappa(
                kk, first_point, first_edge, edge_lengths, post, beta=beta_softmin))(kappa)
            return project_update_against_constraint(u, grad_c_post)

        updates = lax.cond(post_active, proj_post, lambda u: u, updates)

        # SELF constraint (optional but recommended)
        dsoft_self = softmin_self_distance_from_kappa(kappa, first_point, first_edge, edge_lengths, i_indices, j_indices, beta=beta_softmin)
        self_active = (dsoft_self <= d_self_min + 1e-4)

        def proj_self(u):
            grad_c_self = grad(lambda kk: d_self_min - softmin_self_distance_from_kappa(
                kk, first_point, first_edge, edge_lengths, i_indices, j_indices, beta=beta_softmin))(kappa)
            return project_update_against_constraint(u, grad_c_self)

        updates = lax.cond(self_active, proj_self, lambda u: u, updates)

        # --- Feasible step-scaling (now along the wall/tangent) ---
        scale = 1.0
        accepted = False
        for _ in range(max_scale_halvings):
            kappa_trial = optax.apply_updates(kappa, jax.tree_map(lambda u: scale * u, updates))
            d_self = min_self_distance_from_kappa(kappa_trial, first_point, first_edge, edge_lengths, i_indices, j_indices)
            d_post = min_post_distance_from_kappa(kappa_trial, first_point, first_edge, edge_lengths, post)
            L_trial = loss_only(kappa_trial)
            if bool((d_self >= d_self_min) & (d_post >= d_post_min) & jnp.isfinite(L_trial)):
                kappa = kappa_trial
                accepted = True
                break
            scale *= 0.5
        # if no feasible step found, stay at current kappa

        if it % 30 == 0:
            dt = time.time() - start_time; start_time = time.time()
            lk, reg = loss_aux(kappa)
            d_self = min_self_distance_from_kappa(kappa, first_point, first_edge, edge_lengths, i_indices, j_indices)
            d_post = min_post_distance_from_kappa(kappa, first_point, first_edge, edge_lengths, post)
            L = -(lk) + reg
            print(f"dt {dt:.2f} | it {it:5d} | acc {accepted} | loss {float(L): .6e} | "
                  f"LK {float(lk): .6e} | d_self {float(d_self):.4f} | d_post {float(d_post):.4f} | scale {scale:.3f}")
            snapshots.append(kappa)

    snapshots = jnp.stack(snapshots) if len(snapshots) else kappa[None, ...]
    return snapshots


# -----------------------------
# Driver
# -----------------------------
if __name__ == "__main__":
    N = 200

    # Post: single segment with endpoints
    post = jnp.array([[0.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0]])

    # Tentacle: vertical-ish initial centerline offset
    tentacle = jnp.array(np.linspace([0.5, -1, 0.5], [0.5, 2.5, 0.5], N))
    natural_curvature = get_der_curvature(tentacle)
    edge_lengths = get_edge_lengths(tentacle)
    first_point = tentacle[0]
    first_edge  = tentacle[1] - tentacle[0]

    # Slightly perturbed init
    rng = jax.random.PRNGKey(0)
    initial_curvature = natural_curvature + 0.01 * jax.random.normal(rng, natural_curvature.shape)

    # Hard gaps
    total_length = jnp.sum(edge_lengths)
    d_self_min = total_length / 100.0   # self-avoidance gap (skip-neighbor pairs)
    d_post_min = 0.2                    # tentacle–post strict gap

    # Pairs for self-collision checks
    i_indices, j_indices = jnp.triu_indices(N - 1, k=30)
    
    # --------- Optax SGD with strict feasibility ---------
    all_kappa = optimize_curvature_hard_constraints_optax(
        initial_curvature,
        first_point, first_edge, edge_lengths,
        post,
        i_indices, j_indices,
        d_self_min=d_self_min,
        d_post_min=d_post_min,
        lr=1e-5, momentum=0.9, nesterov=True,
        w_kappa=1e-20  # essentially no curvature reg if you want
    )
    print(all_kappa.shape)

    # Reconstruct final tentacle
    final_kappa = all_kappa[-1]
    rec_final, _ = reconstruct_curve_from_curvature_and_length(first_point, first_edge, final_kappa, edge_lengths)

    # Distances
    r1, r2 = rec_final[:-1], rec_final[1:]
    pdist_self = dist_lin_seg_over_ij(r1, r2, i_indices, j_indices)
    A0, A1 = rec_final[:-1], rec_final[1:]
    B0, B1 = post[:-1], post[1:]
    min_d_post = min_distance_between_polylines(A0, A1, B0, B1)

    print(f"Final min self distance (over skipped pairs): {float(jnp.min(pdist_self)):.4e}")
    print(f"Final min tentacle–post distance: {float(min_d_post):.4e}")

    # --------- Polyscope ---------
    ps.init()
    ps.set_up_dir("x_up")
    ps.set_ground_plane_mode("none")

    edges = jnp.stack([jnp.arange(N - 1), jnp.arange(1, N)], axis=-1)
    q_init, _ = reconstruct_curve_from_curvature_and_length(first_point, first_edge, initial_curvature, edge_lengths)

    ps_init = ps.register_curve_network("tentacle_init", q_init, edges)
    ps_init.set_radius(d_self_min * 0.5, relative=False)
    ps_init.set_color((0.2, 0.2, 0.8)); ps_init.set_material("clay")

    ps_curve = ps.register_curve_network("tentacle", rec_final, edges)
    ps_curve.set_radius(d_self_min * 0.5, relative=False)
    ps_curve.set_color((0.8, 0.2, 0.2)); ps_curve.set_material("clay")

    ps_rod = ps.register_curve_network("post", post, np.array([[0,1]]))
    ps_rod.set_radius(d_post_min*0.5, relative=False)
    ps_rod.set_color((0.1,0.1,0.1))

    for k, kappa in enumerate(all_kappa):
        q, _ = reconstruct_curve_from_curvature_and_length(first_point, first_edge, kappa, edge_lengths)
        ps_curve.update_node_positions(q)
        ps.screenshot(f'{movie_dir}/frame_{k:04d}.png', transparent_bg=True)
