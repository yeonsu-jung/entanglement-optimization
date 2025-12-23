import jax
import jax.numpy as jnp
from jax import jit, grad, vmap, config, lax
import time
import matplotlib.pyplot as plt

import polyscope as ps
config.update("jax_enable_x64", True)

TIME_HORIZON = 10001

def fixbound(num):
    """Ensure the number is within the bounds [0, 1]."""
    return jnp.clip(num, 0, 1)

# -----------------------
# Segment-segment distance
# -----------------------
def dist_lin_seg_flat(points):
    d1 = points[3:6] - points[0:3]
    d2 = points[9:12] - points[6:9]
    d12 = points[6:9] - points[0:3]

    D1 = jnp.dot(d1, d1)
    D2 = jnp.dot(d2, d2)
    S1 = jnp.dot(d1, d12)
    S2 = jnp.dot(d2, d12)
    R = jnp.dot(d1, d2)

    den = D1 * D2 - R**2
    
    def case1():
        (t,u) = lax.cond( D1 != 0. , 
                    lambda _: (fixbound(S1/D1),0.),
                    lambda _: lax.cond(D2 != 0.,
                             lambda _: (0.,fixbound(-S2/D2)),
                             lambda _: (0.,0.),
                             None),
                    None)        
        return (t,u)
    
    def case2_1():
        t = 0.
        u = -S2/D2
        uf = fixbound(u)
        (t,u) = lax.cond(uf != u, 
                    lambda _: (fixbound((uf * R + S1) / D1), uf),
                    lambda _: (t, u),
                    None)
        return (t,u)
    
    def case2_2():
        t = fixbound((S1 * D2 - S2 * R) / den)
        u = (t * R - S2) / D2
        uf = fixbound(u)
        (t,u) = lax.cond(uf != u, 
                    lambda _: (fixbound((uf * R + S1) / D1), uf),
                    lambda _: (t, u),
                    None)
        return (t,u)        
    
    def case2():
        (t,u) = lax.cond( den == 0. , 
                    lambda _: case2_1(),                    
                    lambda _: case2_2(),
                    None)        
        return (t,u)
    
    (t,u) = lax.cond( (D1 == 0.) & (D2 == 0.),
                        lambda _: case1(),
                        lambda _: case2(),
                        None)
    
    dist = jnp.linalg.norm(d1 * t - d2 * u - d12)
    return dist

@jit
def dist_lin_seg(point1s, point1e, point2s, point2e):
    d1 = point1e - point1s
    d2 = point2e - point2s
    d12 = point2s - point1s

    D1 = jnp.dot(d1, d1)
    D2 = jnp.dot(d2, d2)
    S1 = jnp.dot(d1, d12)
    S2 = jnp.dot(d2, d12)
    R = jnp.dot(d1, d2)

    den = D1 * D2 - R**2
    
    def case1():
        (t,u) = lax.cond( D1 != 0. , 
                    lambda _: (fixbound(S1/D1),0.),
                    lambda _: lax.cond(D2 != 0.,
                             lambda _: (0.,fixbound(-S2/D2)),
                             lambda _: (0.,0.),
                             None),
                    None)        
        return (t,u)
    
    def case2_1():
        t = 0.
        u = -S2/D2
        uf = fixbound(u)
        (t,u) = lax.cond(uf != u, 
                    lambda _: (fixbound((uf * R + S1) / D1), uf),
                    lambda _: (t, u),
                    None)
        return (t,u)
    
    def case2_2():
        t = fixbound((S1 * D2 - S2 * R) / den)
        u = (t * R - S2) / D2
        uf = fixbound(u)
        (t,u) = lax.cond(uf != u, 
                    lambda _: (fixbound((uf * R + S1) / D1), uf),
                    lambda _: (t, u),
                    None)
        return (t,u)        
    
    def case2():
        (t,u) = lax.cond( den == 0. , 
                    lambda _: case2_1(),                    
                    lambda _: case2_2(),
                    None)        
        return (t,u)
    
    (t,u) = lax.cond( (D1 == 0.) & (D2 == 0.),
                        lambda _: case1(),
                        lambda _: case2(),
                        None)
    
    dist = jnp.linalg.norm(d1 * t - d2 * u - d12)
    return dist

@jit
def dist_lin_seg_over_ij(r1, r2, i_indices, j_indices):
    return vmap(lambda i, j: dist_lin_seg(r1[i], r2[i], r1[j], r2[j]))(i_indices, j_indices)

# For distances between TWO different polylines (all segment pairs)
def min_distance_between_polylines(A0, A1, B0, B1):
    # A0,A1: (NA,3), B0,B1: (NB,3)
    def d_pair(a0,a1,b0,b1):
        return dist_lin_seg(a0,a1,b0,b1)
    D = jax.vmap(lambda a0,a1: jax.vmap(lambda b0,b1: d_pair(a0,a1,b0,b1))(B0,B1))(A0,A1)  # (NA,NB)
    return jnp.min(D)

# --------------------------------
# Your curvature-based kinematics
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

# -------------------------
# Utilities for test curves
# -------------------------
def trefoil_knot(t):
    x = jnp.sin(t) + 2 * jnp.sin(2 * t)
    y = jnp.cos(t) - 2 * jnp.cos(2 * t)
    z = -jnp.sin(3 * t)
    return jnp.stack([x, y, z], axis=-1)

def get_straight_counterpart(positions):
    N = positions.shape[0]
    tangents = positions[1:] - positions[:-1]
    edge_lengths = jnp.linalg.norm(tangents, axis=1)
    cumsum_lengths = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(edge_lengths)])
    first_direction = tangents[0] / jnp.linalg.norm(tangents[0])
    straight_positions = jnp.zeros_like(positions)
    straight_positions = straight_positions.at[0].set(positions[0])
    for i in range(1, N):
        straight_positions = straight_positions.at[i].set(positions[0] + first_direction * cumsum_lengths[i])
    return straight_positions

def get_edge_lengths(positions):
    tangents = positions[1:] - positions[:-1]
    return jnp.linalg.norm(tangents, axis=-1)

# Your discrete "derivative curvature" to initialize
def get_der_curvature(x):
    tangents = x[1:] - x[:-1]
    tangents = tangents / jnp.linalg.norm(tangents, axis=-1, keepdims=True)
    kappa = 2 * jnp.cross(tangents[:-1], tangents[1:]) / (1 + jnp.sum(tangents[:-1] * tangents[1:], axis=-1, keepdims=True))
    return kappa

# -----------------------------------------------
# Your per-segment linking-angle contribution
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

# Vectorized total (pairwise) linking over two polylines
def _segments(P): return P[:-1], P[1:]

def _safe_seg_contrib(p_i, p_ii, p_j, p_jj, eps=1e-12):
    ok_i = jnp.linalg.norm(p_ii - p_i)  > eps
    ok_j = jnp.linalg.norm(p_jj - p_j)  > eps
    def _do(): return compute_linking_number_cartesian(p_i, p_ii, p_j, p_jj)
    return lax.cond(ok_i & ok_j, _do, lambda: 0.0)

_pair_vmap = vmap(
    vmap(_safe_seg_contrib, in_axes=(None, None, 0, 0)),
    in_axes=(0, 0, None, None)
)

def total_linking_number_polyline(A: jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:
    A0, A1 = _segments(A)
    B0, B1 = _segments(B)
    contribs = _pair_vmap(A0, A1, B0, B1)
    contribs = jnp.nan_to_num(contribs, nan=0.0, posinf=0.0, neginf=0.0)
    return jnp.sum(jnp.abs(contribs))

# -----------------------------
# Penalties & objective pieces
# -----------------------------
def self_collision_penalty(q, i_idx, j_idx, diameter):
    r1, r2 = q[:-1], q[1:]
    pdist = dist_lin_seg_over_ij(r1, r2, i_idx, j_idx)
    penalties = jnp.clip(diameter - pdist, a_min=0.0)
    return jnp.sum(penalties**2)

def rod_clearance_penalty(q, rod, dmin):
    # Use segment-segment distances between all tentacle and rod segments
    A0, A1 = q[:-1], q[1:]
    B0, B1 = rod[:-1], rod[1:]
    def d_pair(a0,a1,b0,b1): return dist_lin_seg(a0,a1,b0,b1)
    D = jax.vmap(lambda a0,a1: jax.vmap(lambda b0,b1: d_pair(a0,a1,b0,b1))(B0,B1))(A0,A1)  # (NA,NB)
    penalties = jnp.clip(dmin - D, a_min=0.0)
    return jnp.sum(penalties**2)

# -----------------------------
# Main optimization loop (curv)
# -----------------------------
def gradient_wrt_curvature(initial_curvature,
                           natural_curvature,
                           first_point, first_edge,
                           natural_edge_lengths,
                           i_indices, j_indices,
                           diameter,                 # self-collision rod diameter
                           rod_positions,            # fixed rod polyline
                           rod_clearance=0.05,       # min tentacle-rod distance
                           w_self=10.0,
                           w_rod=10.0,
                           w_kappa=1e-3,
                           step=2e-3):

    # Objective: minimize loss = -(linking) + penalties + reg
    def loss_fn(kappa):
        q, _ = reconstruct_curve_from_curvature_and_length(first_point, first_edge, kappa, natural_edge_lengths)
        lk = total_linking_number_polyline(q, rod_positions)
        self_pen = self_collision_penalty(q, i_indices, j_indices, diameter)
        rod_bending = jnp.sum(jnp.square(kappa - natural_curvature))
        rod_pen  = rod_clearance_penalty(q, rod_positions, rod_clearance)
        reg = w_kappa * jnp.sum(kappa*kappa)
        loss = -(lk) + w_self*self_pen + w_rod*rod_pen + reg + rod_bending
        # loss = w_self*self_pen + w_rod*rod_pen + reg + rod_bending
        return loss, (lk, self_pen, rod_pen, reg, rod_bending)
    
    def proj_fn(kappa):
        q, _ = reconstruct_curve_from_curvature_and_length(first_point, first_edge, kappa, natural_edge_lengths)


    loss_and_grad = jit(grad(lambda k: loss_fn(k)[0]))
    loss_only     = jit(lambda k: loss_fn(k)[0])
    loss_aux      = jit(lambda k: loss_fn(k)[1])

    # xopt = jnp.zeros_like(initial_curvature)  # start straight-ish
    xopt = initial_curvature
    start_time = time.time()
    all_x = []

    for i in range(TIME_HORIZON):
        g = loss_and_grad(xopt)
        # rod_pen  = rod_clearance_penalty(q, rod_positions, rod_clearance)

        # project back


        # simple backtracking line search for stability
        tstep = step
        L0 = loss_only(xopt)
        for _ in range(10):
            xnew = xopt - tstep * g
            L1  = loss_only(xnew)
            if jnp.isfinite(L1) & (L1 <= L0):
                xopt = xnew
                break
            tstep *= 0.5

        if i % 30 == 0:
            elapsed_time = time.time() - start_time
            start_time = time.time()
            lk, self_pen, rod_pen, reg, rod_bending = loss_aux(xopt)
            L = -(lk) + w_self*self_pen + w_rod*rod_pen + reg
            print(f"dt {elapsed_time:.2f} | it {i:5d} | loss {float(L): .6e} | "
                  f"LK {float(lk): .6e} | self {float(self_pen): .3e} | rod {float(rod_pen): .3e} | reg {float(reg): .3e} | bending {float(rod_bending): .3e}")
            all_x.append(xopt)

    all_x = jnp.stack(all_x) if len(all_x) else xopt[None, ...]
    return all_x

# -----------------------------
# Driver
# -----------------------------
if __name__ == "__main__":
    N = 300
    t = jnp.linspace(0, 2 * jnp.pi, N, endpoint=True)
    natural_positions = trefoil_knot(t)

    edges = jnp.stack([jnp.arange(N - 1), jnp.arange(1, N)], axis=-1)
    straight_positions = get_straight_counterpart(natural_positions)  # use as the fixed "rod"
    natural_curvature = get_der_curvature(natural_positions)
    natural_edge_lengths = get_edge_lengths(natural_positions)


    # post position [0,0,0] to [0,0,1]
    import numpy as np

    post = jnp.array([[0.0, 0.0, 0.0],[0.0, 0.0, 1.0]])
    tentacle = jnp.array( np.linspace([0.5, -0.5 ,0.5], [0.5, 9.5 ,0.5], N) )

    
    # lk

    A0,A1 = post[:-1], post[1:]
    B0,B1 = tentacle[:-1], tentacle[1:]
    min_d_rod = min_distance_between_polylines(A0,A1,B0,B1)



    print(f"Initial min distance between post and tentacle: {float(min_d_rod):.4e}")

    natural_curvature = get_der_curvature(tentacle)
    natural_edge_lengths = get_edge_lengths(tentacle)
    first_point = tentacle[0]
    first_edge = tentacle[1] - tentacle[0]
    

    initial_curvature = natural_curvature + 0.01 * jax.random.normal(jax.random.PRNGKey(0), natural_curvature.shape)
    q_init, _ = reconstruct_curve_from_curvature_and_length(first_point, first_edge, initial_curvature, natural_edge_lengths)


    # fig,ax=plt.subplots(subplot_kw={'projection':'3d'})
    # ax.plot(post[:,0], post[:,1], post[:,2], 'g-', lw=4, label='post')
    # ax.plot(tentacle[:,0], tentacle[:,1], tentacle[:,2], 'r-', lw=2, label='tentacle (natural)')
    # ax.plot(q_init[:,0], q_init[:,1], q_init[:,2], 'b--', lw=2, label='tentacle (init curv)')
    # plt.axis('equal')
    # plt.show()



    total_length = jnp.sum(natural_edge_lengths)
    aspect_ratio = 100
    diameter = total_length / aspect_ratio  # self-collision "rod" diameter
    dmin_rod = 0.05                         # tentacle–rod minimum gap

    time_step = 1e-3
    # first_point = natural_positions[0]
    # first_edge = natural_positions[1] - natural_positions[0]

    # skip near neighbors for self-collision pairs
    i_indices, j_indices = jnp.triu_indices(N - 1, k=30)



    # ----- optimize curvature to maximize entanglement wrt straight rod -----
    all_x = gradient_wrt_curvature(
        initial_curvature,
        natural_curvature, first_point, first_edge, natural_edge_lengths,
        i_indices, j_indices, diameter,
        rod_positions=post,
        rod_clearance=dmin_rod,
        w_self=1.0, w_rod=10.0, w_kappa=1e-3,
        step=2e-4
    )
    print(all_x.shape)

    # visualize last frame
    final_kappa = all_x[-1]
    rec_final, _ = reconstruct_curve_from_curvature_and_length(
        first_point,
        first_edge,
        final_kappa,
        natural_edge_lengths
    )

    # optional distances for reporting
    r1, r2 = rec_final[:-1], rec_final[1:]
    pdist_self = dist_lin_seg_over_ij(r1, r2, i_indices, j_indices)
    A0,A1 = rec_final[:-1], rec_final[1:]
    B0,B1 = post[:-1], post[1:]
    min_d_rod = min_distance_between_polylines(A0,A1,B0,B1)

    print(f"Final min self distance (over skipped pairs): {float(jnp.min(pdist_self)):.4e}")
    print(f"Final min tentacle–rod distance: {float(min_d_rod):.4e}")

    # ------------- Polyscope preview (optional) -------------
    # try:
        


    # except Exception as e:
    #     print("Polyscope skipped:", e)

    ps.init()
    ps.set_up_dir("z_up")
    ps.set_ground_plane_mode("none")
    ps_init = ps.register_curve_network("tentacle_init", q_init, edges)
    ps_init.set_radius(diameter * 0.5, relative=False)
    ps_init.set_color((0.2, 0.2, 0.8)); ps_init.set_material("clay")

    ps_curve = ps.register_curve_network("tentacle", rec_final, edges)
    ps_curve.set_radius(diameter * 0.5, relative=False)
    ps_curve.set_color((0.8, 0.2, 0.2)); ps_curve.set_material("clay")

    ps_rod = ps.register_curve_network("rod", post, np.array([[0,1]]))
    ps_rod.set_radius(dmin_rod*0.5, relative=False)
    ps_rod.set_color((0.1,0.1,0.1))

    # ps.show()
    for k,x in enumerate(all_x):
        q, _ = reconstruct_curve_from_curvature_and_length(first_point, first_edge, x, natural_edge_lengths)
        ps_curve.update_node_positions(q)
        ps.screenshot(f'tentacle/frame_{k:04d}.png', transparent_bg=True)


