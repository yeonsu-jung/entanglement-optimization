# %%
from __future__ import annotations

from pathlib import Path
from functools import partial
import os
import numpy as onp

import jax
import jax.numpy as jnp
from jax import jit, grad, vmap, lax
import jax.random as jr

# ---------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------

jax.config.update("jax_enable_x64", True)

CURRENT_FILE = Path(__file__).name
OUT_DIR = Path(CURRENT_FILE).with_suffix("")
OUT_DIR.mkdir(exist_ok=True)
print(f"Running {CURRENT_FILE} -> outputs in {OUT_DIR}/")

# ---------------------------------------------------------------------
# Geometry utilities
# ---------------------------------------------------------------------

@jit
def _clip01(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.clip(x, 0.0, 1.0)

@jit
def _dist_point_segment(x: jnp.ndarray, a: jnp.ndarray, b: jnp.ndarray, eps: float = 1e-12) -> jnp.ndarray:
    v = b - a
    vv = jnp.dot(v, v)
    t = jnp.where(vv > eps, _clip01(jnp.dot(x - a, v) / vv), 0.0)
    c = a + t * v
    return jnp.linalg.norm(x - c)

@jit
def dist_lin_seg(p1s: jnp.ndarray, p1e: jnp.ndarray,
                 p2s: jnp.ndarray, p2e: jnp.ndarray,
                 eps: float = 1e-12) -> jnp.ndarray:
    d1 = p1e - p1s
    d2 = p2e - p2s
    r  = p1s - p2s
    a = jnp.dot(d1, d1)
    e = jnp.dot(d2, d2)
    b = jnp.dot(d1, d2)
    c = jnp.dot(d1, r)
    f = jnp.dot(d2, r)
    det = a * e - b * b
    a_zero = a < eps
    e_zero = e < eps

    def both_points(_):
        return jnp.linalg.norm(p1s - p2s)

    def first_point(_):
        return _dist_point_segment(p1s, p2s, p2e, eps)

    def second_point(_):
        return _dist_point_segment(p2s, p1s, p1e, eps)

    def general_case(_):
        def d2_for(s, t):
            diff = (p1s + s * d1) - (p2s + t * d2)
            return jnp.dot(diff, diff)

        s0 = jnp.where(jnp.abs(det) > eps, (b * f - c * e) / det, 0.0)
        t0 = jnp.where(jnp.abs(det) > eps, (a * f - b * c) / det, 0.0)

        sA = _clip01(s0); tA = _clip01((b * sA + f) / jnp.where(e > eps, e, 1.0))
        tB = _clip01(t0); sB = _clip01((b * tB - c) / jnp.where(a > eps, a, 1.0))
        sC, tC = 0.0, _clip01(f / jnp.where(e > eps, e, 1.0))
        sD, tD = 1.0, _clip01((b + f) / jnp.where(e > eps, e, 1.0))
        tE, sE = 0.0, _clip01(-c / jnp.where(a > eps, a, 1.0))
        tF, sF = 1.0, _clip01((b - c) / jnp.where(a > eps, a, 1.0))

        d2s = jnp.stack([d2_for(sA, tA), d2_for(sB, tB), d2_for(sC, tC),
                         d2_for(sD, tD), d2_for(sE, tE), d2_for(sF, tF)])
        return jnp.sqrt(jnp.min(d2s))

    return lax.cond(a_zero & e_zero, both_points,
           lambda _: lax.cond(a_zero, first_point,
                    lambda __: lax.cond(e_zero, second_point, general_case, operand=None),
                    operand=None),
           operand=None)

# ---------------------------------------------------------------------
# Rod parameterization & helpers
#   Each rod row is [x, y, z, phi, theta] with segment length = seg_len
# ---------------------------------------------------------------------

@jit
def sph_to_dir_scalar(phi: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
    s = jnp.sin(phi)
    return jnp.array([s * jnp.cos(theta), s * jnp.sin(theta), jnp.cos(phi)])

@jit
def endpoints_from_q(q_row: jnp.ndarray, seg_len: float = 1.0) -> tuple[jnp.ndarray, jnp.ndarray]:
    p = q_row[:3]
    u = sph_to_dir_scalar(q_row[3], q_row[4])
    return p, p + seg_len * u

@jit
def create_pairs(m: jnp.ndarray) -> jnp.ndarray:
    N, M = m.shape
    i, j = jnp.triu_indices(N, k=1)
    return jnp.concatenate([m[i], m[j]], axis=1)

@jit
def pairwise_distance_for_pair(q_pair: jnp.ndarray, seg_len: float = 1.0) -> jnp.ndarray:
    q_i = q_pair[:5]
    q_j = q_pair[5:]
    p_i, p_ii = endpoints_from_q(q_i, seg_len)
    p_j, p_jj = endpoints_from_q(q_j, seg_len)
    return dist_lin_seg(p_i, p_ii, p_j, p_jj)

# ---------------------------------------------------------------------
# Entanglement (pairwise ACN-like quantity)
# ---------------------------------------------------------------------

@jit
def _acn_cartesian(p_i, p_ii, p_j, p_jj) -> jnp.ndarray:
    r_ij   = p_i  - p_j
    r_ijj  = p_i  - p_jj
    r_iij  = p_ii - p_j
    r_iijj = p_ii - p_jj

    tol = 1e-6
    n1 = jnp.cross(r_ij,  r_ijj);  n1 = n1/(jnp.linalg.norm(n1)  + tol)
    n2 = jnp.cross(r_ijj, r_iijj); n2 = n2/(jnp.linalg.norm(n2) + tol)
    n3 = jnp.cross(r_iijj,r_iij);  n3 = n3/(jnp.linalg.norm(n3)  + tol)
    n4 = jnp.cross(r_iij, r_ij);   n4 = n4/(jnp.linalg.norm(n4)  + tol)

    clip = lambda x: jnp.clip(x, -1.0, 1.0)
    s12 = jnp.arcsin(clip(jnp.dot(n1, n2)))
    s23 = jnp.arcsin(clip(jnp.dot(n2, n3)))
    s34 = jnp.arcsin(clip(jnp.dot(n3, n4)))
    s41 = jnp.arcsin(clip(jnp.dot(n4, n1)))
    return - (s12 + s23 + s34 + s41) / (4.0 * jnp.pi)

@jit
def acn_for_pair(q_pair: jnp.ndarray, seg_len: float = 1.0) -> jnp.ndarray:
    q_i = q_pair[:5]
    q_j = q_pair[5:]
    p_i, p_ii = endpoints_from_q(q_i, seg_len)
    p_j, p_jj = endpoints_from_q(q_j, seg_len)
    return _acn_cartesian(p_i, p_ii, p_j, p_jj)

@partial(jit, static_argnames=("seg_len",))
def total_abs_acn(q_flat: jnp.ndarray, seg_len: float = 1.0) -> jnp.ndarray:
    q = jnp.reshape(q_flat, (-1, 5))
    pairs = create_pairs(q)
    acn_vals = vmap(lambda qp: jnp.abs(acn_for_pair(qp, seg_len)))(pairs)
    return jnp.sum(acn_vals)

# ---------------------------------------------------------------------
# Collision penalty
# ---------------------------------------------------------------------

@partial(jit, static_argnames=("seg_len",))
def total_harmonic_line(q_flat: jnp.ndarray, col_rad: float, amp: float,
                        seg_len: float = 1.0) -> jnp.ndarray:
    q = jnp.reshape(q_flat, (-1, 5))
    pairs = create_pairs(q)
    thr = 2.0 * col_rad

    def penalty(qp):
        d = pairwise_distance_for_pair(qp, seg_len)
        return jnp.where(d < thr, amp * (d - thr) ** 2, 0.0)

    penal = vmap(penalty)(pairs)
    return jnp.sum(penal)

# ---------------------------------------------------------------------
# Optimizer (FIRE-like, eager / host loop)
# ---------------------------------------------------------------------

def optimize_fire(q0: jnp.ndarray,
                  f: callable,
                  df: callable,
                  Nmax: int,
                  *,
                  atol: float = 1e-4,
                  dt: float = 2e-3,
                  finc: float = 1.1,
                  fdec: float = 0.5,
                  fa: float = 0.99,
                  alpha0: float = 0.1,
                  Ndelay: int = 5,
                  log_every: int = 100,
                  seg_len: float = 1.0) -> tuple[jnp.ndarray, float, int, float]:
    q = q0.copy()
    V = jnp.zeros_like(q)
    F = -df(q)

    alpha = float(alpha0)
    dt_curr = float(dt)
    npos = 0

    for it in range(int(Nmax)):
        P = jnp.vdot(F, V)
        p = float(P)

        Fn = jnp.linalg.norm(F) + 1e-30
        Vn = jnp.linalg.norm(V) + 1e-30
        V = (1.0 - alpha) * V + alpha * (F / Fn) * Vn

        if p > 0.0:
            npos += 1
            if npos > Ndelay:
                dt_curr = min(dt_curr * float(finc), 10.0 * float(dt))
                alpha *= float(fa)
        else:
            npos = 0
            dt_curr *= float(fdec)
            alpha = float(alpha0)
            V = jnp.zeros_like(V)

        V = V + 0.5 * dt_curr * F
        q = q + dt_curr * V
        F = -df(q)
        V = V + 0.5 * dt_curr * F

        err = float(jnp.max(jnp.abs(F)))
        if (it % log_every) == 0:
            q_rows = jnp.reshape(q, (-1, 5))
            i_idx, j_idx = jnp.triu_indices(q_rows.shape[0], k=1)
            def d_ij(i, j):
                p_i, p_ii = endpoints_from_q(q_rows[i], seg_len)
                p_j, p_jj = endpoints_from_q(q_rows[j], seg_len)
                return dist_lin_seg(p_i, p_ii, p_j, p_jj)
            dmins = vmap(d_ij)(i_idx, j_idx)
            print(f"Iter {it:4d}  f={float(f(q)):12.6e}  |F|_inf={err:9.2e}  d_min={float(jnp.min(dmins)):9.6e}")
        if err < float(atol):
            break

    return q, float(f(q)), it, err

# ---------------------------------------------------------------------
# Two-stage driver: entangle then relax
# ---------------------------------------------------------------------

def entangle_then_relax(q_flat: jnp.ndarray,
                        *,
                        seg_len: float = 1.0,
                        entangle_steps: int = 3000,
                        entangle_dt: float = 1e-3,
                        entangle_atol: float = 1e-8,
                        col_rad: float = 2e-3,
                        penalty_amp: float = 1.0,
                        relax_outer: int = 20,
                        relax_steps: int = 50000,
                        relax_dt: float = 1e-4,
                        relax_atol: float = 1e-10) -> jnp.ndarray:
    fA = lambda q: -total_abs_acn(q, seg_len=seg_len)
    dfA = jit(grad(jit(fA)))
    print("\n[Stage A] Entangling (maximize |ACN|)")
    q_mid, f_valA, itA, errA = optimize_fire(
        q_flat, fA, dfA, entangle_steps, atol=entangle_atol, dt=entangle_dt, seg_len=seg_len
    )
    acn_before = float(total_abs_acn(q_flat, seg_len))
    acn_after  = float(total_abs_acn(q_mid, seg_len))
    print(f"[Stage A] |ACN|: {acn_before:.6f} -> {acn_after:.6f}   (f={f_valA:.6e}, iters={itA}, err={errA:.2e})")

    print("\n[Stage B] Collision relaxation")
    for k in range(relax_outer):
        fB = lambda q: total_harmonic_line(q, col_rad=col_rad, amp=penalty_amp, seg_len=seg_len)
        dfB = jit(grad(jit(fB)))
        q_mid, f_valB, itB, errB = optimize_fire(
            q_mid, fB, dfB, relax_steps, atol=relax_atol, dt=relax_dt, seg_len=seg_len
        )
        q_rows = jnp.reshape(q_mid, (-1, 5))
        i_idx, j_idx = jnp.triu_indices(q_rows.shape[0], k=1)
        def d_ij(i, j):
            p_i, p_ii = endpoints_from_q(q_rows[i], seg_len)
            p_j, p_jj = endpoints_from_q(q_rows[j], seg_len)
            return dist_lin_seg(p_i, p_ii, p_j, p_jj)
        dmins = vmap(d_ij)(i_idx, j_idx)
        dmin = float(jnp.min(dmins))
        print(f"[outer {k:02d}] f={f_valB:.6e}  |F|_inf={errB:.2e}  d_min={dmin:.6e}")
        if dmin > 2.0 * col_rad:
            print(f"[outer {k:02d}] Separation achieved (d_min > 2*col_rad).")
            break
    return q_mid

# ---------------------------------------------------------------------
# Random initial state
# ---------------------------------------------------------------------

def random_rods(num_rods: int, key: jax.random.PRNGKey) -> jnp.ndarray:
    k2, k3 = jr.split(key, 2)
    u = jr.uniform(k2, (num_rods,), minval=-1.0, maxval=1.0)  # cos(phi) ~ U[-1,1]
    phi   = jnp.arccos(u)
    theta = jr.uniform(k3, (num_rods,), minval=0.0, maxval=2.0 * jnp.pi)
    centers = jnp.zeros((num_rods, 3))
    return jnp.column_stack([centers, phi, theta])

def create_random_initial_state(num_rods: int, key: jax.random.PRNGKey) -> jnp.ndarray:
    q = random_rods(num_rods, key)
    k = jr.PRNGKey(42)
    jitter = jr.normal(k, (num_rods, 3))
    jitter = jitter / (jnp.linalg.norm(jitter, axis=1, keepdims=True) + 1e-30)
    return q.at[:, :3].add(1e-10 * jitter)

# ---------------------------------------------------------------------
# Alternating entangle / nudge primitives
# ---------------------------------------------------------------------

def min_pair_distance_host(q_flat: jnp.ndarray, seg_len: float = 1.0) -> float:
    q_rows = q_flat.reshape(-1, 5)
    i_idx, j_idx = jnp.triu_indices(q_rows.shape[0], k=1)
    def d_ij(i, j):
        p_i, p_ii = endpoints_from_q(q_rows[i], seg_len)
        p_j, p_jj = endpoints_from_q(q_rows[j], seg_len)
        return dist_lin_seg(p_i, p_ii, p_j, p_jj)
    dmins = vmap(d_ij)(i_idx, j_idx)
    return float(jnp.min(dmins))

def entangle_step(q_flat: jnp.ndarray, *, seg_len: float = 1.0,
                  iters: int = 25, dt: float = 1e-3, atol: float = 1e-8) -> jnp.ndarray:
    fA = lambda q: -total_abs_acn(q, seg_len=seg_len)
    dfA = jit(grad(jit(fA)))
    q_new, _, _, _ = optimize_fire(q_flat, fA, dfA, iters, atol=atol, dt=dt, seg_len=seg_len)
    return q_new

def nudge_step(q_flat: jnp.ndarray, *, seg_len: float = 1.0, col_rad: float = 2e-3, amp: float = 1.0,
               iters: int = 1000, dt: float = 1e-4, atol: float = 1e-10) -> jnp.ndarray:
    fB = lambda q: total_harmonic_line(q, col_rad=col_rad, amp=amp, seg_len=seg_len)
    dfB = jit(grad(jit(fB)))
    q_new, _, _, _ = optimize_fire(q_flat, fB, dfB, iters, atol=atol, dt=dt, seg_len=seg_len, log_every=1000)
    return q_new

def alternate_entangle_nudge(q_flat: jnp.ndarray, *,
    seg_len: float = 1.0,
    col_rad: float = 2e-3,
    penalty_amp: float = 1.0,
    entangle_iters: int = 25,
    entangle_dt: float = 1e-3,
    entangle_atol: float = 1e-8,
    nudge_iters: int = 1000,
    nudge_dt: float = 1e-4,
    nudge_atol: float = 1e-10,
    max_cycles: int = 10,
    dmin_safety: float = 0.0,
    log_every: int = 1,
) -> jnp.ndarray:
    target = 2.0 * col_rad + dmin_safety
    q = q_flat
    for cyc in range(max_cycles):
        q = entangle_step(q, seg_len=seg_len, iters=entangle_iters, dt=entangle_dt, atol=entangle_atol)
        q = nudge_step(q, seg_len=seg_len, col_rad=col_rad, amp=penalty_amp,
                       iters=nudge_iters, dt=nudge_dt, atol=nudge_atol)
        dmin = min_pair_distance_host(q, seg_len)
        acn  = float(total_abs_acn(q, seg_len))
        if (cyc % log_every) == 0:
            print(f"[cycle {cyc:03d}] |ACN|={acn:.6f}  d_min={dmin:.6e}  target>{target:.6e}")
        if dmin >= target:
            print(f"Non-penetration satisfied at cycle {cyc}: d_min={dmin:.6e} ≥ {target:.6e}")
            break
    return q

# ---------------------------------------------------------------------
# Driver that repeats the alternation and records history
# ---------------------------------------------------------------------

def entangle_and_nudge_every(
    *,
    num_rods: int = 20,
    seg_len: float = 1.0,
    alpha: float = 100.0,          # diameter = seg_len/alpha
    penalty_amp: float = 1.0,
    max_outer_loops: int = 10,
    rng_key: jax.random.PRNGKey = jr.PRNGKey(0),
    entangle_iters: int = 1,
    entangle_dt: float = 1e-3,
    entangle_atol: float = 1e-8,
    nudge_iters: int = 1500,
    nudge_dt: float = 1e-2,
    nudge_atol: float = 1e-10,
    max_cycles: int = 10,
    dmin_safety: float = 0.0,
):
    col_rad = seg_len / (2.0 * alpha)
    target = 2.0 * col_rad + dmin_safety

    q = create_random_initial_state(num_rods=num_rods, key=rng_key)
    q_flat = q.reshape(-1).astype(jnp.float64)

    history = []
    for outer in range(max_outer_loops):
        q_flat = alternate_entangle_nudge(
            q_flat,
            seg_len=seg_len,
            col_rad=col_rad,
            penalty_amp=penalty_amp,
            entangle_iters=entangle_iters,
            entangle_dt=entangle_dt,
            entangle_atol=entangle_atol,
            nudge_iters=nudge_iters,
            nudge_dt=nudge_dt,
            nudge_atol=nudge_atol,
            max_cycles=max_cycles,
            dmin_safety=dmin_safety,
        )
        history.append(onp.asarray(q_flat))  # store on CPU for polyscope/saving
        dmin = min_pair_distance_host(q_flat, seg_len)
        print(f"[outer {outer:03d}] d_min={dmin:.6e}  target>{target:.6e}")

    q_final = q_flat.reshape(-1, 5)
    try:
        from visualizations import plot_many_rods
        from matplotlib import pyplot as plt
        ax = plot_many_rods(q_final)
        ax.axis("equal")
        plt.savefig(OUT_DIR / "alt_entangle_nudge.png", dpi=300)
        plt.close()
    except Exception as e:
        print(f"(plot skipped) {e}")

    jnp.save(OUT_DIR / "alt_relaxed_q.npy", q_final)
    jnp.save(OUT_DIR / "alt_relaxed_q_history.npy", onp.stack(history, axis=0))
    return history

# ---------------------------------------------------------------------
# Polyscope helpers (movie)
# ---------------------------------------------------------------------

def prep_for_polyscope(curves: onp.ndarray) -> tuple[onp.ndarray, onp.ndarray, onp.ndarray]:
    """
    curves: (num_rods, Ni, 3) array (each rod can share same Ni here)
    Returns nodes (sum_i Ni, 3), edges (E,2), edge_colors (E,3).
    """
    num_rods = curves.shape[0]
    # palette for rods (interpolated across rods)
    base = onp.array([
        [76, 153, 204], [204, 76, 153], [76, 204, 153], [153, 204, 76],
        [204, 153, 76], [153, 76, 204], [204, 76, 102], [76, 204, 204],
        [204, 204, 76], [102, 76, 204]
    ], dtype=float) / 255.0
    t_base = onp.linspace(0.0, 1.0, base.shape[0])
    t_rods = onp.linspace(0.0, 1.0, num_rods)
    colors_rods = onp.column_stack([
        onp.interp(t_rods, t_base, base[:, c]) for c in range(3)
    ])  # (num_rods,3)

    nodes = []
    edges = []
    edge_cols = []
    offset = 0
    for i in range(num_rods):
        pts = curves[i]  # (Ni,3)
        Ni = pts.shape[0]
        nodes.append(pts)
        # edges for a polyline
        e = onp.column_stack([onp.arange(offset, offset + Ni - 1),
                              onp.arange(offset + 1, offset + Ni)])
        edges.append(e)
        # color: all edges of this rod share same color
        edge_cols.append(onp.repeat(colors_rods[i][None, :], e.shape[0], axis=0))
        offset += Ni
    nodes = onp.vstack(nodes)
    edges = onp.vstack(edges)
    edge_cols = onp.vstack(edge_cols)
    return nodes, edges, edge_cols

def create_movie(qq, rod_diameter, *, seg_len: float = 1.0):
    """
    qq: list/array of q_flat snapshots over time
    """
    try:
        import polyscope as ps
        from transforms import q_to_x  # user-provided
    except Exception as e:
        print(f"(movie skipped; missing deps) {e}")
        return

    ps.init()
    ps.set_autoscale_structures(False)
    ps.set_automatically_compute_scene_extents(False)
    ps.set_ground_plane_mode("none")
    ps.set_length_scale(2.0)

    # first frame
    q0 = jnp.array(qq[0]).reshape(-1, 5)
    x0 = q_to_x(q0)  # expects (num_rods, 2, 3*?) or similar; user-provided
    curves0 = x0.reshape(q0.shape[0], -1, 3)  # (num_rods, Ni, 3)
    nodes, edges, edge_colors = prep_for_polyscope(onp.asarray(curves0))
    ps_curves = ps.register_curve_network("filaments", nodes, edges)
    ps_curves.add_color_quantity("edge_colors", edge_colors, defined_on='edges', enabled=True)
    ps_curves.set_radius(rod_diameter / 2, relative=False)

    # scene box
    sz = 2.0
    low = onp.array((-sz, -sz, -sz))
    high = onp.array((sz, sz, sz))
    ps.set_bounding_box(low, high)
    ps.set_up_dir("z_up")

    (OUT_DIR / "movie").mkdir(exist_ok=True)
    for k, qk in enumerate(qq):
        qk = jnp.array(qk).reshape(-1, 5)
        xk = q_to_x(qk).reshape(qk.shape[0], -1, 3)
        nodes_k, _, _ = prep_for_polyscope(onp.asarray(xk))
        ps_curves.update_node_positions(nodes_k)
        ps.screenshot(str(OUT_DIR / f"movie/step-{k:04d}.png"))

# ---------------------------------------------------------------------
# Demo / script entry
# ---------------------------------------------------------------------

if __name__ == "__main__":
    alpha = 100.0  # rod diameter = seg_len/alpha
    history = entangle_and_nudge_every(alpha=alpha)  # returns list of q_flat snapshots

    create_movie(history, rod_diameter=1.0/alpha)
