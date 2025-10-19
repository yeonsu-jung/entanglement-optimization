# %%
from __future__ import annotations

import os
from pathlib import Path
from functools import partial

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
    """Clamp to [0,1]."""
    return jnp.clip(x, 0.0, 1.0)

@jit
def _dist_point_segment(x: jnp.ndarray, a: jnp.ndarray, b: jnp.ndarray, eps: float = 1e-12) -> jnp.ndarray:
    """Distance from point x to segment [a,b]."""
    v = b - a
    vv = jnp.dot(v, v)
    t = jnp.where(vv > eps, _clip01(jnp.dot(x - a, v) / vv), 0.0)
    c = a + t * v
    return jnp.linalg.norm(x - c)

@jit
def dist_lin_seg(p1s: jnp.ndarray, p1e: jnp.ndarray,
                 p2s: jnp.ndarray, p2e: jnp.ndarray,
                 eps: float = 1e-12) -> jnp.ndarray:
    """
    Shortest distance between two 3D line segments [p1s,p1e] and [p2s,p2e].
    Robust to parallel/degenerate cases; JAX-safe.
    """
    d1 = p1e - p1s
    d2 = p2e - p2s
    r  = p1s - p2s

    a = jnp.dot(d1, d1)        # ||d1||^2
    e = jnp.dot(d2, d2)        # ||d2||^2
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

        # Unconstrained interior solution if well-conditioned
        s0 = jnp.where(jnp.abs(det) > eps, (b * f - c * e) / det, 0.0)
        t0 = jnp.where(jnp.abs(det) > eps, (a * f - b * c) / det, 0.0)

        # Candidate set with clamping
        sA = _clip01(s0)
        tA = _clip01((b * sA + f) / jnp.where(e > eps, e, 1.0))

        tB = _clip01(t0)
        sB = _clip01((b * tB - c) / jnp.where(a > eps, a, 1.0))

        sC, tC = 0.0, _clip01(f / jnp.where(e > eps, e, 1.0))
        sD, tD = 1.0, _clip01((b + f) / jnp.where(e > eps, e, 1.0))
        tE, sE = 0.0, _clip01(-c / jnp.where(a > eps, a, 1.0))
        tF, sF = 1.0, _clip01((b - c) / jnp.where(a > eps, a, 1.0))

        d2s = jnp.stack([
            d2_for(sA, tA),
            d2_for(sB, tB),
            d2_for(sC, tC),
            d2_for(sD, tD),
            d2_for(sE, tE),
            d2_for(sF, tF),
        ])
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
    """Unit vector from spherical angles (physics convention), scalar version."""
    s = jnp.sin(phi)
    return jnp.array([s * jnp.cos(theta), s * jnp.sin(theta), jnp.cos(phi)])

@jit
def sph_to_dir_rows(phi: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
    """Vectorized over rows: phi, theta are (N,), returns (N,3)."""
    s = jnp.sin(phi)
    x = s * jnp.cos(theta)
    y = s * jnp.sin(theta)
    z = jnp.cos(phi)
    return jnp.stack([x, y, z], axis=1)

@jit
def endpoints_from_q(q_row: jnp.ndarray, seg_len: float = 1.0) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Given one rod [x,y,z,phi,theta], return endpoints (p, p+L*u)."""
    p = q_row[:3]
    u = sph_to_dir_scalar(q_row[3], q_row[4])
    return p, p + seg_len * u

@jit
def create_pairs(m: jnp.ndarray) -> jnp.ndarray:
    """(N,M) -> (N*(N-1)/2, 2M) upper-triangular stacked pairs."""
    N, M = m.shape
    i, j = jnp.triu_indices(N, k=1)
    return jnp.concatenate([m[i], m[j]], axis=1)

# ---------------------------------------------------------------------
# Pairwise distances for rods
# ---------------------------------------------------------------------

@jit
def pairwise_distance_for_pair(q_pair: jnp.ndarray, seg_len: float = 1.0) -> jnp.ndarray:
    """q_pair shape (10,) = (xi,yi,zi,phi_i,theta_i,  xj,yj,zj,phi_j,theta_j)."""
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
    """
    A 4-arc spherical polygon based “ACN-like” measure for two segments.
    Returns a signed quantity; use abs() if you want magnitude only.
    """
    r_ij   = p_i  - p_j
    r_ijj  = p_i  - p_jj
    r_iij  = p_ii - p_j
    r_iijj = p_ii - p_jj

    tol = 1e-6
    n1 = jnp.cross(r_ij,  r_ijj);  n1 = n1/(jnp.linalg.norm(n1)  + tol)
    n2 = jnp.cross(r_ijj, r_iijj); n2 = n2/(jnp.linalg.norm(n2) + tol)
    n3 = jnp.cross(r_iijj,r_iij);  n3 = n3/(jnp.linalg.norm(n3)  + tol)
    n4 = jnp.cross(r_iij, r_ij);   n4 = n4/(jnp.linalg.norm(n4)  + tol)

    # avoid nan at boundaries
    clip = lambda x: jnp.clip(x, -1.0 + 0.0, 1.0 - 0.0)
    s12 = jnp.arcsin(clip(jnp.dot(n1, n2)))
    s23 = jnp.arcsin(clip(jnp.dot(n2, n3)))
    s34 = jnp.arcsin(clip(jnp.dot(n3, n4)))
    s41 = jnp.arcsin(clip(jnp.dot(n4, n1)))
    return - (s12 + s23 + s34 + s41) / (4.0 * jnp.pi)

@jit
def acn_for_pair(q_pair: jnp.ndarray, seg_len: float = 1.0) -> jnp.ndarray:
    """Signed ACN-like measure for a pair q_pair (10,)."""
    q_i = q_pair[:5]
    q_j = q_pair[5:]
    p_i, p_ii = endpoints_from_q(q_i, seg_len)
    p_j, p_jj = endpoints_from_q(q_j, seg_len)
    return _acn_cartesian(p_i, p_ii, p_j, p_jj)

@partial(jit, static_argnames=("seg_len",))
def total_abs_acn(q_flat: jnp.ndarray, seg_len: float = 1.0) -> jnp.ndarray:
    """
    Sum of absolute ACN-like quantity over all unordered pairs.
    q_flat: (N*5,)
    """
    q = jnp.reshape(q_flat, (-1, 5))
    pairs = create_pairs(q)  # (P, 10)
    acn_vals = vmap(lambda qp: jnp.abs(acn_for_pair(qp, seg_len)))(pairs)
    return jnp.sum(acn_vals)

# ---------------------------------------------------------------------
# Collision penalty
# ---------------------------------------------------------------------

@partial(jit, static_argnames=("seg_len",))
def total_harmonic_line(q_flat: jnp.ndarray, col_rad: float, amp: float,
                        seg_len: float = 1.0) -> jnp.ndarray:
    """
    Quadratic penalty whenever segment distance < 2*col_rad.
    """
    q = jnp.reshape(q_flat, (-1, 5))
    pairs = create_pairs(q)  # (P, 10)
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
    """
    Simple FIRE loop (host-controlled), with min-distance logging every log_every iters.
    """
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

        # velocity-Verlet-ish
        V = V + 0.5 * dt_curr * F
        q = q + dt_curr * V
        F = -df(q)
        V = V + 0.5 * dt_curr * F

        err = float(jnp.max(jnp.abs(F)))

        if (it % log_every) == 0:
            # min pair distance snapshot
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
# Outer stages: entangle (increase ACN) then relax (separate)
# ---------------------------------------------------------------------

def entangle_then_relax(q_flat: jnp.ndarray,
                        *,
                        seg_len: float = 1.0,
                        # Stage A: entangle (maximize ACN magnitude)
                        entangle_steps: int = 3000,
                        entangle_dt: float = 1e-3,
                        entangle_atol: float = 1e-8,
                        # Stage B: collision relaxation
                        col_rad: float = 2e-3,
                        penalty_amp: float = 1.0,
                        relax_outer: int = 20,
                        relax_steps: int = 50000,
                        relax_dt: float = 1e-4,
                        relax_atol: float = 1e-10) -> jnp.ndarray:
    """
    Stage A: drive up total |ACN| by minimizing -|ACN|.
    Stage B: push rods apart with quadratic contact penalty.
    """
    # Stage A: entangle
    fA = lambda q: -total_abs_acn(q, seg_len=seg_len)
    dfA = jit(grad(jit(fA)))
    print("\n[Stage A] Entangling (maximize |ACN|)")
    q_mid, f_valA, itA, errA = optimize_fire(
        q_flat, fA, dfA, entangle_steps, atol=entangle_atol, dt=entangle_dt, seg_len=seg_len
    )
    acn_before = float(total_abs_acn(q_flat, seg_len))
    acn_after  = float(total_abs_acn(q_mid, seg_len))
    print(f"[Stage A] |ACN|: {acn_before:.6f} -> {acn_after:.6f}   (f={f_valA:.6e}, iters={itA}, err={errA:.2e})")

    # Stage B: relax
    print("\n[Stage B] Collision relaxation")
    for k in range(relax_outer):
        fB = lambda q: total_harmonic_line(q, col_rad=col_rad, amp=penalty_amp, seg_len=seg_len)
        dfB = jit(grad(jit(fB)))

        q_mid, f_valB, itB, errB = optimize_fire(
            q_mid, fB, dfB, relax_steps, atol=relax_atol, dt=relax_dt, seg_len=seg_len
        )

        # Check current min distance
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
    """
    Return (num_rods,5): centers at the origin (or jittered later),
    and uniformly random directions on S^2.
    """
    k2, k3 = jr.split(key, 2)
    u = jr.uniform(k2, (num_rods,), minval=-1.0, maxval=1.0)   # cos(phi) ~ U[-1,1]
    phi   = jnp.arccos(u)
    theta = jr.uniform(k3, (num_rods,), minval=0.0, maxval=2.0 * jnp.pi)
    centers = jnp.zeros((num_rods, 3))
    return jnp.column_stack([centers, phi, theta])


# ---------------------------------------------------------------------
# Demo / script entry
# ---------------------------------------------------------------------

if __name__ == "__main__":
    # Build an initial configuration
    num_rods = 200
    seg_len  = 1.0

    q = random_rods(num_rods, jr.PRNGKey(0))  # (N,5)
    # tiny jitter to centers (to break exact coincidences)
    k = jr.PRNGKey(42)
    jitter = jr.normal(k, (num_rods, 3))
    jitter = jitter / (jnp.linalg.norm(jitter, axis=1, keepdims=True) + 1e-30)
    q = q.at[:, :3].add(1e-10 * jitter)

    # Save an initial plot (optional)
    try:
        from visualizations import plot_many_rods
        from matplotlib import pyplot as plt
        ax = plot_many_rods(q)
        ax.axis("equal")
        plt.savefig(OUT_DIR / "initial_rods.png", dpi=300)
        plt.close()
    except Exception as e:
        print(f"(plot skipped) {e}")

    # Stage A (entangle) + Stage B (relax)
    q0_flat = q.reshape(-1).astype(jnp.float64)
    q_out = entangle_then_relax(
        q0_flat,
        seg_len=seg_len,
        entangle_steps=3000,
        entangle_dt=1e-3,
        entangle_atol=1e-8,
        col_rad=2e-3,
        penalty_amp=1.0,
        relax_outer=20,
        relax_steps=50000,
        relax_dt=1e-4,
        relax_atol=1e-10,
    )

    # Save final figure and state (optional plot)
    try:
        from visualizations import plot_many_rods
        from matplotlib import pyplot as plt
        ax = plot_many_rods(q_out.reshape(-1, 5))
        ax.axis("equal")
        plt.savefig(OUT_DIR / "relaxed_rods.png", dpi=300)
        plt.close()
    except Exception as e:
        print(f"(plot skipped) {e}")

    jnp.save(OUT_DIR / "relaxed_q.npy", q_out)
# %%
