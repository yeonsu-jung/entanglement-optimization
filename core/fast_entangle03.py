# feasibility_newton_rods.py
import sys, os
sys.path.append("../core")

import jax
import jax.numpy as jnp
from jax import jit, vmap, lax

import jax
import jax.numpy as jnp
from jax import vmap, jacrev

from transforms import q_to_x
from protocols import create_nonintersecting_random_rods_contained_pbc

# ======== segment distance (JAX-diffable) ========
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

    # default (parallel/degenerate)
    s = 0.0
    t = _clip01(-e / jnp.where(c > 0.0, c, 1.0))

    def _nonparallel(_):
        s_np = _clip01((b*e - c*d) / denom)
        t_np = _clip01((a*e - b*d) / denom)
        return s_np, t_np

    s, t = lax.cond(denom > 0.0, _nonparallel, lambda _: (s, t), operand=None)

    # coupling from clamping
    t_cl = _clip01((b*s + e) / jnp.where(c > 0.0, c, 1.0))
    s_cl = _clip01((b*t + d) / jnp.where(a > 0.0, a, 1.0))
    s = jnp.where(jnp.abs(t - t_cl) > 0.0, s_cl, s)
    t = jnp.where(jnp.abs(s - s_cl) > 0.0, t_cl, t)

    dP = w0 + u*s - v*t
    return jnp.linalg.norm(dP)

# ======== endpoints from q and pairwise distances ========
@jit
def endpoints_from_q(q):
    """q -> (r1, r2) with shapes (N,3). Assumes q_to_x returns (N,2,3) or (2N,3)."""
    x = q_to_x(q).reshape(-1, 6)  # (N,6)
    return x[:, :3], x[:, 3:]

@jit
def pair_dists(q, i_idx, j_idx):
    r1, r2 = endpoints_from_q(q)
    dist_ij = vmap(lambda i,j: segseg_dist(r1[i], r2[i], r1[j], r2[j]))(i_idx, j_idx)
    return dist_ij

# ======== constraint residuals and energy ========
def make_h_fun(i, j, D):
    """Return a scalar function h_ij(q)=d_ij(q)-D for fixed (i,j)."""
    def h_single(q):
        r1, r2 = endpoints_from_q(q)
        d = segseg_dist(r1[i], r2[i], r1[j], r2[j])
        return d - D
    return h_single

# def build_active_set(q, i_idx, j_idx, D, delta):
#     """Active pairs with d <= D+delta (buffer)."""
#     d = pair_dists(q, i_idx, j_idx)
#     mask = d <= (D + delta)
#     return jnp.where(mask)[0], d  # indices into the flattened pair list, and all d

def residual_vector(q, active_pairs, i_idx, j_idx, D):
    """Residuals r = min(0, h) for active constraints (violations only contribute)."""
    ii = i_idx[active_pairs]
    jj = j_idx[active_pairs]
    # compute h for active
    h_vals = vmap(lambda a,b: make_h_fun(a,b,D))(ii, jj)(q)  # closure vectorization
    r = jnp.minimum(0.0, h_vals)  # negative parts only
    return r, h_vals

# ======== Dense Jacobian (small active set) ========
def jacobian_active(q, active_pairs, i_idx, j_idx, D):
    """Dense J: shape (K, M). K = |active|, M = q.size. Suitable when K is small."""
    ii = i_idx[active_pairs]
    jj = j_idx[active_pairs]

    def grad_h_of_pair(a, b):
        h_fun = make_h_fun(a, b, D)
        return jax.grad(h_fun)  # returns grad function R^M -> R^M

    # Evaluate each row gradient at q
    rows = []
    for a, b in zip(list(ii), list(jj)):
        gh = grad_h_of_pair(int(a), int(b))(q)
        rows.append(gh.reshape(-1))
    J = jnp.stack(rows, axis=0) if rows else jnp.zeros((0, q.size))
    return J

# ======== LM step on violations ========
def lm_step(q, r, J, mu):
    """
    Solve (J^T J + mu I) Δq = -J^T r.
    If no active constraints (J is 0xM), return Δq=0.
    """
    M = q.size
    if J.shape[0] == 0:
        return jnp.zeros_like(q)
    JTJ = J.T @ J
    rhs = -(J.T @ r)
    A = JTJ + mu * jnp.eye(M, dtype=q.dtype)
    dq = jnp.linalg.solve(A, rhs)
    return dq

# ======== backtracking to increase feasibility and decrease violation energy ========
def violation_energy(r):  # 0.5 * ||r||^2
    return 0.5 * jnp.dot(r, r)

def backtrack(q, dq, i_idx, j_idx, D, delta, ls_shrink=0.5, max_ls=20):
    """Backtrack on alpha to ensure (a) phi decreases and (b) min(h) increases."""
    # current residuals & stats
    active_idx, d_now_all = build_active_set(q, i_idx, j_idx, D, delta)
    r_now, h_now = residual_vector(q, active_idx, i_idx, j_idx, D)
    phi_now = violation_energy(r_now)
    min_h_now = jnp.min(h_now) if h_now.size > 0 else jnp.inf

    alpha = 1.0
    q_best = q
    phi_best = phi_now
    min_h_best = min_h_now

    for _ in range(max_ls):
        q_try = q + alpha * dq
        a_idx, _ = build_active_set(q_try, i_idx, j_idx, D, delta)
        r_try, h_try = residual_vector(q_try, a_idx, i_idx, j_idx, D)
        phi_try = violation_energy(r_try)
        min_h_try = jnp.min(h_try) if h_try.size > 0 else jnp.inf

        # Accept if we reduce violation energy and don't worsen the worst gap
        if (phi_try <= phi_now) and (min_h_try >= min_h_now):
            q_best, phi_best, min_h_best = q_try, phi_try, min_h_try
            break
        alpha *= ls_shrink
    return q_best, phi_best, min_h_best, alpha

# ======== Top-level feasibility Newton/LM driver ========
def make_feasible(
    q_init,
    num_rods,
    D,
    delta=0.10,         # active-set buffer (in length units)
    mu=1e-3,            # LM damping
    max_newton=200,     # outer iters
    tol_stop=1e-8,      # stop if phi < tol_stop
    print_every=5
):
    i_idx, j_idx = jnp.triu_indices(num_rods, k=1)
    q = q_init
    history = [q]  # keep whole trajectory (feasibility phase)

    for it in range(max_newton):
        # Active set and residuals
        active_idx, d_all = build_active_set(q, i_idx, j_idx, D, delta)
        r, h = residual_vector(q, active_idx, i_idx, j_idx, D)
        phi = violation_energy(r)
        min_h = (jnp.min(h) if h.size > 0 else jnp.inf)

        # done if all constraints nonnegative (h>=0) or negligible violation
        if (h.size == 0) or (min_h >= 0.0) or (phi < tol_stop):
            if (it % print_every) == 0:
                print(f"[feas] it={it:4d}  phi={float(phi):.3e}  min(h)={float(min_h):.3e}  active={int(active_idx.size)}")
            break

        # Dense Jacobian (OK if active set is small)
        J = jacobian_active(q, active_idx, i_idx, j_idx, D)

        # LM step and backtracking
        dq = lm_step(q, r, J, mu)
        q_new, phi_new, min_h_new, alpha_used = backtrack(q, dq, i_idx, j_idx, D, delta)

        q = q_new
        history.append(q)

        if (it % print_every) == 0:
            print(f"[feas] it={it:4d}  phi={float(phi_new):.3e}  min(h)={float(min_h_new):.3e}  "
                  f"active={int(active_idx.size)}  alpha={float(alpha_used):.2e}")

    return q, history




# assumes you already have: endpoints_from_q(q) and segseg_dist(...)

def h_vals_for_pairs(q, ii, jj, D):
    """Vector h(q) = d_ij(q) - D for the selected pairs (ii, jj). No closures."""
    r1, r2 = endpoints_from_q(q)  # shapes (N,3)
    # vectorized distance for each (i,j)
    dists = vmap(lambda i, j: segseg_dist(r1[i], r2[i], r1[j], r2[j]))(ii, jj)
    return dists - D  # shape (K,)

def residual_vector(q, active_pairs, i_idx, j_idx, D):
    """Residuals r = min(0, h). Works with JIT and vmap."""
    ii = jnp.asarray(i_idx[active_pairs], dtype=jnp.int32)
    jj = jnp.asarray(j_idx[active_pairs], dtype=jnp.int32)
    h = h_vals_for_pairs(q, ii, jj, D)       # (K,)
    r = jnp.minimum(0.0, h)                  # (K,)
    return r, h

def jacobian_active(q, active_pairs, i_idx, j_idx, D):
    """
    Dense Jacobian J = ∂r/∂q of shape (K, q.size).
    We jacobian the vector function h(q) and then mask for r = min(0,h).
    """
    ii = jnp.asarray(i_idx[active_pairs], dtype=jnp.int32)
    jj = jnp.asarray(j_idx[active_pairs], dtype=jnp.int32)

    # J_h has shape (K, *q.shape). We’ll reshape to (K, q.size).
    J_h = jacrev(lambda qq: h_vals_for_pairs(qq, ii, jj, D))(q)  # (K, ...qshape)
    J_h = J_h.reshape(J_h.shape[0], -1)                          # (K, M)

    # r = min(0,h)  → dr/dq = I[h<0] * dh/dq (row-wise masking)
    h = h_vals_for_pairs(q, ii, jj, D)                           # (K,)
    mask = (h < 0.0).astype(q.dtype)[:, None]                    # (K,1)
    J = mask * J_h                                               # (K, M)
    return J

def build_active_set(q, i_idx, j_idx, D, delta):
    d = pair_dists(q, i_idx, j_idx)   # your existing function is fine
    mask = d <= (D + delta)
    return jnp.where(mask)[0], d

# def build_active_set(q, i_idx, j_idx, D, delta):
#     d = pair_dists(q, i_idx, j_idx)   # your existing function is fine
#     mask = d <= (D + delta)
#     return jnp.where(mask)[0], d


# ======== Example usage ========
if __name__ == "__main__":
    NUM_RODS = 300
    ALPHA = 100.0
    D = 1.0 / ALPHA
    CONTAINER = 1.0

    q0 = create_nonintersecting_random_rods_contained_pbc(NUM_RODS, D, CONTAINER)

    q_feas, hist = make_feasible(
        q0,
        num_rods=NUM_RODS,
        D=D,
        delta=0.05*D,       # small buffer around diameter
        mu=1e-3,
        max_newton=200,
        tol_stop=1e-10,
        print_every=5
    )

    # Save full feasibility history
    import numpy as onp
    onp.save("feasibility_history.npy", onp.array(hist))
    print(f"Saved {len(hist)} feasibility iterates → feasibility_history.npy")
