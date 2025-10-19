# jax_nudger_rods.py
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from jax import jit, lax, vmap, random


# ----------------------------
# Geometry helpers (q <-> x)
# q[i] = [x0,y0,z0, theta, phi], |offset| = 1
# x[i] = [x0,y0,z0, x1,y1,z1]  (two endpoints)
# ----------------------------
def sph2cart(theta: jnp.ndarray, phi: jnp.ndarray, r: float = 1.0) -> jnp.ndarray:
    x = r * jnp.sin(theta) * jnp.cos(phi)
    y = r * jnp.sin(theta) * jnp.sin(phi)
    z = r * jnp.cos(theta)
    return jnp.stack([x, y, z], axis=-1)

def cart2sph(x: jnp.ndarray, y: jnp.ndarray, z: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    r = jnp.sqrt(x*x + y*y + z*z)
    # guard division by zero (clip)
    theta = jnp.arccos(jnp.clip(jnp.where(r > 0, z / r, 1.0), -1.0, 1.0))
    phi = jnp.arctan2(y, x)
    return r, theta, phi

def q_to_x(q: jnp.ndarray) -> jnp.ndarray:
    """q: (N*5,) or (N,5) -> x: (N,6) endpoints"""
    q = q.reshape((-1, 5))
    x0 = q[:, :3]
    off = sph2cart(q[:, 3], q[:, 4])  # unit length by construction
    x1 = x0 + off
    return jnp.concatenate([x0, x1], axis=1)

def x_to_q(x: jnp.ndarray) -> jnp.ndarray:
    """x: (N,6) -> q: (N,5)"""
    x = x.reshape((-1, 6))
    x0 = x[:, :3]
    x1 = x[:, 3:]
    off = x1 - x0
    r, theta, phi = cart2sph(off[:, 0], off[:, 1], off[:, 2])
    # r should remain ~1 because we translate both endpoints equally
    q = jnp.concatenate([x0, jnp.stack([theta, phi], axis=1)], axis=1)
    return q


# ---------------------------------------------
# Closest points between two line segments (3D)
# Return the vector from rod j closest point to rod i closest point (r_i - r_j)
# and (optionally) the distance = ||that vector||
# ---------------------------------------------
def _fix01(a: jnp.ndarray) -> jnp.ndarray:
    return jnp.clip(a, 0.0, 1.0)

@jit
def closest_vec_seg_seg(p1s: jnp.ndarray, p1e: jnp.ndarray,
                        p2s: jnp.ndarray, p2e: jnp.ndarray,
                        eps: float = 1e-12) -> jnp.ndarray:
    """
    Returns the minimal vector (r_i - r_j) between two segments.
    If segments intersect, this returns ~0-vector.
    """
    d1 = p1e - p1s
    d2 = p2e - p2s
    r  = p1s - p2s

    a = jnp.dot(d1, d1)            # ||d1||^2
    e = jnp.dot(d2, d2)            # ||d2||^2
    f = jnp.dot(d2, r)
    c = jnp.dot(d1, r)
    b = jnp.dot(d1, d2)
    denom = a * e - b * b

    # Parallel / nearly parallel case handled by branch
    def _non_parallel(_):
        s = _fix01((b * f - c * e) / (denom + eps))
        t = _fix01((a * f - b * c) / (denom + eps))
        return s, t

    def _parallel(_):
        # project r onto d1 and d2 independently
        s = _fix01(-c / (a + eps))     # along d1
        t = _fix01((b * s + f) / (e + eps))
        # clamp t, then recompute s to respect t clamp
        t_cl = _fix01(t)
        s_cl = _fix01((b * t_cl - c) / (a + eps))
        return s_cl, t_cl

    s, t = lax.cond(jnp.abs(denom) > 1e-14, _non_parallel, _parallel, operand=None)

    r1 = p1s + s * d1
    r2 = p2s + t * d2
    return r1 - r2  # vector from j to i at closest points

# batched over pairs
def _pairwise_indices(N: int):
    return jnp.triu_indices(N, k=1)

def _pair_vectors_from_x(x: jnp.ndarray, ii: jnp.ndarray, jj: jnp.ndarray) -> jnp.ndarray:
    """x: (N,6), ii,jj: (M,) -> vecs (M,3) minimal vectors r_i - r_j."""
    r1s = x[ii, 0:3]
    r1e = x[ii, 3:6]
    r2s = x[jj, 0:3]
    r2e = x[jj, 3:6]
    return vmap(closest_vec_seg_seg)(r1s, r1e, r2s, r2e)


# ---------------------------------------------
# Jacobi nudge step for rods (rigid translation)
# ---------------------------------------------
@jit
def nudge_step_rods(q: jnp.ndarray,
                    D: float,
                    step: float = 1.0,
                    underrelax: float = 0.9,
                    eps: float = 1e-12) -> Tuple[jnp.ndarray, float]:
    """
    One Jacobi-style nudge step:
      - Build all pairs (i<j)
      - Compute closest vectors v_ij (from j to i)
      - penetration = relu(D - ||v_ij||)
      - move rod i by +0.5 * pen * u_ij; rod j by -0.5 * pen * u_ij (rigid translation)
    q: (N*5,) or (N,5)
    Returns: q_new, max_penetration
    """
    x = q_to_x(q)                         # (N,6)
    N = x.shape[0]
    ii, jj = _pairwise_indices(N)

    vec = _pair_vectors_from_x(x, ii, jj) # (M,3)
    dij = jnp.linalg.norm(vec, axis=1)    # (M,)
    pen = jnp.maximum(0.0, D - dij)       # (M,)
    u   = vec / (dij[:, None] + eps)      # (M,3)

    # per-pair translations for each rod's center (rigid translate both endpoints)
    move_i = +0.5 * pen[:, None] * u
    move_j = -0.5 * pen[:, None] * u

    # accumulate per-rod translation (3,)
    dC = jnp.zeros((N, 3), dtype=x.dtype)
    dC = dC.at[ii].add(move_i)
    dC = dC.at[jj].add(move_j)

    # expand translation to both endpoints
    dX = jnp.concatenate([dC, dC], axis=1)   # (N,6)
    x_new = x + (step * underrelax) * dX

    q_new = x_to_q(x_new)
    max_pen = jnp.max(pen) if pen.size > 0 else 0.0
    return q_new, max_pen

@jit
def nudge_step_rods_soft(q: jnp.ndarray,
                        D: float,
                        step: float = 1.0,
                        underrelax: float = 0.9,
                        eps: float = 1e-12) -> Tuple[jnp.ndarray, float]:
    """
    One Jacobi-style nudge step:
      - Build all pairs (i<j)
      - Compute closest vectors v_ij (from j to i)
      - penetration = relu(D - ||v_ij||)
      - move rod i by +0.5 * pen * u_ij; rod j by -0.5 * pen * u_ij (rigid translation)
    q: (N*5,) or (N,5)
    Returns: q_new, max_penetration
    """
    x = q_to_x(q)                         # (N,6)
    N = x.shape[0]
    ii, jj = _pairwise_indices(N)

    vec = _pair_vectors_from_x(x, ii, jj) # (M,3)
    dij = jnp.linalg.norm(vec, axis=1)    # (M,)
    pen = jnp.maximum(0.0, D - dij)       # (M,)
    u   = vec / (dij[:, None] + eps)      # (M,3)

    # per-pair translations for each rod's center (rigid translate both endpoints)
    move_i = +0.5 * pen[:, None] * u
    move_j = -0.5 * pen[:, None] * u

    # accumulate per-rod translation (3,)
    dC = jnp.zeros((N, 3), dtype=x.dtype)
    dC = dC.at[ii].add(move_i)
    dC = dC.at[jj].add(move_j)

    # expand translation to both endpoints
    dX = jnp.concatenate([dC, dC], axis=1)   # (N,6)
    x_new = x + (step * underrelax) * dX

    q_new = x_to_q(x_new)
    max_pen = jnp.max(pen) if pen.size > 0 else 0.0
    return q_new, max_pen


def nudge_until_converged(
    q0: jnp.ndarray,
    D: float,
    max_iters: int = 400,
    tol: float = 1e-8,
    step: float = 1.0,
    underrelax: float = 0.9,
    jitter: float = 1e-12,
    key: Optional[jax.random.PRNGKey] = None,
):
    """
    Outer loop (Python). The inner step is JITed.
    """
    q = q0
    if key is not None and jitter > 0:
        # add tiny translation jitter to x to break coincidences
        x = q_to_x(q)
        dx = jitter * random.normal(key, (x.shape[0], 3))
        x = jnp.concatenate([x[:, :3] + dx, x[:, 3:] + dx], axis=1)
        q = x_to_q(x)

    hist = []
    for it in range(max_iters):
        q, max_pen = nudge_step_rods(q, D, step=step, underrelax=underrelax)
        hist.append(float(max_pen))
        if max_pen <= tol:
            break
    return q, it + 1, jnp.array(hist)

def nudge_until_converged_soft(
    q0: jnp.ndarray,
    D: float,
    max_iters: int = 400,
    tol: float = 1e-8,
    step: float = 1.0,
    underrelax: float = 0.9,
    jitter: float = 1e-12,
    key: Optional[jax.random.PRNGKey] = None,
):
    """
    Outer loop (Python). The inner step is JITed.
    """
    q = q0
    if key is not None and jitter > 0:
        # add tiny translation jitter to x to break coincidences
        x = q_to_x(q)
        dx = jitter * random.normal(key, (x.shape[0], 3))
        x = jnp.concatenate([x[:, :3] + dx, x[:, 3:] + dx], axis=1)
        q = x_to_q(x)

    hist = []
    for it in range(max_iters):
        q, max_pen = nudge_step_rods(q, D, step=step, underrelax=underrelax)
        hist.append(float(max_pen))
        if max_pen <= tol:
            break
    return q, it + 1, jnp.array(hist)


# ----------------------------
# Demo: random rods
# ----------------------------
def create_random_rods(num_rods: int, key: jax.random.PRNGKey) -> jnp.ndarray:
    """
    Sample q with start points in [-0.5,0.5]^3 and random unit offsets.
    Center the configuration to mean ~0.
    """
    system_size = 0.5

    k1, k2 = random.split(key)
    x0 = random.uniform(k1, (num_rods, 3), minval=-system_size/2, maxval=system_size/2)
    # uniform directions on S^2 via normalizing Gaussians
    dirs = random.normal(k2, (num_rods, 3))
    dirs = dirs / (jnp.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12)
    # convert to angles
    r, theta, phi = cart2sph(dirs[:, 0], dirs[:, 1], dirs[:, 2])
    q = jnp.concatenate([x0, jnp.stack([theta, phi], axis=1)], axis=1)

    # center so mean of x0 is 0
    x = q_to_x(q)
    center = jnp.mean(x[:, :3], axis=0)
    q = q.at[:, :3].add(-center)
    return q.reshape(-1)


if __name__ == "__main__":
    key = random.PRNGKey(0)
    N   = 10
    D   = 0.01   # target minimum separation between segments (in 3D)

    # q0 = create_random_rods(N, key)

    from protocols import create_intersecting_rods
    q0 = create_intersecting_rods(N)

    x = q_to_x(q0)
    jitter = 1e-3
    dx = jitter * random.normal(key, (x.shape[0], 3))
    q0 = x_to_q(jnp.concatenate([x[:, :3] + dx, x[:, 3:] + dx], axis=1)).reshape(-1)


    q, iters, hist = nudge_until_converged(
        q0, D,
        max_iters=10000,
        tol=1e-6,
        step=1.0,
        underrelax=0.92,
        jitter=1e-12,
        key=key,
    )

    print(f"iters={iters}, final max penetration={hist[-1]:.3e}")

    # Optional quick check of min distance after nudge
    x = q_to_x(q).reshape(-1, 2, 3)
    r1 = x[:, 0, :]
    r2 = x[:, 1, :]
    ii, jj = _pairwise_indices(N)

    def _vec(i, j):
        return closest_vec_seg_seg(r1[i], r2[i], r1[j], r2[j])
    
    vecs = vmap(lambda a, b: _vec(a, b))(ii, jj)
    dists = jnp.linalg.norm(vecs, axis=1)
    print(f"min pair distance = {jnp.min(dists):.6f}, expected ≥ {D}")

    # visualizations
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = q_to_x(q).reshape(-1, 2, 3)
    for i in range(N):
        ax.plot(x[i, :, 0], x[i, :, 1], x[i, :, 2], '-o')
    ax.set_title(f"N={N}, iters={iters}, final max pen={hist[-1]:.3e}")
    plt.axis('equal')
    plt.show()

