"""
Benchmark: all-pairs segment-segment (finite lines) distances.

Two methods:
  1. vmap       – scalar kernel using lax.cond for data-dependent branching
  2. vectorized – broadcasting + jnp.where; covers all three sub-problems:
       (a) Interior × Interior  →  unclamped line-line closest point
       (b) One endpoint clamped →  endpoint projected onto opposite segment
       (c) Both clamped         →  endpoint-endpoint

The key difference:
  lax.cond  needs a *scalar* bool  → fine per-pair, vmapped over the N² pairs
  jnp.where accepts array bools    → evaluates all branches, masks with where
"""
import os
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import sys
import time
import jax
jax.config.update("jax_platforms", "cpu")
jax.config.update("jax_enable_x64", True)

from jax import vmap, jit, lax
from jax import numpy as jnp
import numpy as np


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _safe_norm(v, eps=1e-14):
    """Euclidean norm with small eps to avoid sqrt(0) gradient blow-up."""
    return jnp.sqrt(jnp.sum(v ** 2) + eps)


# ---------------------------------------------------------------------------
# vmap kernel  (lax.cond – branching OK because condition is scalar per call)
# ---------------------------------------------------------------------------

def seg_seg_dist(s1, e1, s2, e2):
    """
    Shortest distance between two finite segments s1→e1 and s2→e2.

    Algorithm (Eberly / Shoemake):
      1. Solve unclamped closest-point parameters (t, u) on the two infinite lines.
      2. Clamp t to [0,1]; recompute u; clamp u to [0,1].
      3. If u was clamped, re-solve t from clamped u and clamp again.
      4. Handle parallel segments (den≈0) separately.
    """
    d1  = e1 - s1
    d2  = e2 - s2
    d12 = s2 - s1

    D1 = jnp.dot(d1, d1)
    D2 = jnp.dot(d2, d2)
    R  = jnp.dot(d1, d2)
    S1 = jnp.dot(d1, d12)
    S2 = jnp.dot(d2, d12)

    den = D1 * D2 - R ** 2
    eps = 1e-10

    # ---- general case ----
    den_safe = jnp.where(jnp.abs(den) > eps, den, 1.0)
    t = jnp.clip(
        jnp.where(jnp.abs(den) > eps, (S1 * D2 - S2 * R) / den_safe, 0.0),
        0.0, 1.0,
    )

    D2_safe = jnp.where(jnp.abs(D2) > eps, D2, 1.0)
    u_raw = (t * R - S2) / D2_safe
    u = jnp.clip(u_raw, 0.0, 1.0)

    # If u was clamped → re-solve t from the clamped u  [endpoint→segment sub-case]
    D1_safe = jnp.where(jnp.abs(D1) > eps, D1, 1.0)
    t_recomp = jnp.clip((u * R + S1) / D1_safe, 0.0, 1.0)
    t, u = lax.cond(
        u_raw != u,
        lambda _: (t_recomp, u),
        lambda _: (t, u),
        None,
    )

    # ---- parallel / degenerate case (den≈0) ----
    u_par = jnp.clip(
        jnp.where(jnp.abs(D2) > eps, -S2 / D2_safe, 0.0),
        0.0, 1.0,
    )
    t, u = lax.cond(
        jnp.abs(den) <= eps,
        lambda _: (0.0, u_par),
        lambda _: (t, u),
        None,
    )

    r1 = s1 + t * d1
    r2 = s2 + u * d2
    return _safe_norm(r1 - r2)


def _seg_seg_dist_x(x1, x2):
    """Thin wrapper: takes two 6-vectors x=(start‖end) and returns distance."""
    return seg_seg_dist(x1[:3], x1[3:], x2[:3], x2[3:])


@jit
def all_pairs_vmap(xs):
    """N×N distances via nested vmap (lax.cond inside the scalar kernel)."""
    return vmap(vmap(_seg_seg_dist_x, (None, 0)), (0, None))(xs, xs)


# ---------------------------------------------------------------------------
# Vectorized kernel  (jnp.where – no lax.cond, pure broadcasting)
#
# The three sub-problems are handled implicitly:
#   (a) interior×interior  →  t,u unclamped from line-line solve
#   (b) endpoint→segment   →  u clamped, t re-solved from clamped u
#   (c) endpoint→endpoint  →  both clamped; jnp.where selects automatically
# ---------------------------------------------------------------------------

@jit
def all_pairs_vectorized(xs):
    """N×N distances via broadcasting + jnp.where (no lax.cond, no Python loops)."""
    ps = xs[:, :3]               # (N, 3) start points
    d  = xs[:, 3:] - xs[:, :3]  # (N, 3) direction vectors

    eps = 1e-10

    # --- scalar products needed for the parametric solve ---
    D = jnp.sum(d * d, axis=-1)                           # (N,)   ||d[i]||²

    R   = d @ d.T                                          # (N, N) d[i]·d[j]
    d12 = ps[None, :, :] - ps[:, None, :]                 # (N, N, 3)  ps[j]-ps[i]

    # S1[i,j] = d[i] · (ps[j] - ps[i])
    S1 = jnp.einsum('id,ijd->ij', d, d12)                 # (N, N)
    # S2[i,j] = d[j] · (ps[j] - ps[i])
    S2 = jnp.einsum('jd,ijd->ij', d, d12)                 # (N, N)

    den = D[:, None] * D[None, :] - R ** 2                # (N, N)

    # --- (a/b) general branch: line-line unclamped t, then iterative clamp ---
    den_safe = jnp.where(jnp.abs(den) > eps, den, 1.0)
    t_raw    = (S1 * D[None, :] - S2 * R) / den_safe
    t        = jnp.clip(t_raw, 0.0, 1.0)

    Dj_safe  = jnp.where(jnp.abs(D) > eps, D, 1.0)       # (N,)
    u_raw    = (t * R - S2) / Dj_safe[None, :]            # (N, N)
    u        = jnp.clip(u_raw, 0.0, 1.0)

    # If u was clamped, re-solve t from clamped u  [endpoint→segment sub-case]
    Di_safe  = jnp.where(jnp.abs(D) > eps, D, 1.0)       # (N,)
    t_recomp = jnp.clip((u * R + S1) / Di_safe[:, None], 0.0, 1.0)
    t_final  = jnp.where(u_raw != u, t_recomp, t)

    # --- parallel branch: t=0, project ps[i] onto seg j ---
    u_par = jnp.clip(-S2 / Dj_safe[None, :], 0.0, 1.0)

    # --- select ---
    is_par = jnp.abs(den) <= eps
    t_out  = jnp.where(is_par, 0.0, t_final)              # (N, N)
    u_out  = jnp.where(is_par, u_par, u)                  # (N, N)

    # --- closest points and distance ---
    # r1[i,j] = ps[i] + t_out[i,j] * d[i]
    # r2[i,j] = ps[j] + u_out[i,j] * d[j]
    r1   = ps[:, None, :] + t_out[:, :, None] * d[:, None, :]   # (N, N, 3)
    r2   = ps[None, :, :] + u_out[:, :, None] * d[None, :, :]   # (N, N, 3)
    diff = r1 - r2                                               # (N, N, 3)

    return jnp.sqrt(jnp.sum(diff ** 2, axis=-1) + 1e-14)        # (N, N)


# ---------------------------------------------------------------------------
# Data generation helpers
# ---------------------------------------------------------------------------

def sph2cart(polar, az):
    return jnp.array([
        jnp.sin(polar) * jnp.cos(az),
        jnp.sin(polar) * jnp.sin(az),
        jnp.cos(polar),
    ])


def make_segments(N, key):
    """Return (N, 6) array of random unit-length segments."""
    key, k1, k2, k3 = jax.random.split(key, 4)
    centers = jax.random.normal(k1, (N, 3))
    thetas  = jax.random.uniform(k2, (N,), minval=0.0, maxval=jnp.pi)
    phis    = jax.random.uniform(k3, (N,), minval=0.0, maxval=2 * jnp.pi)
    dirs    = vmap(sph2cart)(thetas, phis)       # (N, 3)
    return jnp.concatenate([centers - 0.5 * dirs, centers + 0.5 * dirs], axis=1)


def timed(fn, *args, repeats=5):
    fn(*args).block_until_ready()   # JIT compile + warm-up
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(*args).block_until_ready()
        times.append(time.perf_counter() - t0)
    return fn(*args).block_until_ready(), min(times)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    file_path = sys.argv[1]

    xs = np.loadtxt(file_path)
    xs = jnp.array(xs)
    N = xs.shape[0]

    D_vmap, t_vmap = timed(all_pairs_vmap, xs)
    print(f"  nested vmap  (lax.cond) : {t_vmap * 1e3:.2f} ms")

    # D_vec, t_vec = timed(all_pairs_vectorized, xs)
    # print(f"  vectorized  (jnp.where) : {t_vec * 1e3:.2f} ms")

    i, j = jnp.triu_indices(N, k=1)
    actual_min_dist = jnp.min(D_vmap[i, j])
    print(f"True Min Pair Distance: {actual_min_dist:.10f}")


