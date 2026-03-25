"""Minimal physics library for rod entanglement and relaxation.

Rod state q: flattened (N*5,) or shaped (N,5)
  q[i] = [x, y, z, theta, phi]
  where (theta, phi) are spherical angles giving the rod direction:
    u = [sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta)]
  Rod occupies [start, start + u], length = 1.

Endpoints x: shaped (N,6) = [x0, y0, z0, x1, y1, z1]
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import jit, lax, vmap
from functools import partial
import numpy as np

# ── Geometry primitives ────────────────────────────────────────────────────

@jit
def safe_norm(x, axis=None, keepdims=False, eps=1e-14):
    return jnp.sqrt(jnp.sum(jnp.square(x), axis=axis, keepdims=keepdims) + eps)


@jax.custom_jvp
def safe_arcsin(x):
    return jnp.arcsin(jnp.clip(x, -1.0, 1.0))

@safe_arcsin.defjvp
def _safe_arcsin_jvp(primals, tangents):
    (x,) = primals
    (dx,) = tangents
    safe_x_sq = jnp.clip(jnp.square(x), 0.0, 1.0 - 1e-14)
    return safe_arcsin(x), dx / jnp.sqrt(1.0 - safe_x_sq)


@jit
def safe_normalize(x):
    return x / safe_norm(x)


@jit
def _fixbound(x):
    return jnp.clip(x, 0.0, 1.0)


@jit
def _segment_segment_parameters(p1s, p1e, p2s, p2e):
    """Closest-point parameters (t, u) on two line segments."""
    d1  = p1e - p1s
    d2  = p2e - p2s
    d12 = p2s - p1s
    D1 = jnp.dot(d1, d1);  D2 = jnp.dot(d2, d2)
    S1 = jnp.dot(d1, d12); S2 = jnp.dot(d2, d12)
    R  = jnp.dot(d1, d2)
    den = D1 * D2 - R ** 2
    eps = 1e-10

    den_s = jnp.where(jnp.abs(den) > eps, den, 1.0)
    t = _fixbound(jnp.where(jnp.abs(den) > eps,
                             (S1 * D2 - S2 * R) / den_s, 0.0))
    D2_s = jnp.where(jnp.abs(D2) > eps, D2, 1.0)
    u = jnp.where(jnp.abs(D2) > eps, (t * R - S2) / D2_s, 0.0)
    uf = _fixbound(u)

    def _recompute_t(nu):
        D1_s = jnp.where(jnp.abs(D1) > eps, D1, 1.0)
        return _fixbound(jnp.where(jnp.abs(D1) > eps,
                                    (nu * R + S1) / D1_s, 0.0))

    t, u = lax.cond(uf != u,
                    lambda _: (_recompute_t(uf), uf),
                    lambda _: (t, u),
                    None)

    def _degenerate():
        D2_sd = jnp.where(jnp.abs(D2) > eps, D2, 1.0)
        return 0.0, _fixbound(jnp.where(jnp.abs(D2) > eps, -S2 / D2_sd, 0.0))

    t, u = lax.cond(jnp.abs(den) <= eps,
                    lambda _: _degenerate(),
                    lambda _: (t, u),
                    None)
    return t, u


@jit
def dist_lin_seg(p1s, p1e, p2s, p2e):
    t, u = _segment_segment_parameters(p1s, p1e, p2s, p2e)
    return safe_norm(  (p1e - p1s) * t
                     - (p2e - p2s) * u
                     - (p2s - p1s))


@jit
def aabb_overlap_capsule(p1s, p1e, p2s, p2e, threshold):
    return jnp.all(
        (jnp.minimum(p1s, p1e) <= jnp.maximum(p2s, p2e) + threshold) &
        (jnp.minimum(p2s, p2e) <= jnp.maximum(p1s, p1e) + threshold)
    )


# ── Rod coordinate transform ───────────────────────────────────────────────

def sph2cart(theta, phi):
    """(theta, phi) -> unit direction vector (x,y,z)."""
    s = jnp.sin(theta)
    return jnp.stack([s * jnp.cos(phi), s * jnp.sin(phi), jnp.cos(theta)],
                     axis=-1)


def q_to_x(q):
    """q (N*5,) or (N,5) -> endpoints (N,6) = [start | end]."""
    q = jnp.reshape(q, (-1, 5))
    starts = q[:, :3]
    ends   = starts + sph2cart(q[:, 3], q[:, 4])
    return jnp.concatenate([starts, ends], axis=1)


# ── Linking number ─────────────────────────────────────────────────────────

def _lk_pair(rod_i, rod_j):
    """Signed linking number for a single pair (vectorisable)."""
    p_i = rod_i[:3];  p_j = rod_j[:3]
    u_i = sph2cart(rod_i[3], rod_i[4])
    u_j = sph2cart(rod_j[3], rod_j[4])
    p_ii = p_i + u_i;  p_jj = p_j + u_j

    r_ij   = p_i  - p_j
    r_ijj  = p_i  - p_jj
    r_iij  = p_ii - p_j
    r_iijj = p_ii - p_jj

    n1 = safe_normalize(jnp.cross(r_ij,   r_ijj))
    n2 = safe_normalize(jnp.cross(r_ijj,  r_iijj))
    n3 = safe_normalize(jnp.cross(r_iijj, r_iij))
    n4 = safe_normalize(jnp.cross(r_iij,  r_ij))

    s = (safe_arcsin(jnp.dot(n1, n2))
       + safe_arcsin(jnp.dot(n2, n3))
       + safe_arcsin(jnp.dot(n3, n4))
       + safe_arcsin(jnp.dot(n4, n1)))
    # smooth abs to avoid gradient issues at zero
    return -1.0 / (4.0 * jnp.pi) * jnp.sqrt(s * s + 1e-28)


# ── Pairwise indices ───────────────────────────────────────────────────────

def create_pairs(q_mat):
    """Upper-triangle pairs: (N,5) -> (N*(N-1)//2, 10)."""
    N = q_mat.shape[0]
    i, j = jnp.triu_indices(N, k=1)
    return jnp.concatenate([q_mat[i], q_mat[j]], axis=1)


@jit
def _pair_distance(q_pair):
    ri = q_pair[:3];  ui = sph2cart(q_pair[3], q_pair[4])
    rj = q_pair[5:8]; uj = sph2cart(q_pair[8], q_pair[9])
    return dist_lin_seg(ri, ri + ui, rj, rj + uj)


@jit
def all_pairwise_distances(q_pairs):
    return vmap(_pair_distance)(q_pairs)


@jit
def min_pairwise_distance(q):
    q = jnp.reshape(q, (-1, 5))
    pairs = create_pairs(q)
    return jnp.min(all_pairwise_distances(pairs))


def make_min_dist_fn(threshold: float):
    """Return a JIT'd min pairwise distance function.

    The threshold parameter is kept for API compatibility but is no longer
    used internally.  On GPU, lax.cond inside vmap materialises as jnp.where
    (both branches always evaluated), so AABB pruning adds overhead without
    actually skipping work.  Plain full-distance is faster and exact.
    """
    del threshold  # unused on GPU; full distance is cheaper than AABB+where

    return min_pairwise_distance


# ── Potentials ─────────────────────────────────────────────────────────────

def make_entangle_potential(num_rods: int):
    """Return a JIT'd potential f(q) = sum of |LK_ij| over all pairs.

    Uses double-vmap (N×N matrix, mask upper triangle) — faster than
    explicit pair enumeration for large N on GPU.
    """
    mask = jnp.triu(jnp.ones((num_rods, num_rods), dtype=jnp.float64), k=1)

    @jit
    def _f(q_flat):
        q = jnp.reshape(q_flat, (num_rods, 5))
        pot_matrix = vmap(lambda ri: vmap(lambda rj: _lk_pair(ri, rj))(q))(q)
        return jnp.sum(pot_matrix * mask)

    return _f


@jit
def _repulsion_pair(q_pair, col_rad, amp):
    """AABB-pruned harmonic repulsion for a single rod pair."""
    ri = q_pair[:3];  ui = sph2cart(q_pair[3], q_pair[4])
    rj = q_pair[5:8]; uj = sph2cart(q_pair[8], q_pair[9])
    p1s = ri;  p1e = ri + ui
    p2s = rj;  p2e = rj + uj
    threshold = col_rad * 2.0

    def _full():
        d = dist_lin_seg(p1s, p1e, p2s, p2e)
        return lax.cond(d < threshold,
                        lambda _: amp * (d - threshold) ** 2,
                        lambda _: 0.0, None)

    return lax.cond(aabb_overlap_capsule(p1s, p1e, p2s, p2e, threshold),
                    lambda _: _full(), lambda _: 0.0, None)


@jit
def repulsion_potential(q_flat, col_rad, amp):
    """Harmonic repulsion potential with traced col_rad and amp.

    Compiled once per q shape (once per N) — col_rad is a dynamic JAX scalar
    so the same XLA program covers all AR values for a given N.
    """
    q = jnp.reshape(q_flat, (-1, 5))
    pairs = create_pairs(q)
    return jnp.sum(vmap(lambda qp: _repulsion_pair(qp, col_rad, amp))(pairs))


# Gradient w.r.t. q (first argument) — also compiled once per N.
repulsion_gradient = jit(grad(repulsion_potential))
