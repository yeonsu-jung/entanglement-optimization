#!/usr/bin/env python3
# sphere_scp_qp.py
# Sequential Convex Programming for "minimal-nudge" sphere separation.
# Inner QP solved by cyclic projections onto halfspaces (dependency-free).

from __future__ import annotations
from typing import Optional, Tuple

import jax
import jax.numpy as jnp


def _build_pairs(N: int):
    """Upper-triangular index pairs (i<j)."""
    return jnp.triu_indices(N, k=1)


def _halfspace_project_delta(
    dx: jnp.ndarray,  # (N,3)
    i: int,
    j: int,
    u_ij: jnp.ndarray,  # (3,) unit chord at current linearization point
    c_ij: float,        # required RHS: u^T (dx_i - dx_j) >= c_ij
) -> jnp.ndarray:
    """
    Projection of current dx onto halfspace {a^T dx >= c}, where
    a has support on blocks i and j with +u and -u. If violated, move minimally.
    Since ||a||^2 = ||u||^2 + ||-u||^2 = 2 (as ||u||=1), the step is simple.
    """
    val = jnp.dot(u_ij, dx[i] - dx[j])  # current left-hand side
    gap = c_ij - val
    # If not violated, no change
    def no_change(_):
        return dx

    def apply(_):
        # minimal correction along 'a' direction: dx <- dx + ((gap) / ||a||^2) * a
        # => add +delta*u to i, and -delta*u to j, with delta = gap / 2
        delta = 0.5 * gap
        dx2 = dx.at[i].add(delta * u_ij)
        dx2 = dx2.at[j].add(-delta * u_ij)
        return dx2

    return jax.lax.cond(gap > 0.0, apply, no_change, operand=None)


def _project_centroid(dx: jnp.ndarray) -> jnp.ndarray:
    """Project onto equality subspace sum_i dx_i = 0 (remove drift)."""
    mean = jnp.mean(dx, axis=0, keepdims=True)
    return dx - mean


def solve_qp_by_halfspace_projections(
    u_list: jnp.ndarray,     # (P,3) unit directions for each constraint
    ij_list: jnp.ndarray,    # (P,2) int indices for each pair (i,j)
    c_list: jnp.ndarray,     # (P,) RHS per constraint
    N: int,
    max_cycles: int = 50,
    enforce_centroid: bool = True,
) -> Tuple[jnp.ndarray, float]:
    """
    Solve inner QP: min 1/2 ||dx||^2 s.t. u_ij^T (dx_i - dx_j) >= c_ij
    via cyclic projections onto halfspaces. Returns dx and max residual.
    """
    dx = jnp.zeros((N, 3))
    P = u_list.shape[0]

    def one_cycle(dx):
        def body_fun(p, dx_cur):
            i, j = ij_list[p, 0], ij_list[p, 1]
            dx_next = _halfspace_project_delta(dx_cur, int(i), int(j), u_list[p], float(c_list[p]))
            return dx_next

        # loop over all constraints
        dx2 = jax.lax.fori_loop(0, P, body_fun, dx)
        if enforce_centroid:
            dx2 = _project_centroid(dx2)
        return dx2

    # Run multiple cycles
    for _ in range(max_cycles):
        dx = one_cycle(dx)

    # Compute max linearized residual after inner solve
    # residual r = c - u^T (dx_i - dx_j); max over positive r (violations)
    vals = jnp.einsum('pk,pk->p', u_list, dx[ij_list[:, 0]] - dx[ij_list[:, 1]])
    residuals = jnp.maximum(c_list - vals, 0.0)
    max_res = float(jnp.max(residuals))
    return dx, max_res


def scp_project_spheres(
    x0: jnp.ndarray,            # (N,3) initial centers
    D: float,                   # target minimum distance
    max_outer: int = 20,
    inner_cycles: int = 60,     # halfspace-projection cycles per QP
    tol_geom: float = 1e-9,     # tolerance on *true* geometric violation
    tol_lin: float = 1e-10,     # tolerance on linearized residual
    include_margin: float = 0.0,# include also near-active pairs with D - d > -margin
    step_scale: float = 1.0,    # damping on the SCP update (<=1)
    keep_centroid: bool = True,
) -> Tuple[jnp.ndarray, int, float, float]:
    """
    Sequential convex programming loop:
      - linearize constraints at x^k
      - solve QP: min 1/2||dx||^2 s.t. u_ij^T (dx_i - dx_j) >= c_ij
      - update x^{k+1} = x^k + step_scale * dx

    Returns:
        x          : feasible (or near-feasible) positions
        iters_used : outer iterations performed
        max_viol   : max *true* violation (max(D - ||x_i - x_j||, 0))
        rms_disp   : RMS displacement ||x - x0||
    """
    N = x0.shape[0]
    x = x0.copy()
    I, J = _build_pairs(N)
    iters_used = 0

    for k in range(max_outer):
        # Current distances and directions
        rij = x[I] - x[J]                 # (P,3)
        dij = jnp.linalg.norm(rij, axis=1)  # (P,)
        # Unit chords; handle exact coincidences with a fallback axis
        small = dij < 1e-12
        u = jnp.where(
            (~small)[..., None],
            rij / (dij[:, None] + 1e-12),
            jnp.array([1.0, 0.0, 0.0])[None, :],
        )

        # Select violating or near-active pairs for this QP
        phi = D - dij  # >0 means violation right now
        active = phi > -include_margin
        if not bool(jnp.any(active)):
            # already strictly feasible with margin
            break

        u_act = u[active]
        ij_act = jnp.stack([I[active], J[active]], axis=1)
        c_act = phi[active]  # RHS in linearized constraints

        # Solve inner QP (projection onto intersection of halfspaces)
        dx, max_res = solve_qp_by_halfspace_projections(
            u_act, ij_act, c_act, N,
            max_cycles=inner_cycles,
            enforce_centroid=keep_centroid
        )

        # Take damped step (trust region style)
        x = x + step_scale * dx

        # Check true (nonlinear) violation
        rij_new = x[I] - x[J]
        dij_new = jnp.linalg.norm(rij_new, axis=1)
        max_true_viol = float(jnp.maximum(D - dij_new, 0.0).max())

        iters_used = k + 1
        # Stopping if both linearized and true violations are small
        if max_true_viol <= tol_geom and max_res <= tol_lin:
            break

    # Final stats
    rij = x[I] - x[J]
    dij = jnp.linalg.norm(rij, axis=1)
    max_viol = float(jnp.maximum(D - dij, 0.0).max())
    rms_disp = float(jnp.sqrt(jnp.mean(jnp.sum((x - x0) ** 2, axis=1))))
    return x, iters_used, max_viol, rms_disp


def verify_min_dist(x: jnp.ndarray, D: float) -> Tuple[float, float]:
    N = x.shape[0]
    I, J = jnp.triu_indices(N, k=1)
    dij = jnp.linalg.norm(x[I] - x[J], axis=1)
    min_d = float(jnp.min(dij))
    max_viol = float(jnp.maximum(D - dij, 0.0).max())
    return min_d, max_viol


def demo():
    key = jax.random.PRNGKey(0)
    N = 250
    D = 0.06

    key, sub = jax.random.split(key)
    x0 = jax.random.uniform(sub, (N, 3), minval=-0.5, maxval=0.5)

    x, iters, max_viol, rms_disp = scp_project_spheres(
        x0, D,
        max_outer=25,
        inner_cycles=80,
        tol_geom=1e-9,
        tol_lin=1e-10,
        include_margin=0.0,
        step_scale=0.9,       # conservative outer step
        keep_centroid=True,
    )

    mind, chk = verify_min_dist(x, D)

    print("---- SCP sphere QP (linearized) ----")
    print(f"N                : {N}")
    print(f"D (target)       : {D:.6f}")
    print(f"outer iters      : {iters}")
    print(f"min pair dist    : {mind:.9f}")
    print(f"max violation    : {chk:.3e}")
    print(f"RMS displacement : {rms_disp:.6e}")


if __name__ == "__main__":
    demo()
