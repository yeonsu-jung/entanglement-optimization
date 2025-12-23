#!/usr/bin/env python3
# sphere_nudge_01.py
# Minimal-nudge de-overlap for N spheres using pairwise projections (PBD-style).

from __future__ import annotations
from typing import Optional, Tuple

import jax
import jax.numpy as jnp


def project_spheres(
    x0: jnp.ndarray,                 # (N,3) initial centers
    D: float,                        # target minimum distance
    masses: Optional[jnp.ndarray] = None,  # (N,), optional per-sphere mass
    max_iters: int = 200,
    tol: float = 1e-9,
    relax: float = 1.0,              # 1.0 = full pair fix; <1.0 = damped
    keep_centroid: bool = True,
    key: Optional[jax.Array] = None, # PRNG key if you want jitter when d~0
) -> Tuple[jnp.ndarray, int, float, float]:
    """
    Iteratively nudges spheres so all pairwise distances >= D (within tol),
    minimizing total motion by splitting corrections along the chord.

    Returns:
        x          : (N,3) new centers
        iters_used : number of sweeps performed
        max_viol   : final max violation (max(D - d_ij, 0))
        rms_disp   : RMS displacement ||x - x0||
    """
    N = x0.shape[0]
    x = x0.copy()

    # Inverse masses for weighted splits (heavier moves less).
    if masses is None:
        inv_m = jnp.ones((N,))
    else:
        inv_m = 1.0 / jnp.maximum(masses, 1e-12)

    def pair_weights(i: jnp.ndarray, j: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        wi = inv_m[i]
        wj = inv_m[j]
        s = wi + wj
        # Split in proportion to inverse mass (equal if masses equal)
        return wi / s, wj / s

    # All upper-triangular pairs (i<j)
    I, J = jnp.triu_indices(N, k=1)

    small_eps = 1e-12
    rng = key

    iters_used = 0
    max_viol = jnp.inf

    for k in range(max_iters):
        # Pairwise vectors/distances for the current x
        rij = x[I] - x[J]                      # (P,3) where P = N*(N-1)/2
        dij = jnp.linalg.norm(rij, axis=1)     # (P,)

        # Violations (distance too small)
        gap = D - dij
        viol = gap > tol

        if not bool(jnp.any(viol)):
            max_viol = jnp.maximum(gap, 0.0).max()
            iters_used = k
            break

        # Unit chord directions; handle dij~0 by falling back to a fixed axis
        # (optionally, add tiny random jitter if desired)
        u = jnp.where(
            (dij > small_eps)[..., None],
            rij / (dij[:, None] + 1e-12),
            jnp.array([1.0, 0.0, 0.0])[None, :],
        )

        # How much each pair still needs (only for violating pairs)
        corr_mag = relax * jnp.where(viol, gap, 0.0) * 0.5  # split half/half before mass weights

        # Mass-weighted split
        wi, wj = jax.vmap(pair_weights)(I, J)  # (P,)

        # Per-pair displacement vectors
        dxi = (wi * corr_mag)[:, None] * u
        dxj = (wj * corr_mag)[:, None] * u

        # Scatter-add corrections (note opposite signs for j)
        x = x.at[I].add(dxi)
        x = x.at[J].add(-dxj)

        # Keep the new configuration registered to the original (no drift)
        if keep_centroid:
            x = x - jnp.mean(x - x0, axis=0, keepdims=True)

        iters_used = k + 1

    # Final stats
    rij = x[I] - x[J]
    dij = jnp.linalg.norm(rij, axis=1)
    max_viol = jnp.maximum(D - dij, 0.0).max()
    rms_disp = jnp.sqrt(jnp.mean(jnp.sum((x - x0) ** 2, axis=1)))

    return x, int(iters_used), float(max_viol), float(rms_disp)


def verify_min_dist(x: jnp.ndarray, D: float) -> Tuple[float, float]:
    """Return (min_pair_distance, max_violation)."""
    N = x.shape[0]
    I, J = jnp.triu_indices(N, k=1)
    dij = jnp.linalg.norm(x[I] - x[J], axis=1)
    min_d = jnp.min(dij)
    max_viol = jnp.maximum(D - dij, 0.0).max()
    return float(min_d), float(max_viol)


def demo():
    key = jax.random.PRNGKey(0)
    N = 200
    D = 0.08  # target min separation

    # Random initial points in a cube [-0.5,0.5]^3 (likely lots of overlaps)
    key, sub = jax.random.split(key)
    x0 = jax.random.uniform(sub, (N, 3), minval=-0.5, maxval=0.5)

    # Optionally add heterogeneous masses (heavier = moves less)
    # masses = jnp.linspace(0.5, 2.0, N)
    masses = None

    x, iters, max_viol, rms_disp = project_spheres(
        x0,
        D,
        masses=masses,
        max_iters=4000,
        tol=1e-9,
        relax=0.9,
        keep_centroid=True,
        key=key,
    )

    min_d, check_max_viol = verify_min_dist(x, D)

    print("---- sphere_nudge_01 report ----")
    print(f"N                : {N}")
    print(f"D (target)       : {D:.6f}")
    print(f"iters used       : {iters}")
    print(f"min pair dist    : {min_d:.9f}")
    print(f"max violation    : {check_max_viol:.3e}")
    print(f"RMS displacement : {rms_disp:.6e}")


if __name__ == "__main__":
    demo()
