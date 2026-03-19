"""Non-intersecting rod placement.

create_nonintersecting_rods_gpu(num_rods, rod_diameter)
  -> (num_rods, 5) numpy array, q-coordinates, centred at origin.

The outer placement loop is sequential (each rod depends on all prior),
but the distance check uses vmap over existing rods for GPU speed.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, vmap

from physics import dist_lin_seg, sph2cart, q_to_x


@jit
def _min_dist_to_existing(p_i, p_ii, starts, ends, n):
    """Minimum distance from candidate rod to the first n placed rods."""
    dists = vmap(lambda s, e: dist_lin_seg(p_i, p_ii, s, e))(starts, ends)
    mask  = jnp.arange(starts.shape[0]) < n
    return jnp.min(jnp.where(mask, dists, jnp.inf))


def create_nonintersecting_rods_gpu(num_rods: int, rod_diameter: float,
                                     max_attempts: int = 10_000) -> np.ndarray:
    """Place num_rods non-intersecting rods using GPU-batched distance checks.

    Returns (num_rods, 5) float64 array of rod states, centred at origin.
    """
    print(f"Placing {num_rods} non-intersecting rods (diameter={rod_diameter:.6f})…")

    q = np.zeros((num_rods, 5), dtype=np.float64)
    existing_starts = jnp.zeros((num_rods, 3), dtype=jnp.float64)
    existing_ends   = jnp.zeros((num_rods, 3), dtype=jnp.float64)

    for i in range(num_rods):
        created  = False
        attempts = 0

        while not created and attempts < max_attempts:
            x     = np.random.uniform(-1.0, 1.0)
            y     = np.random.uniform(-1.0, 1.0)
            z     = np.random.uniform(-1.0, 1.0)
            theta = np.arccos(np.random.uniform(-1.0, 1.0))
            phi   = np.random.uniform(0.0, 2.0 * np.pi)

            p_i  = jnp.array([x, y, z])
            u_i  = jnp.array([np.sin(theta) * np.cos(phi),
                               np.sin(theta) * np.sin(phi),
                               np.cos(theta)])
            p_ii = p_i + u_i

            if i == 0:
                intersect = False
            else:
                min_d = _min_dist_to_existing(p_i, p_ii,
                                               existing_starts, existing_ends, i)
                intersect = bool(min_d < rod_diameter)

            if not intersect:
                q[i] = [x, y, z, theta, phi]
                existing_starts = existing_starts.at[i].set(p_i)
                existing_ends   = existing_ends.at[i].set(p_ii)
                created = True

            attempts += 1

        if attempts == max_attempts:
            raise RuntimeError(
                f"Failed to place rod {i} after {max_attempts} attempts. "
                "Try reducing num_rods or rod_diameter."
            )

        if i % 200 == 0:
            print(f"  placed {i}/{num_rods}")

    # Centre at origin
    q_jnp  = jnp.array(q, dtype=jnp.float64)
    x_ends = q_to_x(q_jnp)
    centre = np.mean(np.asarray((x_ends[:, :3] + x_ends[:, 3:]) / 2), axis=0)
    q[:, :3] -= centre

    print(f"  done ({num_rods} rods placed)")
    return q
