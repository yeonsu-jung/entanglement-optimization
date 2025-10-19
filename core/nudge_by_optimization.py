# jax_nudger.py
import jax
import jax.numpy as jnp

def _pairwise_indices(N: int):
    return jnp.triu_indices(N, k=1)

# @jax.jit
def nudge_step(X: jnp.ndarray, D: float, step: float = 1.0, underrelax: float = 1.0, eps: float = 1e-12):
    """
    One Jacobi-style nudge step for equal-diameter disks (2D).
    X: (N,2) centers, D: diameter (required min center distance)
    step: scale for the accumulated displacement (usually 1.0)
    underrelax: 0<underrelax<=1 to stabilize large systems (e.g. 0.8~0.95)
    """
    N = X.shape[0]
    ii, jj = _pairwise_indices(N)          # (M,), upper-tri pairs

    rij = X[ii] - X[jj]                    # (M,2)
    dij = jnp.linalg.norm(rij, axis=1)     # (M,)

    # penetration (positive if overlapping)
    pen = jnp.maximum(0.0, D - dij)        # (M,)
    # unit directions (safe with eps)
    uij = rij / (dij[:, None] + eps)       # (M,2)

    # Equal split: push each by pen/2 along ±u
    dX_i =  0.5 * pen[:, None] * uij
    dX_j = -0.5 * pen[:, None] * uij

    # Accumulate to particles (scatter-add)
    dX = jnp.zeros_like(X)
    dX = dX.at[ii].add(dX_i)
    dX = dX.at[jj].add(dX_j)

    X_new = X + (step * underrelax) * dX

    # Report max penetration after this step (for stopping)
    max_pen = jnp.max(pen) if pen.size > 0 else 0.0
    return X_new, max_pen

from typing import Optional


def nudge_until_converged(
    X0: jnp.ndarray,
    D: float,
    max_iters: int = 500,
    tol: float = 1e-8,
    step: float = 1.0,
    underrelax: float = 0.9,
    jitter: float = 1e-12,
    key: Optional[jax.random.PRNGKey] = None,
):
    """
    Pure-Python driver loop (keeps a history list). The inner step is JITed.
    """
    X = X0
    if key is not None and jitter > 0:
        X = X + jitter * jax.random.normal(key, X.shape)

    hist = []
    for it in range(max_iters):
        X, max_pen = nudge_step(X, D, step=step, underrelax=underrelax)
        hist.append(float(max_pen))
        if max_pen <= tol:
            break
    return X, it + 1, jnp.array(hist)


def plot_disks(X, D, title="",ax=None,color='blue'):
    if ax is None:
        plt.figure()
        ax = plt.gca()

    ax.scatter(X[:,0], X[:,1], s=10, color=color)
    # draw disks as circles (lightweight)
    for p in X:
        c = plt.Circle((float(p[0]), float(p[1])), D/2, fill=False, lw=0.5, color=color)
        ax.add_patch(c)
    ax.set_aspect('equal', 'box')
    ax.set_title(title)
    # plt.show()

# demo_nudger.py

import jax
import jax.numpy as jnp

system_size = 20.0
N   = 100
D   = 1   # target diameter (min center distance)

# Example 1: random cloud with overlaps


key = jax.random.key(0)

# Uniform in [0,1]^2 makes lots of overlaps for this D
X0  = jax.random.uniform(key, (N,2)) * system_size

X, iters, hist = nudge_until_converged(
    X0, D,
    max_iters=400,
    tol=1e-6,
    step=1.0,
    underrelax=0.9,
    jitter=1e-12,
    key=key,
)

print(f"[Random] iters={iters}, final max penetration={hist[-1]:.3e}")

# Example 2: dense grid slightly shrunken (strong overlaps), then nudge

import matplotlib.pyplot as plt
displacements = X - X0
# arrow plot
fig, ax = plt.subplots()
plot_disks(X0, D, "Random init (overlapping)", ax=ax, color='gray')
plot_disks(X,  D, "After nudge (non-overlapping)", ax=ax, color='blue')
plt.quiver(X0[:,0], X0[:,1], displacements[:,0], displacements[:,1], angles='xy', scale_units='xy', scale=1)
# plt.xlim(0,system_size)
# plt.ylim(0,system_size)
plt.show()



nx, ny = 9, 9
gx, gy = jnp.meshgrid(jnp.linspace(0, system_size, nx), jnp.linspace(0, system_size, ny), indexing='ij')
Xg = jnp.stack([gx.ravel(), gy.ravel()], axis=1)
# squash the grid to force overlaps
X0_grid = 0.8 * (Xg - 0.5) + 0.5
Dg = 1.0 / max(nx-1, ny-1) * 0.95  # near the original grid spacing

Xg_out, iters_g, hist_g = nudge_until_converged(
    X0_grid, Dg,
    max_iters=2000,
    tol=1e-8,
    step=1.0,
    underrelax=0.92,
    key=jax.random.key(1),
)

print(f"[Grid]   iters={iters_g}, final max penetration={hist_g[-1]:.3e}")
# quick plot (optional)




displacements = Xg_out - X0_grid
# arrow plot
fig, ax = plt.subplots()
plot_disks(X0_grid, D, "Random init (overlapping)", ax=ax, color='gray')
plot_disks(Xg_out,  D, "After nudge (non-overlapping)", ax=ax, color='blue')
plt.quiver(X0_grid[:,0], X0_grid[:,1], displacements[:,0], displacements[:,1], angles='xy', scale_units='xy', scale=1)
# plt.xlim(0,system_size)
# plt.ylim(0,system_size)
plt.show()