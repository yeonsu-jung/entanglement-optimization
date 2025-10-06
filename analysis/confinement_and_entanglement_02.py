# %%

# add ../core
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'core')))

from potentials import total_effective_potential, total_harmonic_line, dist_lin_seg_over_ij
import polyscope as ps
from transforms import q_to_x
from visualizations import prep_for_polyscope

# get current folder and make output folder named the current folder's name
# current filename
current_file = os.path.basename(__file__)
filename = os.path.splitext(current_file)[0]
output_folder = f'{filename}'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f'Output folder: {output_folder}')

from pathlib import Path

output_folder = Path(output_folder)
MOVIE_DIR = f"{output_folder}/movie"
if not os.path.exists(MOVIE_DIR):
    os.makedirs(MOVIE_DIR)
    print(f'Movie folder: {MOVIE_DIR}')


# %%

# confined using PBC

save_folder = f'{output_folder}'


num_rods = 500
alpha = 100
rod_diameter = 1/alpha
container_size = 1

from protocols import create_nonintersecting_random_rods_contained_pbc
q0 = create_nonintersecting_random_rods_contained_pbc(num_rods,rod_diameter, container_size)
# %%
from visualizations import plot_many_rods
plot_many_rods(q0.reshape(-1,5))
# %%

phi = q0[:,3]
theta = q0[:,4]

from matplotlib import pyplot as plt

# x11 = x1 + rod_length*jnp.sin(phi1)*jnp.cos(theta1)
#     y11 = y1 + rod_length*jnp.sin(phi1)*jnp.sin(theta1)
#     z11 = z1 + rod_length*jnp.cos(phi1)
# phi: elevation
# theta: azimuth

plt.figure()
plt.hist(phi, bins=30)
plt.title('phi distribution')

# plt.figure()
# plt.hist(theta, bins=30)
# plt.title('theta distribution')
# %%
from protocols import create_nonintersecting_random_rods
large_num_rods = 1000
big_container_size = 5
# num_rods estimator
# num_rods / container_size^3 * rod_diameter * rod_length ~ 5
large_num_rods_est = int(big_container_size**3 * 5 / (rod_diameter * 1))
print(f'estimated number of rods for big container: {large_num_rods_est}')
# %%
q2 = create_nonintersecting_random_rods_contained_pbc(large_num_rods_est,rod_diameter, big_container_size)

phi = q2[:,3]
theta = q2[:,4]
plt.figure()
plt.hist(phi, bins=30)
plt.title('phi distribution - big container')

# %%
phi = q1[:,3]
theta = q1[:,4]

plt.figure()
plt.hist(phi, bins=30)
plt.title('phi distribution')

# strong domain-topological effects



# %%



# %%

from protocols import create_entangled_rods
import jax

random_keys = jax.random.PRNGKey(11)
# simple grad descent

grad_fn = jax.jit(jax.grad(total_effective_potential))


col_rad = rod_diameter / 2
amp = 100
params = {'col_rad': col_rad,
            'amp': amp}

grad_fn2 = jax.jit(jax.grad(lambda x: total_harmonic_line(x, params)))



from protocols import create_nonintersecting_random_rods_contained_pbc
import jax.numpy as jnp
import numpy as onp




# %%

import numpy as onp
from collections import defaultdict

# --- PBC helpers ---
def box_size_from_C(C):
    return 2.0 * C  # side length

def wrap01(x, L):
    # map coordinate x in R to [0, L)
    return x - L * onp.floor(x / L)

def minimum_image(d, L):
    # component-wise minimum image convention on displacement vector d
    return d - L * onp.round(d / L)

def unwrap_segment_to_same_image(p1, p2, C):
    """Return endpoints in [0,L) where p2 is minimum-image relative to p1."""
    L = box_size_from_C(C)
    # map p1 into [0,L)
    q1 = wrap01(p1 + C, L)  # shift from [-C,C] to [0,2C), then wrap
    # place p2 close to q1 using minimum image
    raw_p2 = p2 + C
    d = minimum_image(raw_p2 - q1, L)
    q2 = q1 + d
    # Both q1, q2 might run slightly out of [0,L) after adding d; bring them in a shared frame:
    # Keep q1 in [0,L), allow q2 to be outside but AABB logic will handle modulo later.
    return q1, q2

def segment_aabb_cells(q1, q2, C, cell_size, inflate):
    """Compute grid cell index ranges overlapped by the segment AABB (inflated) under [0,L) coords.
       Returns ranges per axis and grid dims."""
    L = box_size_from_C(C)
    grid_n = onp.maximum(1, onp.floor(L / cell_size).astype(int))
    # ensure cell size consistent with integer grid
    cell = L / grid_n

    # AABB with inflation
    smin = onp.minimum(q1, q2) - inflate
    smax = onp.maximum(q1, q2) + inflate

    # Handle wrap: AABB might straddle boundary. We’ll compute two possibilities per axis:
    # If smin<0 or smax>=L, split range into two wrapped intervals.
    def axis_ranges(a_min, a_max):
        # Convert to [0,L) by wrapping ends; but split if crossing boundary.
        r = []
        if a_min < 0:
            # [a_min, a_max] maps to [0, a_max] U [a_min+L, L)
            r.append((0.0, min(a_max, L-1e-12)))
            r.append((a_min + L, L))
        elif a_max >= L:
            # [a_min, a_max] maps to [a_min, L) U [0, a_max-L]
            r.append((a_min, L))
            r.append((0.0, (a_max - L)))
        else:
            r.append((a_min, a_max))
        return r

    xr = axis_ranges(smin[0], smax[0])
    yr = axis_ranges(smin[1], smax[1])
    zr = axis_ranges(smin[2], smax[2])

    # Convert to index ranges (inclusive), modulo grid_n
    def to_idx_range(a0, a1):
        i0 = int(onp.floor(a0 / cell))
        i1 = int(onp.floor((max(a1, a0) - 1e-12) / cell))  # include boundary
        # clamp to [0, grid_n-1]
        i0 = max(0, min(grid_n-1, i0))
        i1 = max(0, min(grid_n-1, i1))
        if i1 < i0:
            i0, i1 = i1, i0
        return i0, i1

    x_ranges = [to_idx_range(a0, a1) for (a0, a1) in xr]
    y_ranges = [to_idx_range(a0, a1) for (a0, a1) in yr]
    z_ranges = [to_idx_range(a0, a1) for (a0, a1) in zr]

    return x_ranges, y_ranges, z_ranges, grid_n, cell

def iter_cells_for_aabb(x_ranges, y_ranges, z_ranges):
    # Iterate all combinations of split ranges
    for (ix0, ix1) in x_ranges:
        for (iy0, iy1) in y_ranges:
            for (iz0, iz1) in z_ranges:
                for ix in range(ix0, ix1+1):
                    for iy in range(iy0, iy1+1):
                        for iz in range(iz0, iz1+1):
                            yield (ix, iy, iz)

# --- Main function with spatial hash broad-phase under PBC ---
def create_nonintersecting_random_rods_contained_pbc_broadphase(
    num_rods,
    rod_diameter,
    container_size,
    max_attempts=10000,
    rod_length=1.0,
    cell_size=None,
    progress_every=100
):
    """
    Places num_rods non-intersecting random unit-length rods in a periodic cube [-C, C]^3.

    Broad-phase: uniform grid spatial hash in [0, 2C)^3 with minimum-image unwrapping
    for per-candidate AABB queries. Only rods in overlapping cells are tested with
    exact narrow-phase distance (dist_lin_seg_pbc).

    Args:
        num_rods: int
        rod_diameter: float
        container_size: float = C (half box side)
        max_attempts: int
        rod_length: float (default 1.0 to match your code)
        cell_size: float or None; default ~ rod_length (good heuristic)
        progress_every: int
    Returns:
        q: (N,5) array [x, y, z, phi, theta] of placed rods (subset if early fail).
    """
    C = float(container_size)
    L = box_size_from_C(C)

    if cell_size is None:
        # Heuristic: one rod length per cell; tune as needed (e.g., 0.75*rod_length)
        cell_size = max(rod_length, rod_diameter)

    # Grid bookkeeping
    # Key: (ix,iy,iz) -> list of rod indices
    grid = defaultdict(list)

    q = onp.zeros((num_rods, 5), dtype=onp.float64)

    # Store endpoints for narrow-phase reuse
    p_starts = onp.zeros((num_rods, 3), dtype=onp.float64)
    p_ends   = onp.zeros((num_rods, 3), dtype=onp.float64)

    # Precompute grid params update function
    def aabb_cells_for_segment(p1, p2):
        q1, q2 = unwrap_segment_to_same_image(p1, p2, C)
        inflate = rod_diameter * 0.5
        return segment_aabb_cells(q1, q2, C, cell_size, inflate)

    for i in range(num_rods):
        created = False
        attempts = 0

        while not created and attempts < max_attempts:
            # Sample a random rod
            x = onp.random.uniform(-C, C)
            y = onp.random.uniform(-C, C)
            z = onp.random.uniform(-C, C)
            phi = onp.random.uniform(0, onp.pi)
            theta = onp.random.uniform(0, 2 * onp.pi)

            p_i  = onp.array([x, y, z], dtype=onp.float64)
            dirv = onp.array([
                onp.sin(phi) * onp.cos(theta),
                onp.sin(phi) * onp.sin(theta),
                onp.cos(phi)
            ], dtype=onp.float64)
            p_ii = p_i + rod_length * dirv

            # Find candidate set via grid
            x_ranges, y_ranges, z_ranges, grid_n, cell = aabb_cells_for_segment(p_i, p_ii)

            # Collect unique candidate indices from overlapped cells
            candidate_indices = set()
            for cell_idx in iter_cells_for_aabb(x_ranges, y_ranges, z_ranges):
                candidate_indices.update(grid.get(cell_idx, ()))

            # Narrow-phase: only test against candidates
            intersect = False
            for j in candidate_indices:
                distance = dist_lin_seg_pbc(p_i, p_ii, p_starts[j], p_ends[j], C)
                if distance < rod_diameter:
                    intersect = True
                    break

            if not intersect:
                # Accept this rod
                q[i] = onp.array([x, y, z, phi, theta], dtype=onp.float64)
                p_starts[i] = p_i
                p_ends[i]   = p_ii

                # Insert into grid
                for cell_idx in iter_cells_for_aabb(x_ranges, y_ranges, z_ranges):
                    grid[cell_idx].append(i)

                created = True

            attempts += 1

        if not created:
            print(f"Failed to place all rods without intersection (placed {i}/{num_rods}).")
            return q[:i]

        if (i % progress_every) == 0:
            print(f"Rod {i} placed successfully")

    return q



# %%



import jax
import jax.numpy as jnp

# Clamp to [0,1]
def _clip01(x):
    return jnp.clip(x, 0.0, 1.0)

@jax.jit
def dist_lin_seg(point1s, point1e, point2s, point2e, eps=1e-12):
    """
    Robust shortest distance between two line segments in R^3.
    Branch-light, vectorization-friendly, handles parallel/degenerate segments.
    """
    p = point1s
    q = point2s
    u = point1e - point1s
    v = point2e - point2s
    w0 = p - q

    uu = jnp.dot(u, u)
    vv = jnp.dot(v, v)
    uv = jnp.dot(u, v)
    wu = jnp.dot(w0, u)
    wv = jnp.dot(w0, v)

    # Denominator of the unconstrained solution
    D = uu * vv - uv * uv

    # Unconstrained solution (s*, t*) for infinite lines
    # If D ~ 0 -> nearly parallel; pick s=0 and project q onto v (or vice versa)
    s_uncon = (uv * wv - vv * wu) / jnp.where(jnp.abs(D) < eps, 1.0, D)
    t_uncon = (uu * wv - uv * wu) / jnp.where(jnp.abs(D) < eps, 1.0, D)

    # Fallback for near-parallel: set s_uncon=0, t from projection onto v
    s_uncon = jnp.where(jnp.abs(D) < eps, 0.0, s_uncon)
    t_uncon = jnp.where(jnp.abs(D) < eps, -wv / jnp.where(vv < eps, 1.0, vv), t_uncon)

    # Clamp to segments
    s = _clip01(s_uncon)
    # When clamping s changed it, recompute best t on that boundary
    t = (s * uv + wv) / jnp.where(vv < eps, 1.0, vv)
    t = _clip01(t)

    # If clamping t changed it, recompute s on that boundary
    s_alt = (-wu + t * uv) / jnp.where(uu < eps, 1.0, uu)
    s = jnp.where((t != t_uncon), _clip01(s_alt), s)

    # If both segments degenerate, just distance between points
    both_degenerate = jnp.logical_and(uu < eps, vv < eps)
    # If first degenerate, project p onto second
    s = jnp.where(uu < eps, 0.0, s)
    t = jnp.where(vv < eps, _clip01(-wv / jnp.where(vv < eps, 1.0, vv)), t)

    # Closest points and distance
    dP = w0 + s * u - t * v
    dist = jnp.where(both_degenerate, jnp.linalg.norm(p - q),
                     jnp.linalg.norm(dP))
    return dist

# ---------- Periodic version (minimum-image) ----------
def _min_image(d, L):
    return d - L * jnp.round(d / L)

@jax.jit
def dist_lin_seg_pbc(a0, a1, b0, b1, C, eps=1e-12):
    """
    Shortest distance under PBC in cube [-C, C]^3.
    We unwrap b-segment to the periodic image closest to a0.
    """
    L = 2.0 * C
    # Bring b endpoints near a0
    b0u = a0 + _min_image(b0 - a0, L)
    b1u = a0 + _min_image(b1 - a0, L)
    return dist_lin_seg(a0, a1, b0u, b1u, eps=eps)

# %%

import numpy as onp
import jax
import jax.numpy as jnp
from collections import defaultdict

# ---- Use your JAX distance (from your last message), or the robust one I suggested ----
# from your code:
# @jax.jit
# def dist_lin_seg(...): ...
# @jax.jit
# def dist_lin_seg_pbc(a0, a1, b0, b1, C, eps=1e-12): ...

# Vectorized batch distance under PBC:
# (a0,a1 fixed; b0s,b1s are (K,3))
dist_batch_pbc = jax.jit(
    jax.vmap(lambda b0, b1, a0, a1, C: dist_lin_seg_pbc(a0, a1, b0, b1, C),
             in_axes=(0, 0, None, None, None))
)

# ---------- PBC helpers ----------
def _box_L(C): return 2.0 * C

def _wrap01(x, L):
    return x - L * onp.floor(x / L)

def _min_image(d, L):
    return d - L * onp.round(d / L)

def _unwrap_segment_same_image(p1, p2, C):
    """Return endpoints mapped to [0,L) so that p2 is min-image from p1."""
    L = _box_L(C)
    q1 = _wrap01(p1 + C, L)
    d  = _min_image((p2 + C) - q1, L)
    q2 = q1 + d
    return q1, q2  # in same periodic frame

# ---------- Grid / AABB over PBC ----------
def _grid_params(C, cell_size):
    L = _box_L(C)
    n = max(1, int(onp.floor(L / cell_size)))
    cell = L / n
    return L, n, cell

def _axis_ranges(a_min, a_max, L):
    """Split [a_min,a_max] into up to 2 intervals in [0,L) if it wraps."""
    out = []
    if a_min < 0:
        out.append((0.0, min(a_max, L - 1e-12)))
        out.append((a_min + L, L))
    elif a_max >= L:
        out.append((a_min, L))
        out.append((0.0, a_max - L))
    else:
        out.append((a_min, a_max))
    return out

def _to_idx_range(lo, hi, cell, n):
    i0 = int(onp.floor(lo / cell))
    i1 = int(onp.floor((hi - 1e-12) / cell))
    i0 = max(0, min(n-1, i0))
    i1 = max(0, min(n-1, i1))
    if i1 < i0: i0, i1 = i1, i0
    return i0, i1

def _segment_cells(p0, p1, C, cell_size, inflate):
    """Return an iterator over all grid cells overlapped by segment AABB (+inflate)."""
    L, n, cell = _grid_params(C, cell_size)
    q0, q1 = _unwrap_segment_same_image(p0, p1, C)
    smin = onp.minimum(q0, q1) - inflate
    smax = onp.maximum(q0, q1) + inflate

    xr = _axis_ranges(smin[0], smax[0], L)
    yr = _axis_ranges(smin[1], smax[1], L)
    zr = _axis_ranges(smin[2], smax[2], L)

    for (x0,x1) in xr:
        ix0, ix1 = _to_idx_range(x0, x1, cell, n)
        for (y0,y1) in yr:
            iy0, iy1 = _to_idx_range(y0, y1, cell, n)
            for (z0,z1) in zr:
                iz0, iz1 = _to_idx_range(z0, z1, cell, n)
                for ix in range(ix0, ix1+1):
                    for iy in range(iy0, iy1+1):
                        for iz in range(iz0, iz1+1):
                            yield (ix, iy, iz), n

# ---------- Broad-phase placer with JAX narrow-phase ----------
def create_nonintersecting_random_rods_contained_pbc_broadphase_jax(
    num_rods,
    rod_diameter,
    container_size,
    max_attempts=10000,
    rod_length=1.0,
    cell_size=None,
    progress_every=100,
    rng=None
):
    """
    Broad-phase spatial hash (uniform grid) + JAX vectorized narrow-phase under PBC.
    - Domain: cube [-C, C]^3 with periodic BC.
    - Stores q[i] = [x,y,z,phi,theta].
    """
    C = float(container_size)
    if cell_size is None:
        cell_size = max(rod_length, 0.75 * rod_diameter)  # good starting heuristic

    grid = defaultdict(list)  # (ix,iy,iz) -> [rod indices]
    q = onp.zeros((num_rods, 5), dtype=onp.float64)
    p_starts = onp.zeros((num_rods, 3), dtype=onp.float64)
    p_ends   = onp.zeros((num_rods, 3), dtype=onp.float64)

    rnd = onp.random.RandomState(None if rng is None else rng)

    inflate = 0.5 * rod_diameter

    for i in range(num_rods):
        created = False
        attempts = 0

        while (not created) and (attempts < max_attempts):
            # sample a random rod
            x = rnd.uniform(-C, C)
            y = rnd.uniform(-C, C)
            z = rnd.uniform(-C, C)
            phi   = rnd.uniform(0.0, onp.pi)
            theta = rnd.uniform(0.0, 2.0 * onp.pi)

            p0 = onp.array([x, y, z], dtype=onp.float64)
            d  = onp.array([onp.sin(phi)*onp.cos(theta),
                            onp.sin(phi)*onp.sin(theta),
                            onp.cos(phi)], dtype=onp.float64)
            p1 = p0 + rod_length * d

            # candidate set via grid cells
            cand = set()
            for cell_idx, n in _segment_cells(p0, p1, C, cell_size, inflate):
                cand.update(grid.get(cell_idx, ()))

            if cand:
                # batch narrow-phase with JAX
                b0s = p_starts[list(cand)]
                b1s = p_ends[list(cand)]
                # distances is (K,)
                distances = onp.asarray(dist_batch_pbc(b0s, b1s, p0, p1, C))
                if onp.any(distances < rod_diameter):
                    attempts += 1
                    continue  # reject; try another sample

            # accept
            q[i] = onp.array([x, y, z, phi, theta], dtype=onp.float64)
            p_starts[i] = p0
            p_ends[i]   = p1

            # insert into grid cells
            for cell_idx, n in _segment_cells(p0, p1, C, cell_size, inflate):
                grid[cell_idx].append(i)

            created = True
            attempts += 1

        if not created:
            print(f"Failed to place all rods without intersection (placed {i}/{num_rods}).")
            return q[:i]

        if (i % progress_every) == 0:
            print(f"Rod {i} placed successfully")

    return q

# %%
# example run

q = create_nonintersecting_random_rods_contained_pbc_broadphase_jax(
    num_rods=1000, rod_diameter=0.01, container_size=1.0, rng=42
)

# %%

