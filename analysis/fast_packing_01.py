import time
import numpy as np
import jax
import jax.numpy as jnp
from collections import defaultdict

# ---------------------------------------------------------------------
# 1) Robust segment distance (Euclidean) and PBC wrapper, both JAX-jitted
# ---------------------------------------------------------------------
def _clip01(x): return jnp.clip(x, 0.0, 1.0)

@jax.jit
def dist_lin_seg(p0, p1, q0, q1, eps=1e-12):
    """Distance between two segments p0-p1 and q0-q1 in R^3."""
    u = p1 - p0
    v = q1 - q0
    w0 = p0 - q0

    uu = jnp.dot(u, u)
    vv = jnp.dot(v, v)
    uv = jnp.dot(u, v)
    wu = jnp.dot(w0, u)
    wv = jnp.dot(w0, v)

    D = uu * vv - uv * uv

    # unconstrained (line-line) params
    s_un = (uv * wv - vv * wu) / jnp.where(jnp.abs(D) < eps, 1.0, D)
    t_un = (uu * wv - uv * wu) / jnp.where(jnp.abs(D) < eps, 1.0, D)

    # near-parallel fallback
    s_un = jnp.where(jnp.abs(D) < eps, 0.0, s_un)
    t_un = jnp.where(jnp.abs(D) < eps, -wv / jnp.where(vv < eps, 1.0, vv), t_un)

    # clamp s, recompute t on boundary
    s = _clip01(s_un)
    t = (s * uv + wv) / jnp.where(vv < eps, 1.0, vv)
    t = _clip01(t)

    # if t changed, recompute s-on-boundary
    s_alt = (-wu + t * uv) / jnp.where(uu < eps, 1.0, uu)
    s = jnp.where((t != t_un), _clip01(s_alt), s)

    # degenerate handling
    both_degenerate = jnp.logical_and(uu < eps, vv < eps)
    s = jnp.where(uu < eps, 0.0, s)
    t = jnp.where(vv < eps, _clip01(-wv / jnp.where(vv < eps, 1.0, vv)), t)

    dP = w0 + s * u - t * v
    return jnp.where(both_degenerate, jnp.linalg.norm(p0 - q0), jnp.linalg.norm(dP))

def _min_image(d, L):
    return d - L * jnp.round(d / L)

@jax.jit
def dist_lin_seg_pbc(a0, a1, b0, b1, C, eps=1e-12):
    """Distance under periodic cube [-C, C]^3 by unwrapping b near a0."""
    L = 2.0 * C
    b0u = a0 + _min_image(b0 - a0, L)
    b1u = a0 + _min_image(b1 - a0, L)
    return dist_lin_seg(a0, a1, b0u, b1u, eps=eps)

# Vectorized batch: (b0s,b1s) are (K,3), (a0,a1,C) are scalars
dist_batch_pbc = jax.jit(
    jax.vmap(lambda b0, b1, a0, a1, C: dist_lin_seg_pbc(a0, a1, b0, b1, C),
             in_axes=(0, 0, None, None, None))
)

# Enable 64-bit if you need high precision
jax.config.update("jax_enable_x64", True)

# ---------------------------------------------------------------------
# 2) Broad-phase (uniform grid with PBC AABB) + JAX narrow-phase
# ---------------------------------------------------------------------
def _box_L(C): return 2.0 * C

def _wrap01_np(x, L): return x - L * np.floor(x / L)
def _min_image_np(d, L): return d - L * np.round(d / L)

def _unwrap_segment_same_image_np(p1, p2, C):
    L = _box_L(C)
    q1 = _wrap01_np(p1 + C, L)
    d  = _min_image_np((p2 + C) - q1, L)
    q2 = q1 + d
    return q1, q2

def _grid_params(C, cell_size):
    L = _box_L(C)
    n = max(1, int(np.floor(L / cell_size)))
    cell = L / n
    return L, n, cell

def _axis_ranges(a_min, a_max, L):
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
    i0 = int(np.floor(lo / cell))
    i1 = int(np.floor((hi - 1e-12) / cell))
    i0 = max(0, min(n-1, i0)); i1 = max(0, min(n-1, i1))
    if i1 < i0: i0, i1 = i1, i0
    return i0, i1

def _segment_cells(p0, p1, C, cell_size, inflate):
    L, n, cell = _grid_params(C, cell_size)
    q0, q1 = _unwrap_segment_same_image_np(p0, p1, C)
    smin = np.minimum(q0, q1) - inflate
    smax = np.maximum(q0, q1) + inflate
    for (x0,x1) in _axis_ranges(smin[0], smax[0], L):
        ix0, ix1 = _to_idx_range(x0, x1, cell, n)
        for (y0,y1) in _axis_ranges(smin[1], smax[1], L):
            iy0, iy1 = _to_idx_range(y0, y1, cell, n)
            for (z0,z1) in _axis_ranges(smin[2], smax[2], L):
                iz0, iz1 = _to_idx_range(z0, z1, cell, n)
                for ix in range(ix0, ix1+1):
                    for iy in range(iy0, iy1+1):
                        for iz in range(iz0, iz1+1):
                            yield (ix, iy, iz), n

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
    C = float(container_size)
    if cell_size is None:
        cell_size = max(rod_length, 0.75 * rod_diameter)

    grid = defaultdict(list)  # (ix,iy,iz) -> [rod indices]
    q = np.zeros((num_rods, 5), dtype=np.float64)
    p_starts = np.zeros((num_rods, 3), dtype=np.float64)
    p_ends   = np.zeros((num_rods, 3), dtype=np.float64)
    rnd = np.random.RandomState(None if rng is None else rng)
    inflate = 0.5 * rod_diameter

    for i in range(num_rods):
        created = False
        attempts = 0
        while (not created) and (attempts < max_attempts):
            # sample a random rod
            x = rnd.uniform(-C, C); y = rnd.uniform(-C, C); z = rnd.uniform(-C, C)
            phi = rnd.uniform(0.0, np.pi); theta = rnd.uniform(0.0, 2.0*np.pi)
            p0 = np.array([x, y, z], dtype=np.float64)
            d  = np.array([np.sin(phi)*np.cos(theta),
                           np.sin(phi)*np.sin(theta),
                           np.cos(phi)], dtype=np.float64)
            p1 = p0 + rod_length * d

            # candidate set via grid cells
            cand = set()
            for cell_idx, n in _segment_cells(p0, p1, C, cell_size, inflate):
                cand.update(grid.get(cell_idx, ()))

            if cand:
                b0s = p_starts[list(cand)]
                b1s = p_ends[list(cand)]
                distances = np.asarray(dist_batch_pbc(b0s, b1s, p0, p1, C))
                if np.any(distances < rod_diameter):
                    attempts += 1
                    continue  # reject

            # accept
            q[i] = np.array([x, y, z, phi, theta], dtype=np.float64)
            p_starts[i] = p0; p_ends[i] = p1
            for cell_idx, n in _segment_cells(p0, p1, C, cell_size, inflate):
                grid[cell_idx].append(i)
            created = True
            attempts += 1

        if not created:
            print(f"Failed to place all rods (placed {i}/{num_rods}).")
            return q[:i]

        if (i % progress_every) == 0:
            print(f"Rod {i} placed successfully")

    return q

# ---------------------------------------------------------------------
# 3) Example run + verification
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # Problem params
    C = 5.0                 # half-box size -> box is 10x10x10
    rod_len = 1.0
    alpha = 100.0
    rod_diam = rod_len / alpha  # e.g., slender rods
    N = 20000                  # keep modest for brute-force verification

    print("Building configuration:")
    print(f"  N={N}, C={C}, rod_length={rod_len}, rod_diameter={rod_diam}")

    t0 = time.perf_counter()
    q = create_nonintersecting_random_rods_contained_pbc_broadphase_jax(
        num_rods=N,
        rod_diameter=rod_diam,
        container_size=C,
        rod_length=rod_len,
        cell_size=rod_len,       # heuristic: one rod per cell
        progress_every=50,
        rng=12345
    )
    t1 = time.perf_counter()
    print(f"Placement done in {t1 - t0:.3f} s. Placed {len(q)} rods.")

    # --- Verify (brute force) ---
    # Rebuild endpoints
    x, y, z, phi, theta = [q[:,k] for k in range(5)]
    starts = np.stack([x, y, z], axis=1)
    dirs = np.stack([np.sin(phi)*np.cos(theta),
                     np.sin(phi)*np.sin(theta),
                     np.cos(phi)], axis=1)
    ends = starts + rod_len * dirs

    # All-pairs distances (brute-force) for sanity check
    # (vectorized with vmap in chunks to avoid huge memory)
    def bf_min_violation(starts, ends, C, batch=2048):
        n = starts.shape[0]
        viol = 0
        dmin = np.inf
        for i in range(n):
            # batches of j>i
            js = np.arange(i+1, n)
            if js.size == 0: continue
            b0s = starts[js]
            b1s = ends[js]
            dists = np.asarray(dist_batch_pbc(b0s, b1s, starts[i], ends[i], C))
            dmin = min(dmin, float(dists.min()))
            viol += int(np.sum(dists < rod_diam))
        return dmin, viol

    t2 = time.perf_counter()
    dmin, num_viol = bf_min_violation(starts, ends, C)
    t3 = time.perf_counter()
    print(f"Verification (brute-force) took {t3 - t2:.3f} s.")
    print(f"  Global min distance: {dmin:.6e}")
    print(f"  Violations (< diameter): {num_viol}  (should be 0)")
