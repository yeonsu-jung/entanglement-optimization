# %%
import numpy as np
from numba import njit, int64, float64

# =========================
# PBC & math helpers
# =========================
@njit(inline='always')
def _box_L(C):
    return 2.0 * C

@njit(inline='always')
def _min_image_vec(d, L):
    return d - L * np.round(d / L)

@njit(inline='always')
def _wrap01_vec(x, L):
    return x - L * np.floor(x / L)

@njit(inline='always')
def _unwrap_segment_same_image(p0, p1, C):
    """Map p0 to [0,L) and place p1 in the nearest periodic image to p0."""
    L = _box_L(C)
    q0 = _wrap01_vec(p0 + C, L)
    d  = _min_image_vec((p1 + C) - q0, L)
    q1 = q0 + d
    return q0, q1   # same periodic frame (not re-wrapped)

# =========================
# Grid helpers (CSR style)
# =========================
@njit
def _grid_params(C, cell_size):
    L = _box_L(C)
    nx = max(1, int(np.floor(L / cell_size)))
    ny = nx
    nz = nx
    hx = L / nx
    hy = L / ny
    hz = L / nz
    return L, nx, ny, nz, hx, hy, hz

@njit(inline='always')
def _cell_index(ix, iy, iz, nx, ny, nz):
    # Modulo wrap to respect PBC on the grid indices
    if ix >= nx: ix -= nx
    if iy >= ny: iy -= ny
    if iz >= nz: iz -= nz
    if ix < 0: ix += nx
    if iy < 0: iy += ny
    if iz < 0: iz += nz
    return ix + nx * (iy + ny * iz)

@njit
def _axis_split(a0, a1, L):
    """
    Split [a0,a1] (can be out of [0,L)) into up to 2 intervals inside [0,L).
    Returns (buf[4], count) storing [lo0,hi0, lo1,hi1].
    """
    out = np.empty(4, dtype=float64)
    if a0 < 0.0:
        # [0, a1] U [a0+L, L]
        out[0] = 0.0;        out[1] = min(a1, L - 1e-12)
        out[2] = a0 + L;     out[3] = L
        return out, 2
    elif a1 >= L:
        # [a0, L] U [0, a1-L]
        out[0] = a0;         out[1] = L
        out[2] = 0.0;        out[3] = a1 - L
        return out, 2
    else:
        out[0] = a0;         out[1] = a1
        out[2] = 0.0;        out[3] = 0.0
        return out, 1

@njit(inline='always')
def _to_idx_range(lo, hi, h, n):
    i0 = int(np.floor(lo / h))
    i1 = int(np.floor((hi - 1e-12) / h))
    if i0 < 0: i0 = 0
    if i1 < 0: i1 = 0
    if i0 > n-1: i0 = n-1
    if i1 > n-1: i1 = n-1
    if i1 < i0:
        t = i0; i0 = i1; i1 = t
    return i0, i1





@njit
def _segment_aabb_cell_ranges(p0, p1, C, cell_size, inflate):
    """
    Compute grid index ranges (possibly split across boundaries) for the inflated AABB.
    Returns:
        xr_idx(2,2), nxr,
        yr_idx(2,2), nyr,
        zr_idx(2,2), nzr,
        nx, ny, nz
    """
    # L, nx, ny, nz, hx, hy, hz = _grid_params(C, cell_size)
    # q0, q1 = _unwrap_segment_same_image(p0, p1, C)
    # smin = np.minimum(q0, q1) - inflate
    # smax = np.maximum(q0, q1) + inflate

    L, nx, ny, nz, hx, hy, hz = _grid_params(C, cell_size)

    # Unwrap start; keep true direction for the other endpoint
    q0 = _wrap01_vec(p0 + C, L)
    q1 = q0 + (p1 - p0)   # <-- preserve segment direction

    smin = np.minimum(q0, q1) - inflate
    smax = np.maximum(q0, q1) + inflate


    xr, nxr = _axis_split(smin[0], smax[0], L)
    yr, nyr = _axis_split(smin[1], smax[1], L)
    zr, nzr = _axis_split(smin[2], smax[2], L)

    xr_idx = np.empty((2,2), dtype=int64)
    yr_idx = np.empty((2,2), dtype=int64)
    zr_idx = np.empty((2,2), dtype=int64)

    for a in range(nxr):
        lo = xr[2*a+0]; hi = xr[2*a+1]
        i0, i1 = _to_idx_range(lo, hi, hx, nx)
        xr_idx[a,0] = i0; xr_idx[a,1] = i1
    for a in range(nyr):
        lo = yr[2*a+0]; hi = yr[2*a+1]
        i0, i1 = _to_idx_range(lo, hi, hy, ny)
        yr_idx[a,0] = i0; yr_idx[a,1] = i1
    for a in range(nzr):
        lo = zr[2*a+0]; hi = zr[2*a+1]
        i0, i1 = _to_idx_range(lo, hi, hz, nz)
        zr_idx[a,0] = i0; zr_idx[a,1] = i1

    return xr_idx, nxr, yr_idx, nyr, zr_idx, nzr, nx, ny, nz

@njit
def _grid_insert_segment(rod_id,
                         p0, p1, C, cell_size, inflate,
                         cell_head, link_idx, link_next,
                         nnz_ref, max_entries):
    """
    Insert rod_id into all cells overlapped by its inflated AABB.
    Uses CSR-style linked lists.
    """
    xr_idx, nxr, yr_idx, nyr, zr_idx, nzr, nx, ny, nz = _segment_aabb_cell_ranges(p0, p1, C, cell_size, inflate)

    for ax in range(nxr):
        ix0, ix1 = xr_idx[ax,0], xr_idx[ax,1]
        for ay in range(nyr):
            iy0, iy1 = yr_idx[ay,0], yr_idx[ay,1]
            for az in range(nzr):
                iz0, iz1 = zr_idx[az,0], zr_idx[az,1]
                for ix in range(ix0, ix1+1):
                    for iy in range(iy0, iy1+1):
                        for iz in range(iz0, iz1+1):
                            c = _cell_index(ix, iy, iz, nx, ny, nz)
                            k = nnz_ref[0]
                            if k >= max_entries:
                                return False  # out of capacity
                            link_idx[k]  = rod_id
                            link_next[k] = cell_head[c]
                            cell_head[c] = k
                            nnz_ref[0] = k + 1
    return True

@njit
def _grid_gather_candidates(p0, p1, C, cell_size, inflate,
                            cell_head, link_idx, link_next,
                            seen_stamp, seen, out_buf):
    """
    Gather unique candidate rod indices overlapping the inflated AABB cells.
    Returns the number of candidates written into out_buf.
    """
    xr_idx, nxr, yr_idx, nyr, zr_idx, nzr, nx, ny, nz = _segment_aabb_cell_ranges(p0, p1, C, cell_size, inflate)
    count = 0
    for ax in range(nxr):
        ix0, ix1 = xr_idx[ax,0], xr_idx[ax,1]
        for ay in range(nyr):
            iy0, iy1 = yr_idx[ay,0], yr_idx[ay,1]
            for az in range(nzr):
                iz0, iz1 = zr_idx[az,0], zr_idx[az,1]
                for ix in range(ix0, ix1+1):
                    for iy in range(iy0, iy1+1):
                        for iz in range(iz0, iz1+1):
                            c = _cell_index(ix, iy, iz, nx, ny, nz)
                            e = cell_head[c]
                            while e != -1:
                                j = link_idx[e]
                                if seen[j] != seen_stamp:
                                    seen[j] = seen_stamp
                                    out_buf[count] = j
                                    count += 1
                                e = link_next[e]
    return count



# --- PBC distance: anchor b0 to a0, PRESERVE true direction (b1-b0) ---
@njit(inline='always')
def _dist2_lin_seg_pbc_v2(a0, a1, b0, b1, C, eps=1e-12):
    L = 2.0 * C
    b0u = a0 + (b0 - a0 - L * np.round((b0 - a0)/L))   # min-image b0 wrt a0
    b1u = b0u + (b1 - b0)                               # keep direction
    return _dist2_lin_seg(a0, a1, b0u, b1u, eps)        # your non-PBC dist^2


@njit
def _segment_aabb_cell_ranges_v2(p0, p1, C, cell_size, inflate):
    L, nx, ny, nz, hx, hy, hz = _grid_params(C, cell_size)
    q0 = (p0 + C) - L * np.floor((p0 + C)/L)  # wrap to [0,L)
    q1 = q0 + (p1 - p0)                        # preserve direction
    smin = np.minimum(q0, q1) - inflate
    smax = np.maximum(q0, q1) + inflate

    xr, nxr = _axis_split(smin[0], smax[0], L)
    yr, nyr = _axis_split(smin[1], smax[1], L)
    zr, nzr = _axis_split(smin[2], smax[2], L)

    xr_idx = np.empty((2,2), dtype=np.int64)
    yr_idx = np.empty((2,2), dtype=np.int64)
    zr_idx = np.empty((2,2), dtype=np.int64)

    for a in range(nxr):
        i0 = int(np.floor(xr[2*a+0]/hx)); i1 = int(np.floor((xr[2*a+1]-1e-12)/hx))
        i0 = max(0, min(nx-1, i0)); i1 = max(0, min(nx-1, i1))
        if i1 < i0: i0, i1 = i1, i0
        xr_idx[a,0] = i0; xr_idx[a,1] = i1
    for a in range(nyr):
        i0 = int(np.floor(yr[2*a+0]/hy)); i1 = int(np.floor((yr[2*a+1]-1e-12)/hy))
        i0 = max(0, min(ny-1, i0)); i1 = max(0, min(ny-1, i1))
        if i1 < i0: i0, i1 = i1, i0
        yr_idx[a,0] = i0; yr_idx[a,1] = i1
    for a in range(nzr):
        i0 = int(np.floor(zr[2*a+0]/hz)); i1 = int(np.floor((zr[2*a+1]-1e-12)/hz))
        i0 = max(0, min(nz-1, i0)); i1 = max(0, min(nz-1, i1))
        if i1 < i0: i0, i1 = i1, i0
        zr_idx[a,0] = i0; zr_idx[a,1] = i1

    return xr_idx, nxr, yr_idx, nyr, zr_idx, nzr, nx, ny, nz


@njit
def _grid_gather_candidates_v2(p0, p1, C, cell_size, inflate,
                            cell_head, link_idx, link_next,
                            seen_stamp, seen, out_buf):
    xr_idx, nxr, yr_idx, nyr, zr_idx, nzr, nx, ny, nz = _segment_aabb_cell_ranges_v2(p0, p1, C, cell_size, inflate)
    count = 0
    for ax in range(nxr):
        ix0, ix1 = xr_idx[ax,0], xr_idx[ax,1]
        for ay in range(nyr):
            iy0, iy1 = yr_idx[ay,0], yr_idx[ay,1]
            for az in range(nzr):
                iz0, iz1 = zr_idx[az,0], zr_idx[az,1]
                for ix in range(ix0, ix1+1):
                    for iy in range(iy0, iy1+1):
                        for iz in range(iz0, iz1+1):
                            c = _cell_index(ix, iy, iz, nx, ny, nz)
                            e = cell_head[c]
                            while e != -1:
                                j = link_idx[e]
                                if seen[j] != seen_stamp:
                                    seen[j] = seen_stamp
                                    out_buf[count] = j
                                    count += 1
                                e = link_next[e]
    return count

@njit
def _grid_insert_segment_v2(rod_id, p0, p1, C, cell_size, inflate,
                            cell_head, link_idx, link_next, nnz_ref, max_entries):
    xr_idx, nxr, yr_idx, nyr, zr_idx, nzr, nx, ny, nz = _segment_aabb_cell_ranges_v2(p0, p1, C, cell_size, inflate)
    for ax in range(nxr):
        ix0, ix1 = xr_idx[ax,0], xr_idx[ax,1]
        for ay in range(nyr):
            iy0, iy1 = yr_idx[ay,0], yr_idx[ay,1]
            for az in range(nzr):
                iz0, iz1 = zr_idx[az,0], zr_idx[az,1]
                for ix in range(ix0, ix1+1):
                    for iy in range(iy0, iy1+1):
                        for iz in range(iz0, iz1+1):
                            c = _cell_index(ix, iy, iz, nx, ny, nz)
                            k = nnz_ref[0]
                            if k >= max_entries:
                                return False
                            link_idx[k]  = rod_id
                            link_next[k] = cell_head[c]
                            cell_head[c] = k
                            nnz_ref[0] = k + 1
    return True

# =========================
# Narrow-phase: squared distance under PBC
# =========================
@njit(inline='always')
def _dist2_lin_seg(p0, p1, q0, q1, eps=1e-12):
    """Squared distance between two segments in R^3 (no PBC)."""
    u = p1 - p0
    v = q1 - q0
    w0 = p0 - q0

    uu = u[0]*u[0] + u[1]*u[1] + u[2]*u[2]
    vv = v[0]*v[0] + v[1]*v[1] + v[2]*v[2]
    uv = u[0]*v[0] + u[1]*v[1] + u[2]*v[2]
    wu = w0[0]*u[0] + w0[1]*u[1] + w0[2]*u[2]
    wv = w0[0]*v[0] + w0[1]*v[1] + w0[2]*v[2]

    D = uu * vv - uv * uv
    # unconstrained params
    s = (uv * wv - vv * wu) / (D if abs(D) >= eps else 1.0)
    t = (uu * wv - uv * wu) / (D if abs(D) >= eps else 1.0)

    # near-parallel fallback
    if abs(D) < eps:
        s = 0.0
        t = -wv / (vv if vv >= eps else 1.0)

    # clamp s to [0,1], recompute t on that boundary
    if s < 0.0: s = 0.0
    elif s > 1.0: s = 1.0
    t = (s * uv + wv) / (vv if vv >= eps else 1.0)
    if t < 0.0: t = 0.0
    elif t > 1.0: t = 1.0

    # if t was clamped, recompute s on that boundary
    su = (-wu + t * uv) / (uu if uu >= eps else 1.0)
    if (t < 0.999999 and t > 0.000001) == False:
        if su < 0.0: s = 0.0
        elif su > 1.0: s = 1.0
        else: s = su

    dx = w0[0] + s*u[0] - t*v[0]
    dy = w0[1] + s*u[1] - t*v[1]
    dz = w0[2] + s*u[2] - t*v[2]
    return dx*dx + dy*dy + dz*dz

# @njit(inline='always')
# def _dist2_lin_seg_pbc(a0, a1, b0, b1, C, eps=1e-12):
#     """Squared distance under periodic cube [-C, C]^3 by unwrapping b near a0."""
#     L = _box_L(C)
#     b0u = a0 + _min_image_vec(b0 - a0, L)
#     b1u = a1 + _min_image_vec(b1 - a1, L)  # keep segment direction consistent
#     return _dist2_lin_seg(a0, a1, b0u, b1u, eps)

@njit(inline='always')
def _dist2_lin_seg_pbc(a0, a1, b0, b1, C, eps=1e-12):
    L = _box_L(C)
    # Anchor b0 near a0
    b0u = a0 + _min_image_vec(b0 - a0, L)
    # Preserve the actual segment direction
    b1u = b0u + (b1 - b0)
    return _dist2_lin_seg(a0, a1, b0u, b1u, eps)


# @njit(inline='always')
# def _dist2_lin_seg_pbc(a0, a1, b0, b1, C, eps=1e-12):
#     L = _box_L(C)
#     # unwrap BOTH endpoints of b w.r.t. the SAME anchor (a0)
#     b0u = a0 + _min_image_vec(b0 - a0, L)
#     b1u = a0 + _min_image_vec(b1 - a0, L)
#     return _dist2_lin_seg(a0, a1, b0u, b1u, eps)

# =========================
# Main generator (Numba-native)
# =========================
@njit
def create_nonintersecting_random_rods_contained_pbc_numba(
    num_rods,
    rod_diameter,
    container_size,
    max_attempts,
    rod_length,
    cell_size,
    grid_capacity_multiplier,  # e.g. 48
    seed
):
    C = float(container_size)
    if cell_size <= 0.0:
        cell_size = rod_length

    # Grid & capacity
    L, nx, ny, nz, hx, hy, hz = _grid_params(C, cell_size)
    num_cells = nx * ny * nz
    max_entries = int(grid_capacity_multiplier * num_rods)

    cell_head = np.empty(num_cells, dtype=int64); cell_head[:] = -1
    link_idx  = np.empty(max_entries, dtype=int64)
    link_next = np.empty(max_entries, dtype=int64)
    nnz_ref   = np.zeros(1, dtype=int64)

    q = np.zeros((num_rods, 5), dtype=float64)
    p_starts = np.zeros((num_rods, 3), dtype=float64)
    p_ends   = np.zeros((num_rods, 3), dtype=float64)

    seen    = np.full(num_rods, -1, dtype=int64)
    out_buf = np.empty(num_rods, dtype=int64)

    diam2 = rod_diameter * rod_diameter
    inflate_insert = 0.0              # tight insert
    # inflate_query  = rod_diameter     # conservative query (critical)
    inflate_query = rod_diameter + 1e-12


    np.random.seed(seed)

    placed = 0
    for i in range(num_rods):
        created = False
        attempts = 0

        while (not created) and (attempts < max_attempts):
            # sample
            x = (np.random.random()*2.0 - 1.0)*C
            y = (np.random.random()*2.0 - 1.0)*C
            z = (np.random.random()*2.0 - 1.0)*C
            phi   = np.random.random() * np.pi
            theta = np.random.random() * (2.0*np.pi)

            p0 = np.array([x, y, z], dtype=float64)
            d  = np.array([np.sin(phi)*np.cos(theta),
                           np.sin(phi)*np.sin(theta),
                           np.cos(phi)], dtype=float64)
            p1 = p0 + rod_length * d

            # broad-phase query with full diameter inflate
            seen_stamp = i
            n_cand = _grid_gather_candidates_v2(p0, p1, C, cell_size, inflate_query,
                                             cell_head, link_idx, link_next,
                                             seen_stamp, seen, out_buf)

            # narrow-phase checks
            intersect = False
            for k in range(n_cand):
                j = out_buf[k]
                d2 = _dist2_lin_seg_pbc(p0, p1, p_starts[j], p_ends[j], C)
                if d2 < diam2:
                    intersect = True
                    break

            if not intersect:
                # accept & insert with tight inflate
                q[i,0] = x; q[i,1] = y; q[i,2] = z; q[i,3] = phi; q[i,4] = theta
                p_starts[i,:] = p0
                p_ends[i,:]   = p1
                ok = _grid_insert_segment_v2(i, p0, p1, C, cell_size, inflate_insert,
                                          cell_head, link_idx, link_next,
                                          nnz_ref, max_entries)
                if not ok:
                    return q[:i,:], i  # ran out of grid capacity
                created = True

            attempts += 1

        if not created:
            return q[:i,:], i

        placed += 1

    return q, placed



# Build endpoints from q (x,y,z,phi,theta)
@njit
def _endpoints_from_q(q, rod_length):
    n = q.shape[0]
    p0 = np.empty((n,3), dtype=float64)
    p1 = np.empty((n,3), dtype=float64)
    for i in range(n):
        x = q[i,0]; y = q[i,1]; z = q[i,2]
        phi = q[i,3]; theta = q[i,4]
        sx = np.sin(phi); cx = np.cos(phi)
        ct = np.cos(theta); st = np.sin(theta)
        ux = sx * ct
        uy = sx * st
        uz = cx
        p0[i,0] = x; p0[i,1] = y; p0[i,2] = z
        p1[i,0] = x + rod_length * ux
        p1[i,1] = y + rod_length * uy
        p1[i,2] = z + rod_length * uz
    return p0, p1

@njit
def verify_min_distance_pbc_numba(
    p_starts, p_ends,
    container_size,
    rod_diameter,
    cell_size,
    grid_capacity_multiplier=48
):
    """
    Verify constraints:
    - Returns global min distance and the argmin pair (i,j)
    - Counts #pairs with distance < rod_diameter (violations)
    Uses a CSR grid for near-linear time under PBC.
    """
    C = float(container_size)
    n = p_starts.shape[0]

    # Grid + capacity
    L, nx, ny, nz, hx, hy, hz = _grid_params(C, cell_size)
    num_cells = nx * ny * nz
    max_entries = int(grid_capacity_multiplier * n)

    cell_head = np.empty(num_cells, dtype=int64); cell_head[:] = -1
    link_idx  = np.empty(max_entries, dtype=int64)
    link_next = np.empty(max_entries, dtype=int64)
    nnz_ref   = np.zeros(1, dtype=int64)

    # Insert all rods with ZERO inflation (tight AABBs)
    for i in range(n):
        ok = _grid_insert_segment(
            i, p_starts[i], p_ends[i], C,
            cell_size, 0.0,      # <-- insert with no inflation
            cell_head, link_idx, link_next,
            nnz_ref, max_entries
        )
        if not ok:
            # Out of capacity; signal gracefully
            return np.nan, -1, -1, -1

    # Query with inflation = rod_diameter to catch all potential violators
    seen = np.full(n, -1, dtype=int64)
    out_buf = np.empty(n, dtype=int64)
    diam2 = rod_diameter * rod_diameter

    dmin2 = 1e300
    imin = -1
    jmin = -1
    violations = 0

    for i in range(n):
        # gather candidate neighbors of rod i
        n_cand = _grid_gather_candidates(
            p_starts[i], p_ends[i], C,
            cell_size, rod_diameter,   # <-- query with inflate = diameter
            cell_head, link_idx, link_next,
            i, seen, out_buf           # stamp = i
        )
        # test unique pairs j>i
        for k in range(n_cand):
            j = out_buf[k]
            if j <= i:
                continue
            d2 = _dist2_lin_seg_pbc(p_starts[i], p_ends[i], p_starts[j], p_ends[j], C)
            if d2 < dmin2:
                dmin2 = d2; imin = i; jmin = j
            if d2 < diam2:
                violations += 1

    if imin == -1:
        # n < 2 case
        return np.inf, -1, -1, 0
    return np.sqrt(dmin2), imin, jmin, violations

# Convenience wrapper: take q and rod_length
@njit
def verify_from_q(q, rod_length, container_size, rod_diameter, cell_size, grid_capacity_multiplier=48):
    p0, p1 = _endpoints_from_q(q, rod_length)
    return verify_min_distance_pbc_numba(p0, p1, container_size, rod_diameter, cell_size, grid_capacity_multiplier)

# %%
if __name__ == "__main__":
    # Problem setup
    C = 6.0              # half box => 10^3 volume
    rod_length = 1.0
    alpha = 100.0
    rod_diameter = rod_length / alpha
    # N / C^3 * rod_diameter * rod_length^2 ~ 1
    N = C**3 / (rod_diameter * rod_length**2) * 0.8
    N = int(N)
    # N = 2
    print(N)
# %%
    # N = 20000             # try 1k to see the speedup

    max_attempts = 1000000
    cell_size = rod_length         # good starting heuristic
    # grid_capacity_multiplier = 48  # safe default; increase if you see capacity hits
    grid_capacity_multiplier = 96  # safe default; increase if you see capacity hits
    seed = 12345

    

    q, placed = create_nonintersecting_random_rods_contained_pbc_numba(
        num_rods=N,
        rod_diameter=rod_diameter,
        container_size=C,
        max_attempts=max_attempts,
        rod_length=rod_length,
        cell_size=cell_size,
        grid_capacity_multiplier=grid_capacity_multiplier,
        seed=seed
    )
    print(f"Placed {placed}/{N} rods.")


# %%
    # from visualizations import plot_many_rods
    # plot_many_rods(q.reshape(-1,5))

# %%
# Suppose you already ran:
# q, placed = create_nonintersecting_random_rods_contained_pbc_numba(...)

# Use same cell_size heuristic as placement (or slightly smaller)
    cell_size_verify = rod_length

    dmin, i_min, j_min, num_viol = verify_from_q(
        q[:placed], rod_length, C, rod_diameter, cell_size_verify, 48
    )

    gap = dmin - rod_diameter

    print(f"Min distance: {dmin:.6e}  Gap: {gap:.6e}")
    print(f"Global min distance: {dmin:.6e}")    
    print(f"Argmin pair: (i={i_min}, j={j_min})")
    print(f"Violations (< diameter): {num_viol}  (should be 0)")


# %%
# ---------------------------------------------------------------------@njit
    def audit_pair_min_pbc(i, j, p0, p1, C):
        L = 2.0 * C
        dmin2 = 1e300
        for dx in (-L, 0.0, L):
            for dy in (-L, 0.0, L):
                for dz in (-L, 0.0, L):
                    shift = np.array([dx,dy,dz])
                    d2 = _dist2_lin_seg(p0[i], p1[i], p0[j] + shift, p1[j] + shift)
                    if d2 < dmin2:
                        dmin2 = d2
        return np.sqrt(dmin2)
# %%
    # after placement:
    # q, placed = create_nonintersecting_random_rods_contained_pbc_numba(...)
    p0, p1 = _endpoints_from_q(q[:placed], rod_length)

    # from your log:
    i, j = 6469, 11775  # or whatever pair you want to check
    d_ij = audit_pair_min_pbc(i, j, p0, p1, C)

    print(f"Audit 27-image min distance for ({i},{j}): {d_ij:.9e}")
    print(f"Threshold (rod_diameter):              {rod_diameter:.9e}")
    print(f"Margin d_ij - D:                       {d_ij - rod_diameter:.9e}")


# %%

    



    


# %%
    @njit
    def verify_hardened(q, rod_length, C, rod_diameter, cell_size, grid_capacity_multiplier=48):
        p0, p1 = _endpoints_from_q(q, rod_length)
        # ... build grid with _grid_insert_segment_v2 (tight inflate) ...
        D = rod_diameter
        D2 = D*D
        dmin2 = 1e300; imin=-1; jmin=-1; viol=0
        # for each i: gather with inflate = D using _grid_gather_candidates_v2
        # for each candidate j>i:
        #   d2f = _dist2_lin_seg_pbc_v2(p0[i],p1[i],p0[j],p1[j],C)
        #   if d2f < D2:
        #       d_exact = audit_pair_min_pbc(i,j,p0,p1,C)   # 27 images
        #       if d_exact < D: viol += 1
        #       d2 = d_exact*d_exact
        #   else:
        #       d2 = d2f
        #   if d2 < dmin2: dmin2 = d2; imin=i; jmin=j
        return np.sqrt(dmin2), imin, jmin, viol


    verify_hardened(q[:placed], rod_length, C, rod_diameter, cell_size_verify, 48)