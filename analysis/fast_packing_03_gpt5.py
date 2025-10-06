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
    # component-wise minimum image
    return d - L * np.round(d / L)

@njit(inline='always')
def _wrap01_vec(x, L):
    # map to [0, L)
    return x - L * np.floor(x / L)

# =========================
# Grid helpers (CSR style, PBC aware)
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
    # modulo wrap indices to respect PBC
    if ix >= nx: ix -= nx
    if iy >= ny: iy -= ny
    if iz >= nz: iz -= nz
    if ix < 0: ix += nx
    if iy < 0: iy += ny
    if iz < 0: iz += nz
    return ix + nx * (iy + ny * iz)

@njit
def _axis_split(a0, a1, L):
    """Split [a0,a1] (possibly outside [0,L)) into up to 2 intervals within [0,L)."""
    out = np.empty(4, dtype=float64)
    if a0 < 0.0:
        out[0] = 0.0; out[1] = min(a1, L - 1e-12)
        out[2] = a0 + L; out[3] = L
        return out, 2
    elif a1 >= L:
        out[0] = a0; out[1] = L
        out[2] = 0.0; out[3] = a1 - L
        return out, 2
    else:
        out[0] = a0; out[1] = a1
        out[2] = 0.0; out[3] = 0.0
        return out, 1

@njit(inline='always')
def _to_idx_range(lo, hi, h, n):
    # Inclusive upper edge to avoid cell-plane aliasing
    i0 = int(np.floor(lo / h))
    i1 = int(np.floor(hi / h))
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
    PBC: unwrap start to [0,L) and preserve TRUE direction for end (avoid shape distortion).
    """
    L, nx, ny, nz, hx, hy, hz = _grid_params(C, cell_size)
    q0 = _wrap01_vec(p0 + C, L)
    q1 = q0 + (p1 - p0)   # preserve direction across PBC

    smin = np.minimum(q0, q1) - inflate
    smax = np.maximum(q0, q1) + inflate

    # Tiny grid pad to defeat boundary aliasing on cell planes
    grid_pad = max(1e-12, 1e-9 * cell_size)
    smin -= grid_pad
    smax += grid_pad

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
    Insert rod_id into all cells overlapped by its inflated AABB (CSR-style linked lists).
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

# =========================
# Narrow-phase: squared distance (no PBC) + PBC wrapper
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

@njit(inline='always')
def _dist2_lin_seg_pbc(a0, a1, b0, b1, C, eps=1e-12):
    """
    Squared distance under PBC:
      Anchor b0 near a0 (minimum-image), then PRESERVE the true direction (b1-b0).
    """
    L = _box_L(C)
    b0u = a0 + _min_image_vec(b0 - a0, L)
    b1u = b0u + (b1 - b0)  # keep direction
    return _dist2_lin_seg(a0, a1, b0u, b1u, eps)

# =========================
# Placer (Numba-native, spherocylinder-aware)
# =========================
@njit
def create_nonintersecting_random_rods_contained_pbc_numba(
    num_rods,
    rod_diameter,
    container_size,
    max_attempts,
    rod_length,
    cell_size,
    grid_capacity_multiplier,  # e.g. 48..96
    seed
):
    """
    Fully Numba-native broad-phase + narrow-phase under PBC.
    Spherocylinder-aware AABBs via inflation by r = 0.5*rod_diameter.
    Returns (q, placed) where q is (placed,5) with [x,y,z,phi,theta].
    """
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

    visit_stamp = np.int64(1)

    # Spherocylinder radius r (capsule radius)
    r = 0.5 * rod_diameter
    diam2 = (2.0*r) * (2.0*r)
    inflate_insert = r                   # balanced scheme: insert with r
    inflate_query  = r + 1e-12           # and query with r (+tiny eps)

    # RNG
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

            # gather candidates with inflate r
            # seen_stamp = i

            seen_stamp = visit_stamp
            visit_stamp += 1
            if visit_stamp < 0:
                seen[:] = -1
                visit_stamp = np.int64(1)

            n_cand = _grid_gather_candidates(p0, p1, C, cell_size, inflate_query,
                                             cell_head, link_idx, link_next,
                                             seen_stamp, seen, out_buf)

            # narrow-phase checks (true centerline distance under PBC)
            intersect = False
            for k in range(n_cand):
                j = out_buf[k]
                d2 = _dist2_lin_seg_pbc(p0, p1, p_starts[j], p_ends[j], C)
                if d2 < diam2:   # centerline distance < 2r
                    intersect = True
                    break

            if not intersect:
                # accept & insert with inflate r
                q[i,0] = x; q[i,1] = y; q[i,2] = z; q[i,3] = phi; q[i,4] = theta
                p_starts[i,:] = p0
                p_ends[i,:]   = p1
                ok = _grid_insert_segment(i, p0, p1, C, cell_size, inflate_insert,
                                          cell_head, link_idx, link_next,
                                          nnz_ref, max_entries)
                if not ok:
                    return q[:i,:], i  # capacity exceeded
                created = True

            attempts += 1

        if not created:
            return q[:i,:], i

        placed += 1

    return q, placed

def create_nonintersecting_random_rods_contained_numba(
    num_rods,
    rod_diameter,
    container_size,
    max_attempts,
    rod_length,
    cell_size,
    grid_capacity_multiplier,  # e.g. 48..96
    seed
):
    """
    Fully Numba-native broad-phase + narrow-phase under PBC.
    Spherocylinder-aware AABBs via inflation by r = 0.5*rod_diameter.
    Returns (q, placed) where q is (placed,5) with [x,y,z,phi,theta].
    """
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

    # Spherocylinder radius r (capsule radius)
    r = 0.5 * rod_diameter
    diam2 = (2.0*r) * (2.0*r)
    inflate_insert = r                   # balanced scheme: insert with r
    inflate_query  = r + 1e-12           # and query with r (+tiny eps)

    # RNG
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

            # gather candidates with inflate r
            seen_stamp = i
            n_cand = _grid_gather_candidates(p0, p1, C, cell_size, inflate_query,
                                             cell_head, link_idx, link_next,
                                             seen_stamp, seen, out_buf)

            # narrow-phase checks (true centerline distance under PBC)
            intersect = False
            for k in range(n_cand):
                j = out_buf[k]
                d2 = _dist2_lin_seg(p0, p1, p_starts[j], p_ends[j], C)
                if d2 < diam2:   # centerline distance < 2r
                    intersect = True
                    break

            if not intersect:
                # accept & insert with inflate r
                q[i,0] = x; q[i,1] = y; q[i,2] = z; q[i,3] = phi; q[i,4] = theta
                p_starts[i,:] = p0
                p_ends[i,:]   = p1
                ok = _grid_insert_segment(i, p0, p1, C, cell_size, inflate_insert,
                                          cell_head, link_idx, link_next,
                                          nnz_ref, max_entries)
                if not ok:
                    return q[:i,:], i  # capacity exceeded
                created = True

            attempts += 1

        if not created:
            return q[:i,:], i

        placed += 1

    return q, placed



# =========================
# Verification utilities
# =========================
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
def audit_pair_min_pbc(i, j, p0, p1, C):
    """
    Exact PBC check by enumerating the 27 neighbor images of rod j.
    Returns the true minimum distance between segments i and j.
    """
    L = 2.0 * C
    dmin2 = 1e300
    for dx in (-L, 0.0, L):
        for dy in (-L, 0.0, L):
            for dz in (-L, 0.0, L):
                shift = np.array([dx, dy, dz])
                d2 = _dist2_lin_seg(p0[i], p1[i], p0[j] + shift, p1[j] + shift)
                if d2 < dmin2:
                    dmin2 = d2
    return np.sqrt(dmin2)

@njit
def verify_min_distance_pbc_numba(
    p_starts, p_ends,
    container_size,
    rod_diameter,
    cell_size,
    grid_capacity_multiplier=48
):
    """
    Verify constraints near-linearly using the CSR grid:
      - Returns (global_min_dist, i_min, j_min, violations_count)
      - Counts pairs with centerline distance < 2r (= rod_diameter)
    Uses balanced r/r inflation (insert r, query r).
    """
    C = float(container_size)
    n = p_starts.shape[0]
    r = 0.5 * rod_diameter
    diam2 = (2.0*r)*(2.0*r)

    # Grid + capacity
    L, nx, ny, nz, hx, hy, hz = _grid_params(C, cell_size)
    num_cells = nx * ny * nz
    max_entries = int(grid_capacity_multiplier * n)

    cell_head = np.empty(num_cells, dtype=int64); cell_head[:] = -1
    link_idx  = np.empty(max_entries, dtype=int64)
    link_next = np.empty(max_entries, dtype=int64)
    nnz_ref   = np.zeros(1, dtype=int64)

    # Insert all rods with inflate r (capsule AABB)
    for i in range(n):
        ok = _grid_insert_segment(
            i, p_starts[i], p_ends[i], C,
            cell_size, r,
            cell_head, link_idx, link_next,
            nnz_ref, max_entries
        )
        if not ok:
            return np.nan, -1, -1, -1

    # Query with inflate r
    seen = np.full(n, -1, dtype=int64)
    out_buf = np.empty(n, dtype=int64)

    dmin2 = 1e300
    imin = -1
    jmin = -1
    violations = 0

    for i in range(n):
        n_cand = _grid_gather_candidates(
            p_starts[i], p_ends[i], C,
            cell_size, r,
            cell_head, link_idx, link_next,
            i, seen, out_buf
        )
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
        return np.inf, -1, -1, 0
    return np.sqrt(dmin2), imin, jmin, violations

@njit
def verify_from_q(q, rod_length, container_size, rod_diameter, cell_size, grid_capacity_multiplier=48):
    p0, p1 = _endpoints_from_q(q, rod_length)
    return verify_min_distance_pbc_numba(p0, p1, container_size, rod_diameter, cell_size, grid_capacity_multiplier)

@njit
def verify_hardened(q, rod_length, C, rod_diameter, cell_size, grid_capacity_multiplier=48):
    """
    Same as verify_min_distance_pbc_numba but confirms any flagged pair
    with an exact 27-image audit before counting a violation.
    """
    n = q.shape[0]
    r = 0.5 * rod_diameter
    D = 2.0*r; D2 = D*D

    # endpoints
    p0 = np.empty((n,3), dtype=float64)
    p1 = np.empty((n,3), dtype=float64)
    for i in range(n):
        x,y,z,phi,theta = q[i,0], q[i,1], q[i,2], q[i,3], q[i,4]
        sx = np.sin(phi); cx = np.cos(phi); ct = np.cos(theta); st = np.sin(theta)
        u0 = sx*ct; u1 = sx*st; u2 = cx
        p0[i,0], p0[i,1], p0[i,2] = x, y, z
        p1[i,0], p1[i,1], p1[i,2] = x + rod_length*u0, y + rod_length*u1, z + rod_length*u2

    # grid
    L, nx, ny, nz, hx, hy, hz = _grid_params(C, cell_size)
    num_cells = nx*ny*nz
    max_entries = int(grid_capacity_multiplier * n)
    cell_head = np.empty(num_cells, dtype=int64); cell_head[:] = -1
    link_idx  = np.empty(max_entries, dtype=int64)
    link_next = np.empty(max_entries, dtype=int64)
    nnz_ref   = np.zeros(1, dtype=int64)

    # insert with inflate r
    for i in range(n):
        ok = _grid_insert_segment(i, p0[i], p1[i], C, cell_size, r,
                                  cell_head, link_idx, link_next,
                                  nnz_ref, max_entries)
        if not ok:
            return np.nan, -1, -1, -1

    # verify with gather r; confirm near-threshold with auditor
    seen = np.full(n, -1, dtype=int64)
    out_buf = np.empty(n, dtype=int64)
    dmin2 = 1e300; imin=-1; jmin=-1; viol=0

    for i in range(n):
        n_cand = _grid_gather_candidates(p0[i], p1[i], C, cell_size, r,
                                         cell_head, link_idx, link_next,
                                         i, seen, out_buf)
        for k in range(n_cand):
            j = out_buf[k]
            if j <= i:
                continue
            d2f = _dist2_lin_seg_pbc(p0[i], p1[i], p0[j], p1[j], C)
            if d2f < D2:
                d_exact = audit_pair_min_pbc(i, j, p0, p1, C)
                if d_exact < D:
                    viol += 1
                d2 = d_exact*d_exact
            else:
                d2 = d2f
            if d2 < dmin2:
                dmin2 = d2; imin=i; jmin=j

    if imin == -1:
        return np.inf, -1, -1, 0
    return np.sqrt(dmin2), imin, jmin, viol

# =========================
# Post-placement pruning (drop later rod in each violating pair)
# =========================
@njit
def _compact_q(q, removed):
    n = q.shape[0]
    keep = 0
    for i in range(n):
        if removed[i] == 0:
            keep += 1
    out = np.empty((keep, 5), dtype=q.dtype)
    idx = 0
    for i in range(n):
        if removed[i] == 0:
            out[idx, 0] = q[i, 0]
            out[idx, 1] = q[i, 1]
            out[idx, 2] = q[i, 2]
            out[idx, 3] = q[i, 3]
            out[idx, 4] = q[i, 4]
            idx += 1
    return out, keep

@njit
def prune_overlaps_keep_earlier(q, rod_length, C, rod_diameter, cell_size, grid_capacity_multiplier=48):
    """
    Remove the later rod in every violating pair (centerline distance < 2r).
    Uses balanced r/r AABBs for near-linear time.
    Returns (q_pruned, kept_count, removed_count).
    """
    n = q.shape[0]
    r = 0.5 * rod_diameter
    D2 = (2.0*r)*(2.0*r)

    # endpoints
    p0 = np.empty((n,3), dtype=np.float64)
    p1 = np.empty((n,3), dtype=np.float64)
    for i in range(n):
        x,y,z,phi,theta = q[i,0], q[i,1], q[i,2], q[i,3], q[i,4]
        sx = np.sin(phi); cx = np.cos(phi); ct = np.cos(theta); st = np.sin(theta)
        u0 = sx*ct; u1 = sx*st; u2 = cx
        p0[i,0], p0[i,1], p0[i,2] = x, y, z
        p1[i,0], p1[i,1], p1[i,2] = x + rod_length*u0, y + rod_length*u1, z + rod_length*u2

    # grid (insert with inflate r)
    L, nx, ny, nz, hx, hy, hz = _grid_params(C, cell_size)
    num_cells = nx*ny*nz
    max_entries = int(grid_capacity_multiplier * n)
    cell_head = np.empty(num_cells, dtype=np.int64); cell_head[:] = -1
    link_idx  = np.empty(max_entries, dtype=np.int64)
    link_next = np.empty(max_entries, dtype=np.int64)
    nnz_ref   = np.zeros(1, dtype=np.int64)

    for i in range(n):
        ok = _grid_insert_segment(i, p0[i], p1[i], C, cell_size, r,
                                  cell_head, link_idx, link_next,
                                  nnz_ref, max_entries)
        if not ok:
            # capacity blow-up: keep as-is (caller can rerun with larger multiplier)
            return q, n, 0

    # sweep and mark later rods for removal
    removed = np.zeros(n, dtype=np.uint8)
    seen = np.full(n, -1, dtype=np.int64)
    out_buf = np.empty(n, dtype=np.int64)

    for i in range(n):
        if removed[i] != 0:
            continue
        # gather candidates with inflate r
        n_cand = _grid_gather_candidates(p0[i], p1[i], C, cell_size, r,
                                         cell_head, link_idx, link_next,
                                         i, seen, out_buf)
        for k in range(n_cand):
            j = out_buf[k]
            if j <= i or removed[j] != 0:
                continue
            d2 = _dist2_lin_seg_pbc(p0[i], p1[i], p0[j], p1[j], C)
            if d2 < D2:
                removed[j] = 1  # drop the later rod

    q_out, kept = _compact_q(q, removed)
    removed_count = n - kept
    return q_out, kept, removed_count

# =========================
# Example run
# =========================
if __name__ == "__main__":
    # Problem setup
    C = 6.0                # half-box length => box side = 12.0
    rod_length = 1.0
    alpha = 100.0
    rod_diameter = rod_length / alpha  # 0.01

    # Rough density heuristic: N / C^3 * D * L^2 ~ 0.8
    N = int((C**3) / (rod_diameter * rod_length**2) * 0.8)
    # N = 10000
    print("Target N:", N)

    max_attempts = 1_000_000
    cell_size = rod_length
    grid_capacity_multiplier = 96
    seed = 12345

    # Place
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

    # Verify (fast)
    dmin, i_min, j_min, num_viol = verify_from_q(q[:placed], rod_length, C, rod_diameter, cell_size, 48)
    print(f"Global min distance: {dmin:.6e}")
    print(f"Argmin pair: (i={i_min}, j={j_min})")
    print(f"Violations (< diameter): {num_viol}  (should be 0)")

    # Hardened verify (optional)
    dmin_h, i_h, j_h, num_vh = verify_hardened(q[:placed], rod_length, C, rod_diameter, cell_size, 48)
    print(f"[HARDENED] Global min distance: {dmin_h:.6e}")
    print(f"[HARDENED] Argmin pair: (i={i_h}, j={j_h})")
    print(f"[HARDENED] Violations (< diameter): {num_vh}  (should be 0)")

    # If violations remain, prune and re-verify
    if num_vh > 0:
        q_pruned, kept, removed = prune_overlaps_keep_earlier(q[:placed], rod_length, C, rod_diameter, cell_size, 96)
        print(f"Pruned {removed} rods; kept {kept}.")
        dmin2, i2, j2, viol2 = verify_from_q(q_pruned, rod_length, C, rod_diameter, cell_size, 48)
        print(f"[AFTER PRUNE] Global min distance: {dmin2:.6e}")
        print(f"[AFTER PRUNE] Violations: {viol2}")
        # Optional: audit the previous argmin pair to see the exact PBC min
        if i_min >= 0 and j_min >= 0:
            p0, p1 = _endpoints_from_q(q[:placed], rod_length)
            d_true = audit_pair_min_pbc(i_min, j_min, p0, p1, C)
            print(f"Audit 27-image min distance for ({i_min},{j_min}): {d_true:.6e}  (D={rod_diameter:.6e})")
    else:
        q_pruned = q[:placed]

# %%
    q_pruned.shape

# %%
    from potentials import acn_over_ij
    from transforms import q_to_x
    from jax import numpy as jnp
    
    ii,jj = jnp.triu_indices(q_pruned.shape[0], k=1)
    x_pruned = q_to_x(q_pruned)
    x_pruned = jnp.array(x_pruned)
    x_pruned = x_pruned.astype(jnp.float32)

    r1 = x_pruned[:,:3]
    r2 = x_pruned[:,3:6]
    print(type(x_pruned[0]))
    
    acn_mat = acn_over_ij(r1,r2,ii,jj)
# %%
    print(x_pruned.dtype, x_pruned.flags)

# %%

    import jax
    import time    

    # Fast + fused
    sum_abs_jit = jax.jit(lambda x: jnp.abs(x).sum())

    t0 = time.time()
    res = sum_abs_jit(acn_mat)
    res.block_until_ready()          # IMPORTANT: force computation before timing
    print("sum(|x|) =", float(res), "time:", time.time() - t0, "s")

# %%
    from jax import jit, vmap
    import jax.numpy as jnp

    def compute_linking_number_cartesian(p_i, p_ii, p_j, p_jj):
        # p_i = jnp.array([x_i, y_i, z_i])
        # p_j = jnp.array([x_j, y_j, z_j])
        # u_i = jnp.array([jnp.sin(phi_i)*jnp.cos(theta_i), jnp.sin(phi_i)*jnp.sin(theta_i), jnp.cos(phi_i)])
        # u_j = jnp.array([jnp.sin(phi_j)*jnp.cos(theta_j), jnp.sin(phi_j)*jnp.sin(theta_j), jnp.cos(phi_j)])

        # p_ii = p_i + l*u_i
        # p_jj = p_j + l*u_j

        r_ij = p_i - p_j
        r_ijj = p_i - p_jj
        r_iij = p_ii - p_j
        r_iijj = p_ii - p_jj

        tol = 1e-6
        n1 = jnp.cross(r_ij, r_ijj)
        n1 = n1/(jnp.linalg.norm(n1)+tol)
        n2 = jnp.cross(r_ijj, r_iijj)
        n2 = n2/(jnp.linalg.norm(n2)+tol)
        n3 = jnp.cross(r_iijj, r_iij)
        n3 = n3/(jnp.linalg.norm(n3)+tol)
        n4 = jnp.cross(r_iij, r_ij)
        n4 = n4/(jnp.linalg.norm(n4)+tol)
        
        tol = 0.

        return -1/4/jnp.pi*jnp.abs(jnp.arcsin(  jnp.clip(jnp.dot(n1,n2),-1.+tol,1.-tol))
                                + jnp.arcsin(jnp.clip(jnp.dot(n2,n3),-1.+tol,1.-tol))
                                + jnp.arcsin(jnp.clip(jnp.dot(n3,n4),-1.+tol,1.-tol))
                                + jnp.arcsin(jnp.clip(jnp.dot(n4,n1),-1.+tol,1.-tol)))
    

    @jit
    def acn_over_ij(r1, r2, i_indices, j_indices):
        return vmap(lambda i, j: compute_linking_number_cartesian(r1[i], r2[i], r1[j], r2[j]))(i_indices, j_indices)
    

    entanglement = sum_abs_jit(acn_over_ij(r1,r2,ii,jj))
    print("entanglement =", float(entanglement))
    
    # %%

    N = 10
    C = 1

    NN = np.geomspace(10, 1000, num=10, dtype=int)

    tt = []
    for i in range(len(NN)):
        N = NN[i]
        start = time.time()
        q, placed = create_nonintersecting_random_rods_contained_numba(
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
        tt.append(time.time() - start)
        print("N=", N, "time:", tt[-1], "s")

# %%
    N = int((C**3) / (rod_diameter * rod_length**2) * 10)

# %%
    # N/V*D*L**2 ~ 10


# %%
    import matplotlib.pyplot as plt
    plt.plot(NN, tt, '-o')
    plt.yscale('log')
    plt.xlabel('N')
    plt.ylabel('time (s)')
    plt.show()
        