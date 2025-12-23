# %%
import numpy as np
from numba import njit, int64, float64

# add ../core to sys.path
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'core')))


# ---------- shared grid helpers ----------
@njit
def _grid_params(C, cell_size):
    L = 2.0 * C
    nx = max(1, int(np.floor(L / cell_size)))
    ny = nx
    nz = nx
    hx = L / nx
    hy = L / ny
    hz = L / nz
    return L, nx, ny, nz, hx, hy, hz

@njit(inline='always')
def _to_idx_range(lo, hi, h, n):
    # Inclusive top edge (robust on cell planes)
    i0 = int(np.floor(lo / h))
    i1 = int(np.floor(hi / h))
    if i0 < 0: i0 = 0
    if i1 < 0: i1 = 0
    if i0 > n-1: i0 = n-1
    if i1 > n-1: i1 = n-1
    if i1 < i0:
        t = i0; i0 = i1; i1 = t
    return i0, i1

# ---------- non-PBC AABB→cell ranges (no wrapping, clamped to box) ----------
@njit
def _segment_aabb_cell_ranges_npbc(p0, p1, C, cell_size, inflate):
    """
    Build an inflated AABB in box coords [0,L]^3 (L=2C) and convert to cell index ranges.
    No PBC: we just clamp to the container.
    """
    L, nx, ny, nz, hx, hy, hz = _grid_params(C, cell_size)

    # map world coords [-C,C] -> [0,L]
    q0 = p0 + C
    q1 = p1 + C

    smin = np.minimum(q0, q1) - inflate
    smax = np.maximum(q0, q1) + inflate

    # tiny pad to avoid missing pairs sitting on cell planes
    grid_pad = max(1e-12, 1e-9 * cell_size)
    smin -= grid_pad
    smax += grid_pad

    # clamp to container
    if smax[0] <= 0.0 or smax[1] <= 0.0 or smax[2] <= 0.0:  # fully left/below/front
        return (np.zeros((1,2), np.int64), 0,
                np.zeros((1,2), np.int64), 0,
                np.zeros((1,2), np.int64), 0,
                nx, ny, nz)
    if smin[0] >= L or smin[1] >= L or smin[2] >= L:        # fully right/above/back
        return (np.zeros((1,2), np.int64), 0,
                np.zeros((1,2), np.int64), 0,
                np.zeros((1,2), np.int64), 0,
                nx, ny, nz)

    smin0 = 0.0 if smin[0] < 0.0 else smin[0]
    smin1 = 0.0 if smin[1] < 0.0 else smin[1]
    smin2 = 0.0 if smin[2] < 0.0 else smin[2]
    smax0 = L   if smax[0] > L   else smax[0]
    smax1 = L   if smax[1] > L   else smax[1]
    smax2 = L   if smax[2] > L   else smax[2]

    xr_idx = np.empty((1,2), dtype=int64)
    yr_idx = np.empty((1,2), dtype=int64)
    zr_idx = np.empty((1,2), dtype=int64)
    ix0, ix1 = _to_idx_range(smin0, smax0, hx, nx)
    iy0, iy1 = _to_idx_range(smin1, smax1, hy, ny)
    iz0, iz1 = _to_idx_range(smin2, smax2, hz, nz)
    xr_idx[0,0], xr_idx[0,1] = ix0, ix1
    yr_idx[0,0], yr_idx[0,1] = iy0, iy1
    zr_idx[0,0], zr_idx[0,1] = iz0, iz1
    return xr_idx, 1, yr_idx, 1, zr_idx, 1, nx, ny, nz

@njit
def _grid_insert_segment_npbc(rod_id,
                              p0, p1, C, cell_size, inflate,
                              cell_head, link_idx, link_next,
                              nnz_ref, max_entries):
    xr_idx, nxr, yr_idx, nyr, zr_idx, nzr, nx, ny, nz = _segment_aabb_cell_ranges_npbc(p0, p1, C, cell_size, inflate)
    if nxr == 0 or nyr == 0 or nzr == 0:
        return True  # completely out of box; nothing to insert (OK)
    for ix in range(xr_idx[0,0], xr_idx[0,1]+1):
        for iy in range(yr_idx[0,0], yr_idx[0,1]+1):
            for iz in range(zr_idx[0,0], zr_idx[0,1]+1):
                c = ix + nx*(iy + ny*iz)
                k = nnz_ref[0]
                if k >= max_entries:
                    return False
                link_idx[k]  = rod_id
                link_next[k] = cell_head[c]
                cell_head[c] = k
                nnz_ref[0] = k + 1
    return True

@njit
def _grid_gather_candidates_npbc(p0, p1, C, cell_size, inflate,
                                 cell_head, link_idx, link_next,
                                 seen_stamp, seen, out_buf):
    xr_idx, nxr, yr_idx, nyr, zr_idx, nzr, nx, ny, nz = _segment_aabb_cell_ranges_npbc(p0, p1, C, cell_size, inflate)
    if nxr == 0 or nyr == 0 or nzr == 0:
        return 0
    count = 0
    for ix in range(xr_idx[0,0], xr_idx[0,1]+1):
        for iy in range(yr_idx[0,0], yr_idx[0,1]+1):
            for iz in range(zr_idx[0,0], zr_idx[0,1]+1):
                c = ix + nx*(iy + ny*iz)
                e = cell_head[c]
                while e != -1:
                    j = link_idx[e]
                    if seen[j] != seen_stamp:
                        seen[j] = seen_stamp
                        out_buf[count] = j
                        count += 1
                    e = link_next[e]
    return count

# ---------- narrow-phase (no PBC) ----------
@njit(inline='always')
def _dist2_lin_seg(p0, p1, q0, q1, eps=1e-12):
    u = p1 - p0
    v = q1 - q0
    w0 = p0 - q0

    uu = u[0]*u[0] + u[1]*u[1] + u[2]*u[2]
    vv = v[0]*v[0] + v[1]*v[1] + v[2]*v[2]
    uv = u[0]*v[0] + u[1]*v[1] + u[2]*v[2]
    wu = w0[0]*u[0] + w0[1]*u[1] + w0[2]*u[2]
    wv = w0[0]*v[0] + w0[1]*v[1] + w0[2]*v[2]

    D = uu * vv - uv * uv
    s = (uv * wv - vv * wu) / (D if abs(D) >= eps else 1.0)
    t = (uu * wv - uv * wu) / (D if abs(D) >= eps else 1.0)

    if abs(D) < eps:
        s = 0.0
        t = -wv / (vv if vv >= eps else 1.0)

    if s < 0.0: s = 0.0
    elif s > 1.0: s = 1.0
    t = (s * uv + wv) / (vv if vv >= eps else 1.0)
    if t < 0.0: t = 0.0
    elif t > 1.0: t = 1.0

    su = (-wu + t * uv) / (uu if uu >= eps else 1.0)
    if (t < 0.999999 and t > 0.000001) == False:
        if su < 0.0: s = 0.0
        elif su > 1.0: s = 1.0
        else: s = su

    dx = w0[0] + s*u[0] - t*v[0]
    dy = w0[1] + s*u[1] - t*v[1]
    dz = w0[2] + s*u[2] - t*v[2]
    return dx*dx + dy*dy + dz*dz

# ---------- NON-PBC PLACER (centroid inside the box) ----------
@njit
def create_nonintersecting_random_rods_contained_npbc_numba(
    num_rods,
    rod_diameter,
    container_size,          # C (half-side); box is [-C, C]^3
    max_attempts,
    rod_length,
    cell_size,
    grid_capacity_multiplier,  # e.g. 48..96
    seed
):
    """
    Non-PBC placement. Only the rod centroid is constrained to lie in [-C, C]^3.
    Spherocylinders: centerline distance threshold = 2r = rod_diameter.
    """
    C = float(container_size)
    if cell_size <= 0.0:
        cell_size = rod_length

    # grid
    L, nx, ny, nz, hx, hy, hz = _grid_params(C, cell_size)
    num_cells = nx * ny * nz
    max_entries = int(grid_capacity_multiplier * num_rods)
    cell_head = np.empty(num_cells, dtype=int64); cell_head[:] = -1
    link_idx  = np.empty(max_entries, dtype=int64)
    link_next = np.empty(max_entries, dtype=int64)
    nnz_ref   = np.zeros(1, dtype=int64)

    # outputs & temp
    q = np.zeros((num_rods, 5), dtype=float64)      # [cx,cy,cz,phi,theta] (centroid!)
    p_starts = np.zeros((num_rods, 3), dtype=float64)
    p_ends   = np.zeros((num_rods, 3), dtype=float64)
    seen     = np.full(num_rods, -1, dtype=int64)
    out_buf  = np.empty(num_rods, dtype=int64)

    visit_stamp = np.int64(1)

    r = 0.5 * rod_diameter
    diam2 = (2.0*r) * (2.0*r)
    inflate_insert = r
    inflate_query  = r + 1e-12

    # RNG
    np.random.seed(seed)

    placed = 0
    halfL = 0.5 * rod_length

    for i in range(num_rods):
        created = False
        attempts = 0

        while (not created) and (attempts < max_attempts):
            # sample centroid uniformly inside the box
            cx = (np.random.random()*2.0 - 1.0)*C
            cy = (np.random.random()*2.0 - 1.0)*C
            cz = (np.random.random()*2.0 - 1.0)*C
            phi   = np.random.random() * np.pi
            theta = np.random.random() * (2.0*np.pi)

            u = np.array([np.sin(phi)*np.cos(theta),
                          np.sin(phi)*np.sin(theta),
                          np.cos(phi)], dtype=float64)
            # endpoints (may extend outside the box — that’s OK)
            p0 = np.array([cx, cy, cz], dtype=float64) - halfL * u
            p1 = p0 + rod_length * u

            # gather candidates with capsule AABB (inflate = r)
            seen_stamp = visit_stamp
            visit_stamp += 1
            if visit_stamp < 0:
                seen[:] = -1
                visit_stamp = np.int64(1)

            n_cand = _grid_gather_candidates_npbc(p0, p1, C, cell_size, inflate_query,
                                                  cell_head, link_idx, link_next,
                                                  seen_stamp, seen, out_buf)

            # narrow-phase: true centerline distance (no PBC)
            collide = False
            for k in range(n_cand):
                j = out_buf[k]
                d2 = _dist2_lin_seg(p0, p1, p_starts[j], p_ends[j])
                if d2 < diam2:
                    collide = True
                    break

            if not collide:

                # accept and insert
                q[i,0] = cx; q[i,1] = cy; q[i,2] = cz; q[i,3] = phi; q[i,4] = theta
                p_starts[i,:] = p0
                p_ends[i,:]   = p1
                ok = _grid_insert_segment_npbc(i, p0, p1, C, cell_size, inflate_insert,
                                               cell_head, link_idx, link_next,
                                               nnz_ref, max_entries)
                if not ok:
                    return q[:i,:], i
                created = True

            attempts += 1

        if not created:
            return q[:i,:], i

        placed += 1

    return q, placed

# ---------- Optional: non-PBC verifier ----------
@njit
def _endpoints_from_q_centroid(q, rod_length):
    n = q.shape[0]
    p0 = np.empty((n,3), dtype=float64)
    p1 = np.empty((n,3), dtype=float64)
    halfL = 0.5 * rod_length
    for i in range(n):
        cx,cy,cz,phi,theta = q[i,0], q[i,1], q[i,2], q[i,3], q[i,4]
        u0 = np.sin(phi)*np.cos(theta)
        u1 = np.sin(phi)*np.sin(theta)
        u2 = np.cos(phi)
        p0[i,0] = cx - halfL*u0; p0[i,1] = cy - halfL*u1; p0[i,2] = cz - halfL*u2
        p1[i,0] = cx + halfL*u0; p1[i,1] = cy + halfL*u1; p1[i,2] = cz + halfL*u2
    return p0, p1

@njit
def verify_min_distance_npbc(p_starts, p_ends, C, rod_diameter, cell_size, grid_capacity_multiplier=48):
    n = p_starts.shape[0]
    r = 0.5 * rod_diameter
    D2 = (2.0*r)*(2.0*r)

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
        ok = _grid_insert_segment_npbc(i, p_starts[i], p_ends[i], C, cell_size, r,
                                       cell_head, link_idx, link_next,
                                       nnz_ref, max_entries)
        if not ok:
            return np.nan, -1, -1, -1

    # query with inflate r
    seen = np.full(n, -1, dtype=int64)
    out_buf = np.empty(n, dtype=int64)

    dmin2 = 1e300; imin=-1; jmin=-1; viol=0
    for i in range(n):
        n_cand = _grid_gather_candidates_npbc(p_starts[i], p_ends[i], C, cell_size, r,
                                              cell_head, link_idx, link_next,
                                              i, seen, out_buf)
        for k in range(n_cand):
            j = out_buf[k]
            if j <= i:
                continue
            d2 = _dist2_lin_seg(p_starts[i], p_ends[i], p_starts[j], p_ends[j])
            if d2 < dmin2:
                dmin2 = d2; imin = i; jmin = j
            if d2 < D2:
                viol += 1

    if imin == -1:
        return np.inf, -1, -1, 0
    return np.sqrt(dmin2), imin, jmin, viol

@njit
def verify_from_q_npbc(q, rod_length, C, rod_diameter, cell_size, grid_capacity_multiplier=48):
    p0, p1 = _endpoints_from_q_centroid(q, rod_length)
    return verify_min_distance_npbc(p0, p1, C, rod_diameter, cell_size, grid_capacity_multiplier)

# %%

if __name__ == "__main__":
    # -----------------------------
    # Example run (non-PBC)
    # -----------------------------
    C = 6.0                 # half-side; box is [-C, C]^3
    rod_length = 1.0
    alpha = 100.0
    rod_diameter = rod_length / alpha   # 0.01 here

    # Heuristic target N:  ~ (C^3) / (D * L^2)
    N = int((C**3) / (rod_diameter * rod_length**2) * 1)  # 50% of heuristic density
    print("Target N:", N)

    max_attempts = 1_000_000
    cell_size = rod_length
    grid_capacity_multiplier = 96
    seed = 12345

    # Place rods (centroid constrained to be in the box; endpoints may stick out)
    q, placed = create_nonintersecting_random_rods_contained_npbc_numba(
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

    # Quick sanity: all centroids in [-C, C]^3?
    inside = (
        (q[:placed,0] >= -C) & (q[:placed,0] <= C) &
        (q[:placed,1] >= -C) & (q[:placed,1] <= C) &
        (q[:placed,2] >= -C) & (q[:placed,2] <= C)
    ).all()
    print("Centroids inside box:", bool(inside))

    # Verify min centerline distance and count of violations (< diameter)
    dmin, i_min, j_min, num_viol = verify_from_q_npbc(
        q[:placed], rod_length, C, rod_diameter, cell_size, 48
    )
    gap = dmin - rod_diameter
    print(f"Global min distance: {dmin:.6e}  (gap to D: {gap:.6e})")
    print(f"Argmin pair: (i={i_min}, j={j_min})")
    print(f"Violations (< diameter): {num_viol}  (should be 0)")

    # If you want to inspect that pair’s raw (no-PBC) distance directly:
    if i_min >= 0 and j_min >= 0:
        p0, p1 = _endpoints_from_q_centroid(q[:placed], rod_length)
        d2_pair = _dist2_lin_seg(p0[i_min], p1[i_min], p0[j_min], p1[j_min])
        print(f"Direct segment-segment distance for argmin pair: {np.sqrt(d2_pair):.6e}")
# %%
    C = 6.
    N_max_est = int((C**3) / (rod_diameter * rod_length**2) * 5)    

    NN = np.geomspace(10, N_max_est, num=10, dtype=int)
    tt = []

    import time
    for i in range(len(NN)):
        N = NN[i]
        start = time.time()
        q, placed = create_nonintersecting_random_rods_contained_npbc_numba(
            num_rods=N,
            rod_diameter=rod_diameter,
            container_size=C,
            max_attempts=max_attempts,
            rod_length=rod_length,
            cell_size=cell_size,
            grid_capacity_multiplier=grid_capacity_multiplier,
            seed=seed
        )
        end = time.time()
        elapsed = end - start
        tt.append(elapsed)
        print(f"N={N:6d} placed={placed:6d} time={elapsed:.3f} sec  ({elapsed/N*1e6:.1f} µs/rod)")
    # %%

    # check sanity of last run
    inside = (
        (q[:placed,0] >= -C) & (q[:placed,0] <= C) &
        (q[:placed,1] >= -C) & (q[:placed,1] <= C) &
        (q[:placed,2] >= -C) & (q[:placed,2] <= C)
    ).all()
    print("Centroids inside box:", bool(inside))

    # %%
    dmin, i_min, j_min, num_viol = verify_from_q_npbc(
        q[:placed], rod_length, C, rod_diameter, cell_size, 48
    )
    gap = dmin - rod_diameter
    print(f"Global min distance: {dmin:.6e}  (gap to D: {gap:.6e})")
    print(f"Argmin pair: (i={i_min}, j={j_min})")
    print(f"Violations (< diameter): {num_viol}  (should be 0)")
    # %%
        
    q_select = q[[i_min, j_min], :]

    from visualizations import plot_many_rods
    from matplotlib import pyplot as plt
    plot_many_rods(q_select )
    plt.show()
    # %%
    from transforms import q_to_x
    x = q_to_x(q[:placed])

    p0 = x[i_min,0:3]
    p1 = x[i_min,3:6]
    q0 = x[j_min,0:3]
    q1 = x[j_min,3:6]

    p0 = np.array(p0)
    p1 = np.array(p1)
    q0 = np.array(q0)
    q1 = np.array(q1)
    d2_pair = _dist2_lin_seg(p0, p1, q0, q1)
    print(f"Direct segment-segment distance for argmin pair: {np.sqrt(d2_pair):.6e}")



    # %%
    
    p0 = np.array([-1.0, 0.0, 2.0])
    p1 = np.array([ 1.0, 0.0, 2.0])
    q0 = np.array([ 0.5, -1.0, 0.0])
    q1 = np.array([ 0.5,  1.0, 0.0])

    _dist2_lin_seg(p0,p1,q0,q1)

    # %%

    p0 = np.array([-1.0, 0.0, 0.0])
    p1 = np.array([ 1.0, 0.0, 0.0])
    q0 = np.array([-5.0, 0.0, 0.0])
    q1 = np.array([-1.5, 0.0, 0.0])

    _dist2_lin_seg(p0,p1,q0,q1)


# %%


    # ---------- PBC candidate auditor ----------
    def explain_candidate_miss_pbc(i, j,
                                p_starts, p_ends,
                                C, cell_size,
                                inflate_insert,  # e.g., r
                                inflate_query,   # e.g., r (+tiny eps)
                                rod_diameter):
        """
        Print a human-readable explanation of whether rods i and j
        should have been candidates in the PBC broad-phase.
        Requires your existing PBC helpers:
        - _segment_aabb_cell_ranges (PBC, with splitting & wrapping)
        - _cell_index (PBC wrapping)
        - _grid_params
        - _dist2_lin_seg_pbc
        """

        import numpy as _np

        # --- convenience: listify ranges, enumerate cells ---
        def _ranges_list(ridx, nr):
            return [(int(ridx[a,0]), int(ridx[a,1])) for a in range(nr)]

        def _cells_from_ranges(xr, nxr, yr, nyr, zr, nzr, nx, ny, nz):
            # enumerate all cell ids touched by the AABB
            cells = []
            for ax in range(nxr):
                ix0, ix1 = xr[ax]
                for ay in range(nyr):
                    iy0, iy1 = yr[ay]
                    for az in range(nzr):
                        iz0, iz1 = zr[az]
                        for ix in range(ix0, ix1+1):
                            for iy in range(iy0, iy1+1):
                                for iz in range(iz0, iz1+1):
                                    c = _cell_index(ix, iy, iz, nx, ny, nz)
                                    cells.append(c)
            return _np.array(sorted(set(cells)), dtype=_np.int64)

        def _ranges_overlap(a_list, b_list):
            for a0,a1 in a_list:
                for b0,b1 in b_list:
                    if max(a0,b0) <= min(a1,b1):  # inclusive top edge
                        return True
            return False

        # --- grid params ---
        L, nx, ny, nz, hx, hy, hz = _grid_params(C, cell_size)

        # --- ranges for "inserted" rod j (old), inflated by inflate_insert ---
        xr_j, nxr_j, yr_j, nyr_j, zr_j, nzr_j, *_ = _segment_aabb_cell_ranges(
            p_starts[j], p_ends[j], C, cell_size, inflate_insert
        )
        Xj = _ranges_list(xr_j, nxr_j)
        Yj = _ranges_list(yr_j, nyr_j)
        Zj = _ranges_list(zr_j, nzr_j)

        # --- ranges for "gather" rod i (new), inflated by inflate_query ---
        xr_i, nxr_i, yr_i, nyr_i, zr_i, nzr_i, *_ = _segment_aabb_cell_ranges(
            p_starts[i], p_ends[i], C, cell_size, inflate_query
        )
        Xi = _ranges_list(xr_i, nxr_i)
        Yi = _ranges_list(yr_i, nyr_i)
        Zi = _ranges_list(zr_i, nzr_i)

        # --- per-axis overlaps in index space ---
        ox = _ranges_overlap(Xi, Xj)
        oy = _ranges_overlap(Yi, Yj)
        oz = _ranges_overlap(Zi, Zj)

        # --- (optional) enumerate actual cells and intersect ---
        cells_i = _cells_from_ranges(Xi, len(Xi), Yi, len(Yi), Zi, len(Zi), nx, ny, nz)
        cells_j = _cells_from_ranges(Xj, len(Xj), Yj, len(Yj), Zj, len(Zj), nx, ny, nz)
        share_cells = _np.intersect1d(cells_i, cells_j).size > 0

        # --- narrow-phase truth ---
        D = rod_diameter
        d2 = _dist2_lin_seg_pbc(p_starts[i], p_ends[i], p_starts[j], p_ends[j], C)
        d = float(_np.sqrt(d2))

        # --- report ---
        print(f"[PBC auditor] pair (i={i}, j={j})")
        print(f"  true distance = {d:.6e}   threshold (D) = {D:.6e}   margin = {d-D:.6e}")
        print(f"  grid: nx,ny,nz = {nx},{ny},{nz}   cell_size={cell_size:.6g}   L={L:.6g}")
        print(f"  inflate_insert={inflate_insert:.6g}   inflate_query={inflate_query:.6g}")

        def _fmt_ranges(lbl, R):
            if len(R)==0:
                return f"{lbl}: []"
            return f"{lbl}: " + " U ".join([f"[{a},{b}]" for a,b in R])

        print(_fmt_ranges("  Xi", Xi))
        print(_fmt_ranges("  Xj", Xj))
        print(_fmt_ranges("  Yi", Yi))
        print(_fmt_ranges("  Yj", Yj))
        print(_fmt_ranges("  Zi", Zi))
        print(_fmt_ranges("  Zj", Zj))

        print(f"  per-axis overlap: X={ox}  Y={oy}  Z={oz}")
        print(f"  share any cell?   {share_cells}")

        if d < D and not share_cells:
            # Why did it miss?
            # 1) No overlap on some axis -> AABB index ranges disjoint -> inflate sum too small OR boundary aliasing.
            missing_axes = []
            if not ox: missing_axes.append('X')
            if not oy: missing_axes.append('Y')
            if not oz: missing_axes.append('Z')
            if missing_axes:
                print(f"  REASON: No index overlap along axis/axes: {','.join(missing_axes)}")
                print("          => either (a) inflate_insert + inflate_query < D (capsule overlap not covered),")
                print("             or (b) cell-plane aliasing (fix with inclusive top-edge + tiny grid_pad).")
            else:
                # Overlap per-axis says True but cell sets still disjoint (rare). Usually aliasing on boundaries.
                print("  REASON: Per-axis ranges overlap but cell sets did not intersect -> boundary aliasing.")
                print("          Ensure _to_idx_range uses inclusive top edge and add a small grid_pad.")

# %%
    # ---------- non-PBC candidate auditor ----------
    def explain_candidate_miss_npbc(i, j,
                                    p_starts, p_ends,
                                    C, cell_size,
                                    inflate_insert,   # e.g., r
                                    inflate_query,    # e.g., r (+eps)
                                    rod_diameter):
        """
        Print a human-readable explanation for the non-PBC broad-phase.
        Requires your non-PBC helpers:
        - _segment_aabb_cell_ranges_npbc (no splitting, clamped)
        - _grid_params
        - _dist2_lin_seg
        """
        import numpy as _np

        def _ranges_npbc(p0, p1, infl):
            xr, nxr, yr, nyr, zr, nzr, nx, ny, nz = _segment_aabb_cell_ranges_npbc(p0, p1, C, cell_size, infl)
            assert nxr in (0,1) and nyr in (0,1) and nzr in (0,1)
            if nxr == 0 or nyr == 0 or nzr == 0:
                return [], [], [], nx, ny, nz
            Xi = [(int(xr[0,0]), int(xr[0,1]))]
            Yi = [(int(yr[0,0]), int(yr[0,1]))]
            Zi = [(int(zr[0,0]), int(zr[0,1]))]
            return Xi, Yi, Zi, nx, ny, nz

        def _overlap_1(a, b):
            if len(a)==0 or len(b)==0: return False
            a0,a1 = a[0]; b0,b1 = b[0]
            return max(a0,b0) <= min(a1,b1)

        Xi, Yi, Zi, nx, ny, nz = _ranges_npbc(p_starts[i], p_ends[i], inflate_query)
        Xj, Yj, Zj, *_ = _ranges_npbc(p_starts[j], p_ends[j], inflate_insert)

        ox = _overlap_1(Xi, Xj)
        oy = _overlap_1(Yi, Yj)
        oz = _overlap_1(Zi, Zj)

        D = rod_diameter
        d2 = _dist2_lin_seg(p_starts[i], p_ends[i], p_starts[j], p_ends[j])
        d = float(_np.sqrt(d2))

        print(f"[NPBC auditor] pair (i={i}, j={j})")
        print(f"  true distance = {d:.6e}   threshold (D) = {D:.6e}   margin = {d-D:.6e}")
        print(f"  grid: nx,ny,nz = {nx},{ny},{nz}   cell_size={cell_size:.6g}   L={2.0*C:.6g}")
        print(f"  inflate_insert={inflate_insert:.6g}   inflate_query={inflate_query:.6g}")

        def _fmt(lbl, R):
            return f"{lbl}: []" if len(R)==0 else f"{lbl}: [{R[0][0]},{R[0][1]}]"

        print(_fmt("  Xi", Xi)); print(_fmt("  Xj", Xj))
        print(_fmt("  Yi", Yi)); print(_fmt("  Yj", Yj))
        print(_fmt("  Zi", Zi)); print(_fmt("  Zj", Zj))
        print(f"  per-axis overlap: X={ox}  Y={oy}  Z={oz}")

        if d < D and not (ox and oy and oz):
            missing_axes = []
            if not ox: missing_axes.append('X')
            if not oy: missing_axes.append('Y')
            if not oz: missing_axes.append('Z')
            print(f"  REASON: No index overlap along axis/axes: {','.join(missing_axes)}")
            print("          -> Either inflate sum < D, or the overlap region lies OUTSIDE the container.")
            print("             (non-PBC clamps AABBs to [0,L]; consider a halo of width D if needed.)")


# %%
    # Build endpoints from your centroids q (non-PBC version)
    p0, p1 = _endpoints_from_q_centroid(q[:placed], rod_length)

    r = 0.5 * rod_diameter
    inflate_insert = r
    inflate_query  = r + 1e-12

    explain_candidate_miss_npbc(i_min, j_min,
                                p0, p1,
                                C, cell_size,
                                inflate_insert, inflate_query,
                                rod_diameter)

