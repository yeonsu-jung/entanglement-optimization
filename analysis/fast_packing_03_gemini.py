# %%
import numpy as np
from numba import njit, int64, float64

# A small constant for floating point comparisons at boundaries
EPS = 1e-12

# =============================================================================
# Helper Functions for Periodic Boundary Conditions (PBC) & Math
# =============================================================================

@njit(inline='always')
def _box_L(C):
    """Calculates the full box length L from the half-length C."""
    return 2.0 * C

@njit(inline='always')
def _min_image_vec(d, L):
    """Calculates the displacement vector in the minimum image convention."""
    return d - L * np.round(d / L)

@njit(inline='always')
def _wrap_vec_to_box(x, L):
    """Wraps a coordinate vector into the periodic box [0, L)^3."""
    return x - L * np.floor(x / L)

# =============================================================================
# Grid Helper Functions (for Spatial Hashing)
# =============================================================================

@njit
def _grid_params(C, cell_size):
    """Computes grid dimensions and cell spacings."""
    L = _box_L(C)
    nx = max(1, int(np.floor(L / cell_size)))
    ny = nx; nz = nx
    hx = L / nx; hy = L / ny; hz = L / nz
    return L, nx, ny, nz, hx, hy, hz

@njit(inline='always')
def _cell_index(ix, iy, iz, nx, ny, nz):
    """Computes a 1D cell index from 3D grid indices with PBC wrapping."""
    ix_w = ix % nx; iy_w = iy % ny; iz_w = iz % nz
    return ix_w + nx * (iy_w + ny * iz_w)

@njit
def _axis_split(a0, a1, L):
    """Splits a 1D interval [a0, a1] into up to two intervals inside [0, L)."""
    out = np.empty(4, dtype=float64)
    if a0 < 0.0:
        out[0] = 0.0;     out[1] = min(a1, L - EPS)
        out[2] = a0 + L;  out[3] = L
        return out, 2
    elif a1 >= L:
        out[0] = a0;      out[1] = L
        out[2] = 0.0;     out[3] = a1 - L
        return out, 2
    else:
        out[0] = a0;      out[1] = a1
        return out, 1

@njit(inline='always')
def _to_idx_range(lo, hi, h, n):
    """Converts a spatial range [lo, hi] to a grid index range [i0, i1]."""
    i0 = int(np.floor(lo / h))
    i1 = int(np.floor((hi - EPS) / h))
    i0 = max(0, min(n - 1, i0))
    i1 = max(0, min(n - 1, i1))
    return i0, i1

@njit
def _get_grid_ranges(box_smin, box_smax, C, cell_size):
    """Helper to convert a coordinate-space box into grid index ranges."""
    L, nx, ny, nz, hx, hy, hz = _grid_params(C, cell_size)
    xr, nxr = _axis_split(box_smin[0], box_smax[0], L)
    yr, nyr = _axis_split(box_smin[1], box_smax[1], L)
    zr, nzr = _axis_split(box_smin[2], box_smax[2], L)

    xr_idx = np.empty((2, 2), dtype=int64)
    yr_idx = np.empty((2, 2), dtype=int64)
    zr_idx = np.empty((2, 2), dtype=int64)

    for i in range(nxr):
        xr_idx[i,0],xr_idx[i,1] = _to_idx_range(xr[2*i], xr[2*i+1], hx, nx)
    for i in range(nyr):
        yr_idx[i,0],yr_idx[i,1] = _to_idx_range(yr[2*i], yr[2*i+1], hy, ny)
    for i in range(nzr):
        zr_idx[i,0],zr_idx[i,1] = _to_idx_range(zr[2*i], zr[2*i+1], hz, nz)
        
    return xr_idx, nxr, yr_idx, nyr, zr_idx, nzr, nx, ny, nz

# =============================================================================
# Core Grid Operations (Insert & Gather)
# =============================================================================

@njit
def _grid_insert_segment(rod_id, p0, p1, C, cell_size, inflate,
                         cell_head, link_idx, link_next,
                         entry_counter, max_entries):
    """Inserts a rod_id into cells overlapped by its tight capsule AABB."""
    L = _box_L(C)
    q0 = _wrap_vec_to_box(p0 + C, L)
    q1 = q0 + (p1 - p0)
    smin = np.minimum(q0, q1) - inflate
    smax = np.maximum(q0, q1) + inflate
    
    ranges = _get_grid_ranges(smin, smax, C, cell_size)
    xr_idx, nxr, yr_idx, nyr, zr_idx, nzr, nx, ny, nz = ranges

    for ax in range(nxr):
        ix0, ix1 = xr_idx[ax, 0], xr_idx[ax, 1]
        for ay in range(nyr):
            iy0, iy1 = yr_idx[ay, 0], yr_idx[ay, 1]
            for az in range(nzr):
                iz0, iz1 = zr_idx[az, 0], zr_idx[az, 1]
                for ix in range(ix0, ix1 + 1):
                    for iy in range(iy0, iy1 + 1):
                        for iz in range(iz0, iz1 + 1):
                            c = _cell_index(ix, iy, iz, nx, ny, nz)
                            k = entry_counter[0]
                            if k >= max_entries: return False
                            link_idx[k] = rod_id
                            link_next[k] = cell_head[c]
                            cell_head[c] = k
                            entry_counter[0] = k + 1
    return True

@njit
def _grid_gather_candidates(p0, p1, C, cell_size, rod_length, rod_diameter,
                            cell_head, link_idx, link_next,
                            seen_stamp, seen, out_buf):
    """Gathers candidates using a simple, large, safe search box."""
    L = _box_L(C)
    # Simplified broadphase: Use a large query box centered on the rod's midpoint.
    # The size of the box guarantees it contains the entire rod capsule.
    p_mid = (p0 + p1) / 2.0
    half_size = (rod_length / 2.0) + (rod_diameter / 2.0)
    
    mid_q = _wrap_vec_to_box(p_mid + C, L)
    smin = mid_q - half_size
    smax = mid_q + half_size
    
    ranges = _get_grid_ranges(smin, smax, C, cell_size)
    xr_idx, nxr, yr_idx, nyr, zr_idx, nzr, nx, ny, nz = ranges
    count = 0

    for ax in range(nxr):
        ix0, ix1 = xr_idx[ax, 0], xr_idx[ax, 1]
        for ay in range(nyr):
            iy0, iy1 = yr_idx[ay, 0], yr_idx[ay, 1]
            for az in range(nzr):
                iz0, iz1 = zr_idx[az, 0], zr_idx[az, 1]
                for ix in range(ix0, ix1 + 1):
                    for iy in range(iy0, iy1 + 1):
                        for iz in range(iz0, iz1 + 1):
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

# =============================================================================
# Narrow-Phase: Segment-Segment Distance
# =============================================================================

@njit(inline='always')
def _dist2_lin_seg(p0, p1, q0, q1):
    """Squared distance between two line segments in R^3 (no PBC)."""
    u = p1 - p0; v = q1 - q0; w0 = p0 - q0
    uu = np.dot(u,u); vv = np.dot(v,v); uv = np.dot(u,v)
    wu = np.dot(w0,u); wv = np.dot(w0,v)
    D = uu*vv - uv*uv
    s=0.0; t=0.0
    if abs(D) < EPS:
        t = -wv/vv if vv > EPS else 0.0
    else:
        s = (uv*wv - vv*wu)/D
        t = (uu*wv - uv*wu)/D
    s_clamped = max(0.0, min(1.0, s))
    t_clamped = (s_clamped*uv + wv)/vv if vv > EPS else 0.5
    t_clamped = max(0.0, min(1.0, t_clamped))
    s_recalc = (-wu + t_clamped*uv)/uu if uu > EPS else 0.5
    if t < 0.0 or t > 1.0:
        s_clamped = max(0.0, min(1.0, s_recalc))
    dx = w0[0] + s_clamped*u[0] - t_clamped*v[0]
    dy = w0[1] + s_clamped*u[1] - t_clamped*v[1]
    dz = w0[2] + s_clamped*u[2] - t_clamped*v[2]
    return dx*dx + dy*dy + dz*dz

@njit(inline='always')
def _dist2_lin_seg_pbc(a0, a1, b0, b1, C):
    """Squared distance between two segments under PBC."""
    L = _box_L(C)
    b0u = a0 + _min_image_vec(b0 - a0, L)
    b1u = b0u + (b1 - b0)
    return _dist2_lin_seg(a0, a1, b0u, b1u)

# =============================================================================
# Main Routines: Generation and Verification
# =============================================================================

@njit
def _endpoints_from_q(q, rod_length):
    n = q.shape[0]
    p0 = np.empty((n,3), dtype=float64); p1 = np.empty((n,3), dtype=float64)
    for i in range(n):
        x,y,z,phi,theta = q[i,:]
        sx=np.sin(phi); cx=np.cos(phi); ct=np.cos(theta); st=np.sin(theta)
        ux=sx*ct; uy=sx*st; uz=cx
        p0[i,0]=x; p0[i,1]=y; p0[i,2]=z
        p1[i,0]=x+rod_length*ux; p1[i,1]=y+rod_length*uy; p1[i,2]=z+rod_length*uz
    return p0, p1

@njit
def create_nonintersecting_random_rods(
    num_rods, rod_diameter, container_half_size, max_attempts, rod_length,
    cell_size, grid_capacity_multiplier, seed
):
    C = float(container_half_size)
    if cell_size <= 0.0: cell_size = rod_length

    L, nx, ny, nz, hx, hy, hz = _grid_params(C, cell_size)
    max_entries = int(grid_capacity_multiplier * num_rods)
    cell_head = np.full(nx*ny*nz, -1, dtype=int64)
    link_idx = np.empty(max_entries, dtype=int64)
    link_next = np.empty(max_entries, dtype=int64)
    entry_counter = np.zeros(1, dtype=int64)

    q = np.zeros((num_rods, 5), dtype=float64)
    p_starts = np.zeros((num_rods, 3), dtype=float64)
    p_ends = np.zeros((num_rods, 3), dtype=float64)
    seen = np.full(num_rods, -1, dtype=int64)
    out_buf = np.empty(num_rods, dtype=int64)

    diam2 = rod_diameter * rod_diameter
    capsule_radius = rod_diameter / 2.0

    np.random.seed(seed)
    placed = 0
    for i in range(num_rods):
        created = False
        attempts = 0
        while not created and attempts < max_attempts:
            x,y,z = (np.random.random(3)*2.0-1.0)*C
            phi = np.random.random()*np.pi
            theta = np.random.random()*2.0*np.pi
            p0 = np.array([x,y,z],dtype=float64)
            d = np.array([np.sin(phi)*np.cos(theta),
                          np.sin(phi)*np.sin(theta), np.cos(phi)], dtype=float64)
            p1 = p0 + rod_length*d

            n_cand = _grid_gather_candidates(
                p0, p1, C, cell_size, rod_length, rod_diameter,
                cell_head, link_idx, link_next, i, seen, out_buf
            )

            intersect = False
            for k in range(n_cand):
                j = out_buf[k]
                d2 = _dist2_lin_seg_pbc(p0, p1, p_starts[j], p_ends[j], C)
                if d2 < diam2:
                    intersect = True
                    break
            
            if not intersect:
                q[i,:] = [x,y,z,phi,theta]
                p_starts[i,:]=p0; p_ends[i,:]=p1
                ok = _grid_insert_segment(
                    i, p0, p1, C, cell_size, capsule_radius,
                    cell_head, link_idx, link_next, entry_counter, max_entries
                )
                if not ok:
                    print("Error: Exceeded grid capacity during generation.")
                    return q[:i,:], i
                created = True
                placed += 1
            attempts += 1
        if not created:
            print(f"Warning: Failed to place rod {i} after {max_attempts} attempts.")
            return q[:placed,:], placed
    return q, placed

@njit
def verify_min_distance_pbc(
    q, rod_length, C, rod_diameter, cell_size, grid_capacity_multiplier=48
):
    p_starts, p_ends = _endpoints_from_q(q, rod_length)
    n = p_starts.shape[0]
    if n < 2: return np.inf, -1, -1, 0

    max_entries = int(grid_capacity_multiplier * n)
    L,nx,ny,nz,hx,hy,hz = _grid_params(C, cell_size)
    cell_head = np.full(nx*ny*nz, -1, dtype=int64)
    link_idx = np.empty(max_entries, dtype=int64)
    link_next = np.empty(max_entries, dtype=int64)
    entry_counter = np.zeros(1, dtype=int64)
    capsule_radius = rod_diameter / 2.0
    
    for i in range(n):
        ok = _grid_insert_segment(
            i, p_starts[i], p_ends[i], C, cell_size, capsule_radius,
            cell_head, link_idx, link_next, entry_counter, max_entries
        )
        if not ok:
            print("Error: Exceeded grid capacity during verification.")
            return np.nan, -1, -1, -1

    seen = np.full(n, -1, dtype=int64)
    out_buf = np.empty(n, dtype=int64)
    diam2 = rod_diameter * rod_diameter
    dmin2 = np.inf; imin=-1; jmin=-1; violations = 0

    for i in range(n):
        n_cand = _grid_gather_candidates(
            p_starts[i], p_ends[i], C, cell_size, rod_length, rod_diameter,
            cell_head, link_idx, link_next, i, seen, out_buf
        )
        for k in range(n_cand):
            j = out_buf[k]
            if j <= i: continue
            d2 = _dist2_lin_seg_pbc(p_starts[i],p_ends[i],p_starts[j],p_ends[j],C)
            if d2 < dmin2:
                dmin2 = d2; imin = i; jmin = j
            if d2 < diam2: violations += 1
    return np.sqrt(dmin2), imin, jmin, violations

# =============================================================================
# Main Execution Block
# =============================================================================
if __name__ == "__main__":
    C = 6.0
    rod_length = 1.0
    rod_diameter = rod_length / 100.0
    cell_size = rod_length + rod_diameter
    volume_fraction = 0.1
    rod_volume = (np.pi/4)*rod_diameter**2*rod_length
    box_volume = (2*C)**3
    N = int(volume_fraction * box_volume / rod_volume)
    N = 20000
    
    print(f"Container half-size (C): {C}")
    print(f"Rod length: {rod_length}, Rod diameter: {rod_diameter:.4f}")
    print(f"Targeting {N} rods for a volume fraction of {volume_fraction:.2f}.")
    print(f"Using simplified broadphase with cell size: {cell_size:.4f}")

    q, placed = create_nonintersecting_random_rods(
        num_rods=N,
        rod_diameter=rod_diameter,
        container_half_size=C,
        max_attempts=10000,
        rod_length=rod_length,
        cell_size=cell_size,
        grid_capacity_multiplier=200, # Increased due to simpler, larger query
        seed=12345
    )
    print(f"\nPlaced {placed}/{N} rods successfully.")

    if placed > 1:
        print("\n--- Verifying final configuration ---")
        dmin, i_min, j_min, num_viol = verify_min_distance_pbc(
            q[:placed], rod_length, C, rod_diameter, cell_size
        )

        gap = dmin - rod_diameter
        print(f"Minimum distance found: {dmin:.6e}")
        print(f"Rod diameter threshold: {rod_diameter:.6e}")
        print(f"Closest gap: {gap:.6e}")
        print(f"Closest pair indices: (i={i_min}, j={j_min})")
        print(f"Number of violations (dist < diameter): {num_viol} (should be 0)")