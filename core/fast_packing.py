# %%
"""
Non-PBC random rod placement with Numba broad-phase grid + narrow-phase segment distance.

Key features:
- Places spherocylinders by sampling a centroid in [-C, C]^3; endpoints may extend outside the box.
- Broad-phase: uniform grid with per-segment inflated AABB (non-PBC, clamped).
- Narrow-phase: robust segment–segment distance (no PBC).
- Verifiers to check global minimum centerline distance and count of violations.
- Optional "auditor" functions (non-JIT) to explain why a candidate pair might be missed.
"""

from __future__ import annotations

import os
import sys
import time
import numpy as np
from numba import njit, int64, float64
from utils import setup_directories

# Add ../core to sys.path (for optional demo/plot helpers users may have)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "core")))

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

_F64 = float64
_I64 = int64

_EPS_DIST = 1e-12       # for distance math (parallel checks / denominators)
_GRID_PAD_MIN = 1e-12   # minimum absolute padding in box coords to avoid plane aliasing
_GRID_PAD_REL = 1e-9    # relative padding w.r.t. cell size (used with max above)

# ---------------------------------------------------------------------
# Grid helpers (non-PBC)
# ---------------------------------------------------------------------

@njit
def _grid_params(C: float, cell_size: float):
    """
    Compute grid parameters for a cubic box [-C, C]^3 mapped to [0, 2C]^3.
    Returns: L, nx, ny, nz, hx, hy, hz
    """
    L = 2.0 * C
    nx = max(1, int(np.floor(L / cell_size)))
    ny = nx
    nz = nx
    hx = L / nx
    hy = L / ny
    hz = L / nz
    return L, nx, ny, nz, hx, hy, hz


@njit(inline="always")
def _to_idx_range(lo: float, hi: float, h: float, n: int):
    """
    Map a continuous interval [lo, hi] in [0, L] to discrete cell indices [i0, i1],
    inclusive on the top edge to be robust to cell-plane alignment.
    """
    i0 = int(np.floor(lo / h))
    i1 = int(np.floor(hi / h))
    if i0 < 0:
        i0 = 0
    if i1 < 0:
        i1 = 0
    if i0 > n - 1:
        i0 = n - 1
    if i1 > n - 1:
        i1 = n - 1
    if i1 < i0:
        t = i0
        i0 = i1
        i1 = t
    return i0, i1


# ---------------------------------------------------------------------
# Non-PBC AABB → cell ranges (clamped; no wrapping)
# ---------------------------------------------------------------------

@njit
def _segment_aabb_cell_ranges_npbc(p0, p1, C: float, cell_size: float, inflate: float):
    """
    Build an inflated AABB for a segment in box coords [0, L]^3 and convert to cell index ranges.
    Non-PBC: clamp to container [0, L]^3. Returns (xr_idx, nxr, yr_idx, nyr, zr_idx, nzr, nx, ny, nz).
    If clamped AABB is fully outside, returns zero-length ranges (nxr/nyr/nzr == 0).
    """
    L, nx, ny, nz, hx, hy, hz = _grid_params(C, cell_size)

    # map world coords [-C, C] -> [0, L]
    q0 = p0 + C
    q1 = p1 + C

    smin = np.minimum(q0, q1) - inflate
    smax = np.maximum(q0, q1) + inflate

    # tiny pad to avoid missing pairs on cell planes
    grid_pad = max(_GRID_PAD_MIN, _GRID_PAD_REL * cell_size)
    smin -= grid_pad
    smax += grid_pad

    # trivial reject if wholly out of bounds
    if (smax[0] <= 0.0) or (smax[1] <= 0.0) or (smax[2] <= 0.0):
        return (np.zeros((1, 2), _I64), 0,
                np.zeros((1, 2), _I64), 0,
                np.zeros((1, 2), _I64), 0,
                nx, ny, nz)
    if (smin[0] >= L) or (smin[1] >= L) or (smin[2] >= L):
        return (np.zeros((1, 2), _I64), 0,
                np.zeros((1, 2), _I64), 0,
                np.zeros((1, 2), _I64), 0,
                nx, ny, nz)

    # clamp
    smin0 = 0.0 if smin[0] < 0.0 else smin[0]
    smin1 = 0.0 if smin[1] < 0.0 else smin[1]
    smin2 = 0.0 if smin[2] < 0.0 else smin[2]
    smax0 = L if smax[0] > L else smax[0]
    smax1 = L if smax[1] > L else smax[1]
    smax2 = L if smax[2] > L else smax[2]

    xr_idx = np.empty((1, 2), dtype=_I64)
    yr_idx = np.empty((1, 2), dtype=_I64)
    zr_idx = np.empty((1, 2), dtype=_I64)

    ix0, ix1 = _to_idx_range(smin0, smax0, hx, nx)
    iy0, iy1 = _to_idx_range(smin1, smax1, hy, ny)
    iz0, iz1 = _to_idx_range(smin2, smax2, hz, nz)

    xr_idx[0, 0], xr_idx[0, 1] = ix0, ix1
    yr_idx[0, 0], yr_idx[0, 1] = iy0, iy1
    zr_idx[0, 0], zr_idx[0, 1] = iz0, iz1
    return xr_idx, 1, yr_idx, 1, zr_idx, 1, nx, ny, nz


@njit
def _grid_insert_segment_npbc(rod_id: int,
                              p0, p1,
                              C: float, cell_size: float, inflate: float,
                              cell_head, link_idx, link_next,
                              nnz_ref, max_entries: int) -> bool:
    """
    Insert a segment into all touched cells (non-PBC). Returns False if capacity exceeded.
    """
    xr_idx, nxr, yr_idx, nyr, zr_idx, nzr, nx, ny, nz = _segment_aabb_cell_ranges_npbc(
        p0, p1, C, cell_size, inflate
    )
    if nxr == 0 or nyr == 0 or nzr == 0:
        # completely out of box; nothing to insert (OK)
        return True

    for ix in range(xr_idx[0, 0], xr_idx[0, 1] + 1):
        for iy in range(yr_idx[0, 0], yr_idx[0, 1] + 1):
            for iz in range(zr_idx[0, 0], zr_idx[0, 1] + 1):
                c = ix + nx * (iy + ny * iz)
                k = nnz_ref[0]
                if k >= max_entries:
                    return False
                link_idx[k] = rod_id
                link_next[k] = cell_head[c]
                cell_head[c] = k
                nnz_ref[0] = k + 1
    return True


@njit
def _grid_gather_candidates_npbc(p0, p1,
                                 C: float, cell_size: float, inflate: float,
                                 cell_head, link_idx, link_next,
                                 seen_stamp: int, seen, out_buf) -> int:
    """
    Gather candidate rod indices for a query segment (non-PBC). Returns count in out_buf.
    """
    xr_idx, nxr, yr_idx, nyr, zr_idx, nzr, nx, ny, nz = _segment_aabb_cell_ranges_npbc(
        p0, p1, C, cell_size, inflate
    )
    if nxr == 0 or nyr == 0 or nzr == 0:
        return 0

    count = 0
    for ix in range(xr_idx[0, 0], xr_idx[0, 1] + 1):
        for iy in range(yr_idx[0, 0], yr_idx[0, 1] + 1):
            for iz in range(zr_idx[0, 0], zr_idx[0, 1] + 1):
                c = ix + nx * (iy + ny * iz)
                e = cell_head[c]
                while e != -1:
                    j = link_idx[e]
                    if seen[j] != seen_stamp:
                        seen[j] = seen_stamp
                        out_buf[count] = j
                        count += 1
                    e = link_next[e]
    return count


# ---------------------------------------------------------------------
# Narrow-phase (segment–segment distance; no PBC)
# ---------------------------------------------------------------------

@njit(inline="always")
def _dist2_lin_seg(p0, p1, q0, q1, eps: float = _EPS_DIST) -> float:
    """
    Squared distance between two 3D segments [p0,p1], [q0,q1] (no PBC).
    Robust for nearly parallel segments.
    """
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

    # If nearly parallel, project onto v
    if abs(D) < eps:
        s = 0.0
        t = -wv / (vv if vv >= eps else 1.0)

    # clamp s ∈ [0,1]
    if s < 0.0:
        s = 0.0
    elif s > 1.0:
        s = 1.0

    # recompute t from clamped s
    t = (s * uv + wv) / (vv if vv >= eps else 1.0)
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0

    # final best s from clamped t, unless t is strictly interior
    su = (-wu + t * uv) / (uu if uu >= eps else 1.0)
    if not (t > 0.000001 and t < 0.999999):
        if su < 0.0:
            s = 0.0
        elif su > 1.0:
            s = 1.0
        else:
            s = su

    dx = w0[0] + s*u[0] - t*v[0]
    dy = w0[1] + s*u[1] - t*v[1]
    dz = w0[2] + s*u[2] - t*v[2]
    return dx*dx + dy*dy + dz*dz


# ---------------------------------------------------------------------
# Main placer (non-PBC; centroid inside the box)
# ---------------------------------------------------------------------

@njit
def create_nonintersecting_random_rods_contained_npbc_numba(
    num_rods: int,
    rod_diameter: float,
    container_size: float,        # C (half side); box is [-C, C]^3
    max_attempts: int,
    rod_length: float,
    cell_size: float,
    grid_capacity_multiplier: int,  # 48..96 typical
    seed: int
):
    """
    Place up to `num_rods` spherocylinders with centerlines of length `rod_length` and
    diameter `rod_diameter`, such that centerline–centerline distance ≥ diameter.

    Returns:
        q      : (placed, 5) array of [cx, cy, cz, phi, theta] per rod
        placed : number of successfully placed rods
    """
    C = float(container_size)
    if cell_size <= 0.0:
        cell_size = rod_length

    # grid storage
    L, nx, ny, nz, hx, hy, hz = _grid_params(C, cell_size)
    num_cells = nx * ny * nz
    max_entries = int(grid_capacity_multiplier * num_rods)

    cell_head = np.empty(num_cells, dtype=_I64); cell_head[:] = -1
    link_idx  = np.empty(max_entries, dtype=_I64)
    link_next = np.empty(max_entries, dtype=_I64)
    nnz_ref   = np.zeros(1, dtype=_I64)

    # outputs & temporaries
    q = np.zeros((num_rods, 5), dtype=_F64)    # [cx, cy, cz, phi, theta]
    p_starts = np.zeros((num_rods, 3), dtype=_F64)
    p_ends   = np.zeros((num_rods, 3), dtype=_F64)
    seen     = np.full(num_rods, -1, dtype=_I64)
    out_buf  = np.empty(num_rods, dtype=_I64)

    visit_stamp = np.int64(1)

    r = 0.5 * rod_diameter
    diam2 = (2.0*r) * (2.0*r)
    inflate_insert = r
    inflate_query  = r + _EPS_DIST

    # RNG
    np.random.seed(seed)

    placed = 0
    halfL = 0.5 * rod_length

    total_attempts = 0
    for i in range(num_rods):
        created = False
        attempts = 0

        while (not created) and (attempts < max_attempts):
            # sample centroid uniformly in the box; random orientation on S2
            cx = (np.random.random()*2.0 - 1.0) * C
            cy = (np.random.random()*2.0 - 1.0) * C
            cz = (np.random.random()*2.0 - 1.0) * C
            phi   = np.random.random() * np.pi
            theta = np.random.random() * (2.0 * np.pi)

            u = np.array([np.sin(phi)*np.cos(theta),
                          np.sin(phi)*np.sin(theta),
                          np.cos(phi)], dtype=_F64)

            # endpoints (may extend outside the box; centroid stays inside)
            p0 = np.array([cx, cy, cz], dtype=_F64) - halfL * u
            p1 = p0 + rod_length * u

            # gather candidates via inflated AABB
            seen_stamp = visit_stamp
            visit_stamp += 1
            if visit_stamp < 0:          # wraparound safety
                seen[:] = -1
                visit_stamp = np.int64(1)

            n_cand = _grid_gather_candidates_npbc(
                p0, p1, C, cell_size, inflate_query,
                cell_head, link_idx, link_next,
                seen_stamp, seen, out_buf
            )

            # narrow-phase check
            collide = False
            for k in range(n_cand):
                j = out_buf[k]
                d2 = _dist2_lin_seg(p0, p1, p_starts[j], p_ends[j])
                if d2 < diam2:
                    collide = True
                    break

            if not collide:
                # accept & insert
                q[i, 0] = cx; q[i, 1] = cy; q[i, 2] = cz; q[i, 3] = phi; q[i, 4] = theta
                p_starts[i, :] = p0
                p_ends[i,   :] = p1
                ok = _grid_insert_segment_npbc(
                    i, p0, p1, C, cell_size, inflate_insert,
                    cell_head, link_idx, link_next,
                    nnz_ref, max_entries
                )
                if not ok:
                    return q[:i, :], i, total_attempts
                created = True

            attempts += 1

        if not created:
            return q[:i, :], i, total_attempts

        placed += 1
        total_attempts += attempts

    return q, placed, total_attempts


# ---------------------------------------------------------------------
# Verifiers (non-PBC)
# ---------------------------------------------------------------------

@njit
def _endpoints_from_q_centroid(q, rod_length: float):
    """
    Convert q = [cx, cy, cz, phi, theta] to endpoints arrays p0, p1.
    """
    n = q.shape[0]
    p0 = np.empty((n, 3), dtype=_F64)
    p1 = np.empty((n, 3), dtype=_F64)
    halfL = 0.5 * rod_length
    for i in range(n):
        cx, cy, cz, phi, theta = q[i, 0], q[i, 1], q[i, 2], q[i, 3], q[i, 4]
        ux = np.sin(phi) * np.cos(theta)
        uy = np.sin(phi) * np.sin(theta)
        uz = np.cos(phi)
        p0[i, 0] = cx - halfL * ux; p0[i, 1] = cy - halfL * uy; p0[i, 2] = cz - halfL * uz
        p1[i, 0] = cx + halfL * ux; p1[i, 1] = cy + halfL * uy; p1[i, 2] = cz + halfL * uz
    return p0, p1


@njit
def verify_min_distance_npbc(p_starts, p_ends,
                             C: float, rod_diameter: float,
                             cell_size: float, grid_capacity_multiplier: int = 48):
    """
    Compute global min centerline distance and count of violations (< D) using the same grid.
    Returns: (dmin, i_min, j_min, num_viol)
    """
    n = p_starts.shape[0]
    r = 0.5 * rod_diameter
    D2 = (2.0 * r) * (2.0 * r)

    # grid build
    L, nx, ny, nz, hx, hy, hz = _grid_params(C, cell_size)
    num_cells = nx * ny * nz
    max_entries = int(grid_capacity_multiplier * n)

    cell_head = np.empty(num_cells, dtype=_I64); cell_head[:] = -1
    link_idx  = np.empty(max_entries, dtype=_I64)
    link_next = np.empty(max_entries, dtype=_I64)
    nnz_ref   = np.zeros(1, dtype=_I64)

    # insert with inflate r
    for i in range(n):
        ok = _grid_insert_segment_npbc(
            i, p_starts[i], p_ends[i], C, cell_size, r,
            cell_head, link_idx, link_next, nnz_ref, max_entries
        )
        if not ok:
            return np.nan, -1, -1, -1

    # query with inflate r
    seen = np.full(n, -1, dtype=_I64)
    out_buf = np.empty(n, dtype=_I64)

    dmin2 = 1e300
    imin = -1
    jmin = -1
    viol = 0

    for i in range(n):
        n_cand = _grid_gather_candidates_npbc(
            p_starts[i], p_ends[i], C, cell_size, r,
            cell_head, link_idx, link_next,
            i, seen, out_buf
        )
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
def verify_from_q_npbc(q, rod_length: float, C: float, rod_diameter: float,
                       cell_size: float, grid_capacity_multiplier: int = 48):
    p0, p1 = _endpoints_from_q_centroid(q, rod_length)
    return verify_min_distance_npbc(p0, p1, C, rod_diameter, cell_size, grid_capacity_multiplier)


# ---------------------------------------------------------------------
# Optional: auditors (non-JIT, for debugging candidate misses)
# ---------------------------------------------------------------------

def explain_candidate_miss_npbc(i: int, j: int,
                                p_starts: np.ndarray, p_ends: np.ndarray,
                                C: float, cell_size: float,
                                inflate_insert: float, inflate_query: float,
                                rod_diameter: float):
    """
    Human-readable explanation for non-PBC broad-phase overlap decisions.
    Requires:
      - _segment_aabb_cell_ranges_npbc
      - _grid_params
      - _dist2_lin_seg
    """
    import numpy as _np

    def _ranges_npbc(p0, p1, infl):
        xr, nxr, yr, nyr, zr, nzr, nx, ny, nz = _segment_aabb_cell_ranges_npbc(p0, p1, C, cell_size, infl)
        assert nxr in (0, 1) and nyr in (0, 1) and nzr in (0, 1)
        if nxr == 0 or nyr == 0 or nzr == 0:
            return [], [], [], nx, ny, nz
        Xi = [(int(xr[0, 0]), int(xr[0, 1]))]
        Yi = [(int(yr[0, 0]), int(yr[0, 1]))]
        Zi = [(int(zr[0, 0]), int(zr[0, 1]))]
        return Xi, Yi, Zi, nx, ny, nz

    def _overlap_1(a, b):
        if len(a) == 0 or len(b) == 0:
            return False
        a0, a1 = a[0]
        b0, b1 = b[0]
        return max(a0, b0) <= min(a1, b1)

    Xi, Yi, Zi, nx, ny, nz = _ranges_npbc(p_starts[i], p_ends[i], inflate_query)
    Xj, Yj, Zj, *_ = _ranges_npbc(p_starts[j], p_ends[j], inflate_insert)

    ox = _overlap_1(Xi, Xj)
    oy = _overlap_1(Yi, Yj)
    oz = _overlap_1(Zi, Zj)

    D = rod_diameter
    d2 = _dist2_lin_seg(p_starts[i], p_ends[i], p_starts[j], p_ends[j])
    d = float(_np.sqrt(d2))

    print(f"[NPBC auditor] pair (i={i}, j={j})")
    print(f"  true distance = {d:.6e}   threshold (D) = {D:.6e}   margin = {d - D:.6e}")
    print(f"  grid: nx,ny,nz = {nx},{ny},{nz}   cell_size={cell_size:.6g}   L={2.0*C:.6g}")
    print(f"  inflate_insert={inflate_insert:.6g}   inflate_query={inflate_query:.6g}")

    def _fmt(lbl, R):
        return f"{lbl}: []" if len(R) == 0 else f"{lbl}: [{R[0][0]},{R[0][1]}]"

    print(_fmt("  Xi", Xi)); print(_fmt("  Xj", Xj))
    print(_fmt("  Yi", Yi)); print(_fmt("  Yj", Yj))
    print(_fmt("  Zi", Zi)); print(_fmt("  Zj", Zj))
    print(f"  per-axis overlap: X={ox}  Y={oy}  Z={oz}")

    if d < D and not (ox and oy and oz):
        missing_axes = []
        if not ox: missing_axes.append("X")
        if not oy: missing_axes.append("Y")
        if not oz: missing_axes.append("Z")
        print(f"  REASON: No index overlap along axis/axes: {','.join(missing_axes)}")
        print("          -> Either inflate sum < D, or the overlap region lies OUTSIDE the container.")
        print("             (non-PBC clamps AABBs to [0,L]; consider a halo of width D if needed.)")


# ---------------------------------------------------------------------
# Demo / CLI entry
# ---------------------------------------------------------------------

def _demo_once(C=6.0, rod_length=1.0, alpha=100.0,
               grid_capacity_multiplier=96, max_attempts=1_000_000,
               seed=12345):
    """
    Run a single placement + verify, print stats, and (optionally) quick plot if helpers exist.
    """
    rod_diameter = rod_length / alpha
    # heuristic density ~ (C^3) / (D * L^2); scale as desired
    N = int((C**3) / (rod_diameter * rod_length**2) * 1)
    print("Target N:", N)

    cell_size = rod_length

    q, placed = create_nonintersecting_random_rods_contained_npbc_numba(
        num_rods=N,
        rod_diameter=rod_diameter,
        container_size=C,
        max_attempts=max_attempts,
        rod_length=rod_length,
        cell_size=cell_size,
        grid_capacity_multiplier=grid_capacity_multiplier,
        seed=seed,
    )
    print(f"Placed {placed}/{N} rods.")

    # centroids sanity check
    inside = (
        (q[:placed, 0] >= -C) & (q[:placed, 0] <= C) &
        (q[:placed, 1] >= -C) & (q[:placed, 1] <= C) &
        (q[:placed, 2] >= -C) & (q[:placed, 2] <= C)
    ).all()
    print("Centroids inside box:", bool(inside))

    # verify
    dmin, i_min, j_min, num_viol = verify_from_q_npbc(
        q[:placed], rod_length, C, rod_diameter, cell_size, 48
    )
    gap = dmin - rod_diameter
    print(f"Global min distance: {dmin:.6e}  (gap to D: {gap:.6e})")
    print(f"Argmin pair: (i={i_min}, j={j_min})")
    print(f"Violations (< diameter): {num_viol}  (should be 0)")

    # Optional direct segment–segment check and quick plot (if your helpers exist)
    if i_min >= 0 and j_min >= 0:
        p0, p1 = _endpoints_from_q_centroid(q[:placed], rod_length)
        d2_pair = _dist2_lin_seg(p0[i_min], p1[i_min], p0[j_min], p1[j_min])
        print(f"Direct segment-segment distance for argmin pair: {np.sqrt(d2_pair):.6e}")
        print(f"Distance: {dmin:.6e}   Direct: {np.sqrt(d2_pair):.6e}   Gap: {gap:.6e}")
        print(f'{np.sqrt(d2_pair) - dmin:.6e} (should be near zero)')

        try:
            from visualizations import plot_many_rods
            from matplotlib import pyplot as plt
            plot_many_rods(q[[i_min, j_min], :])
            plt.show()
        except Exception:
            pass

    return q, placed, rod_diameter


def _demo_scaling(C=6.0, rod_length=1.0, alpha=100.0,
                  grid_capacity_multiplier=96, max_attempts=1_000_000,
                  seed=12345, n_points=10):
    """
    Benchmark placement time as N increases (geomspace).
    """
    rod_diameter = rod_length / alpha
    cell_size = rod_length

    N_max_est = int((C**3) / (rod_diameter * rod_length**2) * 5)
    NN = np.geomspace(10, N_max_est, num=n_points, dtype=int)

    for N in NN:
        t0 = time.time()
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
        dt = time.time() - t0
        print(f"N={N:6d} placed={placed:6d} time={dt:.3f} sec  ({dt / max(1, N) * 1e6:.1f} µs/rod)")

    # final verification on last run
    inside = (
        (q[:placed, 0] >= -C) & (q[:placed, 0] <= C) &
        (q[:placed, 1] >= -C) & (q[:placed, 1] <= C) &
        (q[:placed, 2] >= -C) & (q[:placed, 2] <= C)
    ).all()
    print("Centroids inside box:", bool(inside))

    dmin, i_min, j_min, num_viol = verify_from_q_npbc(
        q[:placed], rod_length, C, rod_diameter, cell_size, 48
    )
    gap = dmin - rod_diameter
    print(f"Global min distance: {dmin:.6e}  (gap to D: {gap:.6e})")
    print(f"Argmin pair: (i={i_min}, j={j_min})")
    print(f"Violations (< diameter): {num_viol}  (should be 0)")

    # Optional argmin visualization if your core utilities exist
    try:
        from transforms import q_to_x
        x = q_to_x(q[:placed])
        p0 = x[i_min, 0:3]; p1 = x[i_min, 3:6]
        q0 = x[j_min, 0:3]; q1 = x[j_min, 3:6]
        d2_pair = _dist2_lin_seg(np.array(p0), np.array(p1), np.array(q0), np.array(q1))
        print(f"Direct segment-segment distance for argmin pair: {np.sqrt(d2_pair):.6e}")
        print(f"Distance: {dmin:.6e}   Direct: {np.sqrt(d2_pair):.6e}   Gap: {gap:.6e}")
        print(f'{np.sqrt(d2_pair) - dmin:.6e} (should be near zero)')
    except Exception:
        pass


    # build the SAME endpoints used by verify
    p0v, p1v = _endpoints_from_q_centroid(q[:placed], rod_length)

    # recompute direct distance on the argmin pair from verify
    d2_pair_verify_space = _dist2_lin_seg(p0v[i_min], p1v[i_min], p0v[j_min], p1v[j_min])
    print("Direct distance (verify space):", np.sqrt(d2_pair_verify_space))

    # sanity: brute-force check (slow but definitive for a subset or small N)
    d2 = 1e300; ii=-1; jj=-1
    for i in range(min(placed, 2000)):            # (optionally cap for speed)
        for j in range(i+1, min(placed, 2000)):
            v = _dist2_lin_seg(p0v[i], p1v[i], p0v[j], p1v[j])
            if v < d2:
                d2 = v; ii = i; jj = j
    print("BF min dist:", np.sqrt(d2), "at (i,j)=", ii, jj)


def demo_npbc_combined():
    # Single run + verify
    q, placed, D = _demo_once(C=6.0, rod_length=1.0, alpha=100.0,
                              grid_capacity_multiplier = 96, max_attempts = 1_000_000,
                              seed=12345)

    # Scaling study
    _demo_scaling(C=3.0, rod_length=1.0, alpha=100.0,
                  grid_capacity_multiplier=96, max_attempts=1_000_000,
                  seed=12345, n_points=10)

    # Example auditor usage on the argmin pair (non-PBC)
    # Build endpoints and call explain_candidate_miss_npbc if desired.
    try:
        from transforms import q_to_x
        x = q_to_x(q[:placed])
        # recompute argmin to know indices
        dmin, i_min, j_min, _ = verify_from_q_npbc(q[:placed], 1.0, 6.0, 1.0/100.0, 1.0, 48)
        p0_all = x[:, 0:3].copy()
        p1_all = x[:, 3:6].copy()
        r = 0.5 * (1.0 / 100.0)
        explain_candidate_miss_npbc(i_min, j_min, p0_all, p1_all,
                                    6.0, 1.0, r, r + _EPS_DIST, 1.0/100.0)
    except Exception:
        pass
# %%


if __name__ == "__main__":
    # _demo_scaling(C=3.0, rod_length=1.0, alpha=100.0,
    #               grid_capacity_multiplier=96, max_attempts=1_000_000,
    #               seed=12345, n_points=10)
    output_dir, movie_dir = setup_directories(__file__)

    import sys

    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 100.0
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    print(f"Alpha = {alpha}   Seed = {seed}")

    rod_length = 1.0
    # alpha = 100
    n_points = 30
    max_attempts = 1_000_000_000

    rod_diameter = rod_length / alpha
    cell_size = rod_length
    C = 3.0
    grid_capacity_multiplier = 96
    # seed = 0



    # N_max_est = int((2*C)**3) / (rod_diameter * rod_length**2) * 3.7
    N_max_est = int((2*C)**3) / (rod_diameter * rod_length**2)
    # NN = np.geomspace(10, N_max_est, num=n_points, dtype=int)

    factor = 6.5
    # NN = np.geomspace(10, N_max_est*factor, num=n_points, dtype=float)[::-1].astype(int)

    NN = np.linspace(10, N_max_est*factor, num=n_points, dtype=int)

    # NN = np.linspace(N_max_est*factor, N_max_est*factor*1.5, num=5, dtype=int)

    attempts_list = []
    qq = []
    for N in NN:
        t0 = time.time()
        q, placed, attempts = create_nonintersecting_random_rods_contained_npbc_numba(
            num_rods=int(N),
            rod_diameter=rod_diameter,
            container_size=C,
            max_attempts=max_attempts,
            rod_length=rod_length,
            cell_size=cell_size,
            grid_capacity_multiplier=grid_capacity_multiplier,
            seed=seed
        )
        dt = time.time() - t0
        print(f"N={N:6d} placed={placed:6d} time={dt:.3f} sec  ({dt / max(1, N) * 1e6:.1f} µs/rod)")
        attempts_list.append(attempts)
        qq.append(q)

        # save intermediate
        np.save(f'{output_dir}/placed_rods_npbc_intermediate_N{N}.npy', q)
        # save attempts
        np.save(f'{output_dir}/attempts_npbc_intermediate_N{N}.npy', np.array(attempts))

        

    # final verification on last run
    inside = (
        (q[:placed, 0] >= -C) & (q[:placed, 0] <= C) &
        (q[:placed, 1] >= -C) & (q[:placed, 1] <= C) &
        (q[:placed, 2] >= -C) & (q[:placed, 2] <= C)
    ).all()
    print("Centroids inside box:", bool(inside))

    dmin, i_min, j_min, num_viol = verify_from_q_npbc(
        q[:placed], rod_length, C, rod_diameter, cell_size, 48
    )
    gap = dmin - rod_diameter
    print(f"Global min distance: {dmin:.6e}  (gap to D: {gap:.6e})")
    print(f"Argmin pair: (i={i_min}, j={j_min})")
    print(f"Violations (< diameter): {num_viol}  (should be 0)")


    # save all data
    np.savetxt(f'{output_dir}/attempts_vs_N_npbc.txt', np.column_stack((NN, attempts_list)), header='N Attempts', comments='')
    # save qq
    qq = np.array(qq, dtype=object)
    np.savez_compressed(f'{output_dir}/placed_rods_npbc.npz', NN=NN, qq=qq)    

    from matplotlib import pyplot as plt
    plt.plot(NN/N_max_est, attempts_list, marker='o')
    plt.xlabel('N')
    plt.ylabel('Attempts to place all rods')
    plt.title('Attempts vs N')
    plt.grid(True)
    plt.savefig(f'{output_dir}/attempts_vs_N_npbc.png', dpi=300)
    plt.show()
    


    # Optional argmin visualization if your core utilities exist
    try:
        from transforms import q_to_x
        x = q_to_x(q[:placed])
        p0 = x[i_min, 0:3]; p1 = x[i_min, 3:6]
        q0 = x[j_min, 0:3]; q1 = x[j_min, 3:6]
        d2_pair = _dist2_lin_seg(np.array(p0), np.array(p1), np.array(q0), np.array(q1))
        print(f"Direct segment-segment distance for argmin pair: {np.sqrt(d2_pair):.6e}")
        print(f"Distance: {dmin:.6e}   Direct: {np.sqrt(d2_pair):.6e}   Gap: {gap:.6e}")
        print(f'{np.sqrt(d2_pair) - dmin:.6e} (should be near zero)')
    except Exception:
        pass


    # build the SAME endpoints used by verify
    p0v, p1v = _endpoints_from_q_centroid(q[:placed], rod_length)

    # recompute direct distance on the argmin pair from verify
    d2_pair_verify_space = _dist2_lin_seg(p0v[i_min], p1v[i_min], p0v[j_min], p1v[j_min])
    print("Direct distance (verify space):", np.sqrt(d2_pair_verify_space))

    # sanity: brute-force check (slow but definitive for a subset or small N)
    d2 = 1e300; ii=-1; jj=-1
    for i in range(min(placed, 2000)):            # (optionally cap for speed)
        for j in range(i+1, min(placed, 2000)):
            v = _dist2_lin_seg(p0v[i], p1v[i], p0v[j], p1v[j])
            if v < d2:
                d2 = v; ii = i; jj = j
    print("BF min dist:", np.sqrt(d2), "at (i,j)=", ii, jj)