# %%
from __future__ import annotations

from functools import partial
import numpy as onp
import jax
import jax.numpy as jnp
from jax import jit, grad, vmap, lax


# current filename
import os
current_filename = os.path.basename(__file__)
print(f"Running {current_filename}")

# make folder named after the current filename (without .py)
output_folder = current_filename.replace(".py", "")
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"Created folder: {output_folder}")

# ================================================================
# Geometry utilities
# ================================================================

@jit
def _clip01(x: jnp.ndarray) -> jnp.ndarray:
    """Clamp to [0,1]."""
    return jnp.clip(x, 0.0, 1.0)


@jit
def _dist_point_segment(x: jnp.ndarray, a: jnp.ndarray, b: jnp.ndarray, eps: float = 1e-12) -> jnp.ndarray:
    """Distance from point x to segment [a,b]."""
    v = b - a
    vv = jnp.dot(v, v)
    t = jnp.where(vv > eps, _clip01(jnp.dot(x - a, v) / vv), 0.0)
    c = a + t * v
    return jnp.linalg.norm(x - c)


@jit
def dist_lin_seg(p1s: jnp.ndarray, p1e: jnp.ndarray,
                 p2s: jnp.ndarray, p2e: jnp.ndarray,
                 eps: float = 1e-12) -> jnp.ndarray:
    """
    Shortest distance between two 3D line segments [p1s,p1e] and [p2s,p2e].
    Robust and JAX-safe (no Python conditionals on traced values).
    """
    d1 = p1e - p1s
    d2 = p2e - p2s
    r  = p1s - p2s

    a = jnp.dot(d1, d1)        # ||d1||^2
    e = jnp.dot(d2, d2)        # ||d2||^2
    b = jnp.dot(d1, d2)
    c = jnp.dot(d1, r)
    f = jnp.dot(d2, r)
    det = a * e - b * b

    a_zero = a < eps
    e_zero = e < eps

    def both_points(_):
        return jnp.linalg.norm(p1s - p2s)

    def first_point(_):
        return _dist_point_segment(p1s, p2s, p2e, eps)

    def second_point(_):
        return _dist_point_segment(p2s, p1s, p1e, eps)

    def general_case(_):
        def d2_for(s, t):
            diff = (p1s + s * d1) - (p2s + t * d2)
            return jnp.dot(diff, diff)

        s0 = jnp.where(jnp.abs(det) > eps, (b * f - c * e) / det, 0.0)
        t0 = jnp.where(jnp.abs(det) > eps, (a * f - b * c) / det, 0.0)

        sA = _clip01(s0)
        tA = _clip01((b * sA + f) / jnp.where(e > eps, e, 1.0))

        tB = _clip01(t0)
        sB = _clip01((b * tB - c) / jnp.where(a > eps, a, 1.0))

        sC = 0.0
        tC = _clip01(f / jnp.where(e > eps, e, 1.0))

        sD = 1.0
        tD = _clip01((b + f) / jnp.where(e > eps, e, 1.0))

        tE = 0.0
        sE = _clip01(-c / jnp.where(a > eps, a, 1.0))

        tF = 1.0
        sF = _clip01((b - c) / jnp.where(a > eps, a, 1.0))

        d2s = jnp.stack([
            d2_for(sA, tA),
            d2_for(sB, tB),
            d2_for(sC, tC),
            d2_for(sD, tD),
            d2_for(sE, tE),
            d2_for(sF, tF),
        ])
        return jnp.sqrt(jnp.min(d2s))

    return lax.cond(a_zero & e_zero, both_points,
           lambda _: lax.cond(a_zero, first_point,
                    lambda __: lax.cond(e_zero, second_point, general_case, operand=None),
                    operand=None),
           operand=None)


# ================================================================
# Rod parameterization and pair distances
# ================================================================

@jit
def _sph_to_dir(phi: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
    """Unit vector from spherical angles (physics convention) for scalars."""
    s = jnp.sin(phi)
    return jnp.array([s * jnp.cos(theta), s * jnp.sin(theta), jnp.cos(phi)])


@jit
def _sph_to_dir_batch(phi: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
    """Batch version: phi,theta -> (N,3). Accepts (N,) or (N,1)."""
    phi = jnp.ravel(phi)
    theta = jnp.ravel(theta)
    s = jnp.sin(phi)
    x = s * jnp.cos(theta)
    y = s * jnp.sin(theta)
    z = jnp.cos(phi)
    return jnp.stack([x, y, z], axis=-1)


@jit
def pairwise_distance(q_pair: jnp.ndarray, seg_len: float = 1.0) -> jnp.ndarray:
    """q_pair shape (10,) = (xi,yi,zi,phi_i,theta_i,  xj,yj,zj,phi_j,theta_j)."""
    xi, yi, zi, phi_i, th_i, xj, yj, zj, phi_j, th_j = q_pair
    p_i = jnp.array([xi, yi, zi])
    p_j = jnp.array([xj, yj, zj])
    u_i = _sph_to_dir(phi_i, th_i)
    u_j = _sph_to_dir(phi_j, th_j)
    p_ii = p_i + seg_len * u_i
    p_jj = p_j + seg_len * u_j
    return dist_lin_seg(p_i, p_ii, p_j, p_jj)


@jit
def create_pairs(m: jnp.ndarray) -> jnp.ndarray:
    """Upper-triangular pairs: (N,M) -> (N*(N-1)/2, 2M)."""
    N, M = m.shape
    i, j = jnp.triu_indices(N, k=1)
    return jnp.concatenate([m[i], m[j]], axis=1)


# ================================================================
# Endpoints + batched distance for indexed pairs
# ================================================================

def q_to_endpoints(q_flat: jnp.ndarray, seg_len: float = 1.0) -> tuple[jnp.ndarray, jnp.ndarray]:
    """q_flat (N*5,) -> r1, r2 each (N,3)."""
    q = jnp.reshape(q_flat, (-1, 5))
    p = q[:, :3]
    u = _sph_to_dir_batch(q[:, 3], q[:, 4])
    r1 = p
    r2 = p + seg_len * u
    return r1, r2


vdist = vmap(dist_lin_seg, in_axes=(0, 0, 0, 0))


def pair_dists_indexed(r1: jnp.ndarray, r2: jnp.ndarray, i_idx: jnp.ndarray, j_idx: jnp.ndarray) -> jnp.ndarray:
    """Return distances for pairs (i_idx[k], j_idx[k])."""
    p1s = r1[i_idx]
    p1e = r2[i_idx]
    p2s = r1[j_idx]
    p2e = r2[j_idx]
    return vdist(p1s, p1e, p2s, p2e)


# ================================================================
# Energy (contact penalty)
# ================================================================

@partial(jit, static_argnames=("seg_len",))
def simple_harmonic_line_jump(q_pair: jnp.ndarray, threshold: float, amp: float,
                              seg_len: float = 1.0) -> jnp.ndarray:
    """Quadratic penalty when distance < threshold."""
    d = pairwise_distance(q_pair, seg_len=seg_len)
    return lax.cond(d < threshold,
                    lambda _: amp * (d - threshold) ** 2,
                    lambda _: 0.0,
                    operand=None)


@partial(jit, static_argnames=("seg_len",))
def total_harmonic_line(q_flat: jnp.ndarray, col_rad: float, amp: float,
                        seg_len: float = 1.0) -> jnp.ndarray:
    """All-pairs energy (O(N^2))."""
    q = jnp.reshape(q_flat, (-1, 5))
    pairs = create_pairs(q)
    threshold = 2.0 * col_rad
    penal = vmap(lambda qp: simple_harmonic_line_jump(qp, threshold, amp, seg_len))(pairs)
    return jnp.sum(penal)


@partial(jit, static_argnames=("seg_len",))
def total_harmonic_line_subset(q_flat: jnp.ndarray, col_rad: float, amp: float,
                               i_idx: jnp.ndarray, j_idx: jnp.ndarray,
                               seg_len: float = 1.0) -> jnp.ndarray:
    """Subset energy over candidate pairs from a spatial hash."""
    r1, r2 = q_to_endpoints(q_flat, seg_len=seg_len)
    d = pair_dists_indexed(r1, r2, i_idx, j_idx)
    threshold = 2.0 * col_rad
    penal = jnp.where(d < threshold, amp * (d - threshold) ** 2, 0.0)
    return jnp.sum(penal)


# ================================================================
# Spatial hash (NumPy; rebuilt per outer loop)
# ================================================================

def _grid_key(ix: int, iy: int, iz: int) -> tuple[int, int, int]:
    return (ix, iy, iz)


def build_spatial_hash_pairs(r1: onp.ndarray,
                             r2: onp.ndarray,
                             threshold: float,
                             cell_size: float | None = None) -> tuple[onp.ndarray, onp.ndarray]:
    """
    Build candidate pairs via spatial hashing of AABBs inflated by 'threshold'.

    Parameters
    ----------
    r1, r2 : (N,3) endpoints
    threshold : float            # typically = 2*col_rad (penalty threshold)
    cell_size : float or None    # if None, uses median(seg_len) + threshold

    Returns
    -------
    i_idx, j_idx : int64 arrays of unique candidate pairs (i<j).
    """
    N = r1.shape[0]
    seg_len_est = onp.median(onp.linalg.norm(r2 - r1, axis=1))
    if cell_size is None:
        cell_size = float(seg_len_est + threshold)

    inv_h = 1.0 / cell_size
    grid: dict[tuple[int, int, int], list[int]] = {}

    # Insert segments into all cells overlapped by their inflated AABB
    for i in range(N):
        a = onp.minimum(r1[i], r2[i]) - threshold
        b = onp.maximum(r1[i], r2[i]) + threshold
        ix0, iy0, iz0 = onp.floor(a * inv_h).astype(onp.int64)
        ix1, iy1, iz1 = onp.floor(b * inv_h).astype(onp.int64)
        for ix in range(ix0, ix1 + 1):
            for iy in range(iy0, iy1 + 1):
                for iz in range(iz0, iz1 + 1):
                    grid.setdefault(_grid_key(ix, iy, iz), []).append(i)

    # Unique pairs from each cell
    pair_set: set[tuple[int, int]] = set()
    for inds in grid.values():
        m = len(inds)
        if m < 2:
            continue
        for a in range(m - 1):
            ia = inds[a]
            for b in range(a + 1, m):
                ib = inds[b]
                if ia < ib:
                    pair_set.add((ia, ib))
                else:
                    pair_set.add((ib, ia))

    if not pair_set:
        return onp.empty((0,), dtype=onp.int64), onp.empty((0,), dtype=onp.int64)

    pairs = onp.fromiter(pair_set, dtype=[('i', onp.int64), ('j', onp.int64)], count=len(pair_set))
    return pairs['i'], pairs['j']


# ================================================================
# Optimizers
# ================================================================

def optimize_fire_nonjax_individual(q0: jnp.ndarray,
                                    f: callable,
                                    df: callable,
                                    Nmax: int,
                                    atol: float = 1e-4,
                                    dt: float = 2e-3,
                                    finc: float = 1.1,
                                    fdec: float = 0.5,
                                    fa: float = 0.99,
                                    alpha0: float = 0.1,
                                    Ndelay: int = 5,
                                    log_every: int = 100,
                                    callback=None,
                                    dmin_fn: callable | None = None):
    """Plain FIRE (no line search), with optional d_min logging."""
    q = q0.copy()
    V = jnp.zeros_like(q)
    F = -df(q)

    alpha = float(alpha0)
    dt_curr = float(dt)
    npos = 0
    err = float(jnp.max(jnp.abs(F)))

    for it in range(Nmax):
        P = jnp.vdot(F, V)
        p = float(P)

        Fn = jnp.linalg.norm(F) + 1e-30
        Vn = jnp.linalg.norm(V) + 1e-30
        V = (1.0 - alpha) * V + alpha * (F / Fn) * Vn

        if p > 0.0:
            npos += 1
            if npos > Ndelay:
                dt_curr = min(dt_curr * float(finc), 10.0 * float(dt))
                alpha *= float(fa)
        else:
            npos = 0
            dt_curr *= float(fdec)
            alpha = float(alpha0)
            V = jnp.zeros_like(V)

        V = V + 0.5 * dt_curr * F
        q = q + dt_curr * V
        F = -df(q)
        V = V + 0.5 * dt_curr * F

        err = float(jnp.max(jnp.abs(F)))

        if (it % log_every) == 0:
            dmin_str = ""
            if dmin_fn is not None:
                dmin_val = float(dmin_fn(q))
                dmin_str = f"  d_min={dmin_val:9.3e}"
            print(f"Iter {it:4d}  f={float(f(q)):12.6e}  |F|_inf={err:9.2e}  dt={dt_curr:.2e}{dmin_str}")
            if callback is not None and callback(q, {"iter": it, "d_min": dmin_val if dmin_str else None}):
                print("Callback requested stop.")
                break

        if err < float(atol):
            break

    return q, f(q), npos, err


def optimize_fire_with_linesearch(
    q0: jnp.ndarray,
    f: callable,
    df: callable,
    Nmax: int,
    atol: float = 1e-4,
    dt: float = 2e-3,
    finc: float = 1.1,
    fdec: float = 0.5,
    fa: float = 0.99,
    alpha0: float = 0.1,
    Ndelay: int = 5,
    log_every: int = 100,
    # line-search knobs
    ls_c1: float = 1e-4,
    ls_shrink: float = 0.5,
    ls_max_steps: int = 12,
    ls_min_dt: float = 1e-12,
    callback=None,
    dmin_fn: callable | None = None,
):
    """
    FIRE with Armijo backtracking on velocity-Verlet.
    We align V toward F *before* computing P=<F,V> to avoid the V==0 reset trap.
    """
    q = q0.copy()
    V = jnp.zeros_like(q)
    F = -df(q)
    f_q = f(q)

    alpha = float(alpha0)
    dt_curr = float(dt)
    npos = 0
    err = float(jnp.max(jnp.abs(F)))  # defined even if we continue early

    for it in range(Nmax):
        # --- FIRE mixing: align V toward F ---
        Fn = jnp.linalg.norm(F) + 1e-30
        Vn = jnp.linalg.norm(V)
        mag = jnp.where(Vn > 1e-12, Vn, 1.0)  # avoid zero alignment
        V = (1.0 - alpha) * V + alpha * (F / Fn) * mag

        # Now compute power (after mixing!)
        P = jnp.vdot(F, V)
        p = float(P)

        if p <= 0.0:
            npos = 0
            dt_curr *= float(fdec)
            alpha = float(alpha0)
            V = jnp.zeros_like(V)
            F = -df(q)
            f_q = f(q)
            err = float(jnp.max(jnp.abs(F)))
            if (it % log_every) == 0:
                dmin_str = ""
                if dmin_fn is not None:
                    dmin_val = float(dmin_fn(q))
                    dmin_str = f"  d_min={dmin_val:9.3e}"
                print(f"Iter {it:4d}  f={float(f_q):12.6e}  (P<=0 reset)  dt={dt_curr:.2e}{dmin_str}")
            continue

        # --- Backtracking LS on Verlet position update ---
        dt_try = dt_curr
        ok = False
        ls_steps_used = 0
        for ls_k in range(ls_max_steps):
            q_trial = q + dt_try * V + 0.5 * (dt_try ** 2) * F
            f_trial = f(q_trial)
            ls_steps_used = ls_k + 1
            # Armijo: f(q_trial) <= f(q) - c1 * dt * <F,V>
            if float(f_trial) <= float(f_q) - ls_c1 * dt_try * p:
                ok = True
                break
            dt_try *= float(ls_shrink)
            if dt_try < ls_min_dt:
                ok = True
                break

        # Apply Verlet with accepted dt_try
        V_half = V + 0.5 * dt_try * F
        q = q + dt_try * V_half
        F = -df(q)
        V = V_half + 0.5 * dt_try * F
        f_q = f_trial
        err = float(jnp.max(jnp.abs(F)))

        # Success bookkeeping & FIRE updates
        if ok and ls_steps_used == 1:
            npos += 1
            if npos > Ndelay:
                dt_curr = min(dt_try * float(finc), 10.0 * float(dt))
                alpha *= float(fa)
            else:
                dt_curr = dt_try
        else:
            npos = 0
            dt_curr = dt_try

        if (it % log_every) == 0:
            dmin_str = ""
            if dmin_fn is not None:
                dmin_val = float(dmin_fn(q))
                dmin_str = f"  d_min={dmin_val:9.3e}"
            print(f"Iter {it:4d}  f={float(f_q):12.6e}  |F|_inf={err:9.2e}  dt={dt_curr:.2e}  ls_steps={ls_steps_used}{dmin_str}")

        if err < float(atol):
            break

    return q, f_q, npos, err


# ================================================================
# Outer collision-relaxation (spatial hash + optional LS)
# ================================================================

def collision_relaxation(q_flat: jnp.ndarray,
                         f_in: callable,   # API compatibility (unused directly)
                         params: dict,
                         N_outer: int,
                         Nmax: int,
                         atol: float,
                         dt: float,
                         seg_len: float = 1.0,
                         use_spatial_hash: bool = True,
                         cell_size: float | None = None,
                         callback=None,
                         use_line_search: bool = True) -> jnp.ndarray:
    """
    Repeatedly run FIRE on the penalty objective until all pair distances exceed 2*col_rad.
    If use_spatial_hash=True, the candidate pair list is rebuilt each outer loop using an
    inflated-AABB grid, massively reducing pair count.
    """
    col_rad = float(params["col_rad"])
    amp = float(params["amp"])
    threshold = 2.0 * col_rad

    for k in range(N_outer):
        if use_spatial_hash:
            r1, r2 = q_to_endpoints(q_flat, seg_len=seg_len)
            i_np, j_np = build_spatial_hash_pairs(
                onp.asarray(r1), onp.asarray(r2), threshold=threshold, cell_size=cell_size
            )
            if i_np.size == 0:
                print(f"[outer {k}] Spatial hash found no candidate pairs. Done.")
                break
            i_idx = jnp.asarray(i_np, dtype=jnp.int32)
            j_idx = jnp.asarray(j_np, dtype=jnp.int32)
            f = lambda q: total_harmonic_line_subset(q, col_rad=col_rad, amp=amp,
                                                     i_idx=i_idx, j_idx=j_idx, seg_len=seg_len)

            def dmin_fn(cur_q):
                r1_, r2_ = q_to_endpoints(cur_q, seg_len=seg_len)
                d_ = pair_dists_indexed(r1_, r2_, i_idx, j_idx)
                # if candidate set is empty (unlikely here), return +inf
                return jnp.min(d_) if d_.size > 0 else jnp.array(jnp.inf)
        else:
            f = lambda q: total_harmonic_line(q, col_rad=col_rad, amp=amp, seg_len=seg_len)

            def dmin_fn(cur_q):
                q_rows_ = jnp.reshape(cur_q, (-1, 5))
                pairs_ = create_pairs(q_rows_)
                d_ = vmap(pairwise_distance, in_axes=(0, None))(pairs_, seg_len)
                return jnp.min(d_) if d_.size > 0 else jnp.array(jnp.inf)

        df = jit(grad(jit(f)))

        if use_line_search:
            q_flat, f_val, _, err = optimize_fire_with_linesearch(
                q_flat, f, df, Nmax=Nmax, atol=atol, dt=dt, callback=callback, dmin_fn=dmin_fn
            )
        else:
            q_flat, f_val, _, err = optimize_fire_nonjax_individual(
                q_flat, f, df, Nmax=Nmax, atol=atol, dt=dt, callback=callback, dmin_fn=dmin_fn
            )

        # Separation check using the same candidate set (cheap)
        dmin_now = float(dmin_fn(q_flat))
        if dmin_now > threshold:
            print(f"[outer {k}] Enough push-off: d_min={dmin_now:.6e}")
            break

    return q_flat


# ================================================================
# Demo / script entry
# ================================================================
if __name__ == "__main__":
    # Self-contained demo
    import jax.random as jr

    def random_rods(num_rods: int, key: jax.random.PRNGKey) -> jnp.ndarray:
        """(num_rods,5): x,y,z in [-0.5,0.5]^3 and uniformly random directions on S2."""
        k1, k2, k3 = jr.split(key, 3)
        centers = jr.uniform(k1, (num_rods, 3), minval=-0.5, maxval=0.5)
        u = jr.uniform(k2, (num_rods, 1), minval=-1.0, maxval=1.0)
        phi = jnp.arccos(u)  # polar
        theta = jr.uniform(k3, (num_rods, 1), minval=0.0, maxval=2.0 * jnp.pi)
        return jnp.concatenate([centers, phi, theta], axis=1)

    num_rods = 200
    q = random_rods(num_rods, jr.PRNGKey(0))  # (N,5)

    # tiny jitter to centers
    k = jr.PRNGKey(42)
    jitter = jr.normal(k, q[:, :3].shape)
    jitter = jitter / (jnp.linalg.norm(jitter, axis=1, keepdims=True) + 1e-30)
    q = q.at[:, :3].add(1e-10 * jitter)

    # Params
    params = {"col_rad": 1e-3, "amp": 1.0}
    dt = 1e-3
    N_outer = 5
    Nmax = 500
    atol = 1e-10
    seg_len = 1.0

    # Spatial hash + line search ON
    q_out = collision_relaxation(
        q.flatten(), total_harmonic_line, params, N_outer, Nmax, atol, dt,
        seg_len=seg_len, use_spatial_hash=False, cell_size=None, use_line_search=True
    )

    # If you have your own visualizer:
    from visualizations import plot_many_rods
    ax = plot_many_rods(q_out.reshape(-1,5))
    ax.axis('equal')

    from matplotlib import pyplot as plt
    plt.savefig(f"{output_folder}/final_rods.png", dpi=300)
