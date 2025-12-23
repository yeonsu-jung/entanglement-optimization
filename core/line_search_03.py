# %%
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax import jit, grad, vmap, lax


# ---------------------------------------------------------------------
# Geometry utils
# ---------------------------------------------------------------------

@jit
def _clip01(x: jnp.ndarray) -> jnp.ndarray:
    """Clamp to [0,1]."""
    return jnp.clip(x, 0.0, 1.0)


@jit
def dist_lin_seg(p1s: jnp.ndarray, p1e: jnp.ndarray,
                 p2s: jnp.ndarray, p2e: jnp.ndarray,
                 eps: float = 1e-12) -> jnp.ndarray:
    """
    Shortest distance between two 3D line segments [p1s,p1e], [p2s,p2e].
    Robust to (near-)degenerate segments.

    Returns
    -------
    jnp.ndarray scalar distance.
    """
    d1 = p1e - p1s
    d2 = p2e - p2s
    r  = p1s - p2s

    a = jnp.dot(d1, d1)  # ||d1||^2
    e = jnp.dot(d2, d2)  # ||d2||^2
    f = jnp.dot(d2, r)
    c = jnp.dot(d1, r)
    b = jnp.dot(d1, d2)
    det = a * e - b * b  # denominator

    # Handle degenerate cases
    a_zero = a < eps
    e_zero = e < eps

    def both_degenerate():
        # both segments reduce to points
        return jnp.linalg.norm(p1s - p2s)

    def first_degenerate():
        # project p1s onto seg2
        u = _clip01(-f / e)
        closest = p2s + u * d2
        return jnp.linalg.norm(p1s - closest)

    def second_degenerate():
        # project p2s onto seg1
        t = _clip01(c / a)
        closest = p1s + t * d1
        return jnp.linalg.norm(closest - p2s)

    def general_case():
        # Solve for unconstrained t,u then clamp with endpoint corrections
        def skew_ok():
            t = _clip01((b * f - c * e) / det)
            u = (b * t + f) / e
            u = _clip01(u)
            # re-clip t if u clipped changed the optimum
            t = _clip01((b * u - c) / a) if (u == 0.0) | (u == 1.0) else t
            return t, u

        def parallel_case():
            # When nearly parallel, fall back to projecting one endpoint and clamping
            # Evaluate distance at u in {0,1} and the projected value
            u_proj = _clip01(-f / e)
            # candidates for u
            def dist_for_u(u):
                t = _clip01((b * u - c) / a)
                diff = (p1s + t * d1) - (p2s + u * d2)
                return jnp.dot(diff, diff)

            d0 = dist_for_u(0.0)
            d1_ = dist_for_u(1.0)
            dp = dist_for_u(u_proj)
            u = jnp.where((d0 <= d1_) & (d0 <= dp), 0.0,
                jnp.where((d1_ <= d0) & (d1_ <= dp), 1.0, u_proj))
            t = _clip01((b * u - c) / a)
            return t, u

        t, u = lax.cond(jnp.abs(det) > eps, skew_ok, parallel_case)
        diff = (p1s + t * d1) - (p2s + u * d2)
        return jnp.linalg.norm(diff)

    return lax.cond(a_zero & e_zero,
                    lambda _: both_degenerate(),
                    lambda _: lax.cond(a_zero, lambda __: first_degenerate(),
                                       lambda __: lax.cond(e_zero, lambda ___: second_degenerate(),
                                                           lambda ___: general_case(), None),
                                       None),
                    None)


# ---------------------------------------------------------------------
# Rod parameterization and pair distances
# ---------------------------------------------------------------------

@jit
def _sph_to_dir(phi: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
    """Unit vector from spherical angles (physics convention)."""
    s = jnp.sin(phi)
    return jnp.array([s * jnp.cos(theta), s * jnp.sin(theta), jnp.cos(phi)])


@jit
def pairwise_distance(q_pair: jnp.ndarray, seg_len: float = 1.0) -> jnp.ndarray:
    """
    q_pair shape (10,) = (xi,yi,zi,phi_i,theta_i,  xj,yj,zj,phi_j,theta_j)
    """
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
    """
    Build upper-triangular pairs of rows from m (N,M) -> (N*(N-1)/2, 2M).
    """
    N, M = m.shape
    i, j = jnp.triu_indices(N, k=1)
    return jnp.concatenate([m[i], m[j]], axis=1)


# ---------------------------------------------------------------------
# Energy (contact penalty)
# ---------------------------------------------------------------------

@partial(jit, static_argnames=("seg_len",))
def simple_harmonic_line_jump(q_pair: jnp.ndarray, col_rad2: float, amp: float,
                              seg_len: float = 1.0) -> jnp.ndarray:
    """
    Quadratic penalty when segment distance < 2*col_rad.
    Inputs are arranged as q_pair (10,) like in pairwise_distance.
    """
    d = pairwise_distance(q_pair, seg_len=seg_len)
    # Penalize when d < 2*col_rad
    return lax.cond(d < col_rad2, lambda _: amp * (d - col_rad2) ** 2, lambda _: 0.0, operand=None)


@partial(jit, static_argnames=("seg_len",))
def total_harmonic_line(q_flat: jnp.ndarray, col_rad: float, amp: float,
                        seg_len: float = 1.0) -> jnp.ndarray:
    """
    Total penalty over all unordered rod pairs.
    q_flat has shape (N*5,), representing rows [x,y,z,phi,theta] per rod.
    """
    q = jnp.reshape(q_flat, (-1, 5))
    pairs = create_pairs(q)

    col_rad2 = 2.0 * col_rad
    penal = vmap(lambda qp: simple_harmonic_line_jump(qp, col_rad2, amp, seg_len))(pairs)
    return jnp.sum(penal)


# ---------------------------------------------------------------------
# Optimizer (FIRE-like, simple variant)
# ---------------------------------------------------------------------

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
                                    callback=None):
    """
    Simple FIRE loop in eager mode (not jitted).
    Works with JAX arrays; uses scalar power P = <F,V>.
    """
    q = q0.copy()
    V = jnp.zeros_like(q)
    F = -df(q)

    alpha = alpha0
    dt_curr = dt
    npos = 0

    for it in range(Nmax):
        # Power
        P = jnp.vdot(F, V)

        # Mix velocities toward forces
        V = (1.0 - alpha) * V + alpha * (F / (jnp.linalg.norm(F) + 1e-30)) * (jnp.linalg.norm(V) + 1e-30)

        # Adapt dt and alpha
        if P > 0:
            npos += 1
            if npos > Ndelay:
                dt_curr = jnp.minimum(dt_curr * finc, 10.0 * dt)
                alpha = alpha * fa
        else:
            npos = 0
            dt_curr = dt_curr * fdec
            alpha = alpha0
            V = jnp.zeros_like(V)  # reset velocity when descending wrong way

        # Velocity Verlet-ish step
        V = V + 0.5 * dt_curr * F
        q = q + dt_curr * V
        F = -df(q)
        V = V + 0.5 * dt_curr * F

        err = jnp.max(jnp.abs(F))

        if (it % log_every) == 0:
            # quick min-pair distance snapshot for logging
            q_pairs = create_pairs(jnp.reshape(q, (-1, 5)))
            dmins = vmap(pairwise_distance)(q_pairs)
            print(
                f"Iter {it:4d}  f={float(f(q)):12.6e}  |F|_inf={float(err):9.2e}  d_min={float(jnp.min(dmins)):9.3e}"
            )
            if callback is not None and callback(q, {"iter": it, "min_distance": jnp.min(dmins)}):
                print("Callback requested stop.")
                break

        if err < atol:
            break

    return q, f(q), npos, err


# ---------------------------------------------------------------------
# Outer collision-relaxation loop
# ---------------------------------------------------------------------

def collision_relaxation(q_flat: jnp.ndarray,
                         f_in: callable,
                         params: dict,
                         N_outer: int,
                         Nmax: int,
                         atol: float,
                         dt: float,
                         seg_len: float = 1.0,
                         callback=None) -> jnp.ndarray:
    """
    Repeatedly run FIRE on the penalty objective until all pair distances exceed 2*col_rad.
    """
    col_rad = float(params["col_rad"])
    amp = float(params["amp"])

    for k in range(N_outer):
        f = lambda q: total_harmonic_line(q, col_rad=col_rad, amp=amp, seg_len=seg_len)
        df = jit(grad(jit(f)))

        q_flat, f_val, _, err = optimize_fire_nonjax_individual(
            q_flat, f, df, Nmax=Nmax, atol=atol, dt=dt, callback=callback
        )

        # Check current min distance
        q_rows = jnp.reshape(q_flat, (-1, 5))
        pairs = create_pairs(q_rows)
        dmins = vmap(pairwise_distance)(pairs)
        dmin = jnp.min(dmins)

        if dmin > 2.0 * col_rad:
            print(f"[outer {k}] Enough push-off: d_min={float(dmin):.6e}")
            break

    return q_flat


# ---------------------------------------------------------------------
# Demo / script entry
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # Example driver (kept minimal and self-contained)
    # You can replace these with your own `create_intersecting_rods` and plotting.
    import jax.random as jr

    def random_rods(num_rods: int, key: jax.random.PRNGKey) -> jnp.ndarray:
        """
        Return (num_rods,5): x,y,z in [-0.5,0.5]^3 and uniformly random directions on S2.
        """
        k1, k2, k3 = jr.split(key, 3)
        centers = jr.uniform(k1, (num_rods, 3), minval=-0.5, maxval=0.5)
        # Uniform on S2: sample cos(phi) ~ U[-1,1], theta ~ U[0,2π]
        u = jr.uniform(k2, (num_rods, 1), minval=-1.0, maxval=1.0)
        phi = jnp.arccos(u)  # polar
        theta = jr.uniform(k3, (num_rods, 1), minval=0.0, maxval=2.0 * jnp.pi)
        q = jnp.concatenate([centers, phi, theta], axis=1)
        return q

    num_rods = 100
    q = random_rods(num_rods, jr.PRNGKey(0))  # (N,5)

    # tiny jitter to centers
    k = jr.PRNGKey(42)
    jitter = jr.normal(k, q[:, :3].shape)
    jitter = jitter / (jnp.linalg.norm(jitter, axis=1, keepdims=True) + 1e-30)
    q = q.at[:, :3].add(1e-5 * jitter)

    # Params
    params = {"col_rad": 1e-3, "amp": 1.0}
    dt = 1e-2
    N_outer = 1
    Nmax = 1000
    atol = 1e-4

    # Run relaxation
    q_out = collision_relaxation(
        q.flatten(), total_harmonic_line, params, N_outer, Nmax, atol, dt, seg_len=1.0
    )

    # If you have your own visualizer:
    # from visualizations import plot_many_rods
    # ax = plot_many_rods(q_out.reshape(-1,5))
    # ax.axis('equal')

# %%
