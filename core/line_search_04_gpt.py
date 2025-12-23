# %%
from __future__ import annotations

from functools import partial
import jax
import jax.numpy as jnp
from jax import jit, grad, vmap, lax
import jax.random as jr

# current filename
import os
current_filename = os.path.basename(__file__)
print(f"Running {current_filename}")

# make folder named after the current filename (without .py)
output_folder = current_filename.replace(".py", "")
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"Created folder: {output_folder}")




# ---------------------------------------------------------------------
# Geometry utilities
# ---------------------------------------------------------------------

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
    JAX-safe (no Python branching on traced values) and robust to parallel/degenerate segments.

    Based on candidate-evaluation approach:
      - unconstrained interior solution (if well-conditioned) + clamped variants
      - endpoints of each segment projected onto the other
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
        # helper: distance^2 for (s,t)
        def d2_for(s, t):
            diff = (p1s + s * d1) - (p2s + t * d2)
            return jnp.dot(diff, diff)

        # Unconstrained (interior) solution if well-conditioned
        s0 = jnp.where(jnp.abs(det) > eps, (b * f - c * e) / det, 0.0)
        t0 = jnp.where(jnp.abs(det) > eps, (a * f - b * c) / det, 0.0)

        # Candidate 0: clamp s then recompute/clampt t
        sA = _clip01(s0)
        tA = _clip01((b * sA + f) / jnp.where(e > eps, e, 1.0))

        # Candidate 1: clamp t then recompute/clampt s
        tB = _clip01(t0)
        sB = _clip01((b * tB - c) / jnp.where(a > eps, a, 1.0))

        # Candidate 2: s=0, best t
        sC = 0.0
        tC = _clip01(f / jnp.where(e > eps, e, 1.0))

        # Candidate 3: s=1, best t
        sD = 1.0
        tD = _clip01((b + f) / jnp.where(e > eps, e, 1.0))

        # Candidate 4: t=0, best s
        tE = 0.0
        sE = _clip01(-c / jnp.where(a > eps, a, 1.0))

        # Candidate 5: t=1, best s
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

    # Dispatch without Python boolean conversion
    return lax.cond(a_zero & e_zero, both_points,
           lambda _: lax.cond(a_zero, first_point,
                    lambda __: lax.cond(e_zero, second_point, general_case, operand=None),
                    operand=None),
           operand=None)

@jit
def acn_over_ij(r1, r2, i_indices, j_indices):
    return vmap(lambda i, j: compute_linking_number_cartesian(r1[i], r2[i], r1[j], r2[j]))(i_indices, j_indices)

@jit
def get_entanglement(q):
    r = q[:,:3]
    u = _sph_to_dir(q[:,3], q[:,4])
    r2 = r + u.T
    pairs = create_pairs(r)
    i_indices, j_indices = jnp.triu_indices(r.shape[0], k=1)
    acn_ij = acn_over_ij(r, r2, i_indices, j_indices)
    return jnp.sum(jnp.abs(acn_ij))

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
def simple_harmonic_line_jump(q_pair: jnp.ndarray, threshold: float, amp: float,
                              seg_len: float = 1.0) -> jnp.ndarray:
    """
    Quadratic penalty when segment distance < threshold (typically threshold = 2*col_rad).
    Inputs are arranged as q_pair (10,) like in pairwise_distance.
    """
    d = pairwise_distance(q_pair, seg_len=seg_len)
    return lax.cond(d < threshold,
                    lambda _: amp * (d - threshold) ** 2,
                    lambda _: 0.0,
                    operand=None)


@partial(jit, static_argnames=("seg_len",))
def total_harmonic_line(q_flat: jnp.ndarray, col_rad: float, amp: float,
                        seg_len: float = 1.0) -> jnp.ndarray:
    """
    Total penalty over all unordered rod pairs.
    q_flat has shape (N*5,), representing rows [x,y,z,phi,theta] per rod.
    """
    q = jnp.reshape(q_flat, (-1, 5))
    pairs = create_pairs(q)
    threshold = 2.0 * col_rad
    penal = vmap(lambda qp: simple_harmonic_line_jump(qp, threshold, amp, seg_len))(pairs)
    return jnp.sum(penal)


# ---------------------------------------------------------------------
# Optimizer (FIRE-like, eager / non-jitted)
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
    Avoids Python branching on tracers by casting scalars with float().
    """
    q = q0.copy()
    V = jnp.zeros_like(q)
    F = -df(q)

    alpha = float(alpha0)
    dt_curr = float(dt)
    npos = 0

    for it in range(Nmax):
        # Power
        P = jnp.vdot(F, V)
        p = float(P)

        # Mix velocities toward forces
        Fn = jnp.linalg.norm(F) + 1e-30
        Vn = jnp.linalg.norm(V) + 1e-30
        V = (1.0 - alpha) * V + alpha * (F / Fn) * Vn

        # Adapt dt and alpha
        if p > 0.0:
            npos += 1
            if npos > Ndelay:
                dt_curr = min(dt_curr * float(finc), 10.0 * float(dt))
                alpha *= float(fa)
        else:
            npos = 0
            dt_curr *= float(fdec)
            alpha = float(alpha0)
            V = jnp.zeros_like(V)  # reset velocity

        # Velocity Verlet-ish step
        V = V + 0.5 * dt_curr * F
        q = q + dt_curr * V
        F = -df(q)
        V = V + 0.5 * dt_curr * F

        err = float(jnp.max(jnp.abs(F)))

        if (it % log_every) == 0:
            # quick min-pair distance snapshot for logging
            q_pairs = create_pairs(jnp.reshape(q, (-1, 5)))
            dmins = vmap(pairwise_distance)(q_pairs)
            print(
                f"Iter {it:4d}  f={float(f(q)):12.6e}  |F|_inf={err:9.2e}  d_min={float(jnp.min(dmins)):9.6e}"
            )
            if callback is not None and callback(q, {"iter": it, "min_distance": jnp.min(dmins)}):
                print("Callback requested stop.")
                break

        if err < float(atol):
            break

    return q, f(q), npos, err


# ---------------------------------------------------------------------
# Outer collision-relaxation loop
# ---------------------------------------------------------------------

def collision_relaxation(q_flat: jnp.ndarray,
                         f_in: callable,   # kept for API similarity; not used directly
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
        dmin = float(jnp.min(dmins))
        

        if dmin > 2.0 * col_rad:
            print(f"[outer {k}] Enough push-off: d_min={dmin:.6e}")
            break

    return q_flat

# get entanglement
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

def random_rods(num_rods: int, key: jax.random.PRNGKey) -> jnp.ndarray:
    """
    Return (num_rods,5): x,y,z in [-0.5,0.5]^3 and uniformly random directions on S2.
    """
    k1, k2, k3 = jr.split(key, 3)
    # centers = jr.uniform(k1, (num_rods, 3), minval=-0.5, maxval=0.5)
    centers = jnp.zeros((num_rods, 3))  # all start at origin

    # Uniform on S2: sample cos(phi) ~ U[-1,1], theta ~ U[0,2π]
    u = jr.uniform(k2, (num_rods, 1), minval=-1.0, maxval=1.0)
    phi = jnp.arccos(u)  # polar
    theta = jr.uniform(k3, (num_rods, 1), minval=0.0, maxval=2.0 * jnp.pi)
    return jnp.concatenate([centers, phi, theta], axis=1)




# ---------------------------------------------------------------------
# Demo / script entry
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # Self-contained demo (replace with your own generator/visualizer as needed)    

    num_rods = 200
    q = random_rods(num_rods, jr.PRNGKey(0))  # (N,5)

    # tiny jitter to centers (to break exact coincidences)
    k = jr.PRNGKey(42)
    jitter = jr.normal(k, q[:, :3].shape)
    jitter = jitter / (jnp.linalg.norm(jitter, axis=1, keepdims=True) + 1e-30)
    q = q.at[:, :3].add(1e-10 * jitter)

    # from protocols import create_entangled_rods
    # q_entangled = create_entangled_rods(num_rods,total_effective_potential,random_keys,rod_diameter=(1/AR),Nmax=300,N_outer=5,atol=1e-8,dt=dt,initial_q="non-intersecting",callback=_callback)

    

    

    
    
    # ent = get_entanglement(q)
    # print(f"Initial entanglement: {ent}")
    
    from visualizations import plot_many_rods
    from matplotlib import pyplot as plt
    from potentials import total_effective_potential
    # q_ent = create_entangled_rods(num_rods, total_effective_potential, jr.PRNGKey(1), rod_diameter=0.01, Nmax=300, N_outer=5, atol=1e-8, dt=1e-3, initial_q=q.reshape(-1).astype(jnp.float64))
    # ent2 = get_entanglement(q_ent)

    print(f"Entanglement after entangling: {ent2}")

    
    ax = plot_many_rods(q.reshape(-1, 5))
    ax.axis('equal')
    plt.savefig(f"{output_folder}/initial_rods.png", dpi=300)

    # Params
    params = {"col_rad": 2.e-3, "amp": 1.0}
    dt = 1e-4
    N_outer = 20
    Nmax = 50000
    atol = 1e-10

    # Run relaxation
    q_out = collision_relaxation(
        q.flatten(), total_harmonic_line, params, N_outer, Nmax, atol, dt, seg_len=1.0
    )    

    # If you have your own visualizer:
    
    ax = plot_many_rods(q_out.reshape(-1,5))
    ax.axis('equal')
    plt.savefig(f"{output_folder}/relaxed_rods.png", dpi=300)

    # save q
    jnp.save(f"{output_folder}/relaxed_q.npy", q_out)

# %%

