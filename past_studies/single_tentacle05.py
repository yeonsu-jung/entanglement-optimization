# tentacle_max_linking_points.py
import jax
import jax.numpy as jnp
import numpy as onp

# ===========================
# Your per-segment contribution
# ===========================
def compute_linking_number_cartesian(p_i, p_ii, p_j, p_jj):
    r_ij   = p_i  - p_j
    r_ijj  = p_i  - p_jj
    r_iij  = p_ii - p_j
    r_iijj = p_ii - p_jj

    tol = 1e-6
    n1 = jnp.cross(r_ij,  r_ijj);  n1 = n1 / (jnp.linalg.norm(n1)  + tol)
    n2 = jnp.cross(r_ijj, r_iijj); n2 = n2 / (jnp.linalg.norm(n2) + tol)
    n3 = jnp.cross(r_iijj, r_iij); n3 = n3 / (jnp.linalg.norm(n3) + tol)
    n4 = jnp.cross(r_iij,  r_ij);  n4 = n4 / (jnp.linalg.norm(n4)  + tol)

    tol = 0.0
    return -1.0/(4.0*jnp.pi) * jnp.abs(
        jnp.arcsin(jnp.clip(jnp.dot(n1,n2), -1.0+tol, 1.0-tol)) +
        jnp.arcsin(jnp.clip(jnp.dot(n2,n3), -1.0+tol, 1.0-tol)) +
        jnp.arcsin(jnp.clip(jnp.dot(n3,n4), -1.0+tol, 1.0-tol)) +
        jnp.arcsin(jnp.clip(jnp.dot(n4,n1), -1.0+tol, 1.0-tol))
    )

# ===========================
# Polyline <-> segments utils
# ===========================
def _segments(P: jnp.ndarray):
    return P[:-1], P[1:]  # (N-1,3), (N-1,3)

def _safe_seg_contrib(p_i, p_ii, p_j, p_jj, eps=1e-12):
    ok_i = jnp.linalg.norm(p_ii - p_i)  > eps
    ok_j = jnp.linalg.norm(p_jj - p_j)  > eps
    def _do():
        return compute_linking_number_cartesian(p_i, p_ii, p_j, p_jj)
    return jax.lax.cond(ok_i & ok_j, _do, lambda: 0.0)

_pair_vmap = jax.vmap(
    jax.vmap(_safe_seg_contrib, in_axes=(None, None, 0, 0)),
    in_axes=(0, 0, None, None)
)

def total_linking_number_polyline(A: jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:
    A0, A1 = _segments(A)
    B0, B1 = _segments(B)
    contribs = _pair_vmap(A0, A1, B0, B1)
    contribs = jnp.nan_to_num(contribs, nan=0.0, posinf=0.0, neginf=0.0)
    return jnp.sum(contribs)

# =======================
# Regularizers & penalties
# =======================
def polyline_length(P: jnp.ndarray) -> jnp.ndarray:
    return jnp.sum(jnp.linalg.norm(P[1:] - P[:-1], axis=-1))

def bending_penalty(P: jnp.ndarray) -> jnp.ndarray:
    return jnp.sum(jnp.square(P[:-2] - 2.0*P[1:-1] + P[2:]))

def min_dist_point_polyline(Q: jnp.ndarray, P: jnp.ndarray, eps: float = 1e-9):
    A, B = P[:-1], P[1:]
    AB = B - A
    AB2 = jnp.sum(AB*AB, axis=-1) + eps
    QA = Q[:, None, :] - A[None, :, :]
    t = jnp.sum(QA * AB[None, :, :], axis=-1) / AB2[None, :]
    t = jnp.clip(t, 0.0, 1.0)
    proj = A[None, :, :] + t[..., None] * AB[None, :, :]
    d2 = jnp.sum((Q[:, None, :] - proj)**2, axis=-1)
    return jnp.sqrt(jnp.min(d2, axis=-1))

def clearance_penalty(tent: jnp.ndarray, rod: jnp.ndarray,
                      radius: float = 0.02, k: float = 1e-2) -> jnp.ndarray:
    d = min_dist_point_polyline(tent, rod)
    return k * jnp.sum(jnp.square(jnp.maximum(0.0, radius - d)))

# ====================================
# Objective: maximize linking (min -LK)
# ====================================
def build_objective(rod: jnp.ndarray,
                    base: jnp.ndarray,
                    w_bend=2e-3, w_len=1e-2, L0=1.8,
                    w_clear=1e-2, clear_R=0.02):
    def objective(free_pts: jnp.ndarray):
        tent = jnp.concatenate([base[None, :], free_pts], axis=0)
        lk = total_linking_number_polyline(tent, rod)      # uses YOUR formula
        bend = bending_penalty(tent)
        length = polyline_length(tent)
        clear = clearance_penalty(tent, rod, radius=clear_R, k=w_clear)
        loss = (lk) + w_bend*bend + w_len*(length - L0)**2 + clear
        loss = jnp.nan_to_num(loss, nan=1e6, posinf=1e6, neginf=1e6)
        return loss, (lk, bend, length, clear)
    return objective

# ==========================
# Adam optimizer (with clip)
# ==========================
def make_adam_step(obj_fn, lr=3e-3, b1=0.9, b2=0.999, eps=1e-8, clip=1.0):
    loss_and_grad = jax.value_and_grad(obj_fn, has_aux=True)

    @jax.jit
    def step(params, m, v, t):
        (loss, aux), g = loss_and_grad(params)

        # global-norm clip
        g_norm = jnp.linalg.norm(g)
        scale = jnp.minimum(1.0, clip / (g_norm + 1e-12))
        g = g * scale

        m = b1 * m + (1.0 - b1) * g
        v = b2 * v + (1.0 - b2) * (g * g)
        mhat = m / (1.0 - b1**t)
        vhat = v / (1.0 - b2**t)
        params = params - lr * mhat / (jnp.sqrt(vhat) + eps)

        params = jnp.nan_to_num(params, nan=0.0, posinf=0.0, neginf=0.0)
        return params, m, v, loss, aux
    return step

# ================
# Example setup/run
# ================
def main():
    # Fixed rod: straight line along z (more points = smoother)
    rod = jnp.linspace(jnp.array([0., 0., -0.2]), jnp.array([0., 0., 1.7]), 250)

    # Tentacle base (anchored, off to the side)
    base = jnp.array([0.18, 0.00, 0.00])

    # Tentacle vertices (including base)
    N = 200
    # Initialize a gentle upward arc that leans toward the rod
    x_init = jnp.linspace(base[0],  0.06, N-1)
    y_init = jnp.linspace(base[1],  0.00, N-1)
    z_init = jnp.linspace(base[2],  1.50, N-1)
    free_init = jnp.stack([x_init, y_init, z_init], axis=1)

    # Tiny jitter to avoid coincident geometry
    key = jax.random.key(0)
    free_init = free_init + 1e-4 * jax.random.normal(key, free_init.shape)

    # Objective: maximize linking around rod
    obj = build_objective(rod, base,
                          w_bend=1e4, w_len=1e4, L0=1.8,
                          w_clear=1e-2, clear_R=0.02)

    # Adam with clipping
    adam_step = make_adam_step(lambda P: obj(P), lr=3e-3, clip=1.0)

    free = free_init
    m = jnp.zeros_like(free)
    v = jnp.zeros_like(free)

    iters = 5000
    for t in range(1, iters+1):
        free, m, v, loss, (lk, bend, length, clear) = adam_step(free, m, v, t)
        if t % 100 == 0:
            print(f"iter {t:4d}  loss={float(loss): .6f}  "
                  f"Lk≈{float(lk): .4f}  bend={float(bend):.3e}  "
                  f"len={float(length):.3f}  clear={float(clear):.3e}")

    tentacle_final = jnp.concatenate([base[None, :], free], axis=0)
    final_lk = total_linking_number_polyline(tentacle_final, rod)
    print("\nDone.")
    print("Final (pairwise) linking using your formula:", float(final_lk))
    print("Final length:", float(polyline_length(tentacle_final)))

    # Save arrays if you want to visualize elsewhere
    onp.save("tentacle_final.npy", onp.array(tentacle_final))
    onp.save("rod.npy", onp.array(rod))
    print("Saved: tentacle_final.npy, rod.npy")

    # visualization
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(free_init[:,0], free_init[:,1], free_init[:,2], 'g--', lw=1, label='init tentacle')
        ax.plot(rod[:,0], rod[:,1], rod[:,2], 'k-', lw=2, label='rod')
        ax.plot(tentacle_final[:,0], tentacle_final[:,1], tentacle_final[:,2],
                'r-', lw=2, label='tentacle')
        ax.set_box_aspect([1,1,1])
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.legend()
        ax.axis('equal')
        plt.show()
    except ImportError:
        pass

if __name__ == "__main__":
    main()
