# single_tentacle_points_opt_robust.py
import jax
import jax.numpy as jnp
import numpy as onp

# =========================================
# Discrete Gauss linking integral (robust)
# =========================================
def gauss_linking_number(poly1: jnp.ndarray,
                         poly2: jnp.ndarray,
                         eps2: float = 1e-6) -> jnp.ndarray:
    """
    Midpoint discretization of the Gauss linking integral with masking near the
    singularity. Works for open curves (returns a real value).
    """
    def _segments(P):
        P0, P1 = P[:-1], P[1:]
        return (P1 - P0), 0.5 * (P0 + P1)

    dA, midA = _segments(poly1)               # (NA,3), (NA,3)
    dB, midB = _segments(poly2)               # (NB,3), (NB,3)

    r = midA[:, None, :] - midB[None, :, :]   # (NA,NB,3)
    r2 = jnp.sum(r * r, axis=-1)              # (NA,NB)

    # Mask out very close pairs to avoid 1/||r||^3 blow-ups
    mask = r2 > eps2
    r2_safe = jnp.where(mask, r2, 1.0)        # dummy safe value
    r3 = r2_safe * jnp.sqrt(r2_safe)

    cross = jnp.cross(dA[:, None, :], dB[None, :, :])    # (NA,NB,3)
    num = jnp.einsum('ijk,ijk->ij', cross, r)            # (NA,NB)
    contrib = jnp.where(mask, num / r3, 0.0)

    contrib = jnp.nan_to_num(contrib, nan=0.0, posinf=0.0, neginf=0.0)
    return jnp.sum(contrib) / (4.0 * jnp.pi)

# =======================
# Polyline regularization
# =======================
def polyline_length(P: jnp.ndarray) -> jnp.ndarray:
    return jnp.sum(jnp.linalg.norm(P[1:] - P[:-1], axis=-1))

def bending_penalty(P: jnp.ndarray) -> jnp.ndarray:
    # Discrete curvature via second differences
    return jnp.sum(jnp.square(P[:-2] - 2.0 * P[1:-1] + P[2:]))

# =====================================
# Clearance penalty (tentacle vs. rod)
# =====================================
def min_dist_point_polyline(Q: jnp.ndarray, P: jnp.ndarray, eps: float = 1e-9):
    """
    Q: (M,3) query points, P: (N,3) polyline vertices.
    Returns per-Q minimal distance to polyline segments.
    """
    A = P[:-1]          # (S,3)
    B = P[1:]           # (S,3)
    AB = B - A          # (S,3)
    AB2 = jnp.sum(AB * AB, axis=-1) + eps

    QA = Q[:, None, :] - A[None, :, :]                        # (M,S,3)
    t = jnp.sum(QA * AB[None, :, :], axis=-1) / AB2[None, :]  # (M,S)
    t = jnp.clip(t, 0.0, 1.0)

    proj = A[None, :, :] + t[..., None] * AB[None, :, :]      # (M,S,3)
    d2 = jnp.sum((Q[:, None, :] - proj) ** 2, axis=-1)        # (M,S)
    return jnp.sqrt(jnp.min(d2, axis=-1))                     # (M,)

def clearance_penalty(tent: jnp.ndarray,
                      rod: jnp.ndarray,
                      radius: float = 0.02,
                      k: float = 1e-2) -> jnp.ndarray:
    """
    Soft barrier: penalize tentacle vertices closer than `radius` to the rod.
    """
    d = min_dist_point_polyline(tent, rod)
    return k * jnp.sum(jnp.square(jnp.maximum(0.0, radius - d)))

# ====================================
# Objective builder (base point fixed)
# ====================================
def build_objective_points(rod_polyline: jnp.ndarray,
                           base_point: jnp.ndarray,
                           *,
                           w_bend: float = 1e9,
                           w_len: float = 1e2,
                           L0: float = 1.8,
                           w_clear: float = 1e-2,
                           clearance_R: float = 0.02,
                           eps2: float = 1e-6):
    """
    Returns an objective that takes 'free_points' of shape (N-1,3).
    The full tentacle is [base_point; free_points].
    """
    def objective(free_points: jnp.ndarray):
        tent = jnp.concatenate([base_point[None, :], free_points], axis=0)
        lk = gauss_linking_number(tent, rod_polyline, eps2=eps2)
        bend = bending_penalty(tent)
        length = polyline_length(tent)
        clear = clearance_penalty(tent, rod_polyline, radius=clearance_R, k=w_clear)
        loss = -(lk) + w_bend * bend + w_len * (length - L0) ** 2 + clear
        loss = jnp.nan_to_num(loss, nan=1e6, posinf=1e6, neginf=1e6)
        return loss, (lk, bend, length, clear)
    return objective

# ==========================
# Adam optimizer (with clip)
# ==========================
def make_adam_step(obj_fn,
                   lr: float = 5e-3,
                   b1: float = 0.9,
                   b2: float = 0.999,
                   eps: float = 1e-8,
                   clip: float = 1.0):
    """
    Builds a JITted Adam step with global-norm gradient clipping and NaN guards.
    """
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
        mhat = m / (1.0 - b1 ** t)
        vhat = v / (1.0 - b2 ** t)
        params = params - lr * mhat / (jnp.sqrt(vhat) + eps)

        # sanitize params
        params = jnp.nan_to_num(params, nan=0.0, posinf=0.0, neginf=0.0)
        return params, m, v, loss, aux
    return step

# ================
# Example set-up
# ================
def main():
    # Fixed rod: straight centerline (adjust endpoints as desired)
    rod = jnp.linspace(jnp.array([0., 0., -0.2]), jnp.array([0., 0., 1.7]), 220)

    # Tentacle base (anchored)
    base = jnp.array([0.20, 0.00, 0.00])

    # Number of vertices for the tentacle (including base)
    N = 1000

    # Initial free points: gentle arc toward rod and upward
    x_init = jnp.linspace(base[0], 0.05, N - 1)
    y_init = jnp.linspace(base[1], 0.00, N - 1)
    z_init = jnp.linspace(base[2], 1.50, N - 1)
    free_init = jnp.stack([x_init, y_init, z_init], axis=1)   # (N-1,3)

    # Tiny random jitter to avoid exact coincidences
    key = jax.random.key(0)
    free_init = free_init + 1e-4 * jax.random.normal(key, free_init.shape)

    # Build objective (weights and robust params)
    obj = build_objective_points(
        rod_polyline=rod,
        base_point=base,
        w_bend=2e-3,
        w_len=1e-2,
        L0=1.8,
        w_clear=1e-2,
        clearance_R=0.02,
        eps2=1e-6,
    )

    # Adam optimizer (smaller LR + clipping)
    adam_step = make_adam_step(obj_fn=lambda P: obj(P),
                               lr=5e-5, b1=0.9, b2=0.999, eps=1e-8, clip=1.0)

    # Initialize
    free = free_init
    m = jnp.zeros_like(free)
    v = jnp.zeros_like(free)

    # Optimize
    iters = 5000
    log_every = 100
    for t in range(1, iters + 1):
        free, m, v, loss, (lk, bend, length, clear) = adam_step(free, m, v, t)
        if t % log_every == 0:
            print(f"iter {t:4d}  loss={float(loss): .6f}  "
                  f"Lk≈{float(lk): .4f}  bend={float(bend):.3e}  "
                  f"len={float(length):.3f}  clear={float(clear):.3e}")

    # Final report
    tentacle_final = jnp.concatenate([base[None, :], free], axis=0)
    final_lk = gauss_linking_number(tentacle_final, rod, eps2=1e-6)
    final_len = float(polyline_length(tentacle_final))
    final_clear = float(clearance_penalty(tentacle_final, rod, radius=0.02, k=1e-2))

    print("\nOptimization done.")
    print("Final Gauss linking (open curves):", float(final_lk))
    print("Final tentacle length:", final_len)
    print("Final clearance penalty:", final_clear)

    # Optional: save for plotting elsewhere
    onp.save("tentacle_final.npy", onp.array(tentacle_final))
    onp.save("rod.npy", onp.array(rod))
    print("Saved: tentacle_final.npy, rod.npy")

    # visualization (requires matplotlib)
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(rod[:, 0], rod[:, 1], rod[:, 2], 'r-', label='Rod', lw=2)
        ax.plot(tentacle_final[:, 0], tentacle_final[:, 1], tentacle_final[:, 2],
                'b-', label='Tentacle', lw=2)
        ax.scatter(base[0], base[1], base[2], c='g', s=100, label='Base')
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.set_title('Tentacle Optimization Result')
        ax.legend()
        plt.show()
    except ImportError:
        print("matplotlib not available; skipping visualization.")

    # Optional quick sanity check for NaNs
    assert not onp.isnan(onp.array(final_lk)), "Final LK is NaN"
    assert not onp.isnan(onp.array(final_len)), "Final length is NaN"

if __name__ == "__main__":
    main()
