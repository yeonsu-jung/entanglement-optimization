# single_tentacle_points_opt.py
import jax
import jax.numpy as jnp
import numpy as onp

# =========================================
# Discrete Gauss linking integral (midpoint)
# =========================================
def _segments(P: jnp.ndarray):
    P0 = P[:-1]
    P1 = P[1:]
    dP = P1 - P0
    mid = 0.5 * (P0 + P1)
    return dP, mid

def gauss_linking_number(poly1: jnp.ndarray, poly2: jnp.ndarray,
                         eps2: float = 1e-6):
    """
    Robust midpoint Gauss integral with masking & NaN guards.
    """
    def _segments(P):
        P0, P1 = P[:-1], P[1:]
        return (P1 - P0), 0.5 * (P0 + P1)

    dA, midA = _segments(poly1)               # (NA,3), (NA,3)
    dB, midB = _segments(poly2)               # (NB,3), (NB,3)

    r = midA[:, None, :] - midB[None, :, :]   # (NA,NB,3)
    r2 = jnp.sum(r * r, axis=-1)              # (NA,NB)

    # Mask very close pairs (singular kernel); treat as zero contribution
    mask = r2 > eps2
    r2_safe = jnp.where(mask, r2, 1.0)        # dummy to keep shapes
    r3 = r2_safe * jnp.sqrt(r2_safe)

    cross = jnp.cross(dA[:, None, :], dB[None, :, :])    # (NA,NB,3)
    num = jnp.einsum('ijk,ijk->ij', cross, r)            # (NA,NB)
    contrib = jnp.where(mask, num / r3, 0.0)

    # NaN / Inf guard (paranoid)
    contrib = jnp.nan_to_num(contrib, nan=0.0, posinf=0.0, neginf=0.0)
    return jnp.sum(contrib) / (4.0 * jnp.pi)


# =======================
# Polyline regularization
# =======================
def polyline_length(P: jnp.ndarray) -> jnp.ndarray:
    return jnp.sum(jnp.linalg.norm(P[1:] - P[:-1], axis=-1))

def bending_penalty(P: jnp.ndarray) -> jnp.ndarray:
    # Discrete curvature via second difference
    return jnp.sum(jnp.square(P[:-2] - 2.0 * P[1:-1] + P[2:]))

# ====================================
# Objective builder (base point fixed)
# ====================================
def build_objective_points(rod_polyline: jnp.ndarray,
                           base_point: jnp.ndarray,
                           w_bend: float = 2e-3,
                           w_len: float = 1e-2,
                           L0: float = 1.8):
    """
    Returns objective that takes 'free_points' of shape (N-1,3).
    The full tentacle is [base_point; free_points].
    """
    def objective(free_points: jnp.ndarray):
        tentacle = jnp.concatenate([base_point[None, :], free_points], axis=0)
        lk = gauss_linking_number(tentacle, rod_polyline)
        bend = bending_penalty(tentacle)
        length = polyline_length(tentacle)
        loss = -(lk) + w_bend * bend + w_len * (length - L0) ** 2
        return loss, (lk, bend, length)
    return objective

# ======================
# Simple Adam (JIT-safe)
# ======================
def make_adam_step(obj_fn, lr=3e-2, b1=0.9, b2=0.999, eps=1e-8):
    loss_and_grad = jax.value_and_grad(obj_fn, has_aux=True)

    @jax.jit
    def step(params, m, v, t):
        (loss, aux), g = loss_and_grad(params)
        m = b1 * m + (1.0 - b1) * g
        v = b2 * v + (1.0 - b2) * (g * g)
        mhat = m / (1.0 - b1 ** t)
        vhat = v / (1.0 - b2 ** t)
        params = params - lr * mhat / (jnp.sqrt(vhat) + eps)
        return params, m, v, loss, aux
    return step

# ================
# Example set-up
# ================
def main():
    # Fixed rod: straight centerline (your earlier a_filament)
    rod = jnp.linspace(jnp.array([0., 0., -0.2]), jnp.array([0., 0., 1.7]), 200)

    # Tentacle base (anchored)
    base = jnp.array([0.20, 0.00, 0.00])

    # Number of vertices for the tentacle (including base)
    N = 120
    # Initialize free points as a gentle arc curving toward the rod and upward
    x_init = jnp.linspace(base[0], 0.05, N - 1)
    y_init = jnp.linspace(base[1], 0.00, N - 1)
    z_init = jnp.linspace(base[2], 1.50, N - 1)
    free_init = jnp.stack([x_init, y_init, z_init], axis=1)   # shape (N-1,3)

    # Build objective: maximize LK with regularizers
    obj = build_objective_points(rod, base, w_bend=2e-3, w_len=1e-2, L0=1.8)

    # Adam init
    free = free_init
    m = jnp.zeros_like(free)
    v = jnp.zeros_like(free)
    adam_step = make_adam_step(lambda P: obj(P))

    # Optimize
    iters = 1200
    log_every = 100
    for t in range(1, iters + 1):
        free, m, v, loss, (lk, bend, length) = adam_step(free, m, v, t)
        if t % log_every == 0:
            print(f"iter {t:4d}  loss={float(loss): .6f}  "
                  f"Lk≈{float(lk): .4f}  bend={float(bend):.3e}  len={float(length):.3f}")

    # Final report
    tentacle_final = jnp.concatenate([base[None, :], free], axis=0)
    final_lk = gauss_linking_number(tentacle_final, rod)
    final_len = float(polyline_length(tentacle_final))
    print("\nOptimization done.")
    print("Final Gauss linking (open curves):", float(final_lk))
    print("Final tentacle length:", final_len)

    # (Optional) quick numpy conversion for downstream plotting
    tent_np = onp.array(tentacle_final)
    rod_np = onp.array(rod)
    # You can plot with matplotlib in your own script:
    #   from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    #   import matplotlib.pyplot as plt
    #   fig = plt.figure(); ax = fig.add_subplot(111, projection='3d')
    #   ax.plot(rod_np[:,0], rod_np[:,1], rod_np[:,2])
    #   ax.plot(tent_np[:,0], tent_np[:,1], tent_np[:,2])
    #   plt.show()

if __name__ == "__main__":
    main()
