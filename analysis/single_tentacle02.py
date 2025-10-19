import jax
import jax.numpy as jnp

# ---------- Discrete Gauss linking integral (midpoint rule) ----------
def _segments(polyline: jnp.ndarray):
    P0 = polyline[:-1]
    P1 = polyline[1:]
    dP = P1 - P0
    mid = 0.5 * (P0 + P1)
    return dP, mid

def gauss_linking_number(poly1: jnp.ndarray, poly2: jnp.ndarray, eps: float = 1e-9):
    dA, midA = _segments(poly1)
    dB, midB = _segments(poly2)
    r = midA[:, None, :] - midB[None, :, :]                 # (NA,NB,3)
    r2 = jnp.sum(r * r, axis=-1) + eps
    r3 = r2 * jnp.sqrt(r2)
    cross = jnp.cross(dA[:, None, :], dB[None, :, :])       # (NA,NB,3)
    num = jnp.einsum('ijk,ijk->ij', cross, r)
    return jnp.sum(num / r3) / (4.0 * jnp.pi)

# ---------- Catmull–Rom spline sampler (uniform parameter) ----------
def catmull_rom_sample(ctrl: jnp.ndarray, n_samples: int) -> jnp.ndarray:
    """
    ctrl: (K,3) control points, K>=4. Interpolates P1..P_{K-2}. Ends are clamped.
    """
    K = ctrl.shape[0]
    segs = K - 3
    # Evenly distribute samples across segments
    t_all = jnp.linspace(0.0, segs, n_samples, endpoint=True)
    i = jnp.clip(jnp.floor(t_all).astype(int), 0, segs - 1)
    t = t_all - i

    Pm1 = ctrl[i + 0]   # P_{i}
    P0  = ctrl[i + 1]   # P_{i+1}
    P1  = ctrl[i + 2]   # P_{i+2}
    P2  = ctrl[i + 3]   # P_{i+3}

    t2 = t * t
    t3 = t2 * t
    # Catmull–Rom basis with tension=0.5
    C = 0.5 * (
        (-t3 + 2*t2 - t)[:, None] * Pm1 +
        ( 3*t3 - 5*t2 + 2)[:, None] * P0  +
        (-3*t3 + 4*t2 + t)[:, None] * P1 +
        ( t3 - t2)[:, None] * P2
    )
    return C

# ---------- Utility penalties ----------
def polyline_length(P: jnp.ndarray):
    return jnp.sum(jnp.linalg.norm(P[1:] - P[:-1], axis=-1))

def bending_penalty(P: jnp.ndarray):
    # Discrete second difference ||P_{i-1} - 2P_i + P_{i+1}||^2
    return jnp.sum(jnp.square(P[:-2] - 2*P[1:-1] + P[2:]))

# ---------- Objective: maximize Lk(tentacle, rod) ----------
def build_objective(rod_polyline: jnp.ndarray, base_point: jnp.ndarray,
                    n_samples=200, w_bend=1e-3, w_len=1e-2, L0=1.0):
    def objective(free_ctrl: jnp.ndarray):
        """
        free_ctrl: (K-1,3). The full control stack is [base_point; free_ctrl]
        """
        ctrl = jnp.concatenate([base_point[None, :], free_ctrl], axis=0)
        tentacle = catmull_rom_sample(ctrl, n_samples)
        lk = gauss_linking_number(tentacle, rod_polyline)
        bend = bending_penalty(tentacle)
        length = polyline_length(tentacle)
        loss = -(lk) + w_bend * bend + w_len * (length - L0)**2
        return loss, (lk, bend, length)
    return objective

# ---------- Simple Adam optimizer (pure JAX) ----------
@jax.jit
def adam_step(grad_fn, params, opt_state, m, v, t, lr=2e-2, b1=0.9, b2=0.999, eps=1e-8):
    g, aux = grad_fn(params)
    m = b1 * m + (1 - b1) * g
    v = b2 * v + (1 - b2) * (g * g)
    mhat = m / (1 - b1**t)
    vhat = v / (1 - b2**t)
    params = params - lr * mhat / (jnp.sqrt(vhat) + eps)
    return params, m, v, aux

# ---------- Example usage ----------
# Fixed rod (straight segment or your existing a_filament)
rod = jnp.linspace(jnp.array([0., 0., -0.2]), jnp.array([0., 0., 1.7]), 150)
base = jnp.array([0.2, 0.0, 0.0])  # tentacle base anchored here (change as desired)

# Initialize tentacle control points (K controls total; K>=4)
K = 6
free_init = jnp.stack([
    jnp.linspace(base[0], 0.15, K-1),
    jnp.linspace(base[1], 0.15, K-1),
    jnp.linspace(base[2], 1.00, K-1),
], axis=1)  # simple arcing guess


obj = build_objective(rod, base, n_samples=300, w_bend=2e-3, w_len=1e-2, L0=1.8)

# loss_and_grad returns ((loss, (lk, bend, length)), grad)
loss_and_grad = jax.value_and_grad(lambda p: obj(p), has_aux=True)

# Adam hyperparams
LR = 3e-2; B1 = 0.9; B2 = 0.999; EPS = 1e-8

@jax.jit
def adam_step(params, m, v, t):
    (loss, (lk, bend, length)), g = loss_and_grad(params)
    m = B1 * m + (1.0 - B1) * g
    v = B2 * v + (1.0 - B2) * (g * g)
    mhat = m / (1.0 - B1**t)
    vhat = v / (1.0 - B2**t)
    params = params - LR * mhat / (jnp.sqrt(vhat) + EPS)
    return params, m, v, loss, lk, bend, length

# --- init & loop ---
free = free_init
m = jnp.zeros_like(free)
v = jnp.zeros_like(free)

for t in range(1, 10001):
    free, m, v, loss, lk, bend, length = adam_step(free, m, v, t)
    if t % 100 == 0:
        print(f"iter {t:4d}  loss={float(loss): .6f}  Lk≈{float(lk): .4f}  bend={float(bend):.3e}  len={float(length):.3f}")

ctrl_final = jnp.concatenate([base[None,:], free], axis=0)
tentacle_final = catmull_rom_sample(ctrl_final, 600)
final_lk = gauss_linking_number(tentacle_final, rod)
print("Final Gauss linking (open curves):", float(final_lk))

# visualize with matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig,ax=plt.subplots(subplot_kw={"projection": "3d"}, figsize=(8,6))
ax.plot(rod[:,0], rod[:,1], rod[:,2], 'r-', label='Rod', linewidth=2)
ax.plot(tentacle_final[:,0], tentacle_final[:,1], tentacle_final[:,2], 'b-', label='Tentacle', linewidth=2) 
ax.scatter(ctrl_final[:,0], ctrl_final[:,1], ctrl_final[:,2], c='k', s=50, label='Control Points')
ax.view_init(elev=20., azim=30)
ax.set_box_aspect([1,1,1])
ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
ax.legend()
plt.show()