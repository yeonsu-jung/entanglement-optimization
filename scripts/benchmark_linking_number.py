"""
Benchmark: compute_linking_number vs compute_linking_number_arai
Tests speed, accuracy, and gradient stability.
"""

import sys
import time
import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, grad, vmap

sys.path.insert(0, ".")
from potentials import compute_linking_number, compute_linking_number_arai

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def angles_from_direction(ux, uy, uz):
    """Return (phi, theta) for a unit vector."""
    phi = float(np.arccos(np.clip(uz, -1, 1)))
    theta = float(np.arctan2(uy, ux))
    return phi, theta


def call_both(args):
    orig = compute_linking_number(*args)
    arai = compute_linking_number_arai(*args)
    return float(orig), float(arai)


# ─────────────────────────────────────────────
# Test cases
# ─────────────────────────────────────────────

L = 1.0  # rod length

test_cases = {}

# 1. Perpendicular crossing — clear positive crossing
#    Rod i: along +x from (0,0,0)
#    Rod j: along +y, crosses at (0.5, 0, 0) but offset in z
phi_x, theta_x = angles_from_direction(1, 0, 0)
phi_y, theta_y = angles_from_direction(0, 1, 0)
test_cases["cross_clear"] = (
    0.0, 0.0, 0.0, phi_x, theta_x,   # rod i: start (0,0,0) along +x
    0.5, -0.5, 0.1, phi_y, theta_y,  # rod j: start (0.5,-0.5,0.1) along +y — crosses rod i
    L,
)

# 2. Perpendicular crossing — other handedness (rod j below)
test_cases["cross_clear_neg"] = (
    0.0, 0.0, 0.0,  phi_x, theta_x,
    0.5, -0.5, -0.1, phi_y, theta_y,
    L,
)

# 3. Parallel rods (no crossing) — expect 0
test_cases["parallel"] = (
    0.0, 0.0, 0.0, phi_x, theta_x,
    0.0, 1.0, 0.0, phi_x, theta_x,
    L,
)

# 4. Antiparallel rods — expect 0
phi_neg_x, theta_neg_x = angles_from_direction(-1, 0, 0)
test_cases["antiparallel"] = (
    0.0, 0.0, 0.0,  phi_x,     theta_x,
    0.0, 1.0, 0.0,  phi_neg_x, theta_neg_x,
    L,
)

# 5. Skew, far apart — expect ≈ 0
test_cases["skew_far"] = (
    0.0, 0.0,  0.0, phi_x, theta_x,
    0.0, 10.0, 5.0, phi_y, theta_y,
    L,
)

# 6. Nearly parallel (small angle between rods) — stability test
eps = 1e-4
phi_near, theta_near = angles_from_direction(1, eps, 0)
phi_near /= np.sqrt(1 + eps**2)  # already normalised by arccos
test_cases["nearly_parallel"] = (
    0.0, 0.0, 0.0, phi_x,    theta_x,
    0.0, 1.0, 0.0, phi_near, theta_near,
    L,
)

# 7. Touching / very close (numerically demanding)
test_cases["nearly_touching"] = (
    0.0, 0.0, 0.0,    phi_x, theta_x,
    0.5, -0.5, 1e-5,  phi_y, theta_y,
    L,
)

# 8. Coincident start points (degenerate)
test_cases["coincident_starts"] = (
    0.0, 0.0, 0.0, phi_x, theta_x,
    0.0, 0.0, 0.0, phi_y, theta_y,
    L,
)

# ─────────────────────────────────────────────
# 1. Accuracy
# ─────────────────────────────────────────────

print("=" * 60)
print("ACCURACY")
print("=" * 60)
print(f"{'Case':<22} {'Original':>12} {'Arai':>12} {'Diff':>12}")
print("-" * 60)

accuracy_results = {}
for name, args in test_cases.items():
    orig, arai = call_both(args)
    diff = abs(orig - arai)
    accuracy_results[name] = (orig, arai, diff)
    print(f"{name:<22} {orig:>12.6f} {arai:>12.6f} {diff:>12.6f}")

# ─────────────────────────────────────────────
# 2. Gradient stability
# ─────────────────────────────────────────────

print()
print("=" * 60)
print("GRADIENT STABILITY  (NaN / Inf detection)")
print("=" * 60)
print(f"{'Case':<22} {'Orig NaN/Inf':>14} {'Arai NaN/Inf':>14}")
print("-" * 60)

# Grad w.r.t. all 10 continuous args (not l)
# Wrap to accept a tuple
def orig_fn(xi, yi, zi, pi, ti, xj, yj, zj, pj, tj):
    return compute_linking_number(xi, yi, zi, pi, ti, xj, yj, zj, pj, tj, L)

def arai_fn(xi, yi, zi, pi, ti, xj, yj, zj, pj, tj):
    return compute_linking_number_arai(xi, yi, zi, pi, ti, xj, yj, zj, pj, tj, L)

grad_orig = jit(grad(orig_fn, argnums=tuple(range(10))))
grad_arai = jit(grad(arai_fn, argnums=tuple(range(10))))

for name, args in test_cases.items():
    a = [float(x) for x in args[:10]]
    go = grad_orig(*a)
    ga = grad_arai(*a)
    orig_bad = any(np.isnan(float(g)) or np.isinf(float(g)) for g in go)
    arai_bad = any(np.isnan(float(g)) or np.isinf(float(g)) for g in ga)
    print(f"{name:<22} {'YES' if orig_bad else 'ok':>14} {'YES' if arai_bad else 'ok':>14}")
    if orig_bad:
        print(f"  orig grads: {[float(g) for g in go]}")
    if arai_bad:
        print(f"  arai grads: {[float(g) for g in ga]}")

# ─────────────────────────────────────────────
# 3. Speed (JIT warm-up then timed)
# ─────────────────────────────────────────────

print()
print("=" * 60)
print("SPEED  (JIT warm-up then timed)")
print("=" * 60)

jit_orig = jit(compute_linking_number)
jit_arai = jit(compute_linking_number_arai)

# Warm up
ref_args = test_cases["cross_clear"]
jit_orig(*ref_args).block_until_ready()
jit_arai(*ref_args).block_until_ready()

N_REPEATS = 10_000

for name, args in list(test_cases.items())[:3]:  # subset for speed
    t0 = time.perf_counter()
    for _ in range(N_REPEATS):
        jit_orig(*args).block_until_ready()
    t_orig = (time.perf_counter() - t0) / N_REPEATS * 1e6  # µs

    t0 = time.perf_counter()
    for _ in range(N_REPEATS):
        jit_arai(*args).block_until_ready()
    t_arai = (time.perf_counter() - t0) / N_REPEATS * 1e6  # µs

    print(f"\n  [{name}]")
    print(f"    Original : {t_orig:.3f} µs/call")
    print(f"    Arai     : {t_arai:.3f} µs/call")
    print(f"    Ratio    : {t_arai/t_orig:.2f}x  (>1 means Arai slower)")

# ─────────────────────────────────────────────
# 4. Vmap (batch) speed
# ─────────────────────────────────────────────

print()
print("=" * 60)
print("BATCH SPEED (vmap over 1000 random pairs)")
print("=" * 60)

rng = np.random.default_rng(42)
N = 1000

def rand_args():
    pos = rng.uniform(-2, 2, (N, 6))
    angles = rng.uniform(0, np.pi, (N, 4))
    ls = np.full(N, L)
    return (pos[:, 0], pos[:, 1], pos[:, 2],
            angles[:, 0], angles[:, 1],
            pos[:, 3], pos[:, 4], pos[:, 5],
            angles[:, 2], angles[:, 3],
            ls)

batch_args = rand_args()

vmap_orig = jit(vmap(compute_linking_number))
vmap_arai = jit(vmap(compute_linking_number_arai))

# Warm up
vmap_orig(*batch_args).block_until_ready()
vmap_arai(*batch_args).block_until_ready()

N_BATCH_REPS = 500
t0 = time.perf_counter()
for _ in range(N_BATCH_REPS):
    vmap_orig(*batch_args).block_until_ready()
t_vorig = (time.perf_counter() - t0) / N_BATCH_REPS * 1e3  # ms

t0 = time.perf_counter()
for _ in range(N_BATCH_REPS):
    vmap_arai(*batch_args).block_until_ready()
t_varai = (time.perf_counter() - t0) / N_BATCH_REPS * 1e3  # ms

print(f"  Original  (N={N}): {t_vorig:.3f} ms/batch  ({t_vorig/N*1000:.3f} µs/pair)")
print(f"  Arai      (N={N}): {t_varai:.3f} ms/batch  ({t_varai/N*1000:.3f} µs/pair)")
print(f"  Ratio     : {t_varai/t_vorig:.2f}x")

# ─────────────────────────────────────────────
# 5. Consistency on random pairs
# ─────────────────────────────────────────────

print()
print("=" * 60)
print("CONSISTENCY  (random pairs, max & mean abs diff)")
print("=" * 60)

out_orig = np.array(vmap_orig(*batch_args))
out_arai = np.array(vmap_arai(*batch_args))
diffs = np.abs(out_orig - out_arai)
print(f"  Max  |orig - arai| : {diffs.max():.6f}")
print(f"  Mean |orig - arai| : {diffs.mean():.6f}")
print(f"  Std  |orig - arai| : {diffs.std():.6f}")
print(f"  orig range         : [{out_orig.min():.4f}, {out_orig.max():.4f}]")
print(f"  arai range         : [{out_arai.min():.4f}, {out_arai.max():.4f}]")

# ─────────────────────────────────────────────
# 6. tol sweep in compute_linking_number
# ─────────────────────────────────────────────

print()
print("=" * 60)
print("TOL SWEEP  (normalization softener in compute_linking_number)")
print("=" * 60)

def make_cln_with_tol(tol_val):
    def fn(x_i, y_i, z_i, phi_i, theta_i, x_j, y_j, z_j, phi_j, theta_j, l):
        p_i = jnp.array([x_i, y_i, z_i])
        p_j = jnp.array([x_j, y_j, z_j])
        u_i = jnp.array([jnp.sin(phi_i)*jnp.cos(theta_i), jnp.sin(phi_i)*jnp.sin(theta_i), jnp.cos(phi_i)])
        u_j = jnp.array([jnp.sin(phi_j)*jnp.cos(theta_j), jnp.sin(phi_j)*jnp.sin(theta_j), jnp.cos(phi_j)])
        p_ii = p_i + l * u_i
        p_jj = p_j + l * u_j
        r_ij   = p_i  - p_j
        r_ijj  = p_i  - p_jj
        r_iij  = p_ii - p_j
        r_iijj = p_ii - p_jj
        n1 = jnp.cross(r_ij,   r_ijj);  n1 = n1 / (jnp.linalg.norm(n1) + tol_val)
        n2 = jnp.cross(r_ijj,  r_iijj); n2 = n2 / (jnp.linalg.norm(n2) + tol_val)
        n3 = jnp.cross(r_iijj, r_iij);  n3 = n3 / (jnp.linalg.norm(n3) + tol_val)
        n4 = jnp.cross(r_iij,  r_ij);   n4 = n4 / (jnp.linalg.norm(n4) + tol_val)
        return -1/4/jnp.pi * jnp.abs(
            jnp.arcsin(jnp.clip(jnp.dot(n1, n2), -1., 1.))
          + jnp.arcsin(jnp.clip(jnp.dot(n2, n3), -1., 1.))
          + jnp.arcsin(jnp.clip(jnp.dot(n3, n4), -1., 1.))
          + jnp.arcsin(jnp.clip(jnp.dot(n4, n1), -1., 1.)))
    return fn

tol_values = [1e-3, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 0.0]
focus_cases = ["cross_clear", "nearly_touching", "coincident_starts"]

# Header
header = f"{'tol':<12}" + "".join(f"{c:>20}" for c in focus_cases)
print(header)
print("-" * (12 + 20 * len(focus_cases)))

grad_results = {}
for tol_val in tol_values:
    fn = jit(make_cln_with_tol(tol_val))
    gfn = jit(grad(make_cln_with_tol(tol_val), argnums=tuple(range(10))))
    row = f"{tol_val:<12.2e}"
    grad_row = {}
    for name in focus_cases:
        args = test_cases[name]
        val = float(fn(*args))
        scalar_args = [float(x) for x in args[:10]]
        gs = gfn(*scalar_args, float(args[10]))
        bad = any(np.isnan(float(g)) or np.isinf(float(g)) for g in gs)
        grad_row[name] = bad
        marker = " *NaN-grad*" if bad else ""
        row += f"{val:>12.6f}{marker:>8}"
    grad_results[tol_val] = grad_row
    print(row)

print()
print("  * = gradient contains NaN or Inf at that tol")
print(f"  Arai reference: cross_clear={float(compute_linking_number_arai(*test_cases['cross_clear'])):.6f}"
      f"  nearly_touching={float(compute_linking_number_arai(*test_cases['nearly_touching'])):.6f}"
      f"  coincident_starts={float(compute_linking_number_arai(*test_cases['coincident_starts'])):.6f}")
