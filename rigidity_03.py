"""Rigidity analysis and robust rigid mode extraction.

This script focuses on reliably recovering rigid-body null directions (3 translations + 3 rotations)
from the Jacobian of all pairwise segment-segment distances with respect to endpoint coordinates.

Key steps:
1. Load final configuration and flatten endpoints.
2. Build distance map and its Jacobian using JAX.
3. Perform SVD and adaptively determine nullspace tolerance.
4. Construct analytic rigid modes (translations, rotations about centroid).
5. Orthonormalize modes and verify they lie in (or near) the SVD nullspace.
6. Report quantitative diagnostics (residual norms, projection errors, angles).

Optional extensions (not implemented here):
- Use squared distances to smooth regime transitions.
- Build a sparse rigidity matrix if many rods.
- Incorporate constraints or pinned endpoints.
"""

import sys
sys.path.append('/Users/yeonsu/GitHub/entanglement-optimization-combined/entanglement-optimization/core')

import numpy as np
import jax
import jax.numpy as jnp

from transforms import q_to_x
from potentials import dist_lin_seg_over_ij

DATA_PATH = '/Users/yeonsu/GitHub/entanglement-optimization-combined/entanglement-optimization/entangle_and_nudge_04/q_history.npy'

# ----------------------- Load configuration -----------------------
q_hist = np.load(DATA_PATH)
q = q_hist[-1]
x = q_to_x(q)  # shape (num_rods, 6)
num_rods = x.shape[0]

ii, jj = np.triu_indices(num_rods, k=1)
num_pairs = ii.size

# ---------------- Distance mapping function -----------------------
def all_dists_wrt_x(x_flat: jnp.ndarray) -> jnp.ndarray:
    """Return vector of all pairwise segment distances given flattened endpoints.
    x_flat: shape (6*num_rods,)
    returns: shape (num_pairs,)
    """
    x_arr = x_flat.reshape(num_rods, 6)
    _r1 = x_arr[:, :3]
    _r2 = x_arr[:, 3:]
    return dist_lin_seg_over_ij(_r1, _r2, ii, jj)

x_flat = jnp.asarray(x).reshape(-1)

# Translation invariance sanity
trans_test = all_dists_wrt_x(x_flat + 1.0) - all_dists_wrt_x(x_flat)
print("Max |distance change| under uniform translation:", float(jnp.abs(trans_test).max()))

# -------------------- Jacobian via JAX ----------------------------
J = jax.jacobian(all_dists_wrt_x)(x_flat)  # shape (num_pairs, 6*num_rods)
J_np = np.asarray(J)
m, n = J_np.shape
print(f"Jacobian shape: {J_np.shape}  (rows=pairs={m}, cols=6*num_rods={n})")

# -------------------- SVD & nullspace -----------------------------
U, S, Vh = np.linalg.svd(J_np, full_matrices=True)  # need full Vh (n x n) to access right nullspace for wide matrix
eps = np.finfo(float).eps
base_tol = eps * max(m, n) * (S[0] if S.size else 1.0)

# Adaptive tolerance: also look for a gap near the tail.
if S.size > 1:
    tail_ratios = S[:-1] / (S[1:] + 1e-300)
    # large jump indicates boundary between nonzero and near-zero singular values
    jump_idx = np.argmax(tail_ratios)
    gap_ratio = tail_ratios[jump_idx]
else:
    jump_idx = 0
    gap_ratio = 1.0

adaptive_tol = base_tol
if gap_ratio > 1e6:  # strong evidence of a gap
    adaptive_tol = max(base_tol, S[jump_idx+1] * 10.0)

tol = adaptive_tol
rank = int(np.sum(S > tol))
# For wide matrix (m<n) with full_matrices=True, Vh shape (n,n); rows rank..n-1 span right nullspace
null_basis = Vh[rank:].T  # (n, n-rank)
null_dim = null_basis.shape[1]

print("Singular values (first 12):", S[:12])
print("Base tol:", base_tol, "Adaptive tol:", adaptive_tol)
print("Estimated rank:", rank)
print("Nullspace dimension:", null_dim)

# ------------------ Build rigid motions ---------------------------
def build_rigid_modes(x_flat_np: np.ndarray):
    x_arr = x_flat_np.reshape(num_rods, 6)
    pts = np.concatenate([x_arr[:, :3], x_arr[:, 3:]], axis=0)  # (2N,3)
    center = pts.mean(axis=0)
    rel = pts - center
    modes = []
    labels = []
    # Translations (x,y,z)
    for axis, vec in enumerate(np.eye(3)):
        v = np.zeros_like(x_arr)
        v[:, :3] = vec
        v[:, 3:] = vec
        modes.append(v.reshape(-1))
        labels.append(f"trans_{'xyz'[axis]}")
    # Rotations about x,y,z : delta p = omega x (p - c)
    for axis, omega in enumerate(np.eye(3)):
        rot_disp = np.cross(rel, omega)  # (2N,3)
        v = np.zeros_like(x_arr)
        v[:, :3] = rot_disp[:num_rods]
        v[:, 3:] = rot_disp[num_rods:]
        modes.append(v.reshape(-1))
        labels.append(f"rot_{'xyz'[axis]}")
    M = np.stack(modes, axis=1)  # (6*num_rods, 6)
    return M, labels

rigid_modes, rigid_labels = build_rigid_modes(np.asarray(x_flat))

# Orthonormalize rigid modes (Gram-Schmidt)
def gram_schmidt(A: np.ndarray) -> np.ndarray:
    B = []
    for i in range(A.shape[1]):
        v = A[:, i].copy()
        for b in B:
            v -= np.dot(b, v) * b
        nrm = np.linalg.norm(v)
        if nrm > 0:
            B.append(v / nrm)
    return np.stack(B, axis=1)

rigid_orth = gram_schmidt(rigid_modes)
print("Rigid modes orthonormalized shape:", rigid_orth.shape)

# ------------------ Verification metrics -------------------------
def residual_norms(J: np.ndarray, modes: np.ndarray) -> np.ndarray:
    return np.linalg.norm(J @ modes, axis=0)

residuals = residual_norms(J_np, rigid_orth)
for lab, res in zip(rigid_labels, residuals):
    print(f"Residual |J*v| for {lab}: {res:.3e}")
print("Mean residual:", residuals.mean(), "Max residual:", residuals.max())

# Projection of rigid modes into SVD nullspace
if null_dim > 0:
    # Orthonormalize null_basis for stable projection
    null_orth = gram_schmidt(null_basis) if null_dim > 1 else null_basis
    projections = null_orth @ (null_orth.T @ rigid_orth)  # reconstruct components lying in nullspace
    proj_errors = np.linalg.norm(rigid_orth - projections, axis=0)
    angles = []
    for i in range(rigid_orth.shape[1]):
        v = rigid_orth[:, i]
        vp = projections[:, i]
        num = np.dot(v, vp)
        den = np.linalg.norm(v) * np.linalg.norm(vp) + 1e-300
        angle = np.arccos(np.clip(num / den, -1.0, 1.0)) if np.linalg.norm(vp) > 1e-12 else np.pi/2
        angles.append(angle)
    print("Projection error norms (||v - P_null v||):", proj_errors)
    print("Angles (rad) between rigid modes and their nullspace projection:", angles)
else:
    print("[WARN] Nullspace dimension is zero; tolerance may be too strict.")

# If residuals are tiny but projection errors high, tolerance likely too strict.
if residuals.max() < 1e-8 and null_dim < 6:
    print("[INFO] Rigid residuals small but nullspace missing expected modes; consider relaxing tolerance.")

print("\nSummary:")
print(f"Expected rigid DOF: 6; recovered nullspace dim: {null_dim}; max residual: {residuals.max():.3e}")

if __name__ == '__main__':
    pass  # Script executes on import; no further action.

