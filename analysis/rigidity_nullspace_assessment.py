"""Robust nullspace assessment of the segment-distance Jacobian for N rods.

Features:
- Builds Jacobian J of all pairwise segment distances wrt endpoints (6N DOF).
- Handles overdetermined regime (constraints m >> dof n).
- SVD with full right singular vectors; tolerance sweep across multipliers.
- Reports rank/nullspace dimension vs tolerance, rigid-mode residuals, and overlap of rigid modes with computed nullspace.
"""
import sys
sys.path.append('/Users/yeonsu/GitHub/entanglement-optimization-combined/entanglement-optimization/core')

import numpy as np
import jax
import jax.numpy as jnp

from transforms import q_to_x
from potentials import dist_lin_seg_over_ij

DATA_PATH = '/Users/yeonsu/GitHub/entanglement-optimization-combined/entanglement-optimization/entangle_and_nudge_04/q_history.npy'

# ---------------- Load configuration ----------------
q_hist = np.load(DATA_PATH)
q = q_hist[-1]
x = q_to_x(q)
num_rods = x.shape[0]
print(f"Rods: {num_rods}")

# Pair indices
ii, jj = np.triu_indices(num_rods, k=1)
num_pairs = ii.size
print(f"Pairs: {num_pairs}")

# -------------- Distance map --------------

def all_dists_wrt_x(x_flat: jnp.ndarray) -> jnp.ndarray:
    x_arr = x_flat.reshape(num_rods, 6)
    r1 = x_arr[:, :3]
    r2 = x_arr[:, 3:]
    return dist_lin_seg_over_ij(r1, r2, ii, jj)

x_flat = jnp.asarray(x).reshape(-1)

# -------------- Jacobian --------------
J = jax.jacobian(all_dists_wrt_x)(x_flat)
J_np = np.asarray(J)
m, n = J_np.shape
print(f"Jacobian shape: {J_np.shape}  (m constraints={m}, n dof={n})")

# -------------- Rigid modes --------------

def build_rigid_modes(x_flat_np: np.ndarray):
    x_arr = x_flat_np.reshape(num_rods, 6)
    pts = np.concatenate([x_arr[:, :3], x_arr[:, 3:]], axis=0)  # (2N,3)
    center = pts.mean(axis=0)
    rel = pts - center
    modes = []
    labels = []
    for axis, vec in enumerate(np.eye(3)):
        v = np.zeros_like(x_arr)
        v[:, :3] = vec
        v[:, 3:] = vec
        modes.append(v.reshape(-1))
        labels.append(f"trans_{'xyz'[axis]}")
    for axis, omega in enumerate(np.eye(3)):
        rot_disp = np.cross(rel, omega)
        v = np.zeros_like(x_arr)
        v[:, :3] = rot_disp[:num_rods]
        v[:, 3:] = rot_disp[num_rods:]
        modes.append(v.reshape(-1))
        labels.append(f"rot_{'xyz'[axis]}")
    return np.stack(modes, axis=1), labels

rigid_modes, rigid_labels = build_rigid_modes(np.asarray(x_flat))

# Gram-Schmidt

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

# Residuals for rigid modes
rigid_res = np.linalg.norm(J_np @ rigid_orth, axis=0)
for lab, res in zip(rigid_labels, rigid_res):
    print(f"|J * {lab}| = {res:.3e}")
print(f"Rigid residuals: mean={rigid_res.mean():.3e} max={rigid_res.max():.3e}")

# -------------- SVD and tolerance sweep --------------
U, S, Vh = np.linalg.svd(J_np, full_matrices=True)
base = np.finfo(float).eps * max(m, n) * (S[0] if S.size else 1.0)
scales = [0.1, 1.0, 10.0, 100.0, 1000.0]

print("\nTolerance sweep:")
for s in scales:
    tol = base * s
    rank = int(np.sum(S > tol))
    null_dim = n - rank
    print(f"  tol={tol:.2e}: rank={rank}, null_dim={null_dim}")

# Choose a default tol = base
rank = int(np.sum(S > base))
null_basis = Vh[rank:].T
null_dim = null_basis.shape[1]
print(f"\nSelected tol={base:.2e}: rank={rank}, null_dim={null_dim}")

# Project rigid modes into nullspace and measure alignment
if null_dim > 0:
    null_orth = gram_schmidt(null_basis)
    projections = null_orth @ (null_orth.T @ rigid_orth)
    errs = np.linalg.norm(rigid_orth - projections, axis=0)
    angles = []
    for i in range(rigid_orth.shape[1]):
        v = rigid_orth[:, i]
        vp = projections[:, i]
        num = np.dot(v, vp)
        den = np.linalg.norm(v) * np.linalg.norm(vp) + 1e-300
        angle = np.arccos(np.clip(num / den, -1.0, 1.0)) if np.linalg.norm(vp) > 1e-12 else np.pi/2
        angles.append(angle)
    print("Projection error norms (rigid vs nullspace):", errs)
    print("Angles (rad):", angles)
else:
    print("[INFO] Nullspace empty at selected tolerance; consider lowering tol.")

print("\nDone.")
