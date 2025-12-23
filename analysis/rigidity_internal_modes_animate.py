"""Animate representative internal modes and export frames.

This script:
- Computes internal (non-rigid) nullspace modes for the segment-distance Jacobian.
- Ranks modes by total endpoint displacement and by translation vs rotation-like content.
- Animates a few selected modes with sinusoidal excursions and saves PNG frames.
"""
import sys
sys.path.append('/Users/yeonsu/GitHub/entanglement-optimization-combined/entanglement-optimization/core')

import numpy as np
import jax
import jax.numpy as jnp
from transforms import q_to_x
from potentials import dist_lin_seg_over_ij
from visualizations import prep_for_polyscope

DATA_PATH = '/Users/yeonsu/GitHub/entanglement-optimization-combined/entanglement-optimization/entangle_and_nudge_04/q_history.npy'
OUT_DIR = 'internal_mode_frames'
NUM_FRAMES = 60
AMPLITUDE = 2.0e-2  # excursion amplitude
SELECT_COUNT = 3    # animate top 3 modes by chosen metric
METRIC = 'disp'     # one of {'disp','trans','rot'}

# ---------------- Load config ----------------
q_hist = np.load(DATA_PATH)
q = q_hist[-1]
x = q_to_x(q)
num_rods = x.shape[0]

# Pairs
ii, jj = np.triu_indices(num_rods, k=1)

# Distance map

def all_dists_wrt_x(x_flat: jnp.ndarray) -> jnp.ndarray:
    xa = x_flat.reshape(num_rods,6)
    r1 = xa[:, :3]; r2 = xa[:, 3:]
    return dist_lin_seg_over_ij(r1, r2, ii, jj)

x_flat = jnp.asarray(x).reshape(-1)
J = jax.jacobian(all_dists_wrt_x)(x_flat)
J_np = np.asarray(J)
M, N = J_np.shape
print(f"Jacobian: {J_np.shape}")

# SVD nullspace
U, S, Vh = np.linalg.svd(J_np, full_matrices=True)
base_tol = np.finfo(float).eps * max(M,N) * (S[0] if S.size else 1.0)
rank = int(np.sum(S > base_tol))
null_basis = Vh[rank:].T
null_dim = null_basis.shape[1]
print(f"rank={rank}, null_dim={null_dim}")

# Rigid subspace

def build_rigid_modes(x_flat_np: np.ndarray):
    xa = x_flat_np.reshape(num_rods,6)
    pts = np.concatenate([xa[:, :3], xa[:, 3:]], axis=0)
    c = pts.mean(axis=0)
    rel = pts - c
    modes = []
    for axis, vec in enumerate(np.eye(3)):
        v = np.zeros_like(xa); v[:, :3] = vec; v[:, 3:] = vec
        modes.append(v.reshape(-1))
    for axis, omega in enumerate(np.eye(3)):
        rot_disp = np.cross(rel, omega)
        v = np.zeros_like(xa); v[:, :3] = rot_disp[:num_rods]; v[:, 3:] = rot_disp[num_rods:]
        modes.append(v.reshape(-1))
    return np.stack(modes, axis=1)

rigid_modes = build_rigid_modes(np.asarray(x_flat))

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
null_orth = gram_schmidt(null_basis) if null_dim > 0 else np.zeros((N,0))

# Project out rigid components
internal_vecs = []
for i in range(null_orth.shape[1]):
    v = null_orth[:, i]
    v_r = rigid_orth @ (rigid_orth.T @ v)
    v_int = v - v_r
    n = np.linalg.norm(v_int)
    if n > 1e-10:
        internal_vecs.append(v_int / n)

if not internal_vecs:
    print("[WARN] No internal modes.")
    sys.exit(0)

internal_basis = np.stack(internal_vecs, axis=1)
print(f"Internal modes: {internal_basis.shape[1]}")

# Ranking
v = internal_basis
disp_scores = []
trans_scores = []
rot_scores = []
for i in range(v.shape[1]):
    Vi = v[:, i].reshape(num_rods,6)
    disp_scores.append(np.linalg.norm(Vi))
    v1 = Vi[:, :3]; v2 = Vi[:, 3:]
    trans_scores.append(np.linalg.norm((v1+v2)*0.5))
    rot_scores.append(np.linalg.norm(v2-v1))

if METRIC == 'disp':
    idx = np.argsort(disp_scores)[::-1]
elif METRIC == 'trans':
    idx = np.argsort(trans_scores)[::-1]
else:
    idx = np.argsort(rot_scores)[::-1]

selected = idx[:SELECT_COUNT]
print(f"Selected modes ({METRIC}):", selected.tolist())

# Prepare output dir
import os
os.makedirs(OUT_DIR, exist_ok=True)

# Animate
try:
    import polyscope as ps
    ps.init()
    ps.set_up_dir("z_up")
    nodes0, edges0, _ = prep_for_polyscope(np.asarray(x).reshape(num_rods,-1,3), num_rods)
    ps.register_curve_network("original_rods", nodes0, edges0)
    for k, mi in enumerate(selected):
        mode = v[:, mi].reshape(num_rods,6)
        # create frames
        for t in range(NUM_FRAMES):
            s = np.sin(2*np.pi * t/NUM_FRAMES)
            x_t = np.asarray(x) + (AMPLITUDE * s) * mode
            nodes, edges, _ = prep_for_polyscope(x_t.reshape(num_rods,-1,3), num_rods)
            name = f"internal_mode_{k}_frame_{t:03d}"
            net = ps.register_curve_network(name, nodes, edges)
            ps.screenshot(os.path.join(OUT_DIR, f"{name}.png"))
            ps.remove_curve_network(name)
    print(f"Frames saved in {OUT_DIR}/")
    ps.show()
except Exception as e:
    print("[INFO] Polyscope not available; skipped animation:", e)

print("Done.")
