"""Visualize a single non-rigid (internal) soft mode of the segment-distance Jacobian.

Steps:
1. Load final rod configuration (q_history last entry) and convert to endpoints x (N,6).
2. Build vector of all pairwise segment-segment distances using dist_lin_seg_over_ij.
3. Form Jacobian J (num_pairs x 6N) via JAX.
4. Compute SVD(J) with full right singular vectors; extract nullspace basis.
5. Build analytic rigid-body modes (3 translations + 3 rotations) and orthonormalize.
6. Project nullspace basis vectors to space orthogonal to rigid subspace to get internal modes.
7. Pick one internal mode (largest norm after projection) and apply ±epsilon displacement.
8. Visualize original and perturbed (+) configuration in Polyscope, printing diagnostics.

This focuses strictly on segment distances (not infinite-line approximation).
"""
import sys
sys.path.append('/Users/yeonsu/GitHub/entanglement-optimization-combined/entanglement-optimization/core')

import numpy as np
import jax
import jax.numpy as jnp
from transforms import q_to_x
from potentials import dist_lin_seg_over_ij
from visualizations import prep_for_polyscope

# DATA_PATH = '/Users/yeonsu/GitHub/entanglement-optimization-combined/entanglement-optimization/entangle_and_nudge_04/q_history.npy'
# DATA_PATH = '/Users/yeonsu/GitHub/entanglement-optimization-combined/entanglement-optimization/create_skew_packing/N2_AR50_20251201_144427/q_history.npy'
# DATA_PATH = '/Users/yeonsu/GitHub/entanglement-optimization-combined/entanglement-optimization/create_skew_packing/N2_AR25_20251201_145117/q_history.npy'
# DATA_PATH = '/Users/yeonsu/GitHub/entanglement-optimization-combined/entanglement-optimization/create_skew_packing/N2_AR25_20251201_145117/q_history.npy'
DATA_PATH = '/Users/yeonsu/GitHub/entanglement-optimization-combined/entanglement-optimization/create_skew_packing/N3_AR25_20251201_145726/q_history.npy'
# DATA_PATH = '/Users/yeonsu/GitHub/entanglement-optimization-combined/entanglement-optimization/create_skew_packing/N10_AR25_20251201_145400/q_history.npy'





EPSILON = 2.0e-2  # displacement scale for visualization
INTERNAL_MODE_INDEX = 0  # default internal mode to visualize after filtering

# ---------------- Load configuration ----------------
q_hist = np.load(DATA_PATH)
q = q_hist[-1]
x = q_to_x(q)  # shape (num_rods,6)
num_rods = x.shape[0]

# Pair indices
ii, jj = np.triu_indices(num_rods, k=1)
num_pairs = ii.size

# -------------- Distance map --------------

def all_dists_wrt_x(x_flat: jnp.ndarray) -> jnp.ndarray:
    x_arr = x_flat.reshape(num_rods, 6)
    r1 = x_arr[:, :3]
    r2 = x_arr[:, 3:]
    return dist_lin_seg_over_ij(r1, r2, ii, jj)  # (num_pairs,)

x_flat = jnp.asarray(x).reshape(-1)

# -------------- Jacobian --------------
J = jax.jacobian(all_dists_wrt_x)(x_flat)  # (num_pairs, 6*num_rods)
J_np = np.asarray(J)
M, N = J_np.shape
print(f"Jacobian shape: {J_np.shape} (constraints={M}, dof={N})")

# -------------- SVD & Nullspace --------------
U, S, Vh = np.linalg.svd(J_np, full_matrices=True)
rank = np.sum(S > (np.finfo(float).eps * max(M, N) * S[0]))
null_basis = Vh[rank:].T  # (N, null_dim)
null_dim = null_basis.shape[1]
print(f"Singular values (first 10): {S[:10]}")
print(f"Rank={rank}, Nullspace dimension={null_dim}")

# -------------- Rigid modes --------------

def build_rigid_modes(x_flat_np: np.ndarray):
    x_arr = x_flat_np.reshape(num_rods, 6)
    pts = np.concatenate([x_arr[:, :3], x_arr[:, 3:]], axis=0)  # (2N,3)
    center = pts.mean(axis=0)
    rel = pts - center
    modes = []
    labels = []
    # Translations
    for axis, vec in enumerate(np.eye(3)):
        v = np.zeros_like(x_arr)
        v[:, :3] = vec
        v[:, 3:] = vec
        modes.append(v.reshape(-1))
        labels.append(f"trans_{'xyz'[axis]}")
    # Rotations: delta p = omega x (p - c)
    for axis, omega in enumerate(np.eye(3)):
        rot_disp = np.cross(rel, omega)
        v = np.zeros_like(x_arr)
        v[:, :3] = rot_disp[:num_rods]
        v[:, 3:] = rot_disp[num_rods:]
        modes.append(v.reshape(-1))
        labels.append(f"rot_{'xyz'[axis]}")
    return np.stack(modes, axis=1), labels  # (6*num_rods, 6)

rigid_modes, rigid_labels = build_rigid_modes(np.asarray(x_flat))

# Gram-Schmidt orthonormalization

def gram_schmidt(A: np.ndarray) -> np.ndarray:
    basis = []
    for i in range(A.shape[1]):
        v = A[:, i].copy()
        for b in basis:
            v -= np.dot(b, v) * b
        n = np.linalg.norm(v)
        if n > 0:
            basis.append(v / n)
    return np.stack(basis, axis=1)

rigid_orth = gram_schmidt(rigid_modes)
print(f"Rigid orth shape: {rigid_orth.shape}")

# -------------- Extract internal modes --------------
# Orthonormalize full null basis first
null_orth = gram_schmidt(null_basis) if null_dim > 0 else np.zeros((N,0))

# Project out rigid components
internal_vectors = []
for i in range(null_orth.shape[1]):
    v = null_orth[:, i]
    v_rigid = rigid_orth @ (rigid_orth.T @ v)
    v_int = v - v_rigid
    n_int = np.linalg.norm(v_int)
    if n_int > 1e-10:  # keep meaningful internal component
        internal_vectors.append(v_int / n_int)

if len(internal_vectors) == 0:
    print("[WARN] No internal modes found (all nullspace vectors rigid or below threshold).")
    internal_basis = np.zeros((N,0))
else:
    internal_basis = np.stack(internal_vectors, axis=1)

print(f"Internal (non-rigid) mode count: {internal_basis.shape[1]}")

# -------------- Choose mode to visualize --------------
if internal_basis.shape[1] == 0:
    print("Nothing to visualize; exiting.")
    exit(0)

# Rank internal modes by displacement and by translation vs rotation content
def rank_internal_modes(internal_basis: np.ndarray):
    # total endpoint displacement norm for unit mode scaled by EPSILON
    disp_scores = []
    trans_scores = []
    rot_scores = []
    for i in range(internal_basis.shape[1]):
        v = internal_basis[:, i].reshape(x.shape)
        # displacement magnitude
        disp = np.linalg.norm(v)
        disp_scores.append(disp)
        # translation vs rotation per rod: split endpoints
        v1 = v[:, :3]
        v2 = v[:, 3:]
        # per-rod average translation = (v1 + v2)/2 ; differential rotation-like = (v2 - v1)
        trans_part = (v1 + v2) * 0.5
        rot_part = (v2 - v1)
        trans_scores.append(np.linalg.norm(trans_part))
        rot_scores.append(np.linalg.norm(rot_part))
    # indices sorted by each metric (descending)
    idx_by_disp = np.argsort(disp_scores)[::-1]
    idx_by_trans = np.argsort(trans_scores)[::-1]
    idx_by_rot = np.argsort(rot_scores)[::-1]
    return {
        'disp_scores': disp_scores,
        'trans_scores': trans_scores,
        'rot_scores': rot_scores,
        'idx_by_disp': idx_by_disp,
        'idx_by_trans': idx_by_trans,
        'idx_by_rot': idx_by_rot,
    }

rankings = rank_internal_modes(internal_basis)
print("Top 5 by total displacement:", rankings['idx_by_disp'][:5].tolist())
print("Top 5 by translation content:", rankings['idx_by_trans'][:5].tolist())
print("Top 5 by rotation-like content:", rankings['idx_by_rot'][:5].tolist())

# pick the highest total displacement by default
# INTERNAL_MODE_INDEX = int(rankings['idx_by_disp'][0])
INTERNAL_MODE_INDEX = int(rankings['idx_by_disp'][2])

mode = internal_basis[:, INTERNAL_MODE_INDEX]
print(f"Selected internal mode index {INTERNAL_MODE_INDEX}; ||mode||={np.linalg.norm(mode):.3e}")

# Residual check |J*mode|
residual = np.linalg.norm(J_np @ mode)
print(f"Constraint residual |J*mode|={residual:.3e}")

# -------------- Apply displacement --------------
delta_x = EPSILON * mode.reshape(x.shape)
new_x = x + np.asarray(delta_x)

# -------------- Visualization --------------
try:
    import polyscope as ps
    ps.init()
    ps.set_up_dir("z_up")

    nodes0, edges0, edge_colors0 = prep_for_polyscope(x.reshape(num_rods,-1,3), num_rods)
    nodes1, edges1, edge_colors1 = prep_for_polyscope(new_x.reshape(num_rods,-1,3), num_rods)

    net0 = ps.register_curve_network("original_rods", nodes0, edges0)
    net1 = ps.register_curve_network("internal_mode_+", nodes1, edges1)
    # set colors: original blue, perturbed yellow
    net0.set_color([0.2, 0.4, 0.9])
    net1.set_color([0.95, 0.85, 0.2])
    
    net0.set_radius(0.020,relative=False)
    net1.set_radius(0.020,relative=False)
    
    # Export two screenshots
    import os
    out_dir = "internal_mode_frames"
    os.makedirs(out_dir, exist_ok=True)
    ps.screenshot(os.path.join(out_dir, "frame_0_original.png"))
    ps.screenshot(os.path.join(out_dir, "frame_1_perturbed.png"))
    ps.show()
except Exception as e:
    print(f"[INFO] Polyscope visualization skipped: {e}")

print("Done.")
