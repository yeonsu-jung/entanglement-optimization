"""Visualize a single non-rigid (internal) soft mode of the segment-distance Jacobian in q-space (5D).

Steps:
1. Load final rod configuration (q_history last entry) and convert to q (N,5).
2. Build vector of all pairwise segment-segment distances using dist_lin_seg_over_ij_q.
3. Form Jacobian J (num_pairs x 5N) via JAX.
4. Compute SVD(J) with full right singular vectors; extract nullspace basis.
5. Build analytic rigid-body modes in q-space (3 translations + 3 rotations) and orthonormalize.
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
from transforms import q_to_x, x_to_q
from potentials import dist_lin_seg_over_ij_q
from visualizations import prep_for_polyscope

# DATA_PATH = '/Users/yeonsu/GitHub/entanglement-optimization-combined/entanglement-optimization/entangle_and_nudge_04/q_history.npy'
# DATA_PATH = '/Users/yeonsu/GitHub/entanglement-optimization-combined/entanglement-optimization/create_skew_packing/N2_AR50_20251201_144427/q_history.npy'
# DATA_PATH = '/Users/yeonsu/GitHub/entanglement-optimization-combined/entanglement-optimization/create_skew_packing/N2_AR25_20251201_145117/q_history.npy'
# DATA_PATH = '/Users/yeonsu/GitHub/entanglement-optimization-combined/entanglement-optimization/create_skew_packing/N2_AR25_20251201_145117/q_history.npy'
# DATA_PATH = '/Users/yeonsu/GitHub/entanglement-optimization-combined/entanglement-optimization/create_skew_packing/N3_AR25_20251201_145726/q_history.npy'
# DATA_PATH = '/Users/yeonsu/GitHub/entanglement-optimization-combined/entanglement-optimization/create_skew_packing/N10_AR25_20251201_145400/q_history.npy'
if '--N2' in sys.argv:
    DATA_PATH = '/Users/yeonsu/GitHub/entanglement-optimization-combined/entanglement-optimization/create_skew_packing/N2_AR25_20251201_145117/q_history.npy'
if '--N3' in sys.argv:
    DATA_PATH = '/Users/yeonsu/GitHub/entanglement-optimization-combined/entanglement-optimization/create_skew_packing/N3_AR25_20251201_145726/q_history.npy'
if '--N4' in sys.argv:
    DATA_PATH = '/Users/yeonsu/GitHub/entanglement-optimization-combined/entanglement-optimization/create_skew_packing/N4_AR25_20251201_145733/q_history.npy'
if '--N5' in sys.argv:
    DATA_PATH = '/Users/yeonsu/GitHub/entanglement-optimization-combined/entanglement-optimization/create_skew_packing/N5_AR25_20251201_145651/q_history.npy'
if '--N10' in sys.argv:
    DATA_PATH = '/Users/yeonsu/GitHub/entanglement-optimization-combined/entanglement-optimization/create_skew_packing/N10_AR25_20251201_145400/q_history.npy'
if '--N15' in sys.argv:
    DATA_PATH = '/Users/yeonsu/GitHub/entanglement-optimization-combined/entanglement-optimization/create_skew_packing/N15_AR25_20251201_161430/q_history.npy'
if '--N20' in sys.argv:
    DATA_PATH = '/Users/yeonsu/GitHub/entanglement-optimization-combined/entanglement-optimization/create_skew_packing/N20_AR25_20251201_155152/q_history.npy'
if '--N50' in sys.argv:
    DATA_PATH = '/Users/yeonsu/GitHub/entanglement-optimization-combined/entanglement-optimization/create_skew_packing/N50_AR25_20251201_155223/q_history.npy'
    


# arg has '--visual' to enable polyscope visualization
visual = '--visual' in sys.argv


# DATA_PATH = '/Users/yeonsu/GitHub/entanglement-optimization-combined/entanglement-optimization/create_skew_packing/N10_AR25_20251201_145400/q_history.npy'

EPSILON = 2.0e-2  # displacement scale for visualization
INTERNAL_MODE_INDEX = 0  # default internal mode to visualize after filtering

# ---------------- Load configuration ----------------
q_hist = np.load(DATA_PATH)
q_final = q_hist[-1]
# Ensure q is (N,5)
if q_final.shape[-1] == 6:
    # If stored as x, convert to q
    q = x_to_q(q_final)
else:
    q = q_final.reshape(-1, 5)

num_rods = q.shape[0]

# Pair indices
ii, jj = np.triu_indices(num_rods, k=1)
num_pairs = ii.size

# -------------- Distance map --------------

def all_dists_wrt_q(q_flat: jnp.ndarray) -> jnp.ndarray:
    q_arr = q_flat.reshape(num_rods, 5)
    return dist_lin_seg_over_ij_q(q_arr, ii, jj)  # (num_pairs,)

q_flat = jnp.asarray(q).reshape(-1)

# -------------- Jacobian --------------
J = jax.jacobian(all_dists_wrt_q)(q_flat)  # (num_pairs, 5*num_rods)
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

# -------------- Rigid modes in q-space --------------

def build_rigid_modes_q(q_flat_np: np.ndarray):
    q_arr = q_flat_np.reshape(num_rods, 5)
    # q = (cx, cy, cz, theta, phi)
    # Translations: affect only (cx, cy, cz)
    modes = []
    labels = []
    
    # 1. Translations
    for axis in range(3):
        v = np.zeros_like(q_arr)
        v[:, axis] = 1.0
        modes.append(v.reshape(-1))
        labels.append(f"trans_{'xyz'[axis]}")
        
    # 2. Rotations
    # For small rotation omega, delta_c = omega x c
    # delta_u = omega x u
    # We need to map delta_u to delta_theta, delta_phi
    # u = (sin(theta)cos(phi), sin(theta)sin(phi), cos(theta))
    # du/dtheta = (cos(theta)cos(phi), cos(theta)sin(phi), -sin(theta))
    # du/dphi   = (-sin(theta)sin(phi), sin(theta)cos(phi), 0)
    
    centers = q_arr[:, :3]
    thetas = q_arr[:, 3]
    phis = q_arr[:, 4]
    
    # Precompute u and derivatives
    st, ct = np.sin(thetas), np.cos(thetas)
    sp, cp = np.sin(phis), np.cos(phis)
    
    u = np.stack([st*cp, st*sp, ct], axis=1)
    du_dtheta = np.stack([ct*cp, ct*sp, -st], axis=1)
    du_dphi = np.stack([-st*sp, st*cp, np.zeros_like(st)], axis=1)
    
    # Solve [du_dtheta, du_dphi] [dtheta; dphi] = delta_u
    # This is a 3x2 system per rod. We can use least squares or explicit inversion.
    # Since u, du_dtheta, du_dphi/sin(theta) are orthogonal, we can project.
    # delta_u . du_dtheta = |du_dtheta|^2 dtheta = 1 * dtheta
    # delta_u . du_dphi   = |du_dphi|^2 dphi   = sin^2(theta) dphi
    
    for axis, omega in enumerate(np.eye(3)):
        v = np.zeros_like(q_arr)
        
        # Center displacement: delta_c = omega x c
        delta_c = np.cross(omega, centers)
        v[:, :3] = delta_c
        
        # Director displacement: delta_u = omega x u
        delta_u = np.cross(omega, u)
        
        # Project onto spherical derivatives
        # dtheta = delta_u . du_dtheta
        dtheta = np.sum(delta_u * du_dtheta, axis=1)
        
        # dphi = (delta_u . du_dphi) / sin^2(theta)
        # Handle sin(theta) ~ 0 singularity safely
        sin2 = st**2
        dphi = np.zeros_like(dtheta)
        mask = sin2 > 1e-6
        dphi[mask] = np.sum(delta_u[mask] * du_dphi[mask], axis=1) / sin2[mask]
        
        v[:, 3] = dtheta
        v[:, 4] = dphi
        
        modes.append(v.reshape(-1))
        labels.append(f"rot_{'xyz'[axis]}")
        
    return np.stack(modes, axis=1), labels

rigid_modes, rigid_labels = build_rigid_modes_q(np.asarray(q_flat))

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


if visual:
    import polyscope as ps

    ps.init()
    ps.set_up_dir("z_up")

    x = q_to_x(q)
    nodes0, edges0, edge_colors0 = prep_for_polyscope(x.reshape(num_rods,-1,3), num_rods)
    net0 = ps.register_curve_network("original_rods", nodes0, edges0)

    # set colors: original blue, perturbed yellow
    net0.set_color([0.2, 0.4, 0.9])
    net0.set_radius(0.020,relative=False)
    ps.show()
    

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
        v = internal_basis[:, i].reshape(num_rods, 5)
        # v is (delta_c, delta_theta, delta_phi)
        
        # Approximate displacement magnitude in Cartesian space
        # delta_x ~ delta_c + delta_u * L/2
        # We can just use norm of v for ranking, or project to x-space
        
        # Simple norm in q-space
        disp = np.linalg.norm(v)
        disp_scores.append(disp)
        
        # Translation part: norm of delta_c
        trans_part = v[:, :3]
        trans_scores.append(np.linalg.norm(trans_part))
        
        # Rotation part: norm of (delta_theta, delta_phi)
        rot_part = v[:, 3:]
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
print("Top 5 by total q-norm:", rankings['idx_by_disp'][:5].tolist())
print("Top 5 by translation content:", rankings['idx_by_trans'][:5].tolist())
print("Top 5 by rotation content:", rankings['idx_by_rot'][:5].tolist())

# pick the highest total displacement by default
# INTERNAL_MODE_INDEX = int(rankings['idx_by_disp'][0])
if INTERNAL_MODE_INDEX >= internal_basis.shape[0]:
    INTERNAL_MODE_INDEX = 0
    
mode = internal_basis[:, INTERNAL_MODE_INDEX]
print(f"Selected internal mode index {INTERNAL_MODE_INDEX}; ||mode||={np.linalg.norm(mode):.3e}")

# Residual check |J*mode|
residual = np.linalg.norm(J_np @ mode)
print(f"Constraint residual |J*mode|={residual:.3e}")

# -------------- Apply displacement --------------
delta_q = EPSILON * mode.reshape(num_rods, 5)
new_q = q + np.asarray(delta_q)

# Convert back to x for visualization
x = q_to_x(q)
new_x = q_to_x(new_q)

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
    out_dir = "internal_mode_frames_q"
    os.makedirs(out_dir, exist_ok=True)
    ps.screenshot(os.path.join(out_dir, "frame_0_original.png"))
    ps.screenshot(os.path.join(out_dir, "frame_1_perturbed.png"))
    ps.show()
except Exception as e:
    print(f"[INFO] Polyscope visualization skipped: {e}")

print("Done.")
