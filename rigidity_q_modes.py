"""Rigidity analysis using 5-coordinate (center + spherical angles) representation.

We treat each rod state q_i = (cx, cy, cz, theta, phi). Endpoints are reconstructed with fixed length L.
The Jacobian is built for pairwise segment distances w.r.t. q (5N DOF) rather than endpoints (6N).
This reduces dimensionality and enforces unit-length direction constraint implicitly.
"""
import sys
sys.path.append('/Users/yeonsu/GitHub/entanglement-optimization-combined/entanglement-optimization/core')

import numpy as np
import jax
import jax.numpy as jnp
from transforms import q_to_x
from potentials import dist_lin_seg_over_ij_q

DATA_PATH = '/Users/yeonsu/GitHub/entanglement-optimization-combined/entanglement-optimization/entangle_and_nudge_04/q_history.npy'
ROD_LENGTH = 1.0

# ---------------- Load configuration ----------------
q_hist = np.load(DATA_PATH)
q = q_hist[-1]
q = jnp.asarray(q).reshape(-1,5)  # ensure shape (N,5)
num_rods = q.shape[0]

# Build index pairs
ii, jj = np.triu_indices(num_rods, k=1)
num_pairs = ii.size

# ---------------- Distance map in q-space ----------------

def all_dists_wrt_q(q_flat):
    q_arr = q_flat.reshape(num_rods,5)
    return dist_lin_seg_over_ij_q(q_arr, ii, jj, length=ROD_LENGTH)

q_flat = q.reshape(-1)

# ---------------- Jacobian ----------------
Jq = jax.jacobian(all_dists_wrt_q)(q_flat)  # (num_pairs, 5*num_rods)
Jq_np = np.asarray(Jq)
M, N = Jq_np.shape
print(f"Jacobian(q) shape: {Jq_np.shape}  (constraints={M}, dof={N})")

# ---------------- SVD & nullspace ----------------
U, S, Vh = np.linalg.svd(Jq_np, full_matrices=True)
rank = np.sum(S > (np.finfo(float).eps * max(M,N) * S[0]))
null_basis = Vh[rank:].T  # (N, null_dim)
null_dim = null_basis.shape[1]
print("Singular values (first 10):", S[:10])
print(f"Rank={rank}, Nullspace dim={null_dim}")

# ---------------- Rigid modes in q-space ----------------
# Need to express translations & rotations via variations in center & angles.
# For small rotation δω about axis a, direction vector u changes by a x u.
# Convert that δu to (δtheta, δphi) via local spherical coordinate Jacobian.

from transforms import sph2cart

centers = q[:, :3]
theta = q[:, 3]
phi = q[:, 4]
u = sph2cart(theta, phi)

# Partial derivatives of u wrt theta, phi
# u(theta,phi) = [sin(theta)cos(phi), sin(theta)sin(phi), cos(theta)]
# du/dtheta = [cos(theta)cos(phi), cos(theta)sin(phi), -sin(theta)]
# du/dphi   = [-sin(theta)sin(phi), sin(theta)cos(phi), 0]

du_dtheta = jnp.stack([jnp.cos(theta)*jnp.cos(phi), jnp.cos(theta)*jnp.sin(phi), -jnp.sin(theta)], axis=1)
du_dphi   = jnp.stack([-jnp.sin(theta)*jnp.sin(phi), jnp.sin(theta)*jnp.cos(phi), jnp.zeros_like(theta)], axis=1)

# Build rigid modes: translations modify center only; rotations modify both center and orientation (but pure rotations shouldn't translate centers if rotating about global centroid).
pts = jnp.concatenate([centers - 0.5*ROD_LENGTH*u, centers + 0.5*ROD_LENGTH*u], axis=0)
global_center = pts.mean(axis=0)
rel = centers - global_center  # use center positions for rotation pivot

rigid_modes = []
labels = []

# Translations: d(center)=axis_vec, d(theta)=0, d(phi)=0
for axis, vec in enumerate(np.eye(3)):
    mode = np.zeros((num_rods,5))
    mode[:, :3] = vec  # translate all centers
    rigid_modes.append(mode.reshape(-1))
    labels.append(f"trans_{'xyz'[axis]}")

# Rotations: δcenter = a x rel, δu = a x u; solve for (δtheta, δphi) via least squares in basis {du_dtheta, du_dphi}
for axis, a_vec in enumerate(np.eye(3)):
    a = jnp.array(a_vec)
    delta_center = jnp.cross(a, rel)  # rotation about global center
    delta_u = jnp.cross(a, u)         # desired change in direction
    # Solve for coefficients alpha,beta such that alpha du_dtheta + beta du_dphi ≈ delta_u
    A = jnp.stack([du_dtheta, du_dphi], axis=2)  # (N,3,2)
    # Least squares per rod
    def solve_coeffs(row):
        Mmat = row  # (3,2)
        # normal equations (M^T M) c = M^T delta_u
        MTM = Mmat.T @ Mmat + 1e-12 * jnp.eye(2)
        rhs = Mmat.T @ delta_u_i
        c = jnp.linalg.solve(MTM, rhs)
        return c
    # Vectorized solve
    delta_theta = []
    delta_phi = []
    for i in range(num_rods):
        Mmat = A[i]  # (3,2)
        delta_u_i = delta_u[i]
        MTM = Mmat.T @ Mmat + 1e-12 * jnp.eye(2)
        rhs = Mmat.T @ delta_u_i
        c = jnp.linalg.solve(MTM, rhs)
        delta_theta.append(c[0])
        delta_phi.append(c[1])
    delta_theta = jnp.array(delta_theta)
    delta_phi = jnp.array(delta_phi)
    mode = np.zeros((num_rods,5))
    mode[:, :3] = np.asarray(delta_center)
    mode[:, 3] = np.asarray(delta_theta)
    mode[:, 4] = np.asarray(delta_phi)
    rigid_modes.append(mode.reshape(-1))
    labels.append(f"rot_{'xyz'[axis]}")

rigid_modes = np.stack(rigid_modes, axis=1)  # (5*num_rods, 6)

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
print("Rigid(q) orth shape:", rigid_orth.shape)

# Residuals |Jq * rigid_mode|
for lab, vec in zip(labels, rigid_orth.T):
    res = np.linalg.norm(Jq_np @ vec)
    print(f"Residual |Jq*v| {lab}: {res:.3e}")

# Nullspace orthonormalization
null_orth = gram_schmidt(null_basis) if null_dim > 0 else np.zeros((N,0))

# Extract internal modes
internal = []
for i in range(null_orth.shape[1]):
    v = null_orth[:, i]
    v_r = rigid_orth @ (rigid_orth.T @ v)
    v_int = v - v_r
    n = np.linalg.norm(v_int)
    if n > 1e-10:
        internal.append(v_int / n)
internal_basis = np.stack(internal, axis=1) if internal else np.zeros((N,0))
print("Internal mode count (q-space):", internal_basis.shape[1])

# Pick first internal mode and visualize displacement in Cartesian endpoints
if internal_basis.shape[1] > 0:
    mode = internal_basis[:,0].reshape(num_rods,5)
    # Small step
    EPS = 1e-2
    q_plus = q + EPS * mode
    x_original = q_to_x(q)
    x_plus = q_to_x(q_plus)
    try:
        import polyscope as ps
        from visualizations import prep_for_polyscope
        ps.init()
        ps.set_up_dir("z_up")
        nodes0, edges0, _ = prep_for_polyscope(np.asarray(x_original).reshape(num_rods,-1,3), num_rods)
        nodes1, edges1, _ = prep_for_polyscope(np.asarray(x_plus).reshape(num_rods,-1,3), num_rods)
        ps.register_curve_network("original_rods", nodes0, edges0)
        ps.register_curve_network("q_internal_mode_+", nodes1, edges1)
        ps.show()
    except Exception as e:
        print("[INFO] Polyscope skipped:", e)

print("Done.")
