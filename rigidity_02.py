# %%
import sys
sys.path.append('/Users/yeonsu/GitHub/entanglement-optimization-combined/entanglement-optimization/core')

# %%
import numpy as np
pth = '/Users/yeonsu/GitHub/entanglement-optimization-combined/entanglement-optimization/entangle_and_nudge_04/q_history.npy'
dta = np.load(pth)

q = dta[-1]
# %%
from transforms import q_to_x
x = q_to_x(q)
x.shape

# %%
# d_ij
# dist_lin_seg_over_ij(r1, r2, i_indices, j_indices):

from potentials import dist_lin_seg_over_ij

num_rods = x.shape[0]

# Split endpoints (r1: first endpoint, r2: second endpoint)
r1 = x[:, :3]
r2 = x[:, 3:]

ii, jj = np.triu_indices(num_rods, k=1)
d_ij = dist_lin_seg_over_ij(r1, r2, ii, jj)


# %%
# get jacobian
import jax
import jax.numpy as jnp

# --- Build the full Jacobian J of all d_ij wrt x_flat ---

def all_dists_wrt_x(x_flat):
    """
    Map x_flat (shape 6*num_rods,) to the vector of all pairwise
    distances d_ij (length = num_pairs).
    """
    x_arr = x_flat.reshape(num_rods, 6)
    _r1 = x_arr[:, :3]
    _r2 = x_arr[:, 3:]
    # shape: (num_pairs,)
    return dist_lin_seg_over_ij(_r1, _r2, ii, jj)

x_flat = jnp.asarray(x).reshape(-1)
d_befre = all_dists_wrt_x(x_flat)

x_translated_flat = x_flat + 1.0  # translate all points by 1 in all directions
d_after = all_dists_wrt_x(x_translated_flat)

print("Distance change after translation (should be 0):", d_after - d_befre)

# J has shape (num_pairs, 6*num_rods)
J = jax.jacobian(all_dists_wrt_x)(x_flat)

# %%
ones = jnp.ones_like(x_flat)
# J times ones
J_ones = J @ ones


# %%
J.shape

# Convert to NumPy for linear algebra
J_np = np.asarray(J)
m, n = J_np.shape
print("Jacobian shape:", J_np.shape)  # (num_pairs, 6*num_rods)

# --- Compute a basis of the right-nullspace of J: {v | J v = 0} ---

# SVD-based nullspace
U, S, Vh = np.linalg.svd(J_np, full_matrices=False)

# Use machine epsilon based tolerance (more conservative) rather than an arbitrary scaled threshold
eps = np.finfo(float).eps
tol = eps * max(m, n) * (S[0] if S.size > 0 else 1.0)
rank = np.sum(S > tol)
null_basis = Vh[rank:].T   # shape (n, n-rank)

print("Singular values (first 10):", S[:10])
print("Largest / smallest ratio:", S[0]/S[-1] if S.size>0 and S[-1]>0 else "inf")
print("Tolerance used:", tol)
print("Estimated rank (tol-based):", rank)
print("Nullspace dimension (tol-based):", null_basis.shape[1])

# --- Rigid motion sanity checks -------------------------------------------------
# Expect at least 6 (or fewer if degenerate) null directions: 3 translations + 3 rotations.
# Build candidate rigid motions for endpoints and test J*v ≈ 0.

def build_rigid_modes(x_flat):
    x_arr = x_flat.reshape(num_rods, 6)
    pts = np.concatenate([x_arr[:, :3], x_arr[:, 3:]], axis=0)  # (2N,3)
    center = pts.mean(axis=0)
    rel = pts - center
    modes = []
    labels = []
    # Translations
    for axis, vec in enumerate(np.eye(3)):
        v = np.zeros_like(x_arr)
        v[:, :3] += vec  # translate first endpoints
        v[:, 3:] += vec  # translate second endpoints
        modes.append(v.reshape(-1))
        labels.append(f"trans_{'xyz'[axis]}")
    # Rotations about x,y,z through centroid: delta p = omega x (p - c)
    for axis, omega in enumerate(np.eye(3)):
        rot_disp = np.cross(rel, omega)
        # Split back to endpoints displacements
        v = np.zeros_like(x_arr)
        v[:, :3] = rot_disp[:num_rods]
        v[:, 3:] = rot_disp[num_rods:]
        modes.append(v.reshape(-1))
        labels.append(f"rot_{'xyz'[axis]}")
    return np.stack(modes, axis=1), labels

rigid_modes, rigid_labels = build_rigid_modes(np.asarray(x_flat))
Jv = J_np @ rigid_modes  # shape (num_pairs, 6)
rigid_residuals = np.linalg.norm(Jv, axis=0)
for lab, res in zip(rigid_labels, rigid_residuals):
    print(f"Residual for {lab}: {res:.3e}")

print("Mean rigid residual:", rigid_residuals.mean())
print("Max rigid residual:", rigid_residuals.max())

# If these are not near numerical zero, invariance may be broken by piecewise ops in dist function.
if rigid_residuals.max() > 1e-6:
    print("[WARN] Rigid motion residuals are large; distance function may use clamping or non-invariant operations.")
    print("       Consider re-deriving distance with pure relative vectors to preserve rigid invariance analytically.")

# %%

# Instead of forming the full Jacobian (expensive), get the gradient of the
# scalar objective sum(d_ij) with respect to x directly. This automatically
# accounts for dependence through both endpoints.

def total_distance_wrt_x(x_flat):
    x_arr = x_flat.reshape(num_rods, 6)
    _r1 = x_arr[:, :3]
    _r2 = x_arr[:, 3:]
    return jnp.sum(dist_lin_seg_over_ij(_r1, _r2, ii, jj))

x_flat = jnp.asarray(x).reshape(-1)
g_flat = jax.grad(total_distance_wrt_x)(x_flat)

# %%

# %%

# find delta_x that increases d_ij the most
# Use the gradient direction to increase total pairwise distances the most
delta_x = g_flat
# %%
# normalize delta_x
delta_x = delta_x / (jnp.linalg.norm(delta_x) + 1e-12)

# %%
new_x = jnp.asarray(x) + 1e-1 * delta_x.reshape(x.shape)

# %% polyscope
import polyscope as ps
ps.init()
ps.set_up_dir("z_up")


new_x = new_x.reshape(num_rods, -1, 3)

# Build a single set of nodes and edges for all rods
from visualizations import prep_for_polyscope    

nodes0,edges0,edge_colors0 = prep_for_polyscope(x.reshape(num_rods,-1,3),num_rods)
nodes,edges,edge_colors = prep_for_polyscope(new_x,num_rods)

ps.register_curve_network("original_rods", nodes0, edges0)
ps.register_curve_network("rods", nodes, edges)
ps.show()

# %%
