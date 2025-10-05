# %% From theory 3
import numpy as np
import potentials as pt
import time
import transforms as tf

from matplotlib import pyplot as plt
from jax import numpy as jnp

from visualizations import plot_many_rods
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import svds

from sympy.utilities.lambdify import lambdify
from sympy.parsing.mathematica import parse_mathematica
from sympy import symbols, diff
from scipy.sparse import coo_matrix

# pth = '/Users/yeonsu/GitHub/entanglement-optimization/results/37,178,56/2025-02-18_18_EntangledRelaxedPacking-N0200-AR0200-Scale1/q_relaxed.txt'
pth = '/Users/yeonsu/GitHub/entanglement-optimization/packings/q0.txt'
q = np.loadtxt(pth)

print(q.shape)
# %%
def spherical_to_cartesian(theta, phi):
    return np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta)
    ])

# %%
q = q.reshape(-1,5)
q = jnp.array(q)
num_rods = q.shape[0]

# %%
x = tf.q_to_x(q)
x = jnp.array(x,dtype=jnp.float64)

def analyze_pairwise_dist_entanglement(nodes_at_ith_frame,num_rods):
    i_indices, j_indices = jnp.triu_indices(num_rods, k=1)
    r1 = nodes_at_ith_frame.reshape(-1,6)[:,:3]
    r2 = nodes_at_ith_frame.reshape(-1,6)[:,3:]
    pairwise_acn = pt.acn_over_ij(r1,r2, i_indices, j_indices)
    pairwise_dist = pt.dist_lin_seg_over_ij(r1,r2, i_indices, j_indices)
    dist = jnp.min(pairwise_dist)
    entanglement = jnp.sum(jnp.abs(pairwise_acn)) / (num_rods*(num_rods-1)/2)
    return dist,entanglement,pairwise_dist, pairwise_acn

start = time.time()
d,e,pdist,pacn = analyze_pairwise_dist_entanglement(x,num_rods)
end = time.time()

# %%
# diameter = pdist.min()
diameter = 100
TOL = 1e-20
num_rods_in_contact = []
TOL_SPACE = jnp.geomspace(1e-20,1e-6,30)
for TOL in TOL_SPACE:
    num_rods_in_contact.append(jnp.count_nonzero(pdist < diameter + TOL))
num_rods_in_contact = jnp.array(num_rods_in_contact)
plt.plot(TOL_SPACE,num_rods_in_contact,'o-')

# %%
TOL = .5e-6
num_rods_in_contact = jnp.count_nonzero(pdist < diameter + TOL)
print(num_rods_in_contact)

# %%
sparse_rows = jnp.where(pdist < diameter + TOL)[0]

i_indices, j_indices = jnp.triu_indices(num_rods, k=1)
sparse_i = i_indices[sparse_rows]
sparse_j = j_indices[sparse_rows]
sparse_dist = pdist[sparse_rows]

# %%
# from sympy import Ynm

# %%
d12_expression = parse_mathematica("""
Sqrt[(-((z1 - z2)*Sin[phi1 - phi2]*Sin[theta1]*Sin[theta2]) + 
    (y1 - y2)*(-(Cos[phi1]*Cos[theta2]*Sin[theta1]) + Cos[phi2]*Cos[theta1]*
       Sin[theta2]) + (x1 - x2)*(Cos[theta2]*Sin[phi1]*Sin[theta1] - 
      Cos[theta1]*Sin[phi2]*Sin[theta2]))^2]/
 Sqrt[Sin[phi1 - phi2]^2*Sin[theta1]^2*Sin[theta2]^2 + 
   (Cos[phi1]*Cos[theta2]*Sin[theta1] - Cos[phi2]*Cos[theta1]*Sin[theta2])^2 + 
   (Cos[theta2]*Sin[phi1]*Sin[theta1] - Cos[theta1]*Sin[phi2]*Sin[theta2])^2]
""")

# %%

theta1, phi1, phi2, theta2, x1, y1, z1, x2, y2, z2 = symbols('theta1 phi1 phi2 theta2 x1 y1 z1 x2 y2 z2')

# Differentiate with respect to theta1
d_d12_dtheta1 = diff(d12_expression, theta1)
d_d12_dtheta2 = diff(d12_expression, theta2)
d_d12_dphi1 = diff(d12_expression, phi1)
d_d12_dphi2 = diff(d12_expression, phi2)
d_d12_dx1 = diff(d12_expression, x1)
d_d12_dy1 = diff(d12_expression, y1)
d_d12_dz1 = diff(d12_expression, z1)
d_d12_dx2 = diff(d12_expression, x2)
d_d12_dy2 = diff(d12_expression, y2)
d_d12_dz2 = diff(d12_expression, z2)

# lambdify
d_12_func = lambdify((theta1, phi1, phi2, theta2, x1, y1, z1, x2, y2, z2), d12_expression)
d_d12_dtheta1_func = lambdify((theta1, phi1, phi2, theta2, x1, y1, z1, x2, y2, z2), d_d12_dtheta1)
d_d12_dtheta2_func = lambdify((theta1, phi1, phi2, theta2, x1, y1, z1, x2, y2, z2), d_d12_dtheta2)
d_d12_dphi1_func = lambdify((theta1, phi1, phi2, theta2, x1, y1, z1, x2, y2, z2), d_d12_dphi1)
d_d12_dphi2_func = lambdify((theta1, phi1, phi2, theta2, x1, y1, z1, x2, y2, z2), d_d12_dphi2)
d_d12_dx1_func = lambdify((theta1, phi1, phi2, theta2, x1, y1, z1, x2, y2, z2), d_d12_dx1)
d_d12_dy1_func = lambdify((theta1, phi1, phi2, theta2, x1, y1, z1, x2, y2, z2), d_d12_dy1)
d_d12_dz1_func = lambdify((theta1, phi1, phi2, theta2, x1, y1, z1, x2, y2, z2), d_d12_dz1)
d_d12_dx2_func = lambdify((theta1, phi1, phi2, theta2, x1, y1, z1, x2, y2, z2), d_d12_dx2)
d_d12_dy2_func = lambdify((theta1, phi1, phi2, theta2, x1, y1, z1, x2, y2, z2), d_d12_dy2)
d_d12_dz2_func = lambdify((theta1, phi1, phi2, theta2, x1, y1, z1, x2, y2, z2), d_d12_dz2)

# %%
# construct jacobian J
rows = []
cols = []
vals = []
num_contacts = sparse_i.shape[0]

d_gap = []
for contact_idx in range(num_contacts):
    i = sparse_i[contact_idx]
    j = sparse_j[contact_idx]

    q_i = q[i]
    q_j = q[j]

    theta1_val, phi1_val = q_i[3], q_i[4]
    theta2_val, phi2_val = q_j[3], q_j[4]
    x1_val, y1_val, z1_val = q_i[0], q_i[1], q_i[2]
    x2_val, y2_val, z2_val = q_j[0], q_j[1], q_j[2]

    d_sym = d_12_func(theta1_val, phi1_val, phi2_val, theta2_val, x1_val, y1_val, z1_val, x2_val, y2_val, z2_val)
    d_num = sparse_dist[contact_idx]
    # print(d_sym-d_num)
    d_gap.append(d_sym-d_num)

    # Compute derivatives
    grad_i = [
        d_d12_dx1_func(theta1_val, phi1_val, phi2_val, theta2_val, x1_val, y1_val, z1_val, x2_val, y2_val, z2_val),  # x1
        d_d12_dy1_func(theta1_val, phi1_val, phi2_val, theta2_val, x1_val, y1_val, z1_val, x2_val, y2_val, z2_val),  # y1
        d_d12_dz1_func(theta1_val, phi1_val, phi2_val, theta2_val, x1_val, y1_val, z1_val, x2_val, y2_val, z2_val),  # z1
        d_d12_dtheta1_func(theta1_val, phi1_val, phi2_val, theta2_val, x1_val, y1_val, z1_val, x2_val, y2_val, z2_val),  # theta1
        d_d12_dphi1_func(theta1_val, phi1_val, phi2_val, theta2_val, x1_val, y1_val, z1_val, x2_val, y2_val, z2_val)     # phi1
    ]

    grad_j = [
        d_d12_dx2_func(theta1_val, phi1_val, phi2_val, theta2_val, x1_val, y1_val, z1_val, x2_val, y2_val, z2_val),  # x2
        d_d12_dy2_func(theta1_val, phi1_val, phi2_val, theta2_val, x1_val, y1_val, z1_val, x2_val, y2_val, z2_val),  # y2
        d_d12_dz2_func(theta1_val, phi1_val, phi2_val, theta2_val, x1_val, y1_val, z1_val, x2_val, y2_val, z2_val),  # z2
        d_d12_dtheta2_func(theta1_val, phi1_val, phi2_val, theta2_val, x1_val, y1_val, z1_val, x2_val, y2_val, z2_val),  # theta2
        d_d12_dphi2_func(theta1_val, phi1_val, phi2_val, theta2_val, x1_val, y1_val, z1_val, x2_val, y2_val, z2_val)     # phi2
    ]

    # Fill sparse entries for rod i
    for k in range(5):
        rows.append(contact_idx)
        cols.append(i * 5 + k)
        vals.append(grad_i[k])

    # Fill sparse entries for rod j
    for k in range(5):
        rows.append(contact_idx)
        cols.append(j * 5 + k)
        vals.append(grad_j[k])

# Construct sparse Jacobian matrix
J_sparse = coo_matrix((vals, (rows, cols)), shape=(num_contacts, num_rods * 5))
print("Jacobian shape:", J_sparse.shape)
# %%
# svd
U, S, Vt = svds(J_sparse, k=99)
print("Singular values:", S)
print("U shape:", U.shape)
print("Vt shape:", Vt.shape)
print("Rank of J_sparse:", np.linalg.matrix_rank(J_sparse.toarray()))


# %%

def spherical_to_cartesian(theta, phi):
    return jnp.array([
        jnp.sin(theta) * jnp.cos(phi),
        jnp.sin(theta) * jnp.sin(phi),
        jnp.cos(theta)
    ])

# Step 1: Identify near-zero singular values
threshold = 1e-10
null_mask = S < threshold
null_vectors = Vt[null_mask]

print(f"Number of nullspace vectors: {null_vectors.shape[0]}")

# %%
# Step 2: Check alignment for each rod
total_alignment = 0
for idx, vec in enumerate(null_vectors):
    print(f"\n--- Nullspace vector {idx} ---")
    vec_reshaped = vec.reshape(num_rods, 5)

    aligned_count = 0
    for i in range(num_rods):
        q_i = q[i]
        theta, phi = q_i[3], q_i[4]
        axis = spherical_to_cartesian(theta, phi)

        translation = vec_reshaped[i, :3]  # x, y, z components
        translation_norm = jnp.linalg.norm(translation)

        if translation_norm > 1e-10:
            dir_vector = translation / translation_norm
            alignment = jnp.dot(axis, dir_vector)
            # print(f"Rod {i}: dot(axis, translation_dir) = {alignment:.4f}")

            if jnp.abs(alignment) > 0.99:
                aligned_count += 1
                print(f"Rod {i}: dot(axis, translation_dir) = {alignment:.4f}")
    # print(f"Aligned rods: {aligned_count} / {num_rods}")
    total_alignment += aligned_count

print(f"Total aligned rods: {total_alignment} / {num_rods * null_vectors.shape[0]}")

# %%
# U, S, Vt = svds(J_sparse, k=100, which='SM')



# %%
# we need another type of matrix whose null space gives 


from scipy.optimize import linprog
from scipy.sparse import vstack
import numpy as np

# Convert sparse J to dense (or keep sparse depending on solver)
J_dense = J_sparse.toarray()
num_constraints, num_variables = J_dense.shape

# Objective: maximize sum of distance increases = sum(J delta_q)
c = -np.sum(J_dense, axis=0)  # linprog minimizes, so we negate

# Inequality constraint: J @ delta_q >= 0  -->  -J @ delta_q <= 0
A_ub = -J_dense
b_ub = np.zeros(num_constraints)

# Bound constraint: |delta_q_i| <= 1 for all i
bounds = [(-1, 1)] * num_variables

# Solve the LP
res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

if res.success:
    delta_q = res.x
    gap_increases = J_dense @ delta_q
    num_loosened = np.sum(gap_increases > 1e-6)
    print(f"✅ Found motion increasing distances.")
    print(f"  → Number of constraints strictly loosened: {num_loosened} / {num_constraints}")
    print(f"  → Max gap increase: {gap_increases.max():.4e}")
    print(f"  → Mean gap increase: {gap_increases.mean():.4e}")
else:
    print("❌ No feasible separating motion found.")
