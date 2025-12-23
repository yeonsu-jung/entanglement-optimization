# %%
import cvxpy as cp
import numpy as np

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

# J has shape (num_pairs, 6*num_rods)
J = jax.jacobian(all_dists_wrt_x)(x_flat)

# Convert to NumPy for linear algebra
J_np = np.asarray(J)
m, n = J_np.shape
print("Jacobian shape:", J_np.shape)  # (num_pairs, 6*num_rods)

d_ij_x = dist_lin_seg_over_ij(r1, r2, ii, jj)
D = 0.1

J = J_np          # (M, n)
m = d_ij_x - D    # margins, shape (M,)  where d_ij_x = dist_lin_seg_over_ij(...) at x

n = J.shape[1]
v = cp.Variable(n)

constraints = [J @ v >= -m]  # linearized non-penetration
objective = cp.Minimize(0.5 * cp.sum_squares(v))

prob = cp.Problem(objective, constraints)
prob.solve()

if prob.status in ["optimal", "optimal_inaccurate"]:
    v_safe = np.array(v.value).reshape(-1)
else:
    print("No nontrivial v found in linearized safe region (weird or very tight config).")
