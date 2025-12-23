# %%
import numpy as np
import jax
import jax.numpy as jnp
import cvxpy as cp

# ---------------------------------------------------------
# Assumes you already have:
#   - num_rods
#   - x           : array, shape (num_rods, 6)
#   - ii, jj      : np.triu_indices(num_rods, k=1)
#   - dist_lin_seg_over_ij(r1, r2, ii, jj)
# and you choose:
#   - D           : minimum allowed distance between rod segments
# ---------------------------------------------------------

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
D = 0.01

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


# 1) Wrap distances as a function of x_flat
def all_dists_wrt_x(x_flat):
    """
    x_flat: shape (6*num_rods,)
    returns: d_ij(x) as a 1D array of length M (#pairs)
    """
    x_arr = x_flat.reshape(num_rods, 6)
    r1 = x_arr[:, :3]
    r2 = x_arr[:, 3:]
    return dist_lin_seg_over_ij(r1, r2, ii, jj)  # shape (M,)

# Current flattened configuration
x_flat_j = jnp.asarray(x).reshape(-1)

# Current distances and Jacobian
d_ij_j = all_dists_wrt_x(x_flat_j)                       # (M,)
J_j = jax.jacobian(all_dists_wrt_x)(x_flat_j)            # (M, 6*num_rods)

d_ij = np.asarray(d_ij_j)                                # numpy, (M,)
J_np = np.asarray(J_j)                                   # numpy, (M, 6*num_rods)

print("Jacobian shape J:", J_np.shape)
print("min current distance:", d_ij.min())

# You choose this:
# e.g. D = 2.0 * rod_radius
D = 0.1  # <-- set this appropriately

margins = d_ij - D   # how much slack above D for each pair

# ---------------------------------------------------------
# 2) Linearized safe region and a minimum-norm safe direction
#    Linearized safe region (first order):
#       V_safe_lin = { v | J v >= -margins }
#    We find the min-norm v inside this region.
# ---------------------------------------------------------

def find_min_norm_safe_direction(J, margins, D, margin_buffer=0.0):
    """
    Solve:
        min  0.5 ||v||^2
        s.t. J v >= -(margins - margin_buffer)
    margin_buffer >= 0 shrinks the feasible region a bit to stay away from the boundary.
    """
    M, n = J.shape
    v = cp.Variable(n)

    # We allow distances to decrease by at most 'margins - margin_buffer'
    # so that d_ij(x + v) ≈ d_ij(x) + (J v)_ij >= D + margin_buffer.
    rhs = -(margins - margin_buffer)
    constraints = [J @ v >= rhs]

    objective = cp.Minimize(0.5 * cp.sum_squares(v))
    prob = cp.Problem(objective, constraints)
    prob.solve()

    print("min-norm safe direction status:", prob.status)
    if prob.status not in ["optimal", "optimal_inaccurate"]:
        return None

    v_safe = np.array(v.value).reshape(-1)
    print("||v_safe||_2:", np.linalg.norm(v_safe))
    approx_new_margins = margins + J @ v_safe
    print("min approx new margin:", approx_new_margins.min())
    return v_safe

# Example usage:
v_lin_safe = find_min_norm_safe_direction(J_np, margins, D, margin_buffer=0.0)

# ---------------------------------------------------------
# 3) Exact max step along a direction v (nonlinear check)
#    Given any direction v, find the largest alpha >= 0 such that
#       d_ij(x + alpha * v) >= D  for all pairs (exactly, no linearization).
# ---------------------------------------------------------

def min_distance_at_alpha(x0, v_flat, alpha, ii, jj):
    """
    Compute min d_ij(x0 + alpha * v) exactly (via dist_lin_seg_over_ij).
    """
    x0 = np.asarray(x0)
    v_flat = np.asarray(v_flat)
    x_alpha = x0 + alpha * v_flat.reshape(x0.shape)
    r1 = x_alpha[:, :3]
    r2 = x_alpha[:, 3:]
    d_alpha = dist_lin_seg_over_ij(jnp.asarray(r1), jnp.asarray(r2), ii, jj)
    return float(np.asarray(d_alpha).min())

def max_safe_step_along_direction(x0, v_flat, D, ii, jj,
                                  dist_fun=dist_lin_seg_over_ij,
                                  initial_step=1.0,
                                  n_bisect=40):
    """
    Find max alpha such that d_ij(x0 + alpha * v) >= D for all (i,j),
    using 1D bisection with exact distances.
    """
    x0 = np.asarray(x0)
    v_flat = np.asarray(v_flat)

    # sanity check at alpha = 0
    d0_min = min_distance_at_alpha(x0, v_flat, 0.0, ii, jj)
    if d0_min < D - 1e-12:
        raise RuntimeError(f"Current configuration already violates D; min d = {d0_min}, D = {D}")

    # grow upper bound until we cross D or hit some cap
    lo = 0.0
    hi = initial_step
    for _ in range(20):  # expand up to 20 times
        d_min_hi = min_distance_at_alpha(x0, v_flat, hi, ii, jj)
        if d_min_hi < D:
            break
        lo = hi
        hi *= 2.0
    else:
        # never violated D up to a huge step
        return hi

    # Now we know: at lo it's safe, at hi it's unsafe (or exactly at D)
    for _ in range(n_bisect):
        mid = 0.5 * (lo + hi)
        d_min_mid = min_distance_at_alpha(x0, v_flat, mid, ii, jj)
        if d_min_mid >= D:
            lo = mid
        else:
            hi = mid

    return lo

# ---------------------------------------------------------
# Example: use the linearized safe direction and compute its true safe step
# ---------------------------------------------------------

if v_lin_safe is not None:
    alpha_max = max_safe_step_along_direction(
        x0=x,
        v_flat=v_lin_safe,
        D=D,
        ii=ii,
        jj=jj,
        initial_step=1.0,
        n_bisect=40,
    )
    print("Max safe alpha along v_lin_safe:", alpha_max)

    # Build the new configuration
    new_x = x + alpha_max * v_lin_safe.reshape(x.shape)
    print("min distance at new_x:",
          min_distance_at_alpha(new_x, v_lin_safe*0.0, 0.0, ii, jj))
