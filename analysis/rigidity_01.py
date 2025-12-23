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

ii,jj = np.triu_indices(num_rods,k=1)
d_ij = dist_lin_seg_over_ij(x, x, ii, jj)

# %%
# get jacobian
from jax import jacfwd

jacobian_func = jacfwd(dist_lin_seg_over_ij, argnums=(0,1))

J1, J2 = jacobian_func(x, x, ii, jj)
# %%

# find delta_x that increases d_ij the most
import jax.numpy as jnp
num_vars = x.size
delta_x = jnp.zeros_like(x).reshape(-1)
for idx in range(num_vars):
    e_idx = jnp.zeros_like(x).reshape(-1)
    e_idx = e_idx.at[idx].set(1.0)
    
    # compute increase in d_ij
    increase = jnp.sum(J1[:,idx] + J2[:,idx])
    
    delta_x = delta_x.at[idx].set(increase)
# %%
# normalize delta_x
delta_x = delta_x / jnp.linalg.norm(delta_x)

# %%
new_x = x + 0.1*delta_x.reshape(x.shape)

# %% polyscope
import polyscope as ps
ps.init()
ps.set_up_dir("z_up")

ps_rods = ps.register_curve_network("rods", new_x.reshape(num_rods,-1,3), np.array([np.arange(new_x.reshape(num_rods,-1,3).shape[1]-1), np.arange(1,new_x.reshape(num_rods,-1,3).shape[1])]).T)

ps.show()

# %%
