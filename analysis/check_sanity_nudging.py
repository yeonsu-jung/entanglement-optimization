# %%
import sys
from utils import setup_directories
import numpy as np
# import polyscope as ps
from visualizations import prep_for_polyscope, create_movie_from_q_history
from transforms import q_to_x


import jax.numpy as jnp
from potentials import dist_lin_seg_over_ij, acn_over_ij
import matplotlib.pyplot as plt

sys.path.append('../core')  # to import from parent folder

# current file name
output_folder, MOVIE_DIR = setup_directories(__file__)

NUM_RODS = 200
ROD_DIAMETER = 1 / 100
# %%
pth = '/Users/yeonsu/Downloads/q_history_temp.npy'

qq = np.load(pth)
# %%

import jax.numpy as jnp
ii,jj = jnp.triu_indices(NUM_RODS, k=1)

acn_upper_bound = NUM_RODS*(NUM_RODS-1)/2/2
acn_over_time = []
for q in qq[:-1]:
    x = q_to_x(q)
    r1 = x.reshape(-1, 6)[:,:3]
    r2 = x.reshape(-1, 6)[:,3:]
    acn_mat = acn_over_ij(r1, r2, ii, jj)
    acn_over_time.append(jnp.abs(acn_mat).sum())

# %%
pth2 = '/Users/yeonsu/GitHub/mujoco-balls/data/6,7,8/2025-02-16_17_EntangledRelaxedPacking-N0200-AR0100-Scale1/qq.npy'
qq2 = np.load(pth2)
qq2 = qq2.reshape(-1,NUM_RODS*5)
q2 = qq2[-1]
# %%
acn_over_time2 = []
for q in qq2:    
    x = q_to_x(q)
    r1 = x.reshape(-1, 6)[:,:3]
    r2 = x.reshape(-1, 6)[:,3:]
    acn_mat = acn_over_ij(r1, r2, ii, jj)
    acn_over_time2.append(jnp.abs(acn_mat).sum())
# %%
plt.plot(acn_over_time)
plt.axhline(acn_upper_bound, color='red', linestyle='--', label='Upper Bound')
# plt.xscale('log')
plt.xlabel('Time Step')
plt.ylabel('Total |ACN|')
# %%

plt.plot(acn_over_time2)
plt.axhline(acn_upper_bound, color='red', linestyle='--', label='Upper Bound')
# plt.xscale('log')
plt.xlabel('Time Step')
plt.ylabel('Total |ACN|')

# %%

plt.plot(acn_over_time)
plt.plot(acn_over_time2)
plt.axhline(acn_upper_bound, color='red', linestyle='--', label='Upper Bound')
# plt.xscale('log')
plt.xlabel('Time Step')
plt.ylabel('Total |ACN|')
# %%

min_dist_over_time = []
for q in qq[:-1]:
    x = q_to_x(q)
    r1 = x.reshape(-1, 6)[:,:3]
    r2 = x.reshape(-1, 6)[:,3:]
    d_mat = dist_lin_seg_over_ij(r1, r2, ii, jj)
    min_dist_over_time.append(jnp.min(d_mat))
# %%
plt.plot(min_dist_over_time)
plt.xlabel('Time Step')
plt.ylabel('Minimum Distance Between Rods')

# %%
q0 = qq[-1]

create_movie_from_q_history(qq, MOVIE_DIR, NUM_RODS, ROD_DIAMETER)


# %%
pth2 = '/Users/yeonsu/GitHub/mujoco-balls/data/6,7,8/2025-02-16_17_EntangledRelaxedPacking-N0200-AR0100-Scale1/qq.npy'
qq2 = np.load(pth2)
q2 = qq2[-1]

num_frames = len(qq2)
qq2 = qq2.reshape(-1,NUM_RODS*5)
qq2.shape
# %%
create_movie_from_q_history(qq2, MOVIE_DIR, NUM_RODS, ROD_DIAMETER)

# %%

x1 = q_to_x(qq[-1])
x2 = q_to_x(qq2[-1])

r11 = x1.reshape(-1, 6)[:,:3]
r12 = x1.reshape(-1, 6)[:,3:]

r21 = x2.reshape(-1, 6)[:,:3]
r22 = x2.reshape(-1, 6)[:,3:]



ii,jj = jnp.triu_indices(NUM_RODS, k=1)



d_dist_1 = dist_lin_seg_over_ij(r11, r12, ii, jj)
d_dist_2 = dist_lin_seg_over_ij(r21, r22, ii, jj)

# %%

plt.hist(d_dist_1, bins=100, range=(0,0.5), alpha=0.5, label='1')
plt.hist(d_dist_2, bins=100, range=(0,0.5), alpha=0.5, label='2')
plt.legend()
plt.show()

# %%
jnp.min(d_dist_1), jnp.min(d_dist_2)

# %%


acn_dist_1 = acn_over_ij(r11, r12, ii, jj)
acn_dist_2 = acn_over_ij(r21, r22, ii, jj)

e1 = jnp.abs(acn_dist_1).sum()
e2 = jnp.abs(acn_dist_2).sum()

# %%
200*199/2/2
# %%
plt.hist(acn_dist_1, bins=100, range=(-0.5,0.5), alpha=0.5, label='1')
plt.hist(acn_dist_2, bins=100, range=(-0.5,0.5), alpha=0.5, label='2')
plt.legend()
plt.show()

