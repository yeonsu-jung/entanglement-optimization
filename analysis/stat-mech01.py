# %%
from pathlib import Path

import jax
import jax.numpy as jnp
from protocols4 import create_random_rods
from visualizations import plot_many_rods
from potentials import dist_lin_seg_over_ij
from transforms import q_to_x


alpha = 100
rod_length = 1.0
rod_diameter = rod_length/alpha
print(f'rod_diameter: {rod_diameter}')

# N/V = 

V = 1.5**3
Z = 10
N = V*Z/(rod_diameter*rod_length**2)
N = int(N)
print(f'Number of rods N: {N}')

prng_key=jax.random.PRNGKey(11)
rods = create_random_rods(N,prng_key,size=10)
rods = rods.reshape((N,-1))
rods.shape
# %%
x = q_to_x(rods)
jnp.max(x)
jnp.min(x)


# %%

plot_many_rods(rods)

# %%
# distance matrix


x = q_to_x(rods)
x = jnp.array(x,dtype=jnp.float64)
r1 = x.reshape(-1,6)[:,:3]
r2 = x.reshape(-1,6)[:,3:]
num_rods = rods.shape[0]

i_indices, j_indices = jnp.triu_indices(num_rods, k=1)
dists = dist_lin_seg_over_ij(r1,r2, i_indices, j_indices)
dists.shape

# reshape into a matrix
dist_matrix = jnp.zeros((num_rods,num_rods))
dist_matrix = dist_matrix.at[i_indices,j_indices].set(dists)
dist_matrix = dist_matrix + dist_matrix.T
dist_matrix = dist_matrix.at[jnp.diag_indices(num_rods)].set(0.0)
dist_matrix

# %%
import matplotlib.pyplot as plt
plt.hist(dists,bins=30)
plt.xlabel('pairwise distance')
plt.ylabel('count')
plt.title('Histogram of pairwise distances between rods')
plt.show()

# %%
D = rod_diameter
num_contacts = jnp.count_nonzero(dists < D)
print(f'Number of rod pairs with distance < {D}: {num_contacts}')
print(f'Number of rods: {num_rods}')

# %%

prng_key=jax.random.PRNGKey(11)


rod_length = 1.0
rod_diameter = rod_length/alpha

alpha = 100

V = 1.5**3
Z = 10

num_is = 30
num_contacts_list = []
for i in range(num_is):
    
    N = V*Z/(rod_diameter*rod_length**2)*(i+1)/10
    N = int(N)

    print(f'Number of rods N: {N}')

    rods = create_random_rods(N,prng_key)
    rods = rods.reshape((N,-1))    

    x = q_to_x(rods)
    x = jnp.array(x,dtype=jnp.float64)
    r1 = x.reshape(-1,6)[:,:3]
    r2 = x.reshape(-1,6)[:,3:]
    num_rods = rods.shape[0]

    i_indices, j_indices = jnp.triu_indices(num_rods, k=1)
    dists = dist_lin_seg_over_ij(r1,r2, i_indices, j_indices)

    num_contacts_list.append(jnp.count_nonzero(dists < rod_diameter))

# %%
plt.loglog(jnp.arange(1,num_is+1),num_contacts_list,'o-')

# get the power

x = jnp.arange(1,num_is+1)
y = jnp.array(num_contacts_list)
coeffs = jnp.polyfit(jnp.log(x), jnp.log(y), 1)
print(f'Power law exponent: {coeffs[0]}')
plt.loglog(x, jnp.exp(jnp.polyval(coeffs, jnp.log(x))), '--', label=f'fit: {coeffs[0]:.2f}')
plt.xlabel('Number of rods')
plt.ylabel('Number of contacts')
plt.legend()
