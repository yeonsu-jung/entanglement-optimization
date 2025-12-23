# %%
from jax import jit
import jax.numpy as jnp

# %%
# pick up random number from spherical coordinates
from protocols import create_intersecting_rods
from visualizations import plot_many_rods
num_rods = 100
q = create_intersecting_rods(num_rods)

ax = plot_many_rods(q.reshape(-1,5))
ax.axis('equal')


# %%
# nudging
q = q.reshape(-1,5)
centers = q[:,0:3]

# give a gaussian noise to the centers
import jax
key = jax.random.PRNGKey(0)
noise = jax.random.normal(key,shape=centers.shape)
noise = noise / jnp.linalg.norm(noise,axis=1,keepdims=True)
noise = 1.e-5 * noise

new_centers = centers + noise
q = q.at[:,0:3].set(new_centers)

# %%
plot_many_rods(q.reshape(-1,5))
# %%

# %%
from protocols import relax_collision

params = {}
params["col_rad"] = 1.e-3
params["amp"] = 1
dt = 1.e-2
N_outer = 1
Nmax = 1000

q = relax_collision(q.flatten(),dt,params,N_outer,Nmax,callback=None)