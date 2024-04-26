import jax.numpy as jnp
import numpy as np

import jax
jax.config.update("jax_enable_x64", True)

def sph2cart(phi,theta):
    x = np.sin(phi)*np.cos(theta)
    y = np.sin(phi)*np.sin(theta)
    z = np.cos(phi)
    return np.array([x,y,z]).transpose()

def q_to_x(q):
    # q = jnp.array(q)
    q = q.reshape((-1,5))
    x = jnp.zeros((q.shape[0],6))
    x = x.at[:,:3].set(q[:,:3])
    x = x.at[:,3:6].set(sph2cart(q[:,3],q[:,4]) + x[:,0:3])
    return x

def q_to_u(q):
    # q = jnp.array(q)
    q = q.reshape((-1,5))
    # u = jnp.zeros((q.shape[0],3))
    u = sph2cart(q[:,3],q[:,4])
    return u