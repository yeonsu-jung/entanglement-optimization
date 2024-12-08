import jax.numpy as jnp
import numpy as np

import jax
jax.config.update("jax_enable_x64", True)

def sph2cart(phi,theta):
    x = jnp.sin(phi)*jnp.cos(theta)
    y = jnp.sin(phi)*jnp.sin(theta)
    z = jnp.cos(phi)
    return jnp.array([x,y,z]).transpose()

def cart2sph(u):
    x = u[:,0]
    y = u[:,1]
    z = u[:,2]
    
    hxy = jnp.hypot(x, y)
    r = jnp.hypot(hxy, z)
    theta = jnp.arctan2(hxy, z)  # Polar angle (inclination)
    phi = jnp.arctan2(y, x)      # Azimuthal angle
    return r, theta, phi

def q_to_x(q):
    # q = jnp.array(q)
    q = q.reshape((-1,5))
    x = jnp.zeros((q.shape[0],6))
    x = x.at[:,:3].set(q[:,:3])
    x = x.at[:,3:6].set(sph2cart(q[:,3],q[:,4]) + x[:,0:3])
    return x

def x_to_q(x):
    # x = jnp.array(x)
    x = x.reshape((-1,6))
    q = jnp.zeros((x.shape[0],5))
    q = q.at[:,:3].set(x[:,:3])
    _,th,phi=cart2sph(x[:,3:6] - x[:,:3])
    q = q.at[:,3:5].set(jnp.array([th,phi]).transpose())
    return q

# jnp.array([jnp.sin(phi_i)*jnp.cos(theta_i), jnp.sin(phi_i)*jnp.sin(theta_i), jnp.cos(phi_i)])

def q_to_u(q):
    # q = jnp.array(q)
    q = q.reshape((-1,5))
    # u = jnp.zeros((q.shape[0],3))
    u = sph2cart(q[:,3],q[:,4])
    return u

def x_to_epairs(x,num_rods):
    # x: (num_rods*num_vertices*3,) array
    
    x_mat = x.reshape(num_rods,-1,3)
    num_vertices = x_mat.shape[1]

    e = jnp.zeros((num_rods,(num_vertices-1)*6))
    
    for i in range(num_rods):
        x_i = x_mat[i,:,:]
        e_i = jnp.concatenate([x_i[1:,:], x_i[:-1,:]],axis=1).flatten()
        e = e.at[i,:].set(e_i)
        
    return e

def x_to_rpairs(x,num_rods):
    # x: (num_rods*num_vertices*3,) array
    
    x_mat = x.reshape(num_rods,-1,3)
    num_vertices = x_mat.shape[1]

    r = jnp.zeros((num_rods,num_vertices*3))
    
    for i in range(num_rods):
        x_i = x[i,:]
        r = r.at[i,:].set(x_i)
        
    return r

def vert_to_edge(vert):
    edge = np.hstack([vert[1:,:],vert[:-1,:]])
    return edge