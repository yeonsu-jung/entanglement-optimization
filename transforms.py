import jax.numpy as jnp
import numpy as np

import jax
jax.config.update("jax_enable_x64", True)

# def sph2cart(phi,theta):
#     x = jnp.sin(phi)*jnp.cos(theta)
#     y = jnp.sin(phi)*jnp.sin(theta)
#     z = jnp.cos(phi)
#     return jnp.array([x,y,z]).transpose()

# def cart2sph(u):
#     x = u[:,0]
#     y = u[:,1]
#     z = u[:,2]
    
#     hxy = jnp.hypot(x, y)
#     r = jnp.hypot(hxy, z)
#     theta = jnp.arctan2(hxy, z)  # Polar angle (inclination)
#     phi = jnp.arctan2(y, x)      # Azimuthal angle
#     return r, theta, phi



def sph2cart(theta, phi, r=1):
    x = r * jnp.sin(theta) * jnp.cos(phi)
    y = r * jnp.sin(theta) * jnp.sin(phi)
    z = r * jnp.cos(theta)
    return jnp.stack([x, y, z], axis=-1)

def cart2sph(x, y, z):
    r = jnp.sqrt(x**2 + y**2 + z**2)
    theta = jnp.arccos(jnp.clip(z / r, -1.0, 1.0))  # Polar angle
    phi = jnp.arctan2(y, x)  # Azimuthal angle
    return r, theta, phi

def q_to_x(q):
    q = q.reshape((-1, 5))
    x0 = q[:, :3]
    offsets = sph2cart(q[:, 3], q[:, 4])
    x1 = x0 + offsets
    x = jnp.concatenate([x0, x1], axis=1)
    return x

def x_to_q(x):
    x = x.reshape((-1, 6))
    x0 = x[:, :3]
    x1 = x[:, 3:]
    offsets = x1 - x0
    r, theta, phi = cart2sph(offsets[:, 0], offsets[:, 1], offsets[:, 2])
    q = jnp.concatenate([x0, jnp.stack([theta, phi], axis=1)], axis=1)
    return q

# def q_to_x(q):
#     # q = jnp.array(q)
#     q = q.reshape((-1,5))
#     x = jnp.zeros((q.shape[0],6))
#     x = x.at[:,:3].set(q[:,:3])
#     x = x.at[:,3:6].set(sph2cart(q[:,3],q[:,4]) + x[:,0:3])
#     return x

# def x_to_q(x):
#     # x = jnp.array(x)
#     x = x.reshape((-1,6))
#     q = jnp.zeros((x.shape[0],5))
#     q = q.at[:,:3].set(x[:,:3])
#     _,th,phi=cart2sph(x[:,3:6] - x[:,:3])
#     q = q.at[:,3:5].set(jnp.array([th,phi]).transpose())
#     return q

# def q_to_x(q):
#     q = q.reshape((-1, 5))
#     x0 = q[:, :3]  # Extract initial positions
#     offset = sph2cart(q[:, 3], q[:, 4])  # Convert spherical to Cartesian
#     x1 = x0 + offset  # Compute the second position
#     x = jnp.concatenate([x0, x1], axis=1)  # Concatenate into a single array
#     return x

# def x_to_q(x):
#     x = x.reshape((-1, 6))
#     x0 = x[:, :3]  # Extract first set of positions
#     x1 = x[:, 3:6]  # Extract second set of positions
#     offset = x1 - x0  # Compute the offset vector
#     r, th, phi = cart2sph(offset[:, 0], offset[:, 1], offset[:, 2])  # Convert Cartesian to spherical
#     q = jnp.concatenate([x0, jnp.stack([th, phi], axis=1)], axis=1)  # Concatenate into a single array
#     return q

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