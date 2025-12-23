# jax_nudger.py
import jax
import jax.numpy as jnp
from typing import Optional
from jax import jit, lax

from jax import grad,random,jit


import matplotlib.pyplot as plt
from pathlib import Path
# from protocols import create_random_rods
# from potentials import dist_lin_seg
# from transforms import q_to_x, x_to_q


def fixbound(num):
    """Ensure the number is within the bounds [0, 1]."""
    return jnp.clip(num, 0, 1)
###########

@jit
def dist_lin_seg(point1s, point1e, point2s, point2e):
    """Calculate the shortest distance between two line segments using JAX with cond."""
    d1 = point1e - point1s
    d2 = point2e - point2s
    d12 = point2s - point1s

    D1 = jnp.dot(d1, d1)
    D2 = jnp.dot(d2, d2)
    S1 = jnp.dot(d1, d12)
    S2 = jnp.dot(d2, d12)
    R = jnp.dot(d1, d2)

    den = D1 * D2 - R**2
    
    def case1():
        (t,u) = lax.cond( D1 != 0. , 
                    lambda _: (fixbound(S1/D1),0.),
                    lambda _: lax.cond(D2 != 0.,
                             lambda _: (0.,fixbound(-S2/D2)),
                             lambda _: (0.,0.),
                             None),
                    None)        
        return (t,u)
    
    def case2_1():
        t = 0.
        u = -S2/D2
        uf = fixbound(u)
        
        (t,u) = lax.cond(uf != u, 
                    lambda _: (fixbound((uf * R + S1) / D1), uf),
                    lambda _: (t, u),
                    None)
        
        return (t,u)
    
    def case2_2():
        t = fixbound((S1 * D2 - S2 * R) / den)
        u = (t * R - S2) / D2
        uf = fixbound(u)
        
        (t,u) = lax.cond(uf != u, 
                    lambda _: (fixbound((uf * R + S1) / D1), uf),
                    lambda _: (t, u),
                    None)
        
        return (t,u)        
    
    def case2():
        (t,u) = lax.cond( den == 0. , 
                    lambda _: case2_1(),                    
                    lambda _: case2_2(),
                    None)        
        return (t,u)
    
    (t,u) = lax.cond( (D1 == 0.) & (D2 == 0.),
                        lambda _: case1(),
                        lambda _: case2(),
                        None)
    
    dist = jnp.linalg.norm(d1 * t - d2 * u - d12)
    
    return dist





from jax import jacfwd, jacrev, vmap

def _pairwise_indices(N: int):
    return jnp.triu_indices(N, k=1)



# @jax.jit
def nudge_step(q: jnp.ndarray, jacobian_fn: jnp.ndarray, D: float, step: float = 1.0, underrelax: float = 1.0, eps: float = 1e-12):
    """
    One Jacobi-style nudge step for equal-diameter disks (2D).
    X: (N,2) centers, D: diameter (required min center distance)
    step: scale for the accumulated displacement (usually 1.0)
    underrelax: 0<underrelax<=1 to stabilize large systems (e.g. 0.8~0.95)
    """
    N = q.shape[0]
    ii, jj = _pairwise_indices(N)          # (M,), upper-tri pairs
    
    # x = q_to_x(q).reshape(-1, 2, 3)
    # r1 = x[:,0,:]
    # r2 = x[:,1,:]

    # rij = X[ii] - X[jj]                    # (M,2)
    # dij = jnp.linalg.norm(rij, axis=1)     # (M,)

    # dist_func_vmap = vmap(dist_lin_seg_vector, in_axes=(0,0,0,0))

    rij = jacobian_fn(q) # (M, M)
    dij = jnp.linalg.norm(rij, axis=1)     # (M,)

    # penetration (positive if overlapping)
    pen = jnp.maximum(0.0, D - dij)        # (M,)
    # unit directions (safe with eps)
    uij = rij / (dij[:, None] + eps)       # (M,2)

    # Equal split: push each by pen/2 along ±u
    dX_i =  0.5 * pen[:, None] * uij
    dX_j = -0.5 * pen[:, None] * uij

    # Accumulate to particles (scatter-add)
    dX = jnp.zeros_like(x)
    dX = dX.at[ii].add(dX_i)
    dX = dX.at[jj].add(dX_j)

    X_new = x + (step * underrelax) * dX
    q_new = x_to_q(X_new.reshape(-1,6))

    # Report max penetration after this step (for stopping)
    max_pen = jnp.max(pen) if pen.size > 0 else 0.0
    return X_new, max_pen

def fixbound(num):
    """Ensure the number is within the bounds [0, 1]."""
    return jnp.clip(num, 0, 1)

@jit
# def dist_lin_seg(point1s, point1e, point2s, point2e):
def dist_lin_seg(x_pair):
    p1s_x, p1s_y, p1s_z, p1e_x, p1e_y, p1e_z, p2s_x, p2s_y, p2s_z, p2e_x, p2e_y, p2e_z = x_pair
    """Calculate the shortest distance between two line segments using JAX with cond."""
    d1 = jnp.array([p1e_x - p1s_x, p1e_y - p1s_y, p1e_z - p1s_z])
    d2 = jnp.array([p2e_x - p2s_x, p2e_y - p2s_y, p2e_z - p2s_z])
    d12 = jnp.array([p2s_x - p1s_x, p2s_y - p1s_y, p2s_z - p1s_z])
    # d1 = point1e - point1s
    # d2 = point2e - point2s
    # d12 = point2s - point1s

    D1 = jnp.dot(d1, d1)
    D2 = jnp.dot(d2, d2)
    S1 = jnp.dot(d1, d12)
    S2 = jnp.dot(d2, d12)
    R = jnp.dot(d1, d2)

    den = D1 * D2 - R**2
    
    def case1():
        (t,u) = lax.cond( D1 != 0. , 
                    lambda _: (fixbound(S1/D1),0.),
                    lambda _: lax.cond(D2 != 0.,
                             lambda _: (0.,fixbound(-S2/D2)),
                             lambda _: (0.,0.),
                             None),
                    None)        
        return (t,u)
    
    def case2_1():
        t = 0.
        u = -S2/D2
        uf = fixbound(u)
        
        (t,u) = lax.cond(uf != u, 
                    lambda _: (fixbound((uf * R + S1) / D1), uf),
                    lambda _: (t, u),
                    None)
        
        return (t,u)
    
    def case2_2():
        t = fixbound((S1 * D2 - S2 * R) / den)
        u = (t * R - S2) / D2
        uf = fixbound(u)
        
        (t,u) = lax.cond(uf != u, 
                    lambda _: (fixbound((uf * R + S1) / D1), uf),
                    lambda _: (t, u),
                    None)
        
        return (t,u)        
    
    def case2():
        (t,u) = lax.cond( den == 0. , 
                    lambda _: case2_1(),                    
                    lambda _: case2_2(),
                    None)        
        return (t,u)
    
    (t,u) = lax.cond( (D1 == 0.) & (D2 == 0.),
                        lambda _: case1(),
                        lambda _: case2(),
                        None)
    
    dist = jnp.linalg.norm(d1 * t - d2 * u - d12)
    
    return dist

def dist_lin_seg_vector(point1s, point1e, point2s, point2e):
    """Calculate the shortest distance between two line segments using JAX with cond."""
    d1 = point1e - point1s
    d2 = point2e - point2s
    d12 = point2s - point1s

    D1 = jnp.dot(d1, d1)
    D2 = jnp.dot(d2, d2)
    S1 = jnp.dot(d1, d12)
    S2 = jnp.dot(d2, d12)
    R = jnp.dot(d1, d2)

    den = D1 * D2 - R**2
    
    def case1():
        (t,u) = lax.cond( D1 != 0. , 
                    lambda _: (fixbound(S1/D1),0.),
                    lambda _: lax.cond(D2 != 0.,
                             lambda _: (0.,fixbound(-S2/D2)),
                             lambda _: (0.,0.),
                             None),
                    None)        
        return (t,u)
    
    def case2_1():
        t = 0.
        u = -S2/D2
        uf = fixbound(u)
        
        (t,u) = lax.cond(uf != u, 
                    lambda _: (fixbound((uf * R + S1) / D1), uf),
                    lambda _: (t, u),
                    None)
        
        return (t,u)
    
    def case2_2():
        t = fixbound((S1 * D2 - S2 * R) / den)
        u = (t * R - S2) / D2
        uf = fixbound(u)
        
        (t,u) = lax.cond(uf != u, 
                    lambda _: (fixbound((uf * R + S1) / D1), uf),
                    lambda _: (t, u),
                    None)
        
        return (t,u)        
    
    def case2():
        (t,u) = lax.cond( den == 0. , 
                    lambda _: case2_1(),                    
                    lambda _: case2_2(),
                    None)        
        return (t,u)
    
    (t,u) = lax.cond( (D1 == 0.) & (D2 == 0.),
                        lambda _: case1(),
                        lambda _: case2(),
                        None)
    
    # dist = jnp.linalg.norm(d1 * t - d2 * u - d12)    
    r1 = d1 * t + point1s
    r2 = d2 * u + point2s
    # return t,u,r1,r2, r1-r2, jnp.linalg.norm(r1 - r2)
    return r1 - r2

def create_pairs(m):
    N, M = m.shape
    # Get the upper triangular indices excluding the diagonal
    i, j = jnp.triu_indices(N, k=1)
    # Retrieve rows for each index in the pairs
    m_i = m[i]  # Shape will be (N(N-1)/2, M)
    m_j = m[j]  # Shape will be (N(N-1)/2, M)
    # Concatenate the rows from each pair horizontally
    m_pairs = jnp.concatenate([m_i, m_j], axis=1)  # Resulting shape will be (N(N-1)/2, 2M)
    return m_pairs

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

def create_random_rods(num_rods,random_keys):
    # create jnp random array
    key = random.key(random_keys[0])
    p1s = random.uniform(key, (num_rods,3), minval=-0.5, maxval=0.5)
    key = random.key(random_keys[1])
    phi1 = random.uniform(key, (num_rods,1), minval=0., maxval=jnp.pi)
    key = random.key(random_keys[2])
    theta1 = random.uniform(key, (num_rods,1), minval=0., maxval=2*jnp.pi)
    q0 = jnp.concatenate([p1s, phi1, theta1], axis=1)
    
    x0 = q_to_x(q0)
    center = jnp.mean(x0[:,:3],axis=0)
    # q0[:,:3] = q0[:,:3] - center    
    q0 = q0.at[:,:3].set(q0[:,:3] - center)
    
    q0 = q0.flatten()
    q0 = jnp.array(q0,dtype=jnp.float64)
    return q0


if __name__ == "__main__":

    num_rods = 20
    random_keys = [0,0,0]
    q = create_random_rods(num_rods, random_keys)    

    # @jit
    vmap_dist = vmap(dist_lin_seg)
    def distance_from_q(q,ii,jj):
        x = q_to_x(q).reshape(-1, 2, 3)
        r1 = x[:,0,:]
        r2 = x[:,1,:]

        # ii, jj = _pairwise_indices(num_rods)

        x_pairs = jnp.concatenate([r1[ii], r2[ii], r1[jj], r2[jj]], axis=1)
        dists = vmap_dist(x_pairs)
        return dists
    
    # ii,jj as input
    

    D = 1./100.
    ii, jj = _pairwise_indices(num_rods)
    dij = distance_from_q(q, ii, jj)
    
    dij.shape
    contact_candidate = dij < D

    # contact pairs
    ii_contact = ii[contact_candidate]
    jj_contact = jj[contact_candidate]

    dij_contact = distance_from_q(q, ii_contact, jj_contact)

    # direction of nudging?
    jacobian_fn = jit(jacfwd(distance_from_q, argnums=0), static_argnums=(1,2))
    jacobian_example = jacobian_fn(q, ii_contact, jj_contact)
    print(jacobian_example.shape)  # should be (M, N*5)

    