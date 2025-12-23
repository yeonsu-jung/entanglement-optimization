# %%
# todo: given a rod (fixed entirely) and a tentacle (fixed at one end), get the configuration of tentacle maximizing the entanglement with the rod

import jax.numpy as jnp
import numpy as np
from matplotlib import pyplot as plt

def compute_linking_number_cartesian(p_i, p_ii, p_j, p_jj):
    # p_i = jnp.array([x_i, y_i, z_i])
    # p_j = jnp.array([x_j, y_j, z_j])
    # u_i = jnp.array([jnp.sin(phi_i)*jnp.cos(theta_i), jnp.sin(phi_i)*jnp.sin(theta_i), jnp.cos(phi_i)])
    # u_j = jnp.array([jnp.sin(phi_j)*jnp.cos(theta_j), jnp.sin(phi_j)*jnp.sin(theta_j), jnp.cos(phi_j)])

    # p_ii = p_i + l*u_i
    # p_jj = p_j + l*u_j

    r_ij = p_i - p_j
    r_ijj = p_i - p_jj
    r_iij = p_ii - p_j
    r_iijj = p_ii - p_jj

    tol = 1e-6
    n1 = jnp.cross(r_ij, r_ijj)
    n1 = n1/(jnp.linalg.norm(n1)+tol)
    n2 = jnp.cross(r_ijj, r_iijj)
    n2 = n2/(jnp.linalg.norm(n2)+tol)
    n3 = jnp.cross(r_iijj, r_iij)
    n3 = n3/(jnp.linalg.norm(n3)+tol)
    n4 = jnp.cross(r_iij, r_ij)
    n4 = n4/(jnp.linalg.norm(n4)+tol)
    
    tol = 0.

    return -1/4/jnp.pi*jnp.abs(jnp.arcsin(  jnp.clip(jnp.dot(n1,n2),-1.+tol,1.-tol))
                               + jnp.arcsin(jnp.clip(jnp.dot(n2,n3),-1.+tol,1.-tol))
                               + jnp.arcsin(jnp.clip(jnp.dot(n3,n4),-1.+tol,1.-tol))
                               + jnp.arcsin(jnp.clip(jnp.dot(n4,n1),-1.+tol,1.-tol)))


# a_filament = jnp.array()
a_filament = np.linspace([0, 0, -0.2], [0, 0, 1.7], num=100)
# %%
a_filament = jnp.array(a_filament)
# %%
# or helix
def create_helix(start, radius, pitch, num_points, num_turns):
    t = jnp.linspace(0, num_turns * 2 * jnp.pi, num_points)
    x = start[0] + radius * jnp.cos(t)
    y = start[1] + radius * jnp.sin(t)
    z = start[2] + (pitch / (2 * jnp.pi)) * t
    return jnp.vstack((x, y, z)).T

a_helix = create_helix(start=jnp.array([0, 0, 0]), radius=0.1, pitch=0.5, num_points=100, num_turns=3)
# %%
fig,ax=plt.subplots(subplot_kw={'projection': '3d'})
ax.plot(a_filament[:,0], a_filament[:,1], a_filament[:,2])
ax.plot(a_helix[:,0], a_helix[:,1], a_helix[:,2])
ax.view_init(elev=0., azim=30)
plt.show()  

# %%
# get entanglement of the helix


p_i = a_filament[0,:]
p_ii = a_filament[1,:]

p_j = a_helix[0,:]
p_jj = a_helix[1,:]


# just between first two segments
compute_linking_number_cartesian(p_i, p_ii, p_j, p_jj)

# how can we get the total linking number between two filaments?


# %%


import jax
import jax.numpy as jnp

def _segments(polyline: jnp.ndarray):
    """Return segment vectors and midpoints for a polyline of shape (N,3)."""
    P0 = polyline[:-1]
    P1 = polyline[1:]
    dP = P1 - P0                 # (M,3)
    mid = 0.5 * (P0 + P1)        # (M,3)
    return dP, mid

def gauss_linking_number(poly1: jnp.ndarray, poly2: jnp.ndarray, eps: float = 1e-12):
    """
    Discrete Gauss linking integral (midpoint rule).
    poly1, poly2: (N,3) and (M,3) arrays of vertices.
    Returns a scalar (float32/float64) that approaches the true Lk as the mesh is refined.
    """
    dA, midA = _segments(poly1)        # (NA,3), (NA,3)
    dB, midB = _segments(poly2)        # (NB,3), (NB,3)

    # Broadcast over all segment pairs
    r = midA[:, None, :] - midB[None, :, :]          # (NA,NB,3)
    r2 = jnp.sum(r * r, axis=-1) + eps               # (NA,NB)
    r3 = r2 * jnp.sqrt(r2)                           # (NA,NB)

    cross = jnp.cross(dA[:, None, :], dB[None, :, :])        # (NA,NB,3)
    num = jnp.einsum('ijk,ijk->ij', cross, r)                # (NA,NB)

    contrib = num / r3                                       # (NA,NB)
    return jnp.sum(contrib) / (4.0 * jnp.pi)

# Example: total “entanglement” between your straight rod and helix
Lk_open = gauss_linking_number(a_filament, a_helix)
print("Gauss linking (open curves):", float(Lk_open))

# %%

# Sketch (not full app):
from jax import grad

def tentacle_polyline_from_ctrl(ctrl_pts, n_samples=200):
    # e.g., Catmull–Rom or cubic B-spline sampler returning (n_samples,3)
    # ... implement your favorite spline here ...
    raise NotImplementedError

def objective(ctrl_pts, rod_polyline):
    tentacle = tentacle_polyline_from_ctrl(ctrl_pts)
    lk = gauss_linking_number(tentacle, rod_polyline)
    bend = jnp.sum(jnp.square(tentacle[:-2] - 2*tentacle[1:-1] + tentacle[2:]))
    length = jnp.sum(jnp.linalg.norm(tentacle[1:] - tentacle[:-1], axis=-1))
    return -(lk) + 1e-3*bend + 1e-2*(length - 1.0)**2  # minus because we maximize lk

g_obj = grad(objective)
# Then run your favorite JAX optimizer (Optax Adam/FIRE/etc.), fixing the base point in ctrl_pts.

