# %%
import numpy as np
import potentials as pt
import time
import transforms as tf

from matplotlib import pyplot as plt
from jax import numpy as jnp

from visualizations import plot_many_rods
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import svds

pth = '/Users/yeonsu/GitHub/entanglement-optimization/results/37,178,56/2025-02-18_18_EntangledRelaxedPacking-N0200-AR0200-Scale1/q_relaxed.txt'
q = np.loadtxt(pth)

print(q.shape)
# %%
def spherical_to_cartesian(theta, phi):
    return np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta)
    ])

# %%
q = q.reshape(-1,5)
q = jnp.array(q)

q.shape
num_rods = q.shape[0]

# %%
fig,ax= plt.subplots(subplot_kw={'projection': '3d'})
halflength = 1.0


plot_many_rods(q,ax=ax)

# %%
# old ways

start = time.time()
q_pairs = pt.create_pairs(jnp.reshape(q,(-1,5)))
d = pt.all_pairwise_distances(q_pairs)
end = time.time()
print(f"Time taken to compute pairwise distance: {end - start:.4f} seconds")

# %%
# new ways
x = tf.q_to_x(q)
x = jnp.array(x,dtype=jnp.float64)

def analyze_pairwise_dist_entanglement(nodes_at_ith_frame,num_rods):
    i_indices, j_indices = jnp.triu_indices(num_rods, k=1)
    r1 = nodes_at_ith_frame.reshape(-1,6)[:,:3]
    r2 = nodes_at_ith_frame.reshape(-1,6)[:,3:]
    pairwise_acn = pt.acn_over_ij(r1,r2, i_indices, j_indices)
    pairwise_dist = pt.dist_lin_seg_over_ij(r1,r2, i_indices, j_indices)
    dist = jnp.min(pairwise_dist)
    entanglement = jnp.sum(jnp.abs(pairwise_acn)) / (num_rods*(num_rods-1)/2)
    return dist,entanglement,pairwise_dist, pairwise_acn

start = time.time()
d,e,pdist,pacn = analyze_pairwise_dist_entanglement(x,num_rods)
end = time.time()


# %%
diameter = pdist.min()
TOL = 1e-20
num_rods_in_contact = []
TOL_SPACE = jnp.geomspace(1e-20,1e-6,30)
for TOL in TOL_SPACE:
    num_rods_in_contact.append(jnp.count_nonzero(pdist < diameter + TOL))
num_rods_in_contact = jnp.array(num_rods_in_contact)

plt.plot(TOL_SPACE,num_rods_in_contact,'o-')

# %%
TOL = .5e-6
num_rods_in_contact = jnp.count_nonzero(pdist < diameter + TOL)
print(num_rods_in_contact)

# %%
sparse_rows = jnp.where(pdist < diameter + TOL)[0]

i_indices, j_indices = jnp.triu_indices(num_rods, k=1)
sparse_i = i_indices[sparse_rows]
sparse_j = j_indices[sparse_rows]
sparse_dist = pdist[sparse_rows]

# %%
from sympy import Ynm
from sympy.utilities.lambdify import lambdify
from sympy.parsing.mathematica import parse_mathematica

d12_expression = parse_mathematica("""
Sqrt[(-((z1 - z2)*Sin[phi1 - phi2]*Sin[theta1]*Sin[theta2]) + 
    (y1 - y2)*(-(Cos[phi1]*Cos[theta2]*Sin[theta1]) + Cos[phi2]*Cos[theta1]*
       Sin[theta2]) + (x1 - x2)*(Cos[theta2]*Sin[phi1]*Sin[theta1] - 
      Cos[theta1]*Sin[phi2]*Sin[theta2]))^2]/
 Sqrt[Sin[phi1 - phi2]^2*Sin[theta1]^2*Sin[theta2]^2 + 
   (Cos[phi1]*Cos[theta2]*Sin[theta1] - Cos[phi2]*Cos[theta1]*Sin[theta2])^2 + 
   (Cos[theta2]*Sin[phi1]*Sin[theta1] - Cos[theta1]*Sin[phi2]*Sin[theta2])^2]
""")

# %%
d12_expression

# %%
from sympy import symbols, diff

theta1, phi1, phi2, theta2, x1, y1, z1, x2, y2, z2 = symbols('theta1 phi1 phi2 theta2 x1 y1 z1 x2 y2 z2')

# Differentiate with respect to theta1
d_d12_dtheta1 = diff(d12_expression, theta1)
d_d12_dtheta2 = diff(d12_expression, theta2)
d_d12_dphi1 = diff(d12_expression, phi1)
d_d12_dphi2 = diff(d12_expression, phi2)
d_d12_dx1 = diff(d12_expression, x1)
d_d12_dy1 = diff(d12_expression, y1)
d_d12_dz1 = diff(d12_expression, z1)
d_d12_dx2 = diff(d12_expression, x2)
d_d12_dy2 = diff(d12_expression, y2)
d_d12_dz2 = diff(d12_expression, z2)
# d_d12_dtheta1 = d_d12_dtheta1.simplify()
# d_d12_dtheta2 = d_d12_dtheta2.simplify()
# d_d12_dphi1 = d_d12_dphi1.simplify()
# d_d12_dphi2 = d_d12_dphi2.simplify()
# d_d12_dx1 = d_d12_dx1.simplify()
# d_d12_dy1 = d_d12_dy1.simplify()
# d_d12_dz1 = d_d12_dz1.simplify()
# d_d12_dx2 = d_d12_dx2.simplify()
# d_d12_dy2 = d_d12_dy2.simplify()
# d_d12_dz2 = d_d12_dz2.simplify()


# %%
# lambdify
d_12_func = lambdify((theta1, phi1, phi2, theta2, x1, y1, z1, x2, y2, z2), d12_expression)
d_d12_dtheta1_func = lambdify((theta1, phi1, phi2, theta2, x1, y1, z1, x2, y2, z2), d_d12_dtheta1)
d_d12_dtheta2_func = lambdify((theta1, phi1, phi2, theta2, x1, y1, z1, x2, y2, z2), d_d12_dtheta2)
d_d12_dphi1_func = lambdify((theta1, phi1, phi2, theta2, x1, y1, z1, x2, y2, z2), d_d12_dphi1)
d_d12_dphi2_func = lambdify((theta1, phi1, phi2, theta2, x1, y1, z1, x2, y2, z2), d_d12_dphi2)
d_d12_dx1_func = lambdify((theta1, phi1, phi2, theta2, x1, y1, z1, x2, y2, z2), d_d12_dx1)
d_d12_dy1_func = lambdify((theta1, phi1, phi2, theta2, x1, y1, z1, x2, y2, z2), d_d12_dy1)
d_d12_dz1_func = lambdify((theta1, phi1, phi2, theta2, x1, y1, z1, x2, y2, z2), d_d12_dz1)
d_d12_dx2_func = lambdify((theta1, phi1, phi2, theta2, x1, y1, z1, x2, y2, z2), d_d12_dx2)
d_d12_dy2_func = lambdify((theta1, phi1, phi2, theta2, x1, y1, z1, x2, y2, z2), d_d12_dy2)
d_d12_dz2_func = lambdify((theta1, phi1, phi2, theta2, x1, y1, z1, x2, y2, z2), d_d12_dz2)



# %%
# Test the function with some values
theta1_val = np.pi/2
phi1_val = 0.

u1 = spherical_to_cartesian(theta1_val, phi1_val)
print(u1)

theta2_val = np.pi/2
phi2_val = np.pi/2

u2 = spherical_to_cartesian(theta2_val, phi2_val)
print(u2)

x1_val = 0.5
y1_val = 0.5
z1_val = 0.1

x2_val = 0.5
y2_val = 0.5
z2_val = 0.5


d = d_12_func(theta1_val, phi1_val, phi2_val, theta2_val, x1_val, y1_val, z1_val, x2_val, y2_val, z2_val)

dddtheta1 = d_d12_dtheta1_func(theta1_val, phi1_val, phi2_val, theta2_val, x1_val, y1_val, z1_val, x2_val, y2_val, z2_val)
dddtheta2 = d_d12_dtheta2_func(theta1_val, phi1_val, phi2_val, theta2_val, x1_val, y1_val, z1_val, x2_val, y2_val, z2_val)
dddphi1 = d_d12_dphi1_func(theta1_val, phi1_val, phi2_val, theta2_val, x1_val, y1_val, z1_val, x2_val, y2_val, z2_val)
dddphi2 = d_d12_dphi2_func(theta1_val, phi1_val, phi2_val, theta2_val, x1_val, y1_val, z1_val, x2_val, y2_val, z2_val)
dddtheta1 = d_d12_dtheta1_func(theta1_val, phi1_val, phi2_val, theta2_val, x1_val, y1_val, z1_val, x2_val, y2_val, z2_val)

dddx1 = d_d12_dx1_func(theta1_val, phi1_val, phi2_val, theta2_val, x1_val, y1_val, z1_val, x2_val, y2_val, z2_val)
dddy1 = d_d12_dy1_func(theta1_val, phi1_val, phi2_val, theta2_val, x1_val, y1_val, z1_val, x2_val, y2_val, z2_val)
dddz1 = d_d12_dz1_func(theta1_val, phi1_val, phi2_val, theta2_val, x1_val, y1_val, z1_val, x2_val, y2_val, z2_val)
dddx2 = d_d12_dx2_func(theta1_val, phi1_val, phi2_val, theta2_val, x1_val, y1_val, z1_val, x2_val, y2_val, z2_val)
dddy2 = d_d12_dy2_func(theta1_val, phi1_val, phi2_val, theta2_val, x1_val, y1_val, z1_val, x2_val, y2_val, z2_val)
dddz2 = d_d12_dz2_func(theta1_val, phi1_val, phi2_val, theta2_val, x1_val, y1_val, z1_val, x2_val, y2_val, z2_val)

print(d)

# %%
print(dddtheta1)
print(dddtheta2)
print(dddphi1)
print(dddphi2)

print(dddx1)
print(dddy1)
print(dddz1)
print(dddx2)
print(dddy2)
print(dddz2)


# %%
sparse_i = i_indices[sparse_rows]
sparse_j = j_indices[sparse_rows]
sparse_dist = pdist[sparse_rows]


# %%
# construct a vector (upper triangle?)
num_contacts = len(sparse_i)
for i in range(num_contacts):
    1

# %%
J = []
for i in range(num_contacts):
    # print(f"Contact {i}:")
    # print(f"  Rod 1: {sparse_i[i]}")
    # print(f"  Rod 2: {sparse_j[i]}")
    # print(f"  Distance: {sparse_dist[i]}")
    # print()


    q_i = q[sparse_i[i]]
    q_j = q[sparse_j[i]]
    
    theta1_val = q_i[3]
    phi1_val = q_i[4]
    theta2_val = q_j[3]
    phi2_val = q_j[4]
    x1_val = q_i[0]
    y1_val = q_i[1]
    z1_val = q_i[2]
    x2_val = q_j[0]
    y2_val = q_j[1]
    z2_val = q_j[2]
    d = d_12_func(theta1_val, phi1_val, phi2_val, theta2_val, x1_val, y1_val, z1_val, x2_val, y2_val, z2_val)

    dddtheta1 = d_d12_dtheta1_func(theta1_val, phi1_val, phi2_val, theta2_val, x1_val, y1_val, z1_val, x2_val, y2_val, z2_val)
    dddtheta2 = d_d12_dtheta2_func(theta1_val, phi1_val, phi2_val, theta2_val, x1_val, y1_val, z1_val, x2_val, y2_val, z2_val)
    dddphi1 = d_d12_dphi1_func(theta1_val, phi1_val, phi2_val, theta2_val, x1_val, y1_val, z1_val, x2_val, y2_val, z2_val)
    dddphi2 = d_d12_dphi2_func(theta1_val, phi1_val, phi2_val, theta2_val, x1_val, y1_val, z1_val, x2_val, y2_val, z2_val)

    dddx1 = d_d12_dx1_func(theta1_val, phi1_val, phi2_val, theta2_val, x1_val, y1_val, z1_val, x2_val, y2_val, z2_val)
    dddy1 = d_d12_dy1_func(theta1_val, phi1_val, phi2_val, theta2_val, x1_val, y1_val, z1_val, x2_val, y2_val, z2_val)
    dddz1 = d_d12_dz1_func(theta1_val, phi1_val, phi2_val, theta2_val, x1_val, y1_val, z1_val, x2_val, y2_val, z2_val)

    dddx2 = d_d12_dx2_func(theta1_val, phi1_val, phi2_val, theta2_val, x1_val, y1_val, z1_val, x2_val, y2_val, z2_val)
    dddy2 = d_d12_dy2_func(theta1_val, phi1_val, phi2_val, theta2_val, x1_val, y1_val, z1_val, x2_val, y2_val, z2_val)
    dddz2 = d_d12_dz2_func(theta1_val, phi1_val, phi2_val, theta2_val, x1_val, y1_val, z1_val, x2_val, y2_val, z2_val)

    # J.append([dddtheta1, dddtheta2, dddphi1, dddphi2, dddx1, dddy1, dddz1, dddx2, dddy2, dddz2])
    # J.append([dddtheta1, dddtheta2, dddphi1, dddphi2, dddx1, dddy1, dddz1, dddx2, dddy2, dddz2])
J = np.array(J)
print(J.shape)

# %%




# %%




# %%
# J = np.vstack([
#     approx_fprime(params, lambda p: constraint_function(p)[i], epsilon=eps)
#     for i in range(len(constraint_function(params)))
# ])
J = []
for i in range(10):
    for j in range(i+1, 10):
        dddtheta1 = d_d12_dtheta1_func(theta1_val, phi1_val, phi2_val, theta2_val, x1_val, y1_val, z1_val, x2_val, y2_val, z2_val)
        dddtheta2 = d_d12_dtheta2_func(theta1_val, phi1_val, phi2_val, theta2_val, x1_val, y1_val, z1_val, x2_val, y2_val, z2_val)
        dddphi1 = d_d12_dphi1_func(theta1_val, phi1_val, phi2_val, theta2_val, x1_val, y1_val, z1_val, x2_val, y2_val, z2_val)
        dddphi2 = d_d12_dphi2_func(theta1_val, phi1_val, phi2_val, theta2_val, x1_val, y1_val, z1_val, x2_val, y2_val, z2_val)
        dddtheta1 = d_d12_dtheta1_func(theta1_val, phi1_val, phi2_val, theta2_val, x1_val, y1_val, z1_val, x2_val, y2_val, z2_val)

        dddx1 = d_d12_dx1_func(theta1_val, phi1_val, phi2_val, theta2_val, x1_val, y1_val, z1_val, x2_val, y2_val, z2_val)
        dddy1 = d_d12_dy1_func(theta1_val, phi1_val, phi2_val, theta2_val, x1_val, y1_val, z1_val, x2_val, y2_val, z2_val)
        dddz1 = d_d12_dz1_func(theta1_val, phi1_val, phi2_val, theta2_val, x1_val, y1_val, z1_val, x2_val, y2_val, z2_val)
        dddx2 = d_d12_dx2_func(theta1_val, phi1_val, phi2_val, theta2_val, x1_val, y1_val, z1_val, x2_val, y2_val, z2_val)
        dddy2 = d_d12_dy2_func(theta1_val, phi1_val, phi2_val, theta2_val, x1_val, y1_val, z1_val, x2_val, y2_val, z2_val)
        dddz2 = d_d12_dz2_func(theta1_val, phi1_val, phi2_val, theta2_val, x1_val, y1_val, z1_val, x2_val, y2_val, z2_val)

        J.append([dddtheta1, dddtheta2, dddphi1, dddphi2, dddx1, dddy1, dddz1, dddx2, dddy2, dddz2])
J = np.array(J)
print(J.shape)

# %%


u2 = spherical_to_cartesian(theta2_val, phi2_val)
x2_val = x2_val + u2[0]
y2_val = y2_val + u2[1]
z2_val = z2_val + u2[2]

result = d_d12_dtheta1_func(theta1_val, phi1_val, phi2_val, theta2_val, x1_val, y1_val, z1_val, x2_val, y2_val, z2_val)
print(result)
