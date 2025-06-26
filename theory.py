# %%
import numpy as np
from scipy.optimize import approx_fprime
from numpy.linalg import svd

# --- Helper Functions ---

def spherical_to_cartesian(theta, phi):
    return np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta)
    ])

def line_distance_param(p1, theta1, phi1, p2, theta2, phi2):
    u1 = spherical_to_cartesian(theta1, phi1)
    u2 = spherical_to_cartesian(theta2, phi2)
    cross = np.cross(u1, u2)
    norm_cross = np.linalg.norm(cross)
    diff = p2 - p1
    if norm_cross < 1e-10:
        return np.linalg.norm(np.cross(diff, u1))
    else:
        return np.abs(np.dot(diff, cross)) / norm_cross

def all_pairwise_distances(params, num_lines):
    distances = []
    lines = []
    for i in range(num_lines):
        x, y, z, theta, phi = params[5*i:5*(i+1)]
        lines.append((np.array([x, y, z]), theta, phi))
    for i in range(num_lines):
        for j in range(i+1, num_lines):
            r1, t1, p1 = lines[i]
            r2, t2, p2 = lines[j]
            d = line_distance_param(r1, t1, p1, r2, t2, p2)
            distances.append(d)
    return np.array(distances)

# --- Setup ---

N = 15  # Number of lines
np.random.seed(0)

# Random positions and angles for each line
# positions = np.random.randn(N, 3)
# angles = np.random.rand(N, 2) * np.array([np.pi, 2 * np.pi])  # theta in [0, pi], phi in [0, 2pi]
# params = np.hstack([np.hstack([positions[i], angles[i]]) for i in range(N)])
# params.shape

# %%
# Define constraint function
def constraint_function(p):
    return all_pairwise_distances(p, N)

# Compute Jacobian matrix of constraints
import time

start = time.time()
eps = np.sqrt(np.finfo(float).eps)


# J = np.vstack([
#     approx_fprime(params, lambda p: constraint_function(p)[i], epsilon=eps)
#     for i in range(len(constraint_function(params)))
# ])
J = []



end = time.time()
print(f"Time taken to compute Jacobian: {end - start:.4f} seconds")
# %%
J.shape
# %%
# 15 choose 2

# %%
# from matplotlib import pyplot as plt
# fig,ax= plt.subplots(subplot_kw={'projection': '3d'})
# halflength = 1.0
# for i in range(N):
#     x, y, z, theta, phi = params[5*i:5*(i+1)]

#     x_start = x - halflength * np.sin(theta) * np.cos(phi)
#     y_start = y - halflength * np.sin(theta) * np.sin(phi)
#     z_start = z - halflength * np.cos(theta)

#     x_end = x + halflength * np.sin(theta) * np.cos(phi)
#     y_end = y + halflength * np.sin(theta) * np.sin(phi)
#     z_end = z + halflength * np.cos(theta)
#     ax.plot([x_start, x_end], [y_start, y_end], [z_start, z_end], color='b', alpha=0.5)
    
# plt.show()


# Compute nullspace of Jacobian
_, svals, vh = svd(J)
tol = 1e-8
rank = np.sum(svals > tol)
nullspace = vh[rank:]  # Each row is a nullspace basis vector

# Output results
print(f"Number of lines: {N}")
print(f"Total parameters: {params.size}")
print(f"Number of distance constraints: {len(constraint_function(params))}")
print(f"Rank of Jacobian: {rank}")
print(f"Nullspace dimension (infinitesimal motions that preserve all distances): {nullspace.shape[0]}")
