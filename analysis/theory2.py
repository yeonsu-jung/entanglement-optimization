# Re-import necessary packages after code state reset
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
N = 4
np.random.seed(0)
positions = np.random.randn(N, 3)
angles = np.random.rand(N, 2) * np.array([np.pi, 2 * np.pi])
params = np.hstack([np.hstack([positions[i], angles[i]]) for i in range(N)])

# Constraint and Jacobian
def constraint_function(p):
    return all_pairwise_distances(p, N)

eps = np.sqrt(np.finfo(float).eps)
J = np.vstack([
    approx_fprime(params, lambda p: constraint_function(p)[i], epsilon=eps)
    for i in range(len(constraint_function(params)))
])
_, svals, vh = svd(J)
tol = 1e-8
rank = np.sum(svals > tol)
nullspace = vh[rank:]

# --- Rigid body modes ---
def generate_rigid_body_modes(N, params):
    dof = 5 * N
    modes = []

    # 3 translations
    for axis in range(3):
        v = np.zeros(dof)
        for i in range(N):
            v[5 * i + axis] = 1.0
        modes.append(v)

    # 3 rotations
    for axis in range(3):
        v = np.zeros(dof)
        for i in range(N):
            x, y, z, theta, phi = params[5 * i:5 * (i + 1)]
            r = np.array([x, y, z])
            u = spherical_to_cartesian(theta, phi)

            rot_axis = np.zeros(3)
            rot_axis[axis] = 1.0

            dr = np.cross(rot_axis, r)
            du = np.cross(rot_axis, u)

            v[5 * i + 0:5 * i + 3] = dr

            eps = 1e-6
            def direction_to_theta_phi(u_new):
                normed = u_new / np.linalg.norm(u_new)
                th = np.arccos(normed[2])
                ph = np.arctan2(normed[1], normed[0])
                return np.array([th, ph])

            th0, ph0 = theta, phi
            th1, ph1 = direction_to_theta_phi(u + eps * du)
            dtheta = (th1 - th0) / eps
            dphi = (ph1 - ph0) / eps

            v[5 * i + 3] = dtheta
            v[5 * i + 4] = dphi

        modes.append(v)

    return np.array(modes)

rigid_modes = generate_rigid_body_modes(N, params)

# --- Projection errors of rigid motions onto nullspace ---
projections = []
for mode in rigid_modes:
    proj = nullspace.T @ (nullspace @ mode)
    error = np.linalg.norm(mode - proj)
    projections.append(error)

projections
