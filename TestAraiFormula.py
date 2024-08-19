# %%
import numpy as np

def compute_linking_number(x_i, y_i, z_i, phi_i, theta_i, x_j, y_j, z_j, phi_j, theta_j, l):
    p_i = np.array([x_i, y_i, z_i])
    p_j = np.array([x_j, y_j, z_j])
    u_i = np.array([np.sin(phi_i)*np.cos(theta_i), np.sin(phi_i)*np.sin(theta_i), np.cos(phi_i)])
    u_j = np.array([np.sin(phi_j)*np.cos(theta_j), np.sin(phi_j)*np.sin(theta_j), np.cos(phi_j)])

    p_ii = p_i + l*u_i
    p_jj = p_j + l*u_j
    
    a = p_i - p_j
    b = p_i - p_jj
    c = p_ii - p_jj
    d = p_ii - p_j

    cross_bc = np.cross(b, c)
    cross_da = np.cross(d, a)

    term1 = np.arctan2(np.dot(a, cross_bc),
                    (np.linalg.norm(a) * np.linalg.norm(b) * np.linalg.norm(c) +
                     np.dot(a, b) * np.linalg.norm(c) +
                     np.dot(c, a) * np.linalg.norm(b) +
                     np.dot(b, c) * np.linalg.norm(a)))

    term2 = np.arctan2(np.dot(c, cross_da),
                    (np.linalg.norm(c) * np.linalg.norm(d) * np.linalg.norm(a) +
                     np.dot(c, d) * np.linalg.norm(a) +
                     np.dot(a, c) * np.linalg.norm(d) +
                     np.dot(d, a) * np.linalg.norm(c)))

    lk_ij = 1 / (2 * np.pi) * (term1 + term2)
    return lk_ij


def compute_linking_number2(p_i,p_ii,p_j,p_jj):
    
    a = p_i - p_j
    b = p_i - p_jj
    c = p_ii - p_jj
    d = p_ii - p_j

    cross_bc = np.cross(b, c)
    cross_da = np.cross(d, a)

    term1 = np.arctan2(np.dot(a, cross_bc),
                    (np.linalg.norm(a) * np.linalg.norm(b) * np.linalg.norm(c) +
                     np.dot(a, b) * np.linalg.norm(c) +
                     np.dot(c, a) * np.linalg.norm(b) +
                     np.dot(b, c) * np.linalg.norm(a)))

    term2 = np.arctan2(np.dot(c, cross_da),
                    (np.linalg.norm(c) * np.linalg.norm(d) * np.linalg.norm(a) +
                     np.dot(c, d) * np.linalg.norm(a) +
                     np.dot(a, c) * np.linalg.norm(d) +
                     np.dot(d, a) * np.linalg.norm(c)))

    lk_ij = 1 / (2 * np.pi) * (term1 + term2)
    return lk_ij




x_1 = 0
y_1 = 0
z_1 = 0
phi_1 = 0
theta_1 = 0

x_2 = 0
y_2 = -0.5
z_2 = 0.001
phi_2 = np.pi/2
theta_2 = np.pi/2

l = 1

lk_ij = compute_linking_number(x_1, y_1, z_1, phi_1, theta_1, x_2, y_2, z_2, phi_2, theta_2, l)
print(lk_ij)
# %%
p_i = np.array([-1,0,0])
p_ii = np.array([1,0,0])
p_j = np.array([0,-1,-0.01])
p_jj = np.array([0,1,-0.01])

lk = compute_linking_number2(p_i,p_ii,p_j,p_jj)
print(lk)

# %%
