# %%
import numpy as np

def skew_line_closest_points(x, y, z, theta, phi):
    # Define direction vectors
    v1 = np.array([0.0, 0.0, 1.0])  # Line 1 direction
    v2 = np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta)
    ])  # Line 2 direction

    # Compute scalar t such that the vector between lines is orthogonal to both directions
    sin_theta_sq = np.sin(theta)**2
    dot_xy = x * v2[0] + y * v2[1]
    t = -dot_xy / sin_theta_sq if sin_theta_sq > 1e-10 else 0.0
    s = z + t * v2[2]

    # Compute the points on each line
    r1 = s * v1
    r2 = np.array([x, y, z]) + t * v2

    # Distance vector from Line 1 to Line 2
    distance_vector = r2 - r1

    # Midpoint between the two closest points (point of contact)
    contact_point = 0.5 * (r1 + r2)

    return distance_vector, contact_point

# Example usage
x_val, y_val, z_val = 1.0, 2.0, 3.0
theta_val = np.pi / 3
phi_val = np.pi / 4

d_vec, p_contact = skew_line_closest_points(x_val, y_val, z_val, theta_val, phi_val)
print("Distance vector:", d_vec)
print("Point of contact:", p_contact)

# %%
from sympy.utilities.lambdify import lambdify
from sympy.parsing.mathematica import parse_mathematica
from sympy import symbols, diff

theta1, phi1, phi2, theta2, x1, y1, z1, x2, y2, z2 = symbols('theta1 phi1 phi2 theta2 x1 y1 z1 x2 y2 z2')
x,y,z,theta,phi= symbols('x y z theta phi')

# mx_expr = parse_mathematica("(3*x + x*Cos[2*phi] + y*Sin[2*phi])/4")
# my_expr = parse_mathematica("(3*y - y*Cos[2*phi] + x*Sin[2*phi])/4")
# mz_expr = parse_mathematica("z")

# (x - (x + y)*Cos[phi]^2)/2, (y - (x + y)*Cos[phi]*Sin[phi])/2, z - (x + y)*Cos[phi]*Cot[theta]
mx_expr = parse_mathematica("(x - (x + y)*Cos[phi]^2)/2")
my_expr = parse_mathematica("(y - (x + y)*Cos[phi]*Sin[phi])/2")
mz_expr = parse_mathematica("z - (x + y)*Cos[phi]*Cot[theta]")

# %%
mx_func = lambdify((x,y,z,theta,phi), mx_expr)
my_func = lambdify((x,y,z,theta,phi), my_expr)
mz_func = lambdify((x,y,z,theta,phi), mz_expr)

# %%


p_contact_sym = [mx_func(x_val,y_val,z_val,theta_val,phi_val)
,my_func(x_val,y_val,z_val,theta_val,phi_val)
,mz_func(x_val,y_val,z_val,theta_val,phi_val)]

print(p_contact_sym)


# %%
def d_ij(x,y,phi):
    return -x*np.cos(phi) + y*np.sin(phi)

def d_ij_vec(x,y,phi):
    return np.array([
        -(-x*np.cos(phi) + y*np.sin(phi))*np.cos(phi),
        (-x*np.sin(phi) + y*np.cos(phi))*np.sin(phi),
        0
    ])

d_ij_vec_num = d_ij_vec(x_val,y_val,phi_val)
d_ij_num = d_ij(x_val,y_val,phi_val)

print(d_ij_vec_num)
print(d_vec)

# %%
print(d_ij_num)
print(np.linalg.norm(d_vec))

# %%



# %%