# %%
import numpy as np
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

theta2_val = np.pi/2
phi2_val = np.pi/2

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
def spherical_to_cartesian(theta, phi):
    return np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta)
    ])

u2 = spherical_to_cartesian(theta2_val, phi2_val)

x2_val += 10000*u2[0]
y2_val += 10000*u2[1]
z2_val += 10000*u2[2]

d = d_12_func(theta1_val, phi1_val, phi2_val, theta2_val, x1_val, y1_val, z1_val, x2_val, y2_val, z2_val)

print(d)
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
