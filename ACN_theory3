# %%
import numpy as np
from potentials import compute_linking_number
import numpy as np
import scipy.integrate as integrate
from matplotlib import pyplot as plt

# From area formula

def acn_from_area_formula(d_ij, theta, ai, aj):

    l = 1
    x1,y1,z1 = -l/2+(ai-0.5),0,0
    phi_i,theta_i = np.pi/2,0
    phi_j,theta_j = np.pi/2,theta

    u_j = np.array([np.sin(phi_j)*np.cos(theta_j), np.sin(phi_j)*np.sin(theta_j), np.cos(phi_j)])

    z2 = d_ij
    x2 = -l/2*u_j[0]+(aj-0.5)*u_j[0]
    y2 = -l/2*u_j[1]+(aj-0.5)*u_j[1]

    lk = compute_linking_number(x1, y1, z1, phi_i, theta_i, x2, y2, z2, phi_j, theta_j, l)
    return np.abs(lk)

# compute_linking_number(x_i, y_i, z_i, phi_i, theta_i, x_j, y_j, z_j, phi_j, theta_j, l):
# from antiderivative

def acn_from_antiderivative(d_ij, theta, ai, aj):


    # This doesn't work - why?
    # def func(x, y):
    #     numerator = (x * y + d_ij**2 * np.cos(theta))
    #     denominator = (d_ij * np.sqrt(x**2 + y**2 - 2 * x * y * np.cos(theta) + d_ij**2 * (np.sin(theta))**2))

    #     return -np.arctan(numerator / denominator) / (4 * np.pi)

    def func(x, y):
        numerator = np.sin(theta) * (x * y + d_ij**2 * np.cos(theta) / np.sin(theta)**2)
        denominator = np.sqrt(d_ij**2 + x**2 + y**2 - 2 * np.cos(theta) * x * y)*d_ij
        
        return -np.arctan(numerator / denominator) / (4 * np.pi)
    
    return np.abs(func(1-ai,1-aj) - func(-ai,1-aj) - func(1-ai,-aj) + func(-ai,-aj))

# %% from numeical integration 
def acn_from_numerical_integration(d_ij, theta, ai, aj):

    # Define the function foo(x, y)
    def foo(x, y, d_ij, theta):
        # return d_ij * np.sin(theta)**2 / (x**2 + y**2 - 2*x*y*np.cos(theta) + d_ij**2 * np.sin(theta)**2)**(3/2)
        return d_ij*np.sin(theta)/(d_ij**2 + x**2 + y**2 - 2*x*y*np.cos(theta))**(3/2)/4/np.pi

    # Define limits of integration
    x_lower = -ai  # example lower bound for x
    x_upper = 1-ai  # example upper bound for x
    y_lower = -aj  # example lower bound for y
    y_upper = 1-aj  # example upper bound for y

    # Perform double integration
    result, error = integrate.dblquad(foo, y_lower, y_upper, lambda x: x_lower, lambda x: x_upper, args=(d_ij, theta))
    return result
# %%
d_ij = 0.1
theta = np.pi/2
ai = 0.5
aj = 0.5

print(acn_from_area_formula(d_ij, theta, ai, aj))
print(acn_from_antiderivative(d_ij, theta, ai, aj))
print(acn_from_numerical_integration(d_ij, theta, ai, aj))
# %%
ai_list = np.linspace(-3,4.,100)

import time
start = time.time()
for d_ij in [0.001, 0.01, 0.1, 0.2, 0.3, 0.4]:
    acn_results = []
    for ai in ai_list:
        # acn_results.append(acn_from_area_formula(d_ij, theta, ai, aj)) # mid
        acn_results.append(acn_from_antiderivative(d_ij, theta, ai, aj)) # fastest
        # acn_results.append(acn_from_numerical_integration(d_ij, theta, ai, aj)) # slowest
    plt.plot(ai_list, acn_results,'-')

print("Time taken:", time.time()-start)

# %%



# %%
