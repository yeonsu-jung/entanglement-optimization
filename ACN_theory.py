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

d_ij = 0.1
theta = np.pi/2*0.1
ai = 0.1
aj = 0.5

print(acn_from_area_formula(d_ij, theta, ai, aj))
print(acn_from_antiderivative(d_ij, theta, ai, aj))
print(acn_from_numerical_integration(d_ij, theta, ai, aj))
# %%
import numpy as np
import matplotlib.pyplot as plt

# Dictionary to define the number of points for each variable
variable_points = {
    'd_ij': 7,  # e.g., use 7 points for d_ij
    'theta': 6, # use 6 points for theta
    'ai': 8,    # use 8 points for ai
    'aj': 5     # use 5 points for aj
}

# Generate ranges dynamically
d_ij_vals = np.linspace(0.1, 1, variable_points['d_ij'])
theta_vals = np.linspace(0.01, np.pi/2, variable_points['theta'])
ai_vals = np.linspace(0.1, 0.9, variable_points['ai'])
aj_vals = np.linspace(0.1, 0.9, variable_points['aj'])

# Initialize arrays dynamically based on the sizes
acn1 = np.zeros((len(d_ij_vals), len(theta_vals), len(ai_vals), len(aj_vals)))
acn2 = np.zeros((len(d_ij_vals), len(theta_vals), len(ai_vals), len(aj_vals)))
acn3 = np.zeros((len(d_ij_vals), len(theta_vals), len(ai_vals), len(aj_vals)))

# Iterate over the ranges and compute ACN values for each method
for i, d_ij in enumerate(d_ij_vals):
    for j, theta in enumerate(theta_vals):
        for k, ai in enumerate(ai_vals):
            for l, aj in enumerate(aj_vals):
                acn1[i, j, k, l] = acn_from_area_formula(d_ij, theta, ai, aj)
                acn2[i, j, k, l] = acn_from_antiderivative(d_ij, theta, ai, aj)
                acn3[i, j, k, l] = acn_from_numerical_integration(d_ij, theta, ai, aj)
# %%
markers = ['o', 's', '^']
# Function to create a single plot with overlapping curves for comparison
def plot_acn_comparison(variable_vals, variable_label, fixed_vals, fixed_indices):
    plt.figure(figsize=(2.5, 2))
    
    # Plot for each ACN method on the same plot
    k = 0
    offset = 0
    for acn, label in zip(
        [acn1, acn2, acn3],
        ['Area formula', 'Analytic integration', 'Numerical integration']  # Labels for each method
    ):
        # Take the mean over the fixed indices to compare across the given variable
        acn_mean = np.mean(acn, axis=fixed_indices)*(1 + offset*k)  # Add a small offset for better visualization
        print(acn_mean)
        plt.plot(variable_vals, acn_mean, label=label, marker=markers[k])
        k += 1
    
    # Labels and title
    plt.xlabel(variable_label)
    plt.ylabel('ACN')
    plt.title(f'ACN Comparison vs {variable_label}')
    
    # Show legend for each method
    # plt.legend()
    
    # Display the plot
    plt.tight_layout()
    plt.show()

# Example plotting comparisons
# For 'd_ij', we fix theta, ai, and aj (axes 1, 2, 3)
plot_acn_comparison(d_ij_vals, 'd_ij', (theta_vals, ai_vals, aj_vals), (1, 2, 3))

# For 'theta', we fix d_ij, ai, and aj (axes 0, 2, 3)
plot_acn_comparison(theta_vals, 'theta', (d_ij_vals, ai_vals, aj_vals), (0, 2, 3))

# For 'ai', we fix d_ij, theta, and aj (axes 0, 1, 3)
plot_acn_comparison(ai_vals, 'ai', (d_ij_vals, theta_vals, aj_vals), (0, 1, 3))

# For 'aj', we fix d_ij, theta, and ai (axes 0, 1, 2)
plot_acn_comparison(aj_vals, 'aj', (d_ij_vals, theta_vals, ai_vals), (0, 1, 2))