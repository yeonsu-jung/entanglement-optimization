# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
from scipy.optimize import minimize

# Function to compute Bezier curve points
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
from scipy.optimize import minimize

# Function to compute Bezier curve points
def bezier_curve(control_points, n_points=1000):
    n = len(control_points) - 1
    t = np.linspace(0, 1, n_points)
    curve = np.zeros((n_points, 3))
    for i in range(n + 1):
        binomial_coeff = comb(n, i)
        curve += np.outer(binomial_coeff * (t ** i) * ((1 - t) ** (n - i)), control_points[i])
    return curve

# Function to compute the first derivative of the Bezier curve
def bezier_curve_first_derivative(control_points, n_points=1000):
    n = len(control_points) - 1
    t = np.linspace(0, 1, n_points)
    first_derivative = np.zeros((n_points, 3))
    for i in range(n):
        binomial_coeff = comb(n-1, i)
        first_derivative += np.outer(binomial_coeff * (t ** i) * ((1 - t) ** (n-1-i)) * n, (control_points[i+1] - control_points[i]))
    return first_derivative

# Function to compute the second derivative of the Bezier curve
def bezier_curve_second_derivative(control_points, n_points=1000):
    n = len(control_points) - 1
    t = np.linspace(0, 1, n_points)
    second_derivative = np.zeros((n_points, 3))
    for i in range(n-1):
        binomial_coeff = comb(n-2, i)
        second_derivative += np.outer(binomial_coeff * (t ** i) * ((1 - t) ** (n-2-i)) * n * (n-1), (control_points[i+2] - 2*control_points[i+1] + control_points[i]))
    return second_derivative

# Function to compute the scalar curvature of the Bezier curve
def scalar_curvature(control_points, n_points=1000):
    first_derivative = bezier_curve_first_derivative(control_points, n_points)
    second_derivative = bezier_curve_second_derivative(control_points, n_points)

    curvature = np.zeros(n_points)
    for i in range(n_points):
        # Cross product of first and second derivatives
        cross_product = np.cross(first_derivative[i], second_derivative[i])
        curvature[i] = np.linalg.norm(cross_product) / (np.linalg.norm(first_derivative[i])**3)
    
    return curvature

# Function to get initial guess for control points
def initial_control_points(points1, points2, n_control_points):
    total_points = np.vstack((points1, points2))
    indices = np.linspace(0, len(total_points) - 1, n_control_points).astype(int)
    return total_points[indices]

# Error function to minimize, including curvature constraint
def error_function(variable_control_points_flat, points1, points2, n_control_points, max_curvature):
    control_points = variable_control_points_flat.reshape((n_control_points, 3))
    bezier_points = bezier_curve(control_points, len(points1) + len(points2))
    bezier_derivative = bezier_curve_first_derivative(control_points, len(points1) + len(points2))

    combined_points = np.vstack((points1, points2))
    
    # Compute the sum of squared distances between the original points and the Bezier points
    error = np.sum(np.linalg.norm(combined_points - bezier_points, axis=1)**2)
    
    # Compute the curvature penalty
    curvature = scalar_curvature(control_points, len(points1) + len(points2))
    curvature_penalty = np.sum(np.maximum(0, curvature - max_curvature)**2)

    return error + curvature_penalty

# Function to optimize Bezier curve with junction point
def optimize_bezier_curve_with_junction(points1, points2, n_control_points, max_curvature):
    initial_guess = initial_control_points(points1, points2, n_control_points)
    variable_control_points_flat = initial_guess.flatten()

    # Perform optimization
    result = minimize(error_function, variable_control_points_flat, args=(points1, points2, n_control_points, max_curvature), method='L-BFGS-B')
    optimized_control_points = result.x.reshape((n_control_points, 3))

    # Generate the optimized Bezier curve and its derivative
    smooth_points = bezier_curve(optimized_control_points, len(points1) + len(points2))
    tangent_vectors = bezier_curve_first_derivative(optimized_control_points, len(points1) + len(points2))
    curvature = scalar_curvature(optimized_control_points, len(points1) + len(points2))
    
    return optimized_control_points, smooth_points, tangent_vectors, curvature



def main():        
        
    # Example Nx3 numpy arrays for two curves (use your own data here)
    N = 100  # Example number of points for the first curve
    M = 100  # Example number of points for the second curve
    t1 = np.linspace(0, 1, N)
    t2 = np.linspace(1, 2, M)
    x1 = np.sin(2 * np.pi * t1)  # Example x-coordinates for the first curve
    y1 = np.cos(2 * np.pi * t1)  # Example y-coordinates for the first curve
    z1 = t1                      # Example z-coordinates for the first curve
    x2 = np.sin(2 * np.pi * t2)  # Example x-coordinates for the second curve
    y2 = np.cos(2 * np.pi * t2)  # Example y-coordinates for the second curve
    z2 = t2                      # Example z-coordinates for the second curve

    points1 = np.vstack((x1, y1, z1)).T
    points2 = np.vstack((x2, y2, z2)).T

    # Set the number of control points and maximum allowed curvature
    n_control_points = 7
    max_curvature = 0.1

    # Optimize the Bezier curve
    optimized_control_points, smooth_points, tangent_vectors, curvature = optimize_bezier_curve_with_junction(points1, points2, n_control_points, max_curvature)

    # Plot the original points, optimized Bezier curve, and tangent vectors
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(points1[:, 0], points1[:, 1], points1[:, 2], 'r.', label='Original Curve 1')
    ax.plot(points2[:, 0], points2[:, 1], points2[:, 2], 'g.', label='Original Curve 2')
    ax.plot(smooth_points[:, 0], smooth_points[:, 1], smooth_points[:, 2], 'b-', label='Optimized Bezier Curve')
    ax.scatter(optimized_control_points[:, 0], optimized_control_points[:, 1], optimized_control_points[:, 2], color='y', label='Optimized Control Points')

    # Plot tangent vectors at the ends
    tangent_scale = 0.1
    start_tangent = smooth_points[0] + tangent_vectors[0] * tangent_scale
    end_tangent = smooth_points[-1] + tangent_vectors[-1] * tangent_scale

    ax.quiver(smooth_points[0, 0], smooth_points[0, 1], smooth_points[0, 2],
            tangent_vectors[0, 0], tangent_vectors[0, 1], tangent_vectors[0, 2],
            color='m', length=tangent_scale, normalize=True, label='Start Tangent')
    ax.quiver(smooth_points[-1, 0], smooth_points[-1, 1], smooth_points[-1, 2],
            tangent_vectors[-1, 0], tangent_vectors[-1, 1], tangent_vectors[-1, 2],
            color='c', length=tangent_scale, normalize=True, label='End Tangent')

    # Color the curve by curvature
    norm_curvature = (curvature - np.min(curvature)) / (np.max(curvature) - np.min(curvature))  # Normalize curvature
    colors = plt.cm.viridis(norm_curvature)  # Map to colors

    for i in range(len(smooth_points)-1):
        ax.plot(smooth_points[i:i+2, 0], smooth_points[i:i+2, 1], smooth_points[i:i+2, 2], color=colors[i])

    ax.legend()
    plt.show()

if __name__ == '__main__':
    main()
