import numpy as np

# Define the gradient of the objective function f(x)
def gradient_f(x):
    # Example: Gradient of f(x) = (x - 3)^2
    return 2 * (x - 3)

# Define the projection function using (I - nn^T) v
def projection_g(v, n):
    # n must be a unit vector (n.T @ n = 1)
    n = n / np.linalg.norm(n)  # Ensure n is normalized
    projection_matrix = np.eye(len(n)) - np.outer(n, n)
    return projection_matrix @ v

# Projected Gradient Descent function with the (I - nn^T) projection
def projected_gradient_descent(grad_f, proj_g, initial_x, n, learning_rate=0.1, max_iter=100, tol=1e-6):
    x = initial_x
    for i in range(max_iter):
        # Compute the gradient
        grad = grad_f(x)
        
        # Gradient descent step
        x_new = x - learning_rate * grad
        
        # Projection step: project x_new using (I - nn^T)
        x_new = proj_g(x_new, n)
        
        # Check for convergence
        if np.linalg.norm(x_new - x) < tol:
            print(f"Converged after {i} iterations")
            break
        
        x = x_new

    return x

# Example usage
initial_x = np.array([2.0, 2.0])
n = np.array([1.0, 0.0])  # Example unit vector along the x-axis
solution = projected_gradient_descent(gradient_f, projection_g, initial_x, n)
print(f"Solution: {solution}")
