import numpy as np
from scipy.optimize import minimize

# Function to compute the cosine of the angle between two vectors
def cosine_angle(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# Function to compute the minimum angle (in radians) between any pair of lines
def min_angle(lines):
    n = len(lines)
    min_theta = np.pi
    for i in range(n):
        for j in range(i + 1, n):
            cos_theta = cosine_angle(lines[i], lines[j])
            theta = np.arccos(np.clip(cos_theta, -1, 1))  # Ensure cos_theta stays within [-1, 1]
            min_theta = min(min_theta, theta)
    return min_theta

# Objective function to minimize the negative of the minimum angle (we want to maximize the angle)
def objective(flat_lines):
    n = len(flat_lines) // 3
    lines = flat_lines.reshape((n, 3))
    return -min_angle(lines)

# Constraint to ensure all vectors are unit vectors
def unit_constraint(flat_lines):
    n = len(flat_lines) // 3
    lines = flat_lines.reshape((n, 3))
    return np.array([np.linalg.norm(line) - 1 for line in lines])

# Initial guess: random unit vectors
def random_unit_vector():
    vec = np.random.randn(3)
    return vec / np.linalg.norm(vec)

# Number of lines (you can change this to test with more lines)
num_lines = 10
initial_lines = np.array([random_unit_vector() for _ in range(num_lines)]).flatten()

# Set up constraints for the optimization
constraints = [{'type': 'eq', 'fun': unit_constraint}]

# Perform the optimization
result = minimize(objective, initial_lines, constraints=constraints, method='SLSQP')

# Extract optimized lines
optimized_lines = result.x.reshape((num_lines, 3))

# Print the optimized lines and the maximum minimum angle
print("Optimized Lines (unit vectors):")
print(optimized_lines)
print(f"Maximum minimum angle: {np.degrees(min_angle(optimized_lines)):.2f} degrees")
