# %%
import numpy as np
from scipy.optimize import linprog

# Define parameters
m1, m2 = 1.0, 1.0  # Masses of the two bodies
g = 9.81  # Gravity
timestep = 0.01  # Time step
v1, v2 = np.array([0.0, 0.0]), np.array([0.0, 0.0])  # Initial velocities of the two bodies
p1, p2 = np.array([0.0, 1.0]), np.array([0.0, 0.0])  # Initial positions
r1, r2 = 0.5, 0.5  # Radii of the two bodies (assuming circular objects)

# LCP solver function using linprog from scipy
def solve_lcp(M, q):
    n = len(q)
    P = np.zeros((n, n))
    A_eq = np.block([[M, np.eye(n)]])
    b_eq = np.zeros(n)
    c = np.hstack([np.zeros(n), np.ones(n)])  # Linear program objective
    bounds = [(0, None) for _ in range(n)] + [(None, None) for _ in range(n)]  # Constraints for the LCP

    # Solve linear program
    result = linprog(c, A_eq=A_eq, b_eq=q, bounds=bounds, method='highs')

    if result.success:
        return result.x[:n]
    else:
        raise ValueError("LCP solver failed to converge")

# Simulation loop
for step in range(100):
    # Relative position and distance between the two bodies
    delta_p = p2 - p1
    distance = np.linalg.norm(delta_p)
    penetration = r1 + r2 - distance

    if penetration > 0:
        # Compute contact normal and relative velocity
        normal = delta_p / distance
        rel_velocity = v2 - v1

        # Build the LCP matrices
        # Normal direction constraint: impulse should push bodies apart
        M = np.array([[1/m1 + 1/m2]])  # LCP mass matrix (1D in normal direction)
        q = np.array([-np.dot(rel_velocity, normal) - penetration / timestep])

        # Solve LCP for normal impulse
        impulse_normal = solve_lcp(M, q)

        # Apply impulses to update velocities
        v1 -= impulse_normal * normal / m1
        v2 += impulse_normal * normal / m2

    # Apply gravity to update velocities
    v1 += np.array([0, -g * timestep])
    v2 += np.array([0, -g * timestep])

    # Update positions
    p1 += v1 * timestep
    p2 += v2 * timestep

    print(f"Step {step}, p1: {p1}, p2: {p2}")

print("Simulation complete!")

# %%
