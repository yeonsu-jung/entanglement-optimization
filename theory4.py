from sympy import symbols, sqrt, sin, cos, Matrix

# Define variables for 10 rods
rod_vars = [symbols(f'x{i} y{i} z{i} theta{i} phi{i}') for i in range(10)]
flat_vars = [v for group in rod_vars for v in group]  # Flattened 50-variable vector

# Define one pairwise function
def d_ij(v1, v2):
    x1, y1, z1, theta1, phi1 = v1
    x2, y2, z2, theta2, phi2 = v2
    return (
        sqrt(
            sin(phi1 - phi2)**2 * sin(theta1)**2 * sin(theta2)**2 +
            (cos(phi1)*cos(theta2)*sin(theta1) - cos(phi2)*cos(theta1)*sin(theta2))**2 +
            (cos(theta2)*sin(phi1)*sin(theta1) - cos(theta1)*sin(phi2)*sin(theta2))**2
        )
        /
        (
            -((z1 - z2)*sin(phi1 - phi2)*sin(theta1)*sin(theta2)) +
            (y1 - y2)*(-cos(phi1)*cos(theta2)*sin(theta1) + cos(phi2)*cos(theta1)*sin(theta2)) +
            (x1 - x2)*(cos(theta2)*sin(phi1)*sin(theta1) - cos(theta1)*sin(phi2)*sin(theta2))
        )
    )

# Build vector of all 45 distances
d_list = []
pairs = []
for i in range(10):
    for j in range(i+1, 10):
        d_list.append(d_ij(rod_vars[i], rod_vars[j]))
        pairs.append((i, j))

# Stack into vector function
d_vector = Matrix(d_list)

# Compute full Jacobian (45x50)
J = d_vector.jacobian(flat_vars)

print(J)
