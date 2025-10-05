# %%
# %matplotlib qt
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter1d

def generate_random_unit_vector():
    """
    Generates a random unit vector in 3D space.

    Returns:
    np.ndarray: A 3D unit vector.
    """
    random_vector = np.random.randn(3)
    return random_vector / np.linalg.norm(random_vector)

def generate_random_point_in_unit_sphere():
    """
    Generates a random point within the unit sphere.

    Returns:
    np.ndarray: A 3D point.
    """
    while True:
        point = np.random.uniform(-1, 1, 3)
        if np.linalg.norm(point) <= 1:
            return point

def reflect_point_in_sphere(point):
    """
    Reflects a point that lies outside the unit sphere back into the sphere.

    Parameters:
    point (np.ndarray): A 3D point.

    Returns:
    np.ndarray: A 3D point reflected back inside the unit sphere.
    """
    norm = np.linalg.norm(point)
    if norm > 1:
        point = point / norm  # Normalize and bring it to the surface of the sphere
    return point

def generate_wlc_3d(N, Lp, segment_length=0.01):
    """
    Generates a 3D worm-like chain with N segments and persistence length Lp.

    Parameters:
    N (int): Number of segments.
    Lp (float): Persistence length.
    segment_length (float): Length of each segment.

    Returns:
    np.ndarray: Array of shape (N, 3) representing the coordinates of the chain.
    """
    # Initialize the first tangent vector to a random direction
    tangents = np.zeros((N, 3))
    tangents[0] = generate_random_unit_vector()

    # Generate correlated random orientations
    for i in range(1, N):
        angle = np.random.normal(scale=np.sqrt(segment_length / Lp))
        axis = np.random.randn(3)
        axis /= np.linalg.norm(axis)
        rotation_matrix = _rotation_matrix(axis, angle)
        tangents[i] = np.dot(rotation_matrix, tangents[i-1])

    # Integrate tangent vectors to get positions
    positions = np.zeros((N, 3))
    positions[0] = generate_random_point_in_unit_sphere()

    for i in range(1, N):
        next_position = positions[i-1] + tangents[i] * segment_length
        positions[i] = reflect_point_in_sphere(next_position)

    return positions

def _rotation_matrix(axis, angle):
    """
    Generates a rotation matrix for rotating around a given axis by a certain angle.

    Parameters:
    axis (np.ndarray): The axis to rotate around.
    angle (float): The angle to rotate by.

    Returns:
    np.ndarray: The rotation matrix.
    """
    axis = axis / np.linalg.norm(axis)
    a = np.cos(angle / 2)
    b, c, d = -axis * np.sin(angle / 2)
    return np.array([[a*a + b*b - c*c - d*d, 2*(b*c - a*d), 2*(b*d + a*c)],
                     [2*(b*c + a*d), a*a + c*c - b*b - d*d, 2*(c*d - a*b)],
                     [2*(b*d - a*c), 2*(c*d + a*b), a*a + d*d - b*b - c*c]])

if __name__ == '__main__':    
    # Parameters
    M = 1  # Number of chains
    N = 100  # Number of segments per chain
    Lp = 1.  # Persistence length
    segment_length = 1  # Length of each segment

    # Plotting the 3D curves
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('Worm-Like Chains in 3D within Unit Sphere')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Generate and plot M WLC chains in 3D
    for _ in range(M):
        wlc_3d = generate_wlc_3d(N, Lp, segment_length)
        # Smooth out with Gaussian kernel
        sigma = 10
        wlc_3d_smoothed = gaussian_filter1d(wlc_3d, sigma=sigma, axis=0)
        ax.plot(wlc_3d_smoothed[:, 0], wlc_3d_smoothed[:, 1], wlc_3d_smoothed[:, 2], lw=1)

    # Plot the unit sphere for reference
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color='r', alpha=0.1)
    ax.axis('equal')
    plt.show()
# %%