# %%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generate_random_unit_vector():
    """
    Generates a random unit vector in 3D space.

    Returns:
    np.ndarray: A 3D unit vector.
    """
    random_vector = np.random.randn(3)
    return random_vector / np.linalg.norm(random_vector)

def generate_random_initial_point(N, segment_length):
    """
    Generates a random initial point within a box with side length N*segment_length centered at the origin.

    Parameters:
    N (int): Number of segments.
    segment_length (float): Length of each segment.

    Returns:
    np.ndarray: A 3D point.
    """
    half_length = (N * segment_length) / 2
    return np.random.uniform(-half_length, half_length, size=3)

def generate_wlc_3d(N, Lp, segment_length=1.0):
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
    positions[0] = generate_random_initial_point(N, segment_length)

    for i in range(1, N):
        positions[i] = positions[i-1] + tangents[i] * segment_length

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

# Parameters
N = 1000  # Number of segments
Lp = 5.0  # Persistence length
segment_length = 1.0  # Length of each segment

# Generate the WLC chain in 3D
wlc_3d = generate_wlc_3d(N, Lp, segment_length)

# smooth out with Gaussian kernel
from scipy.ndimage import gaussian_filter1d
sigma = 10
wlc_3d_smoothed = gaussian_filter1d(wlc_3d, sigma=sigma, axis=0)
# %%
# Plotting the 3D curve
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot(wlc_3d_smoothed[:, 0], wlc_3d_smoothed[:, 1], wlc_3d_smoothed[:, 2], lw=1)
ax.set_title('Worm-Like Chain in 3D')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()
