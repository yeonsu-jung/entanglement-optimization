# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

# Parameters
N = 21                  # Number of balls (odd for symmetry)
v0 = 1.0                # Initial velocity
T = 600                 # Total number of frames
d = 1.0                 # Initial spacing
r = 0.3                 # Radius of balls (for drawing)

# Ball states
positions = np.arange(N) * d
velocities = np.zeros(N)
velocities[N // 2] = v0

# For animation
positions_history = []

# Naive time-stepping simulation (small dt)
for t in range(T):
    # Store positions
    positions_history.append(positions.copy())

    # Update positions
    positions += velocities

    # Handle collisions (perfectly elastic)
    for i in range(N - 1):
        if positions[i + 1] - positions[i] < 2 * r:
            velocities[i], velocities[i + 1] = velocities[i + 1], velocities[i]
            # Separate overlapping balls
            overlap = 2 * r - (positions[i + 1] - positions[i])
            positions[i] -= overlap / 2
            positions[i + 1] += overlap / 2

positions_history = np.array(positions_history)

# %%
positions_history.shape

# %%
k = 0
for position in positions_history[:30]:
    # print(position)
    y = np.zeros_like(position)
    plt.plot(position, y,'o')

    plt.xlim(0, 2*N * d)
    plt.savefig(f'chain_collisions/frame_{k:04d}.png'.format(0))
    plt.close('all')
    k += 1
    