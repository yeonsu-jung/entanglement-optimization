
# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle

# Simulation parameters
N = 100                   # Number of balls
L = 10.0                 # Size of the box
r = 0.4                  # Radius of each ball
T = 500                  # Total frames
dt = 0.05                # Time step

# Initialize positions (non-overlapping)
positions = []
while len(positions) < N:
    candidate = np.random.uniform(r, L - r, size=2)
    if all(np.linalg.norm(candidate - p) > 2 * r for p in positions):
        positions.append(candidate)
positions = np.array(positions)

# Initialize random velocities
velocities = np.random.uniform(-2, 2, size=(N, 2))

# Set up figure
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-9*L, 10*L)
ax.set_ylim(-9*L, 10*L)
ax.set_aspect('equal')
ax.axis('off')
circles = [Circle((0, 0), radius=r, color='blue') for _ in range(N)]
for c in circles:
    ax.add_patch(c)

def handle_collisions(positions, velocities, restitution=0.8):  # e < 1
    for i in range(N):
        for j in range(i + 1, N):
            delta = positions[i] - positions[j]
            dist = np.linalg.norm(delta)
            if dist < 2 * r:
                n = delta / dist
                v_rel = np.dot(velocities[i] - velocities[j], n)
                if v_rel < 0:
                    impulse = (1 + restitution) * v_rel / 2
                    velocities[i] -= impulse * n
                    velocities[j] += impulse * n

def update(frame):
    global positions, velocities
    positions += velocities * dt
    handle_collisions(positions, velocities, restitution=0.)
    for i, c in enumerate(circles):
        c.center = positions[i]
    return circles

ani = FuncAnimation(fig, update, frames=T, interval=30, blit=True)

# Save or show
ani.save("bouncing_balls_2d.mp4", fps=30, dpi=150)
print("Saved animation as bouncing_balls_2d.mp4")
plt.show()
