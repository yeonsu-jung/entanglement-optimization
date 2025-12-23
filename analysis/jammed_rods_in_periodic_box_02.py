# %%
from pathlib import Path

output_dir = Path(Path(__file__).stem)
output_dir.mkdir(parents=True, exist_ok=True)

# %%
pth = '/Users/yeonsu/GitHub/rod-dynamics-3d/initial-configs/rods932.csv'

# read csv
import numpy as np

# get first rows with # as header
metadata = []
with open(pth, 'r') as f:
    for line in f:
        if line.startswith('#'):
            metadata.append(line.strip())
        else:
            break

# get
# rod_length
# rod_diameter
# box_size
# placed

rod_length = float(metadata[0].split('=')[1])
rod_diameter = float(metadata[1].split('=')[1])
box_size = float(metadata[4].split('=')[1])
placed = int(metadata[7].split('=')[1])
    
# %%
# skip the rows starting with # and x0
rods_in_shape = np.loadtxt(pth, delimiter=',', skiprows=10)
rods_in_shape = rods_in_shape.reshape(932,-1,3)

# %%
import sys

sys.path.append('/Users/yeonsu/GitHub/entanglement-optimization-combined/entanglement-optimization/core')

# %%


# %%

from visualizations import prep_for_polyscope

import polyscope as ps
ps.init()
ps.set_up_dir("z_up")
nodes0, edges0, _ = prep_for_polyscope(rods_in_shape,932)
ps_rods = ps.register_curve_network("rods", nodes0, edges0)
ps.show()

# %% Nematic order and director distribution on S2
import jax.numpy as jnp
from core.transforms import x_to_q
from core.potentials import compute_nematic_order, directors_from_q, create_pairs, all_pairwise_angles
from matplotlib import pyplot as plt
from matplotlib import cm

# Convert endpoints to (N,6) then to (N,5) q = (cx,cy,cz, theta, phi)
assert rods_in_shape.shape[1] >= 2, "Expected at least two vertices per rod"
x_endpoints = rods_in_shape[:, :2, :].reshape(rods_in_shape.shape[0], 6)
q = x_to_q(jnp.array(x_endpoints))

# Compute nematic tensor and order
Q_avg, Q, S, evals, evecs = compute_nematic_order(q)
print("N rods:", x_endpoints.shape[0])
print("Nematic order S:", float(S))
print("Eigenvalues:", list(map(float, evals)))
print("Principal director:", list(map(float, np.array(evecs)[:, -1])))

# Pairwise minimal angles (optional diagnostic)
q_pairs = create_pairs(q)
angles = all_pairwise_angles(q_pairs)
print("Mean minimal pairwise angle (deg):", float(jnp.mean(angles) * 180/np.pi))

# Build hemisphere distribution (fold u and -u)
u = np.array(directors_from_q(q))
u_fold = np.where(u[:,2:3] < 0, -u, u)
xh, yh, zh = u_fold[:,0], u_fold[:,1], u_fold[:,2]
hxy = np.hypot(xh, yh)
theta = np.arctan2(hxy, zh)              # [0, π/2]
phi = (np.arctan2(yh, xh) + 2*np.pi) % (2*np.pi)

Btheta, Bphi = 36, 72
H, theta_edges, phi_edges = np.histogram2d(theta, phi,
                                           bins=[Btheta, Bphi],
                                           range=[[0, np.pi/2], [0, 2*np.pi]],
                                           density=False)
H = H.astype(float); H /= H.sum() + 1e-12

# Plot dots on the unit hemisphere and export PNG/SVG (axis='equal')
fig = plt.figure(figsize=(6,5))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xh, yh, zh, s=8, c='k', alpha=0.8)
ax.set_box_aspect([1,1,1])  # axis='equal' for 3D
ax.set_title("Directors on S2 (hemisphere), points")
ax.set_axis_off()
plt.tight_layout()
fig.savefig(f"{output_dir}/directors_s2_points.png", dpi=200)
fig.savefig(f"{output_dir}/directors_s2_points.svg")
plt.close(fig)

print("Saved nematic outputs and S2 scatter: directors_s2_points.png, directors_s2_points.svg")


# %%
