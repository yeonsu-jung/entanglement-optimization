
# %%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import discrete_rod as drod


# Parameters
R = 1  # Radius of the sphere
n_turns = 5  # Number of turns on the sphere
n_points = 100  # Number of points on the curve

# Generate parameter values
theta = np.linspace(0, np.pi, n_points)  # Polar angle
phi = n_turns * np.linspace(0, 2 * np.pi, n_points)  # Azimuthal angle

t = theta
# Parametric equations for the curve
x = R * np.sin(theta) * np.cos(phi)
y = R * np.sin(theta) * np.sin(phi)
z = R * np.cos(theta)

# Plotting the curve
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z, label='Spiral on a sphere')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()

plt.show()
# %%
vertices = np.array([x,y,z]).T
tangents = np.diff(vertices,axis=0)
arc_length = np.cumsum(np.linalg.norm(tangents,axis=1))
tangents /= np.linalg.norm(tangents,axis=1)[:,None]

# initial material frame
m1 = np.cross(tangents[0],np.array([0.0,0.0,-1.0]))
m1 = m1 / np.linalg.norm(m1)
m2 = np.cross(tangents[0],m1)

# parallel transport
ne = tangents.shape[0]
d1 = np.zeros_like(tangents)
d2 = np.zeros_like(tangents)

d1[0] = m1
d2[0] = m2

for i in range(ne-1):
    a = d1[i]
    b = tangents[i]
    c = tangents[i+1]
    d = drod.parallel_transport(a,b,c)
    d1[i+1] = d
    d2[i+1] = np.cross(c,d)

t_prev = tangents[:-1,:]
t_curr = tangents[1:,:]

curvature_binormal = 2*np.cross(t_prev,t_curr)/(1+np.sum(t_prev*t_curr,axis=1))[:,None]

kappa_1 = np.sum(d1[:-1]*curvature_binormal,axis=1)
kappa_2 = np.sum(d2[:-1]*curvature_binormal,axis=1)
kappa = np.sqrt(kappa_1**2 + kappa_2**2)
np.max(kappa)
# %%
plt.close('all')
fig,ax=plt.subplots()
ax.plot(arc_length[:-1],kappa_1)
ax.plot(arc_length[:-1],kappa_2)
ax.plot(arc_length[:-1],kappa)
# %%
plt.close('all')
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
colors = plt.cm.viridis((kappa - kappa.min()) / (kappa.max() - kappa.min()))
for i in range(len(t) - 2):
    ax.plot(x[i:i+2], y[i:i+2], z[i:i+2],'o-',color=colors[i], linewidth=2)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()
# %%
arrow_length = 0.1
plt.close('all')
fig,ax=plt.subplots(subplot_kw={'projection':'3d'})
ax.plot(vertices[:,0],vertices[:,1],vertices[:,2])
ax.quiver(vertices[:-1,0],vertices[:-1,1],vertices[:-1,2],tangents[:,0],tangents[:,1],tangents[:,2],length=arrow_length,colors='b')
ax.quiver(vertices[:-1,0],vertices[:-1,1],vertices[:-1,2],d1[:,0],d1[:,1],d1[:,2],length=arrow_length,colors='r')
ax.quiver(vertices[:-1,0],vertices[:-1,1],vertices[:-1,2],d2[:,0],d2[:,1],d2[:,2],length=arrow_length,colors='g')
ax.axis('equal')

# %%
plt.close('all')
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create a colormap based on the parameter t
kappa = np.sqrt(kappa_1**2 + kappa_2**2)
# add an element to the end of the array to make the colors match the number of vertices
kappa = np.append(kappa, kappa[-1])
colors = plt.cm.viridis((kappa - kappa.min()) / (kappa.max() - kappa.min()))


for i in range(len(t) - 1):
    ax.plot(x[i:i+2], y[i:i+2], z[i:i+2], color=colors[i])

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.axis('equal')

plt.show()
# %%




# %%
initial_vertex = np.array([[0,0,0]])
initial_tangent = np.array([[0,1,0]])
# Example usage:
# Assume kappa_1, kappa_2, initial_vertex, initial_tangent, and arc_length are defined
scale_factor = 0
reconstructed_vertices = drod.reconstruct_vertices(kappa_1*scale_factor, kappa_2*scale_factor, initial_vertex, initial_tangent, arc_length)

import polyscope as ps
from visualizations import prep_for_polyscope

ps.init()
ps.set_ground_plane_mode("none")

nodes = reconstructed_vertices
edges = np.array([[i, i + 1] for i in range(len(nodes) - 1)])
min_z = np.min(nodes[:,2])
rod_diameter = 10/100

ps_curves = ps.register_curve_network("filaments",nodes,edges)
ps_curves.set_radius(rod_diameter/2,relative=False)
ps.set_up_dir("y_up")
# ps.show()
# %%
output_path = 'ball_spiral'
if not os.path.exists(output_path):
    os.makedirs(output_path)
    
i = 0
for scale_factor in np.linspace(0,1,100):
    reconstructed_vertices = drod.reconstruct_vertices(kappa_1*scale_factor, kappa_2*scale_factor, initial_vertex, initial_tangent, arc_length)
    ps_curves.update_node_positions(reconstructed_vertices)
    ps.screenshot(f'{output_path}/frame_{i:04d}.png',transparent_bg=False)
    i += 1
# %%