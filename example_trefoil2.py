# %%
%matplotlib qt
import numpy as np
import matplotlib.pyplot as plt
import discrete_rod as drod

# %%
def trefoil_knot(t):
    x = np.sin(t) + 2 * np.sin(2 * t)
    y = np.cos(t) - 2 * np.cos(2 * t)
    z = np.sin(3 * t)
    return x, y, z

num_vertices = 120
t_power = 1

t = np.linspace(0, (2 * np.pi)**t_power, num_vertices)
t = t**(1/t_power)

# plt.close('all')
# plt.plot(t)

# %%
# TODO: deal with tight ends
x, y, z = trefoil_knot(t)

plt.close('all')
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
colors = plt.cm.viridis((t - t.min()) / (t.max() - t.min()))
for i in range(len(t) - 1):
    ax.plot(x[i:i+2], y[i:i+2], z[i:i+2], color=colors[i], linewidth=2)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

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
initial_vertex = np.array([[0,0,0]])
initial_tangent = np.array([[0,1,0]])
# Example usage:
# Assume kappa_1, kappa_2, initial_vertex, initial_tangent, and arc_length are defined
reconstructed_vertices = drod.reconstruct_vertices(kappa_1, kappa_2, initial_vertex, initial_tangent, arc_length)






# %%
plt.close('all')
fig = plt.figure()
ax=fig.add_subplot(111,projection='3d')
ax.plot(reconstructed_vertices[:,0],reconstructed_vertices[:,1],reconstructed_vertices[:,2])
# %%




# %%
np.savetxt(f'vertices_trefoil.txt', vertices, delimiter=',')
# %%
arrow_length = 0.5
plt.close('all')
fig,ax=plt.subplots(subplot_kw={'projection':'3d'})
ax.plot(vertices[:,0],vertices[:,1],vertices[:,2])
ax.quiver(vertices[:-1,0],vertices[:-1,1],vertices[:-1,2],tangents[:,0],tangents[:,1],tangents[:,2],length=arrow_length,colors='b')
ax.quiver(vertices[:-1,0],vertices[:-1,1],vertices[:-1,2],d1[:,0],d1[:,1],d1[:,2],length=arrow_length,colors='r')
ax.quiver(vertices[:-1,0],vertices[:-1,1],vertices[:-1,2],d2[:,0],d2[:,1],d2[:,2],length=arrow_length,colors='g')
ax.axis('equal')
plt.savefig(f'material_frame_trefoil.png')
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
kappa_both = np.array([kappa_1,kappa_2]).T
np.savetxt(f'curvature_binormal_projected_trefoil.txt', kappa_both, delimiter=',')
# %%
plt.close('all')
plt.plot(arc_length[:-1],kappa_1,'o-')
plt.plot(arc_length[:-1],kappa_2,'o-')
plt.xlabel('Arc length (a.u.)')
plt.ylabel('Curvature 1 (a.u.)')
# %%
# magnitude / angle representation
# zero angle was arbitrarily chosen; you can add any number to it
# here I added 90 deg, so that the angle starts from zero

curvature_angle = np.arctan2(kappa_2,kappa_1)*180/np.pi + 90 # you can add any number here.
curvature_magnitude = np.linalg.norm(kappa_both,axis=1)

np.savetxt(f'curvature_angle_trefoil.txt', curvature_angle, delimiter=',')
np.savetxt(f'curvature_magnitude_trefoil.txt', curvature_magnitude, delimiter=',')

# %% plot curvature angle
plt.close('all')

plt.plot(arc_length[:-1],curvature_angle,'o-')
plt.xlabel('Arc length (a.u.)')
plt.ylabel('Curvature angle (deg)')
plt.savefig(f'curvature_angle_trefoil.png')
# %% plot curvature magnitude
plt.close('all')

plt.plot(arc_length[1:],curvature_magnitude,'o-')
plt.xlabel('Arc length (a.u.)')
plt.ylabel('Curvature magnitude (deg)')
plt.savefig(f'curvature_magnitude_trefoil.png')


# %%

edge_lengths = np.linalg.norm(np.diff(vertices,axis=0),axis=1)
# make a straight line with vertices with the above edge lengths
a_straight_line = np.zeros((vertices.shape[0],3))
a_straight_line[0] = vertices[0]
for i in range(1,len(a_straight_line)):
    a_straight_line[i] = a_straight_line[i-1] + edge_lengths[i-1] * np.array([0,0,-1])

plt.close('all')
fig,ax=plt.subplots(subplot_kw={'projection':'3d'})
ax.plot(a_straight_line[:,0],a_straight_line[:,1],a_straight_line[:,2],'o-')

a_straight_line *= 0.3/np.linalg.norm(a_straight_line[-1] - a_straight_line[0])
# %%
np.linalg.norm(a_straight_line[-1] - a_straight_line[0])
# %%
a_straight_line.shape
# %%
np.savetxt('vertices_artificial_straight_trefoil.txt', a_straight_line, delimiter=',')

# %%
tmp = np.append(0,kappa_1)
tmp = np.append(tmp,0)

tmp2 = np.append(0,kappa_2)
tmp2 = np.append(tmp2,0)

kappa_both = np.array([tmp,tmp2]).T

np.savetxt(f'kappa_initial_trefoil.txt', kappa_both, delimiter=' ')
np.savetxt('vertices_artificial_straight_trefoil.txt', a_straight_line, delimiter=',')

