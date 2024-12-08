# %%
%matplotlib qt
import numpy as np
import matplotlib.pyplot as plt
from generate_wlc_on_s2 import generate_wlc_3d,gaussian_filter1d


# %%

def parallel_transport(d1_1, t1, t2):
    b = np.cross(t1, t2)
    
    if np.linalg.norm(b) == 0:
        return d1_1
    else:
        b = b / np.linalg.norm(b)
        b = b - np.dot(b, t1) * t1
        b = b / np.linalg.norm(b)
        b = b - np.dot(b, t1) * t2
        b = b / np.linalg.norm(b)
        
        n1 = np.cross(t1, b)
        n2 = np.cross(t2, b)
        
        d1_2 = np.dot(d1_1, t1) * t2 + np.dot(d1_1, n1) * n2 + np.dot(d1_1, b) * b
        d1_2 = d1_2 - np.dot(d1_2, t2) * t2
        d1_2 = d1_2 / np.linalg.norm(d1_2)
        
        return d1_2

def compute_space_parallel(tangent, ne):
    d1 = np.zeros_like(tangent)
    d2 = np.zeros_like(tangent)

    t0 = tangent[0]
    t1 = np.array([0.0, 0.0, -1.0])
    d1Tmp = np.cross(t0, t1)

    if np.abs(np.linalg.norm(d1Tmp)) < 1.0e-6:
        t1 = np.array([0.0, 1.0, 0.0])
        d1Tmp = np.cross(t0, t1)

    d1[0] = d1Tmp
    d2[0] = np.cross(t0, d1Tmp)

    for i in range(ne - 1):
        a = d1[i]
        b = tangent[i]
        c = tangent[i + 1]
        d = parallel_transport(a, b, c)
        d1[i + 1] = d
        d2[i + 1] = np.cross(c, d)
    
    return d1, d2


def rotate_axis_angle(v, z, theta):
    if theta != 0:
        cs = np.cos(theta)
        ss = np.sin(theta)
        v = cs * v + ss * np.cross(z, v) + np.dot(z, v) * (1.0 - cs) * z
    return v



def signed_angle(u, v, n):
    w = np.cross(u, v)
    angle = np.arctan2(np.linalg.norm(w), np.dot(u, v))
    if np.dot(n, w) < 0:
        return -angle
    else:
        return angle


def get_ref_twist(tangent, d1, ne, ref_twist_old):
    ref_twist = np.zeros(ne)
    
    for i in range(1, ne):
        u0 = d1[i - 1]
        u1 = d1[i]
        t0 = tangent[i - 1]
        t1 = tangent[i]

        ut = parallel_transport(u0, t0, t1)
        rotate_axis_angle(ut, t1, ref_twist_old[i])

        sgnAngle = signed_angle(ut, u1, t1)
        ref_twist[i] = ref_twist_old[i] + sgnAngle
    
    return ref_twist


def compute_twist_bar(x, ref_twist, ne):
    twist_bar = np.zeros(ne)
    for i in range(1, ne):
        theta_i = x[4 * (i - 1) + 3]
        theta_f = x[4 * i + 3]
        twist_bar[i] = theta_f - theta_i + ref_twist[i]
    return twist_bar


def compute_material_director(angle_list,d1,d2):
    ne = vertices.shape[0]
    for i in range(ne):
        # angle = x[4 * i + 3]
        angle = angle_list[i]
        cs = np.cos(angle)
        ss = np.sin(angle)
        m1[i, :] = cs * d1[i, :] + ss * d2[i, :]
        m2[i, :] = -ss * d1[i, :] + cs * d2[i, :]
# %%
def trefoil_knot(t):
    x = np.sin(t) + 2 * np.sin(2 * t)
    y = np.cos(t) - 2 * np.cos(2 * t)
    z = np.sin(3 * t)
    return x, y, z

num_vertices = 120
t_power = 0.25

t = np.linspace(0, (2 * np.pi)**t_power, num_vertices)
t = t**(1/t_power)

plt.close('all')
plt.plot(t)
# %%
# TODO: deal with tight ends
x, y, z = trefoil_knot(t)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z)
plt.show()

# %%
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
colors = plt.cm.viridis((t - t.min()) / (t.max() - t.min()))
for i in range(len(t) - 1):
    ax.plot(x[i:i+2], y[i:i+2], z[i:i+2], color=colors[i])

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
    d = parallel_transport(a,b,c)
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

