# %%
# %matplotlib qt
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

# %%
# create vertices for a helix
# num turns = 3
# num vertices per turn = 20

# Number of turns and vertices per turn
num_turns = 3
num_vertices_per_turn = 20
num_vertices = num_turns * num_vertices_per_turn

# Initialize the vertices array
vertices = np.zeros((num_vertices, 3))

# Generate the helix vertices
for i in range(num_turns):
    theta = np.linspace(0, 2 * np.pi, num_vertices_per_turn, endpoint=False) + i * 2 * np.pi
    z = np.linspace(i, i + 1, num_vertices_per_turn, endpoint=False)
    x = np.cos(theta)
    y = np.sin(theta)
    vertices[i * num_vertices_per_turn:(i + 1) * num_vertices_per_turn, 0] = x
    vertices[i * num_vertices_per_turn:(i + 1) * num_vertices_per_turn, 1] = y
    vertices[i * num_vertices_per_turn:(i + 1) * num_vertices_per_turn, 2] = z


vertices.shape
# %%
# np.savetxt(f'vertices_n1_skip{skip_factor}.txt', vertices, delimiter=',')
np.savetxt(f'vertices_helix.txt', vertices, delimiter=',')
# %%

fig,ax=plt.subplots(subplot_kw={'projection':'3d'})
ax.plot(vertices[:,0],vertices[:,1],vertices[:,2],'o-')
ax.axis('equal')

# %%
fig,ax=plt.subplots()
ax.plot(what_is_this)
# %%
# put frame on the curve
# i need to find material frame





# %%

# curvature
# anything else?
# tangents = np.gradient(vertices,axis=0)
tangents = np.diff(vertices,axis=0)
rod_length = np.linalg.norm(tangents,axis=1)
tangents /= np.linalg.norm(tangents,axis=1)[:,None]

t_prev = tangents[:-1,:]
t_curr = tangents[1:,:]

curvature_binormal = 2*np.cross(t_prev,t_curr)/(1+np.sum(t_prev*t_curr,axis=1))[:,None]
# %%

tangents = np.diff(vertices,axis=0)
tangents /= np.linalg.norm(tangents,axis=1)[:,None]

# initial material frame
m1 = np.cross(tangents[0],np.array([0.0,0.0,-1.0]))
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
    
# %%
kappa_1 = np.sum(d1[:-1]*curvature_binormal,axis=1)
kappa_2 = np.sum(d2[:-1]*curvature_binormal,axis=1)

# %%
kappa_both = np.array([kappa_1,kappa_2]).T

np.savetxt('curvature_binormal_projected.txt', kappa_both, delimiter=',') 

# %%
from numba import jit
@jit(nopython=True)
def pdist2(rr1,rr2):
    n = rr1.shape[0]
    m = rr2.shape[0]
    dist_matrix = np.zeros((n,m))
    for i in range(n):
        for j in range(m):
            dist_matrix[i,j] = np.linalg.norm(rr1[i] - rr2[j])
    return dist_matrix

dist_mat = pdist2(vertices,vertices)
# %%
np.min(dist_mat[dist_mat>0])
    
# %%

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
wlc_3d = generate_wlc_3d(N, Lp, segment_length)

sigma = 5
wlc_3d_smoothed = gaussian_filter1d(wlc_3d, sigma=sigma, axis=0)
# ax.plot(wlc_3d[:, 0], wlc_3d[:, 1], wlc_3d[:, 2], lw=1)
ax.plot(wlc_3d_smoothed[:, 0], wlc_3d_smoothed[:, 1], wlc_3d_smoothed[:, 2], lw=1)

vertices = wlc_3d_smoothed
np.savetxt('vertices.txt', vertices, delimiter=',')
# %%
# random curve

tangents = np.diff(vertices,axis=0)
tangents /= np.linalg.norm(tangents,axis=1)[:,None]

# initial material frame
m1 = np.cross(tangents[0],np.array([0.0,0.0,-1.0]))
m2 = np.cross(tangents[0],m1)

# parallel transport
ne = tangents.shape[0]
d1 = np.zeros_like(tangents)
d2 = np.zeros_like(tangents)

d1[0] = m1
d2[0] = m2

ref_twist = get_ref_twist(tangents,d1,ne,np.zeros(ne))


for i in range(ne-1):
    a = d1[i]
    b = tangents[i]
    c = tangents[i+1]
    d = parallel_transport(a,b,c)
    d1[i+1] = d
    d2[i+1] = np.cross(c,d)
# %%
# for (int i = 0; i < ne; i++):
m1 = np.zeros_like(tangents)
m2 = np.zeros_like(tangents)
for i in range(ne):
    angle = 0
    cs = np.cos(angle)
    ss = np.sin(angle)
    m1[i] = cs * d1[i] + ss * d2[i]
    m2[i] = -ss * d1[i] + cs * d2[i]


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
# curvature binormal
curvature_binormal = 2*np.cross(tangents[:-1],tangents[1:])/(1+np.sum(tangents[:-1]*tangents[1:],axis=1))[:,None]

kappa_1 = np.sum(d1[:-1]*curvature_binormal,axis=1)
kappa_2 = np.sum(d2[:-1]*curvature_binormal,axis=1)
# %%
plt.plot(kappa_1)

# %%
kappa_1 = np.sum(d1[:-1]*curvature_binormal,axis=1)
kappa_2 = np.sum(d2[:-1]*curvature_binormal,axis=1)
    
# %%arrow_length = 0.1
plt.close('all')
arrow_length = 0.01
fig,ax=plt.subplots(subplot_kw={'projection':'3d'})
ax.plot(vertices[:,0],vertices[:,1],vertices[:,2])
ax.quiver(vertices[:-1,0],vertices[:-1,1],vertices[:-1,2],tangents[:,0],tangents[:,1],tangents[:,2],length=arrow_length,colors='b')
ax.quiver(vertices[:-1,0],vertices[:-1,1],vertices[:-1,2],d1[:,0],d1[:,1],d1[:,2],length=arrow_length,colors='r')
ax.quiver(vertices[:-1,0],vertices[:-1,1],vertices[:-1,2],d2[:,0],d2[:,1],d2[:,2],length=arrow_length,colors='g')
ax.axis('equal')




# %%
fig,ax=plt.subplots()
ax.plot(what_is_this)
ax.plot(np.linalg.norm(curvature_binormal,axis=1))

# %%
scalar_curvature = np.linalg.norm(curvature_binormal,axis=1)
# %%
np.savetxt('scalar_curvature.txt', scalar_curvature, delimiter=',')


# %%
import polyscope as ps

# np.arange(vertices.shape[0]-1)
# [0,1],[1,2],...,[n-2,n-1]
edge_labels = np.array([np.arange(vertices.shape[0]-1),np.arange(1,vertices.shape[0])]).T   

ps.init()
ps_knot = ps.register_curve_network("knot", vertices, edge_labels)

num_edges = vertices.shape[0]-1
vals_edge = np.ones((num_edges,3))
for i in range(1,num_edges-1):
    vals_edge[i] = plt.cm.coolwarm(scalar_curvature[i]/np.max(scalar_curvature))[:3]

ps_knot.add_color_quantity(f"scalar_curvature", vals_edge, defined_on='edges', enabled=True)

ps.set_up_dir('z_up')


ps.show()

# %%
