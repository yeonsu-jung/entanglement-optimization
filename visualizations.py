import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
from jax import lax
from optimization import optimize_fire2,optimize_fire_debug
from potentials import effective_potential,total_effective_potential,total_effective_potential_ref
import numpy as np
from matplotlib import pyplot as plt
import time

def set_3d_plot():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    return fig,ax

def plot_contacts(q,i,neighbors):
    qs = jnp.reshape(q,(-1,5))
    # 3d plots    
    plot_many_rods(jnp.reshape(qs[i,:],(-1,5)),opt_dict={"color":'r','linewidth':2})
    plot_many_rods(qs[neighbors,:])
    
    return 1

def plot_many_rods(q,ax=None,opt_dict={}):
    if ax is None:
        fig,ax=set_3d_plot()
        
    N = q.shape[0]
    for i in range(N):        
        plot_rod(q[i,:],opt_dict)
        
    return ax

def plot_rod(q_single,opt_dict):
    q_np = np.array(q_single)
    x1 = q_np[0]
    y1 = q_np[1]
    z1 = q_np[2]
    phi1 = q_np[3]
    theta1 = q_np[4]
    rod_length = 1.

    x11 = x1 + rod_length*jnp.sin(phi1)*jnp.cos(theta1)
    y11 = y1 + rod_length*jnp.sin(phi1)*jnp.sin(theta1)
    z11 = z1 + rod_length*jnp.cos(phi1)
    plt.plot([x1, x11], [y1, y11], [z1, z11],**opt_dict)
    
def plot_rods(q):
    q_np = np.array(q)
    x1 = q_np[0]
    y1 = q_np[1]
    z1 = q_np[2]
    phi1 = q_np[3]
    theta1 = q_np[4]
    rod_length = 1.

    x11 = x1 + rod_length*jnp.sin(phi1)*jnp.cos(theta1)
    y11 = y1 + rod_length*jnp.sin(phi1)*jnp.sin(theta1)
    z11 = z1 + rod_length*jnp.cos(phi1)

    x2 = q_np[5]
    y2 = q_np[6]
    z2 = q_np[7]
    phi2 = q_np[8]
    theta2 = q_np[9]

    x22 = x2 + rod_length*jnp.sin(phi2)*jnp.cos(theta2)
    y22 = y2 + rod_length*jnp.sin(phi2)*jnp.sin(theta2)
    z22 = z2 + rod_length*jnp.cos(phi2)

    # 3d plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.plot([x1, x11], [y1, y11], [z1, z11])
    ax.plot([x2, x22], [y2, y22], [z2, z22])
    
def plot_single_rod(single_rod, *args, ax=None, **kwargs):
    if ax is None:
        fig,ax = set_3d_plot()
    ax.plot(single_rod[:,0],single_rod[:,1],single_rod[:,2],*args,**kwargs)
    return ax
    
def plot_curves(curve,ax,params={}):
    ax.plot(curve[:,0],curve[:,1],curve[:,2],**params)    
    return 1

def plot_many_curves(curves,num_rods,ax,params={}):
    if curves.ndim == 1:
        curves = curves.reshape(num_rods,-1,3)
    elif curves.ndim == 3:
        num_vertices = curves.shape[1]//3//num_rods    
        curves = curves.reshape((num_rods,-1))
    else:
        print('Input must be 1d or 3d array')
        return -1
    
    for i in range(num_rods):
        plot_curves(curves[i,:],ax,params)
    return 1

def plot_edges(edges,ax=None,params={}):
    N = edges.shape[0]
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
    for i in range(N):
        ax.plot([edges[i,0],edges[i,3]],[edges[i,1],edges[i,4]],[edges[i,2],edges[i,5]],**params)    
    

def plot_centerline_with_container(centerlines,svd_cylinders,i,ax):
    cl = centerlines[i]
    cyl = svd_cylinders[i,:]

    cyl_diam = cyl[6]
    cyl_e1 = cyl[0:3]
    cyl_e2 = cyl[3:6]
    cyl_cen = (cyl_e1+cyl_e2)/2
    cyl_len = np.linalg.norm(cyl_e1-cyl_e2)
    cyl_axis = (cyl_e2-cyl_e1)/cyl_len

    Xc, Yc, Zc = data_for_cylinder_along_z(0, 0, cyl_diam, cyl_len/2)
    # Compute the rotation matrix
    rotation_matrix = rotation_matrix_from_vectors(np.array([0, 0, 1]), cyl_axis) 
    # Rotate the cylinder
    Xc_rot, Yc_rot, Zc_rot = rotate_grid(Xc, Yc, Zc, rotation_matrix)
    Xc_rot = Xc_rot + cyl_cen[0]
    Yc_rot = Yc_rot + cyl_cen[1]
    Zc_rot = Zc_rot + cyl_cen[2]
    
    bounding_box = np.array([np.min(cl, axis=0), np.max(cl, axis=0)])
    ax.plot_surface(Xc_rot, Yc_rot, Zc_rot, alpha=0.5)
    ax.plot(cl[:,0], cl[:,1], cl[:,2], color='r')
    # ax.scatter(cyl_e1[0], cyl_e1[1], cyl_e1[2], color='g')
    # zoom in
    # ax.set_xlim(bounding_box[:,0])
    # ax.set_ylim(bounding_box[:,1])
    # ax.set_zlim(bounding_box[:,2])

def data_for_cylinder_along_z(center_x, center_y, radius, height_z):
    z = np.linspace(-height_z, height_z, 50)
    theta = np.linspace(0, 2 * np.pi, 50)
    theta_grid, z_grid = np.meshgrid(theta, z)
    x_grid = radius * np.cos(theta_grid) + center_x
    y_grid = radius * np.sin(theta_grid) + center_y
    return x_grid, y_grid, z_grid

def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2 """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat @ kmat * ((1 - c) / (s ** 2))
    return rotation_matrix

def rotate_grid(X, Y, Z, rotation_matrix):
    shape = X.shape
    grid = np.vstack([X.ravel(), Y.ravel(), Z.ravel()])
    rotated_grid = rotation_matrix @ grid
    X_rot, Y_rot, Z_rot = rotated_grid.reshape(3, *shape)
    return X_rot, Y_rot, Z_rot
