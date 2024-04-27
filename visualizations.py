import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
from jax import lax
from optimization import optimize_fire2,optimize_fire_debug
from potentials import effective_potential,total_effective_potential,total_effective_potential_ref
import numpy as onp
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

def plot_many_rods(q,opt_dict={}):
    N = q.shape[0]
    for i in range(N):        
        plot_rod(q[i,:],opt_dict)
        
    return 1
def plot_rod(q_single,opt_dict):
    q_onp = onp.array(q_single)
    x1 = q_onp[0]
    y1 = q_onp[1]
    z1 = q_onp[2]
    phi1 = q_onp[3]
    theta1 = q_onp[4]
    rod_length = 1.

    x11 = x1 + rod_length*jnp.sin(phi1)*jnp.cos(theta1)
    y11 = y1 + rod_length*jnp.sin(phi1)*jnp.sin(theta1)
    z11 = z1 + rod_length*jnp.cos(phi1)
    plt.plot([x1, x11], [y1, y11], [z1, z11],**opt_dict)
    
def plot_rods(q):
    q_onp = onp.array(q)
    x1 = q_onp[0]
    y1 = q_onp[1]
    z1 = q_onp[2]
    phi1 = q_onp[3]
    theta1 = q_onp[4]
    rod_length = 1.

    x11 = x1 + rod_length*jnp.sin(phi1)*jnp.cos(theta1)
    y11 = y1 + rod_length*jnp.sin(phi1)*jnp.sin(theta1)
    z11 = z1 + rod_length*jnp.cos(phi1)

    x2 = q_onp[5]
    y2 = q_onp[6]
    z2 = q_onp[7]
    phi2 = q_onp[8]
    theta2 = q_onp[9]

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

def plot_edges(edges,ax=None):
    N = edges.shape[0]
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
    for i in range(N):
        ax.plot([edges[i,0],edges[i,3]],[edges[i,1],edges[i,4]],[edges[i,2],edges[i,5]])    
    

