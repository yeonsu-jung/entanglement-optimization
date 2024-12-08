# %%
# %matplotlib qt
import matplotlib
import numpy as np
import jax.numpy as jnp
import jax
from jax import jit
from potentials import create_pairs, all_pairwise_distances, dist_lin_seg, all_distances_between_curves2
from matplotlib import pyplot as plt

from visualizations import set_3d_plot, plot_edges, plot_many_curves, plot_many_rods
from data_io import read_data, import_from_dismech, import_from_dismech_hook
from transforms import q_to_u, q_to_x, x_to_rpairs, x_to_epairs,vert_to_edge
import numba

import matplotlib.animation as animation
import sys
import os
import time

import re

import glob
import networkx as nx

jax.config.update("jax_enable_x64", True)


from numba import jit as njit


def guess_contact_point(fi1,fi2):
    fi1x = fi1[0]
    fi1y = fi1[1]
    fi1z = fi1[2]
    fi2x = fi2[0]
    fi2y = fi2[1]
    fi2z = fi2[2]
    
    force_tol = 1e-6
    if ((fi1x**2 + fi1y**2 + fi1z**2) < force_tol) and ((fi2x**2 + fi2y**2 + fi2z**2) < force_tol):
        return np.nan
    
    x_frac = fi2x/(fi1x+fi2x)
    y_frac = fi2y/(fi1y+fi2y)
    z_frac = fi2z/(fi1z+fi2z)
    
    return x_frac

def process_contact_data(single_contact_info,curr_nodes):
    # dismech originally computes force as a gradient of potential
    # thus here we need to change its sign
    rod_i = int(single_contact_info[4])
    rod_j = int(single_contact_info[5])
    node_i1 = int(single_contact_info[0])
    node_i2 = int(single_contact_info[2])
    node_j1 = int(single_contact_info[1])
    node_j2 = int(single_contact_info[3])

    fi1 = -single_contact_info[6:9]
    fi2 = -single_contact_info[9:12]
    fj1 = -single_contact_info[12:15]
    fj2 = -single_contact_info[15:18]

    ni1 = curr_nodes[rod_i][node_i1]
    ni2 = curr_nodes[rod_i][node_i2]
    nj1 = curr_nodes[rod_j][node_j1]
    nj2 = curr_nodes[rod_j][node_j2]
    
    contact_point_i = guess_contact_point(fi1,fi2) * (ni2-ni1) + ni1
    contact_force_i = fi1 + fi2
    contact_point_j = guess_contact_point(fj1,fj2) * (nj2-nj1) + nj1
    contact_force_j = fj1 + fj2
    
    log_contact_force_i = np.sign(contact_force_i) * np.log(np.abs(contact_force_i) + 1e-6)
    log_contact_force_j = np.sign(contact_force_j) * np.log(np.abs(contact_force_j) + 1e-6)
    
    contact_info = {"rod_i":rod_i,
                    "rod_j":rod_j,
                    "node_i1":node_i1,
                    "node_i2":node_i2,
                    "node_j1":node_j1,
                    "node_j2":node_j2,
                    "contact_point_i":contact_point_i,
                    "contact_force_i":contact_force_i,
                    "contact_point_j":contact_point_j,
                    "contact_force_j":contact_force_j,
                    "log_contact_force_i":log_contact_force_i,
                    "log_contact_force_j":log_contact_force_j,
                    "ni1":ni1,
                    "ni2":ni2,
                    "nj1":nj1,
                    "nj2":nj2,
                    "fi1":fi1,
                    "fi2":fi2,
                    "fj1":fj1,
                    "fj2":fj2}
    
    return contact_info

def get_curr_force_essentials(curr_force_all_info,curr_nodes):
    num_total_contacts = len(curr_force_all_info)
    curr_force_essentials = np.zeros((num_total_contacts,6))
    for query_index in range(num_total_contacts):
        single_contact_info = curr_force_all_info[query_index]
        contact_info = process_contact_data(single_contact_info,curr_nodes)        
        pi = contact_info['contact_point_i']
        pj = contact_info['contact_point_j']
        cij = (pi+pj)/2
        fij = contact_info['contact_force_i']        
        curr_force_essentials[query_index] = np.array([cij[0],cij[1],cij[2],fij[0],fij[1],fij[2]])
    return curr_force_essentials        

def get_local_fields_at_a_point(centerlines, point, R, h, rod_diameter, rod_length):    
    _,labels,edges_all_in_one = get_edges_labels_from_centerlines(centerlines)
    
    I_local = sample_edges_locally_and_return_indices(edges_all_in_one,point,R)
    unique_labels_in_sphere = np.unique(labels[I_local])    
    ee_list = collect_local_edges(edges_all_in_one,labels,unique_labels_in_sphere)        
    
    total_edges = np.vstack(ee_list)
    edge_length = np.linalg.norm(total_edges[:,3:6] - total_edges[:,:3],axis=1)                
    
    number_of_local_curves = len(ee_list)
    local_volume_fraction = np.sum(edge_length)*(np.pi*rod_diameter**2/4)/(4/3*np.pi*R**3)                    
    
    local_orientational_order = compute_local_orientational_order(ee_list)
    
    lk_mat = compute_local_lk(ee_list)    
    local_average_crossing_number = np.sum(np.abs(lk_mat[np.triu_indices(lk_mat.shape[0],k=1)]))
    
    print(f'Number of local curves: {number_of_local_curves}')
    print(f'Local volume fraction: {local_volume_fraction}')
    print(f'Local orientational order: {local_orientational_order}')
    print(f'Local average crossing number: {local_average_crossing_number}')
    
    return number_of_local_curves, local_volume_fraction, local_orientational_order, local_average_crossing_number
    
def get_edges_labels_from_centerlines(centerlines):
        edges = []
        for i in range(centerlines.shape[0]):
            rr = centerlines[i]
            edges.append(np.hstack([rr[:-1],rr[1:]]))
        labels = label_edges(edges)
        edges_vcat = np.vstack(edges)
        return edges,labels,edges_vcat
    
def sample_edges_locally_and_return_indices(edges,center,R):
    I1 = np.linalg.norm((edges[:,:3] - center), axis=1) < R
    I2 = np.linalg.norm((edges[:,3:6] - center), axis=1) < R
    I_sphere_segments = I1 & I2
    return I_sphere_segments    

def collect_local_edges(edges,labels,unique_labels_in_sphere):
    ee_list = []
    for lb in unique_labels_in_sphere:                    
        I = labels == lb
        ee_list.append(edges[I,:])
    return ee_list
    
def compute_local_orientational_order(ee_list):                    
    S = np.zeros((3,3))                    
    total_edges = np.vstack(ee_list)
    for i in range(len(total_edges)):
        orientation_i = total_edges[i,3:6] - total_edges[i,:3]
        orientation_i /= np.linalg.norm(orientation_i)
        S += np.outer(orientation_i,orientation_i)

    eigenvalues,eigenvectors = np.linalg.eig( 3/2*(S/len(total_edges) - 1/3*np.eye(3)) )
    I = np.argmax(np.abs(eigenvalues))
    
    # all_orientations = total_edges[:,3:6] - total_edges[:,:3]
    # all_orientations /= np.linalg.norm(all_orientations,axis=1).reshape(-1,1)
    # fig,ax=plt.subplots(subplot_kw={'projection':'3d'})
    # for i in range(len(total_edges)):
    #     ax.plot(all_orientations[i,0],all_orientations[i,1],all_orientations[i,2],'k.')
    # ax.plot([0,eigenvectors[0,I]], [0,eigenvectors[1,I]], [0,eigenvectors[2,I]],'r')
    # ax.axis('equal')
    
    return eigenvalues[I]

def unpack_centerlines(centerlines):
    unpacked = np.vstack(centerlines)
    labels = np.zeros(unpacked.shape[0],dtype=np.int64)
    start_idx = 0
    for i,cl in enumerate(centerlines):
        end_idx = start_idx + cl.shape[0]
        labels[start_idx:end_idx] = i
        start_idx = end_idx
    return unpacked,labels

def label_centerlines(centerlines):
    labeled_centerlines = []
    labels = []
    for i,cl in enumerate(centerlines):
        n = cl.shape[0]
        lb = np.ones(n,dtype=np.int64)*i
        
        labels.append(lb)
        labeled_cl = np.hstack([cl,lb.reshape(-1,1)])
        labeled_centerlines.append(labeled_cl)
        
    return np.array(labeled_centerlines),np.hstack(np.array(labels))

def label_edges(edges):
    labels = []
    for i,ee in enumerate(edges):
        n = ee.shape[0]
        lb = np.ones(n,dtype=np.int64)*i        
        labels.append(lb)
        
    return np.hstack(np.array(labels))

def get_local_fields_over_domain(centerlines, R, h, rod_diameter, rod_length):
    # R: radius of bounding spheres
    # h: grid spacing
    edges = []
    for i in range(centerlines.shape[0]):
        rr = centerlines[i]
        edges.append(np.hstack([rr[:-1],rr[1:]]))
    labels = label_edges(edges)
    edges_vcat = np.vstack(edges)

    a = np.max(centerlines[:, :, 0]) - np.min(centerlines[:, :, 0])
    b = np.max(centerlines[:, :, 1]) - np.min(centerlines[:, :, 1])
    c = np.max(centerlines[:, :, 2]) - np.min(centerlines[:, :, 2])
    x_min = np.min(centerlines[:, :, 0])
    y_min = np.min(centerlines[:, :, 1])
    z_min = np.min(centerlines[:, :, 2])
    
    center_x = np.arange(R, a-R, h) + x_min
    center_y = np.arange(R, b-R, h) + y_min
    center_z = np.arange(R, c-R, h) + z_min

    num_x = center_x.size
    num_y = center_y.size
    num_z = center_z.size
    num_rods = len(centerlines)

    print(f'Number of rods: {num_rods}')
    print(f'Size of the map: {num_x}, {num_y}, {num_z}')
    t_start = time.time()
    
    n_field = np.full((num_x,num_y,num_z),np.nan)
    phi_field = np.full((num_x,num_y,num_z),np.nan)
    e_field = np.full((num_x,num_y,num_z),np.nan)
    S_field = np.full((num_x,num_y,num_z),np.nan)
    
    # local_fields = []
    for k in range(num_z):
        I_slab_segments = (np.abs(edges_vcat[:, 2] - center_z[k]) < 1.1 * R) & (np.abs(edges_vcat[:, 5] - center_z[k]) < 1.1 * R)
        # unique_labels = np.unique(labels[I_slab_segments])
        labeled_edges_in_slab = edges_vcat[I_slab_segments,:]
        labels_in_slab = labels[I_slab_segments]
        
        if labeled_edges_in_slab.shape[0] == 0:
            continue
        
        for i in range(num_x):
            for j in range(num_y):
                
                center = np.array([center_x[i], center_y[j], center_z[k]])
                
                I1 = np.linalg.norm((labeled_edges_in_slab[:,:3] - center), axis=1) < R
                I2 = np.linalg.norm((labeled_edges_in_slab[:,3:6] - center), axis=1) < R
                I_sphere_segments = I1 & I2
                if np.count_nonzero(I_sphere_segments) == 0:
                    continue
                
                # cf. for a box
                # I1 = np.abs(labeled_edges_in_slab[:,0] - center[0]) < R
                # I2 = np.abs(labeled_edges_in_slab[:,1] - center[1]) < R
                # I3 = np.abs(labeled_edges_in_slab[:,2] - center[2]) < R
                # I4 = np.abs(labeled_edges_in_slab[:,3] - center[0]) < R
                # I5 = np.abs(labeled_edges_in_slab[:,4] - center[1]) < R
                # I6 = np.abs(labeled_edges_in_slab[:,5] - center[2]) < R
                # I_box_segments = I1 & I2 & I3 & I4 & I5 & I6
                
                labeled_segments_in_sphere = labeled_edges_in_slab[I_sphere_segments]
                labels_in_sphere = labels_in_slab[I_sphere_segments]
                unique_labels_in_sphere = np.unique(labels_in_slab[I_sphere_segments])
                
                ee_list = []
                for lb in unique_labels_in_sphere:                    
                    I = labels_in_sphere == lb
                    ee_list.append(labeled_segments_in_sphere[I,:])
                    
                # we got edges in the sphere now
                
                
                total_edges = np.vstack(ee_list)
                edge_length = np.linalg.norm(total_edges[:,3:6] - total_edges[:,:3],axis=1)                
                    
                lk_mat = compute_local_lk(ee_list)
                number_of_local_curves = len(ee_list)
                local_volume_fraction = np.sum(edge_length)*(np.pi*rod_diameter**2/4)/(4/3*np.pi*R**3)                
                local_average_crossing_number = np.sum(np.abs(lk_mat[np.triu_indices(lk_mat.shape[0],k=1)]))
                local_orientational_order = compute_local_orientational_order(ee_list)
                
                n_field[i,j,k] = number_of_local_curves
                phi_field[i,j,k] = local_volume_fraction
                e_field[i,j,k] = local_average_crossing_number
                S_field[i,j,k] = local_orientational_order                

        print(f'Z-Layer: {k+1}/{num_z} \t Loop time: {time.time() - t_start:.2f} \t Elapsed time: {time.time() - t_start:.2f}')

    return n_field, phi_field, e_field, S_field, center_x, center_y, center_z


def import_all_log(alllog_pth):
    with open(alllog_pth) as f:
        lines = f.readlines()
        
    time_line = []
    node_list = []
    contact_list = []
    for i,line in enumerate(lines):
        if 'Time' in line:
            time_line.append(float(line.split('Time: ')[-1].rstrip('\n')))
            
        if 'Node' in line:
            next_line = lines[i+1]                       
            node_list.append(np.array([float(x) for x in next_line.split(',')]))
            
        if 'Force' in line:
            next_line = lines[i+1]
            if next_line == "\n":
                contact_list.append(np.array([]))
            else:
                contact_list.append(np.array([float(x) for x in next_line.split(',')]))
                
    return time_line, node_list, contact_list

def parse_path_string(pth):
    
    filename = pth.split('/')[-1]        
    file_id = filename.split('-mu')[0]
    
    surfix_match = re.search(r'\d{8}-\d{6}', filename)
    surfix = surfix_match.group(0) if surfix_match else None
    
    num_rods_match = re.search(r'-N(\d+)-', filename)
    num_rods = int(num_rods_match.group(1)) if num_rods_match else None
    
    return file_id, surfix, num_rods

@njit(nopython=True)
def compute_linking_number_for_edges(e_i,e_j):
    r_ij = e_i[0:3] - e_j[0:3]
    r_ijj = e_i[0:3] - e_j[3:6]
    r_iij = e_i[3:6] - e_j[0:3]
    r_iijj = e_i[3:6] - e_j[3:6]    
    
    # p_i = e_i[0:3]
    # p_ii = e_i[3:6]
    # p_j = e_j[0:3]
    # p_jj = e_j[3:6]
    
    # r_ij = p_i - p_j
    # r_ijj = p_i - p_jj
    # r_iij = p_ii - p_j
    # r_iijj = p_ii - p_jj

    tol = 1e-6
    n1 = np.cross(r_ij, r_ijj)
    n1 = n1/(np.linalg.norm(n1)+tol)
    n2 = np.cross(r_ijj, r_iijj)
    n2 = n2/(np.linalg.norm(n2)+tol)
    n3 = np.cross(r_iijj, r_iij)
    n3 = n3/(np.linalg.norm(n3)+tol)
    n4 = np.cross(r_iij, r_ij)
    n4 = n4/(np.linalg.norm(n4)+tol)
    
    tol = 1e-6
    return -1/4/np.pi*np.abs(np.arcsin(  my_clip(my_dot(n1,n2),-1.+tol,1.-tol))
                               + np.arcsin(my_clip(my_dot(n2,n3),-1.+tol,1.-tol))
                               + np.arcsin(my_clip(my_dot(n3,n4),-1.+tol,1.-tol))
                               + np.arcsin(my_clip(my_dot(n4,n1),-1.+tol,1.-tol)))    
    
@njit(nopython=True)
def compute_curves_lk(ee_i,ee_j):
    lk = 0
    num_edges_i = ee_i.shape[0]
    num_edges_j = ee_j.shape[0]
    for i in range(num_edges_i):
        e_i = ee_i[i]                        
        for j in range(num_edges_j):                            
            e_j = ee_j[j]
            lk += compute_linking_number_for_edges(e_i, e_j)
    return lk
    
# @njit(nopython=True)
def compute_local_lk(ee_list):
    # u and k are arrays of shape (N,3)
    num_distinct_labels = len(ee_list)
    lk_mat = np.full((num_distinct_labels,num_distinct_labels),np.nan)
    for i in range(num_distinct_labels):
        e_i = ee_list[i]
        for j in range(i+1,num_distinct_labels):
            e_j = ee_list[j]
            lk_mat[i,j] = compute_curves_lk(e_i, e_j)
            
    return lk_mat

@njit(nopython=True)
def compute_linking_number(p_i,p_ii,p_j,p_jj):
    r_ij = p_i - p_j
    r_ijj = p_i - p_jj
    r_iij = p_ii - p_j
    r_iijj = p_ii - p_jj

    tol = 1e-6
    n1 = np.cross(r_ij, r_ijj)
    n1 = n1/(np.linalg.norm(n1)+tol)
    n2 = np.cross(r_ijj, r_iijj)
    n2 = n2/(np.linalg.norm(n2)+tol)
    n3 = np.cross(r_iijj, r_iij)
    n3 = n3/(np.linalg.norm(n3)+tol)
    n4 = np.cross(r_iij, r_ij)
    n4 = n4/(np.linalg.norm(n4)+tol)
    
    tol = 1e-6
    return -1/4/np.pi*np.abs(np.arcsin(  my_clip(my_dot(n1,n2),-1.+tol,1.-tol))
                               + np.arcsin(my_clip(my_dot(n2,n3),-1.+tol,1.-tol))
                               + np.arcsin(my_clip(my_dot(n3,n4),-1.+tol,1.-tol))
                               + np.arcsin(my_clip(my_dot(n4,n1),-1.+tol,1.-tol)))

def compute_linking_number_jax(p_i1,p_i2,p_j1,p_j2):
    r_i1j1 = p_i1 - p_j1
    r_i1j2 = p_i1 - p_j2
    r_i2j1 = p_i2 - p_j1
    r_i2j2 = p_i2 - p_j2

    tol = 1e-6
    n1 = jnp.cross(r_i1j1, r_i1j2)
    n1 = n1/(jnp.linalg.norm(n1)+tol)
    n2 = jnp.cross(r_i1j2, r_i2j2)
    n2 = n2/(jnp.linalg.norm(n2)+tol)
    n3 = jnp.cross(r_i2j2, r_i2j1)
    n3 = n3/(jnp.linalg.norm(n3)+tol)
    n4 = jnp.cross(r_i2j1, r_i1j1)
    n4 = n4/(jnp.linalg.norm(n4)+tol)
    
    return (-1/4/jnp.pi)*jnp.abs(jnp.arcsin(jnp.clip(jnp.dot(n1,n2),-1.+tol,1.-tol))
                               + jnp.arcsin(jnp.clip(jnp.dot(n2,n3),-1.+tol,1.-tol))
                               + jnp.arcsin(jnp.clip(jnp.dot(n3,n4),-1.+tol,1.-tol))
                               + jnp.arcsin(jnp.clip(jnp.dot(n4,n1),-1.+tol,1.-tol)))
    
compute_linking_number_jax_batched = jax.jit(jax.vmap(jax.vmap(compute_linking_number_jax, (0,0,None,None)), (None,None,0,0)))
# %%
from protocols import create_random_rods
from transforms import q_to_x
from visualizations import plot_edges

num_rods = 15000
rods = create_random_rods(num_rods)
rods = q_to_x(rods)

# %%
fig,ax=plt.subplots(subplot_kw={'projection':'3d'})
plot_edges(rods,ax=ax)
# %%
compute_linking_number_jax_batched(rods[:,:3], rods[:,3:], rods[:,:3], rods[:,3:]).shape
# %%
