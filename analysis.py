# %%
# %matplotlib qt
import matplotlib
import numpy as np
import jax.numpy as jnp
import jax
from jax import jit
from potentials import create_pairs, all_pairwise_distances, dist_lin_seg, all_distances_between_curves2
from matplotlib import pyplot as plt
from data_io import import_all_log, parse_path_string
from visualizations import set_3d_plot, plot_edges, plot_many_curves, plot_many_rods
from data_io import read_data, import_from_dismech, import_from_dismech_hook
from transforms import q_to_u, q_to_x, x_to_rpairs, x_to_epairs,vert_to_edge
import numba
from distances import lumelsky_dist_vec

import matplotlib.animation as animation
import sys
import os
import time

import re

import glob
import networkx as nx
import filamentFields

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
    
# numba

@njit(nopython=True)
def my_clip(x, xmin, xmax):
    if x < xmin:
        return xmin
    elif x > xmax:
        return xmax
    else:
        return x
    
@njit(nopython=True)
def my_dot(a,b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

@njit(nopython=True)
def my_cross(a,b):
    return np.array([a[1]*b[2] - a[2]*b[1],
                     a[2]*b[0] - a[0]*b[2],
                     a[0]*b[1] - a[1]*b[0]])
    

def find_contacts(q,rod_radius):
    # q is an one dimensional array
    # qs are the degrees of freedom of the rods, qs = reshape(q,(-1,5))
    # q_pairs is a list of pairs of indices of rods that are in contact
    q_pairs = create_pairs(jnp.reshape(q,(-1,5)))    
    d = all_pairwise_distances(q_pairs)
    contact_indices = np.where(d < 2*rod_radius)
    
    num_rods = q.shape[0]//5
    i, j = jnp.triu_indices(num_rods, k=1)
    contacts = jnp.array([i[contact_indices], j[contact_indices]]).transpose()
    num_contacts = contacts.shape[0]
    
    avg_num_contacts_per_rod = num_contacts/num_rods
    
    # print(np.where((contacts[:,0] == 0) or (contacts[:,1] == 0)))
        
    contact_degrees = np.zeros(num_rods)
    neighbors = []
    
    # TO DO: make it faster?
    for i in range(num_rods):
        # nnz
        contact_degrees[i] = jnp.count_nonzero(contacts[:,0] == i) + jnp.count_nonzero(contacts[:,1] == i)
        neighbors.append(jnp.concatenate([contacts[contacts[:,0] == i,1], contacts[contacts[:,1] == i,0]]))
    
    print('Number of contacts: ', num_contacts)
    print('Average number of contacts per rod: ', avg_num_contacts_per_rod)
    print('Avg. contact degrees: ', np.mean(contact_degrees))
    
    return contacts, neighbors, contact_degrees

def example_contacts():
    pth = '/Users/yeonsu/Data/entangled_rods_N300_relaxed_22-04-2024_00-36-18.txt'
    q = read_data(pth)    
    rod_radius = 0.08 # TO DO: read from file    
    contacts = find_contacts(q,rod_radius)
        
def calculate_oreintational_order(q):
    x = q_to_x(q)
    u = q_to_u(q)
    num_rods = x.shape[0]
    
    print(u)
    S = 1   
    return S

def distance_check():
    from data_io import import_from_dismech
    from potentials import distance_between_two_curves, all_distnaces_between_curves    
    
    sim_id = '20240426-215217_node_20240427-014524'
    root_dir = '/Users/yeonsu/Data/from-cluster'
    pth = f'{root_dir}/{sim_id}.csv'
    num_rods = 100
    
    spatial_data,timepoints = import_from_dismech(pth,num_rods)
    spatial_data = jnp.array(spatial_data, dtype=jnp.float64)
    
    from potentials import distance_between_two_curves, all_distnaces_between_curves    
    num_vertices = spatial_data.shape[1]//(3*num_rods)
    
    import time
    
    start = time.time()
    d = all_distnaces_between_curves(spatial_data[-1,:])
    now = time.time()
    print(f'Elapsed time: {now-start}')
    
    rod_radius = 2
    print(f"Number of contacts: {jnp.count_nonzero(d < 2*rod_radius*1.5)}")
    print(f"Min distance: {jnp.min(d)}")
    print(f"Distance median: {jnp.median(d)}")
    
    plt.hist(d,bins=100)
    plt.show()
    
    return 1

def length_of_polygonal_curve(nodes):
    tan = nodes[1:,:] - nodes[:-1,:]
    length = np.sum(np.linalg.norm(tan,axis=1))
    return length

def curvature_of_polygonal_curve(nodes):
    tan2 = nodes[2:,:] - nodes[1:-1,:]    
    tan1 = nodes[1:-1,:] - nodes[:-2,:]
    
    nom = np.linalg.norm(2*np.cross(tan1,tan2,axis=1),axis=1)
    den = np.sum(tan1*tan2,axis=1)
    curvature = np.sum(nom/den)
    return curvature

def curvature_check(dof_at_a_time,num_rods):
    nodes_for_single_curve = dof_at_a_time.reshape((num_rods,-1,3))
    curvature_list = []
    for i in range(num_rods):
        curvature = curvature_of_polygonal_curve(nodes_for_single_curve[i,:])
        curvature_list.append(curvature)
    return curvature_list

def length_check(dof_at_a_time,num_rods):
    nodes_for_single_curve = dof_at_a_time.reshape((num_rods,-1,3))
    length_list = []
    for i in range(num_rods):
        length = length_of_polygonal_curve(nodes_for_single_curve[i,:])
        length_list.append(length)
        
    length_list = np.array(length_list)
    return length_list

def main2():
    root_dir = '/Users/yeonsu/Data/from-cluster'
    data_id = '20240425-215943_node_20240426-150758'
    
    from data_io import import_from_dismech
    pth = f'{root_dir}/{data_id}.csv'
    num_rods = 100
    nodes_over_time, timepoints = import_from_dismech(pth,num_rods)
    print(nodes_over_time.shape)
    
    nodes_at_a_time = nodes_over_time[0,:]
    print(nodes_at_a_time.shape)
    
    num_vertices = nodes_over_time.shape[1]//3//num_rods
    nodes_in_matrix = nodes_at_a_time.reshape((num_rods,-1))
    
    from visualizations import plot_many_curves
    plot_many_curves(nodes_in_matrix)
    
    plt.show()
    return 1

def main3():
    sim_id = '20240426-215217_node_20240429-221150'
    root_dir = '/Users/yeonsu/Data/from-cluster'
    pth = f'{root_dir}/{sim_id}.csv'
    num_rods = 100
    
    from data_io import import_from_dismech
    nodes_over_time, timepoints = import_from_dismech(pth,num_rods)
    print(nodes_over_time.shape)
    q1 = nodes_over_time[0,:]
    
    from visualizations import plot_many_curves,set_3d_plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    plot_many_curves(q1,num_rods,ax)
    plt.show()    
    
    dist= np.linalg.norm(nodes_over_time - q1,axis=1)
    plt.plot(timepoints,dist)
    plt.show()
    
    # params = {'marker': '.-'}
    fig,ax = set_3d_plot()
    plot_many_curves(nodes_over_time[0,:],num_rods,ax)
    plt.show()    
    
def check_curvature(nodes_over_time,timepoints):
    # check curvature
    num_rods = 100
    num_timepoints = nodes_over_time.shape[0]    
    curvature_over_time = []
    total_curvature_over_time = []
    selected_timepoints = []    
    avg_length_over_time = []
    
    for i in range(0,num_timepoints,100):
        curvature_over_time.append(curvature_check(nodes_over_time[i,:],num_rods))
        total_curvature_over_time.append(np.sum(curvature_over_time[-1]))
        selected_timepoints.append(timepoints[i])        
        avg_length_over_time.append(np.mean(length_check(nodes_over_time[i,:],num_rods)))        
        print(f"Iteration: {i}/{num_timepoints}")
    
    total_curvature_over_time = np.array(total_curvature_over_time)
    selected_timepoints = np.array(selected_timepoints)
    
    plt.plot(selected_timepoints,total_curvature_over_time)
    plt.xlabel('Simulation time')
    plt.ylabel('Total curvature')
    plt.show()
    return 1

def check_length(selected_timepoints,avg_length_over_time):    
    avg_length_over_time = np.array(avg_length_over_time)
    plt.plot(selected_timepoints,avg_length_over_time)
    plt.xlabel('Simulation time')
    plt.ylabel('Average length')
    
def inspect_edge_level():
    pth = '/Users/yeonsu/Documents/GitHub/dismech-rods-main/runs/20240503-1542_COMPILE_/log_files/EntangledRelaxedPackingXYZ_node_20240503-154355.csv'
    dta = np.loadtxt(pth,delimiter=',')
    nodes = dta[1:]
    nodes_mat = nodes.reshape(-1,30)
    
    print(nodes_mat.shape)
    
    r1 = nodes_mat[248,:]
    r2 = nodes_mat[265,:]
    
    print(r1.shape)
    
    print(nodes_mat[250,:])
    
    e1 = r1.reshape(-1,3)[3,:]
    e11 = r1.reshape(-1,3)[4,:]
    
    e2 = r2.reshape(-1,3)[2,:]
    e22 = r2.reshape(-1,3)[3,:]
    
    d_e = dist_lin_seg(e1,e11,e2,e22)
    print(d_e)

    
def self_avoiding_pairs(e_i,e_j):
    # e_i, e_j: (num_rods,num_edges*6) array
    e_i_mat = e_i.reshape(-1,6)
    e_j_mat = e_j.reshape(-1,6)
    
    pairs = jnp.concatenate([e_i_mat,e_j_mat],axis=1)
    
    return pairs

@jit
def distances_between_curve_edges(pairs):
    from jax import vmap
    d = vmap(dist_lin_seg)(pairs[:,0:3], pairs[:,3:6], pairs[:,6:9], pairs[:,9:12])
    return d
    
def debugging_distance():
    num_rods = 300    
    pth = '/Users/yeonsu/Data/from-cluster/EntangledRelaxedPackingHookScaled_N100_r0.500000_node_20240503-223905.csv'
    
    dta = np.loadtxt(pth,delimiter=',')
    nodes = dta[1:]
    nodes_mat = nodes.reshape(-1,30)
    
    e = x_to_epairs(nodes_mat,num_rods)
    r = x_to_rpairs(nodes_mat,num_rods)
    
    i, j = jnp.triu_indices(num_rods, k=1)
    r_i = r[i]  # Shape will be (N(N-1)/2, M)
    r_j = r[j]  # Shape will be (N(N-1)/2, M)
    
    curve_pairs = jnp.concatenate([r_i,r_j],axis=1)
    half_size = curve_pairs.shape[1]//2
    pairs1 = curve_pairs[:,:half_size]
    pairs2 = curve_pairs[:,half_size:]
    
    d = all_distances_between_curves2(pairs1,pairs2)
    print(jnp.min(d))
    return 0

def inspect_packing(pth):
    dta = np.loadtxt(pth,delimiter=',')
    
    # if one dimensional array
    if len(dta.shape) == 1:
        q = dta[7:]
        q_mat = q.reshape(-1,30)
        
        print(q_mat[0,:])
    elif len(dta.shape) == 2:
        nodes = dta[:,1:]
        nodes_mat = nodes.reshape(-1,30)
        
        print(nodes_mat.shape)
        
        
    
    
    nodes = dta[1:]
    nodes_mat = nodes.reshape(-1,30)
    
    print(nodes_mat.shape)
    
    r1 = nodes_mat[248,:]
    r2 = nodes_mat[265,:]
    
    print(r1.shape)
    
    print(nodes_mat[250,:])
    
    e1 = r1.reshape(-1,3)[3,:]
    e11 = r1.reshape(-1,3)[4,:]
    
    e2 = r2.reshape(-1,3)[2,:]
    e22 = r2.reshape(-1,3)[3,:]
    
    d_e = dist_lin_seg(e1,e11,e2,e22)
    print(d_e)
    
def create_curve_pairs(curves):
        # every edge in a curve will be paired with every edge in another curve
        # each edge in a curve will not be paired with another edge in the same curve
        num_rods = curves.shape[0]
        r = x_to_rpairs(curves,num_rods)
        i, j = jnp.triu_indices(num_rods, k=1)
        r_i = r[i]  # Shape will be (N(N-1)/2, M)
        r_j = r[j]  # Shape will be (N(N-1)/2, M)
    
        curve_pairs = jnp.concatenate([r_i,r_j],axis=1)
        half_size = curve_pairs.shape[1]//2
        pairs1 = curve_pairs[:,:half_size]
        pairs2 = curve_pairs[:,half_size:]
        return pairs1, pairs2, i ,j

def inspect_dismech_nodes(pth,zoom,start_column=1,max_rows=100000,row_skip=1,visualize=0):
    parsed_info = parse_filename(pth)
    
    nodes_over_time, timepoints, num_vertices = import_from_dismech_hook(pth,parsed_info["num_rods"],start_col = start_column, max_rows = max_rows, row_skip=row_skip)
            
    curves = nodes_over_time[0,:].reshape(parsed_info['num_rods'],-1) # curves are num_rods x num_vertices x 3 array
    pairs1,pairs2,i,j = create_curve_pairs(curves)
    d = all_distances_between_curves2(pairs1,pairs2)
    
    rod_radius = parsed_info['rod_radius']
    idx_in_contact = np.unique(np.vstack([i[d < rod_radius*2.05], j[d < rod_radius*2.05]]))
    
    # log file
    logfiledir = f'/Users/yeonsu/Data/analysis/{parsed_info["batch_id"]}/{parsed_info["date_time"]}'
    if not os.path.exists(logfiledir):
        os.makedirs(logfiledir)
        
    logfilepath = f'{logfiledir}/{parsed_info["sim_id"]}_log.txt'    
    with open(logfilepath,'w') as f:        
        f.write(f'File path: {parsed_info["pth"]}\n')
        f.write(f'Simulation ID: {parsed_info["sim_id"]}\n')
        f.write(f'Num. rods: {parsed_info["num_rods"]}\n')
        f.write(f'Rod radius: {parsed_info["rod_radius"]}\n')
        f.write(f'Rod length: {parsed_info["rod_length"]}\n')
        f.write(f'Aspect ratio: {parsed_info["AR"]}\n')
        f.write(f'Shape of nodes_over_time: {nodes_over_time.shape}\n')
        f.write(f'Final time point: {timepoints[-1]} sec\n')
        f.write(f'Number of vertices: {num_vertices}\n')
        f.write(f'Min distance: {jnp.min(d)}\n')
        f.write(f'Number of contacts at the last frame: {jnp.count_nonzero(d < 2*parsed_info["rod_radius"]*1.05)}\n')
        f.write(f'Min distance: {jnp.min(d)}\n')
    # save animation
    cluster_size_list = create_animation_with_label(pth,nodes_over_time,timepoints,zoom)
    
    # fig,ax=plt.subplots(figsize=(4,3))
    # ax.plot(timepoints,cluster_size_list)
    # ax.set_xlabel('Time')
    # ax.set_ylabel('Number of rods in the cluster')
    
    # outdir = f'/Users/yeonsu/Figures/{parsed_info["batch_id"]}/{parsed_info["date_time"]}'
    # if not os.path.exists(outdir):
    #     os.makedirs(outdir)    
    # plt.savefig(f'{outdir}/{parsed_info["sim_id"]}_cluster_size.png',dpi=300)
    
    # dataout = np.vstack([timepoints,cluster_size_list]).T
    # np.savetxt(f'{logfiledir}/{parsed_info["sim_id"]}_cluster_size.txt',dataout)
    
    return 0
    
def create_animation_with_label(pth,nodes_over_time,timepoints,zoom):
    parsed_info = parse_filename(pth)
    num_rods = parsed_info['num_rods']
    dt = timepoints[1] - timepoints[0] # assuming uniform time steps
    title_string = ''
    tokens = parsed_info['sim_id'].split('_')[0].split('-')
    for token in tokens:
        if 'N' in token:
            title_string += f'{token}_'
        if 'AR' in token:
            title_string += f'{token}_'
        if 'mu' in token:
            title_string += f'{token}_'
        if 'visc' in token:
            title_string += f'{token}_'
        if 'amp' in token:
            title_string += f'{token}'
    num_frames = nodes_over_time.shape[0]
    cluster_size_list = []
    
    rod_radius = parsed_info['rod_radius']
    
    fig,ax=set_3d_plot()
    plot_many_curves(nodes_over_time[0,:],num_rods,ax,params={'color':'k','alpha':0.2})
    
    def update(frame):
        ax.clear()        
        plot_many_curves(nodes_over_time[frame,:],num_rods,ax)
        print(frame)
        # curves = nodes_over_time[frame,:].reshape(parsed_info['num_rods'],-1)
        # pairs1,pairs2,i,j = create_curve_pairs(curves)
        # d = all_distances_between_curves2(pairs1,pairs2)
        # rods_in_contact = np.unique(np.vstack([i[d < rod_radius*2.05], j[d < rod_radius*2.05]]))
        # rods_not_in_contact = np.setdiff1d(np.arange(num_rods),rods_in_contact)
        
        # nodes_at_a_time_matrix = nodes_over_time[frame,:].reshape(num_rods,-1)
        # if len(rods_in_contact) > 0:
        #     plot_many_curves(nodes_at_a_time_matrix[rods_in_contact,:].flatten(),len(rods_in_contact),ax,params={'color':'k','alpha':0.2})
        # if len(rods_not_in_contact) > 0:
        #     plot_many_curves(nodes_at_a_time_matrix[rods_not_in_contact,:].flatten(),len(rods_not_in_contact),ax)
        # cluster_size_list.append(len(rods_in_contact))
    
        ax.set_title(title_string,fontsize=10)        
        ax.text2D(0.05, 0.95, f't={timepoints[frame]}', transform=ax.transAxes)
        ax.set_xlim(-parsed_info["rod_length"]/zoom,parsed_info["rod_length"]/zoom)
        ax.set_ylim(-parsed_info["rod_length"]/zoom,parsed_info["rod_length"]/zoom)
        ax.set_zlim(-parsed_info["rod_length"]/zoom,parsed_info["rod_length"]/zoom)        
        return ax
    
    ani = animation.FuncAnimation(fig=fig, func=update, frames=np.arange(1,nodes_over_time.shape[0],1), interval=30, )    
    FFwriter = animation.FFMpegWriter(fps=10)
    outpath = f'/Users/yeonsu/Videos/{parsed_info["batch_id"]}/{parsed_info["date_time"]}'
    if not os.path.exists(outpath):
        os.makedirs(outpath)        
    ani.save(f'{outpath}/{parsed_info["sim_id"]}_zoom{zoom}_dt{dt}.mp4', writer = FFwriter)
    # close figure
    plt.close()
    return np.array(cluster_size_list)

        
def create_animation(pth,nodes_over_time,timepoints,zoom):
    parsed_info = parse_filename(pth)
    num_rods = parsed_info['num_rods']
    dt = timepoints[1] - timepoints[0] # assuming uniform time steps
    title_string = ''
    tokens = parsed_info['sim_id'].split('_')[0].split('-')
    for token in tokens:
        if 'N' in token:
            title_string += f'{token}_'
        if 'AR' in token:
            title_string += f'{token}_'
        if 'mu' in token:
            title_string += f'{token}_'
        if 'visc' in token:
            title_string += f'{token}_'
        if 'amp' in token:
            title_string += f'{token}'
    num_frames = nodes_over_time.shape[0]
    cluster_size_list = []
    
    fig,ax=set_3d_plot()
    plot_many_curves(nodes_over_time[0,:],num_rods,ax)
    def update(frame):
        ax.clear()        
        plot_many_curves(nodes_over_time[frame,:],num_rods,ax)
        ax.set_title(title_string,fontsize=10)        
        ax.text2D(0.05, 0.95, f't={timepoints[frame]}', transform=ax.transAxes)
        ax.set_xlim(-parsed_info["rod_length"]/zoom,parsed_info["rod_length"]/zoom)
        ax.set_ylim(-parsed_info["rod_length"]/zoom,parsed_info["rod_length"]/zoom)
        ax.set_zlim(-parsed_info["rod_length"]/zoom,parsed_info["rod_length"]/zoom)        
        return ax
    
    ani = animation.FuncAnimation(fig=fig, func=update, frames=np.arange(0,nodes_over_time.shape[0],1), interval=30, )    
    FFwriter = animation.FFMpegWriter(fps=10)
    outpath = f'/Users/yeonsu/Videos/{parsed_info["batch_id"]}/{parsed_info["date_time"]}'
    if not os.path.exists(outpath):
        os.makedirs(outpath)        
    ani.save(f'{outpath}/{parsed_info["sim_id"]}_zoom{zoom}_dt{dt}.mp4', writer = FFwriter)
    return 0
    
def create_animation_images(pth,nodes_over_time,timepoints,zoom=1,offset=[0,0,0],params={}):
    parsed_info = parse_filename(pth)
    num_rods = parsed_info['num_rods']
    dt = timepoints[1] - timepoints[0] # assuming uniform time steps
    title_string = ''
    tokens = parsed_info['sim_id'].split('_')[0].split('-')
    for token in tokens:
        if 'N' in token:
            title_string += f'{token}_'
        if 'AR' in token:
            title_string += f'{token}_'
        if 'mu' in token:
            title_string += f'{token}_'
        if 'visc' in token:
            title_string += f'{token}_'
        if 'amp' in token:
            title_string += f'{token}'    
    num_frames = nodes_over_time.shape[0]
        
    cluster_size_list = []
    
    def create_folder_with_numbering(outpath):
        if not os.path.exists(outpath):
            os.makedirs(outpath)
        else:
            # Regular expression to match the base path and the numbering
            match = re.match(r"^(.*?)(?: \(([0-9]+)\))?$", outpath)
            base_path = match.group(1)
            index = 1 if match.group(2) is None else int(match.group(2)) + 1
            
            # Find the next available numbered folder
            while os.path.exists(f"{base_path} ({index})"):
                index += 1
            
            # Create the new folder with the next available number
            os.makedirs(f"{base_path} ({index})")
            outpath = f"{base_path} ({index})"
        return outpath
    
    outpath = f'/Users/yeonsu/Videos/{parsed_info["sim_id"]}'
    outpath = create_folder_with_numbering(outpath)
        
    fig,ax=set_3d_plot()
    for frame in range(num_frames):
        plot_many_curves(nodes_over_time[frame,:],num_rods,ax,params=params)
        ax.set_title(title_string,fontsize=10)        
        ax.text2D(0.05, 0.95, f't={timepoints[frame]}', transform=ax.transAxes)
        ax.set_xlim((-parsed_info["rod_length"])/zoom+offset[0],(parsed_info["rod_length"])/zoom+offset[0])
        ax.set_ylim((-parsed_info["rod_length"])/zoom+offset[1],(parsed_info["rod_length"])/zoom+offset[1])
        ax.set_zlim((-parsed_info["rod_length"])/zoom+offset[2],(parsed_info["rod_length"])/zoom+offset[2])
        plt.savefig(f'{outpath}/frame{frame:03d}.png',dpi=300)
        ax.clear()
    
    return 0
    
def parse_filename(pth):
    sim_id = pth.split('/')[-1].split('.csv')[0]    
    date_time = 0
    batch_id = 0
    
    tokens = pth.split('/')[:-1]
    
    for token in tokens:
        if re.match(r'^\d+-\d+$',token):            
            date_time = token        
        if re.match(r'^[A-Za-z]+,$',token):            
            batch_id = token
        
    tmp = pth.split('_node')[0]
    tmp = tmp.split('/')[-1].split('.csv')[0].split('-')
    tmp = tmp[1:]
    num_rods = [int(i.split('N')[-1]) for i in tmp if 'N' in i][0]
    AR = [float(i.split('AR')[-1]) for i in tmp if 'AR' in i][0]
    rod_length = [float(i.split('Scale')[-1]) for i in tmp if 'Scale' in i][0]        
    rod_radius = rod_length/AR/2
    
    parsed_info = {'pth': pth,
                   'sim_id': sim_id,
                   'num_rods': num_rods,
                   'rod_radius': rod_radius,
                   'AR': AR,
                   'rod_length': rod_length,                   
                   'batch_id': batch_id,
                   'date_time': date_time}
    
    return parsed_info

def analyze_single_data(pth):    
    parsed_info = parse_filename(pth)
    dta = np.loadtxt(pth,delimiter=',')
    start_column=1
    max_rows=1000000
    row_skip=1
    zoom = 1
    
    if len(dta.shape) == 1:
        inspect_packing(pth)
    elif len(dta.shape) == 2:
        inspect_dismech_nodes(pth,zoom,start_column=start_column,max_rows=max_rows,row_skip=row_skip)
        
def analyze_batch_data(pth):
    t_start = time.time()
    for fname in glob.glob(pth):
        parsed_info = parse_filename(fname)
        print(f"Analyzing: {parsed_info['sim_id']}")
        print(f"Elapsed time for single analysis: {time.time() - t_start}")
        analyze_single_data(fname)
        
    t_elapsed = time.time() - t_start
    print(f"Elapsed time: {t_elapsed}")
        
def calculate_rod_correlation(nodes_at_a_time,num_rods,num_vertices):
    # nodes_at_a_time
    # given a set of nodes, calculate the correlation between rods
    # nodes_at_a_time: (num_rods*num_vertices*3,) array
    
    # TO DO: which is better; reshape or indexing?
    
    
    # definition of correlation: 
    
    rod_correlation = 1;
    return rod_correlation
    
# @numba.jit(nopython=True)
def calculate_correlation(pos_pairs1,pos_pairs2,vel_pairs1,vel_pairs2,num_pairs):
    # pos_pairs1, pos_pairs2: (num_pairs,6) arrays
    # vel_pairs1, vel_pairs2: (num_pairs,6) arrays
    # to do: compare with a nested loop implementation    
    correlations = np.zeros((num_pairs,2))
    i = 0
    for (pos1,pos2,vel1,vel2) in zip(pos_pairs1,pos_pairs2,vel_pairs1,vel_pairs2):
        d_pos = np.linalg.norm(pos1-pos2)
        d_vel = np.linalg.norm(vel1-vel2)
        correlations[i,0] = d_pos
        correlations[i,1] = d_vel
        i = i + 1
    return correlations

@numba.jit(nopython=True)
def calculate_correlation2(nodes_at_a_time,node_velocities, num_rods):    
    corr_matrix = np.zeros((num_rods,num_rods))
    for i in range(num_rods):
        pos1 = nodes_at_a_time[i,:].reshape(-1,3)
        vel1 = node_velocities[i,:].reshape(-1,3)
        
        for j in range(i+1,num_rods):            
            pos2 = nodes_at_a_time[j,:].reshape(-1,3)
            vel2 = node_velocities[j,:].reshape(-1,3)
            
            corr = 0
            for k in range(0,num_vertices):
                pos_ik = pos1[k,:]    
                vel_ij = vel1[k,:]
                for l in range(0,num_vertices):
                    pos_jl = pos2[l,:]
                    vel_jl = vel2[l,:]
                    corr += jnp.linalg.norm(pos_ik-pos_jl) + jnp.linalg.norm(vel_ij-vel_jl)
                    
                # store d_pos and d_vel
            corr_matrix[i,j] = corr
    
    return corr

def analyze_correlation(nodes_at_a_time,nodes_at_next_time, num_rods):
    # actually needs to consider dt
    node_velocity = nodes_at_next_time - nodes_at_a_time    
    num_vertices = node_velocity.shape[0]//3//num_rods
    
    node_velocity_matrix = node_velocity.reshape(num_rods,-1) # curves are num_rods x num_vertices x 3 array
    
    position_pairs1,position_pairs2 = create_curve_pairs(nodes_at_a_time.reshape(num_rods,-1))
    velocity_pairs1,velocity_pairs2 = create_curve_pairs(node_velocity_matrix)
    
    d = all_distances_between_curves2(position_pairs1,position_pairs2)
    print(f"Min distance: {jnp.min(d)}")
    
    rod_radius = parsed_info['rod_radius']
    print(np.count_nonzero(d < 2.1*rod_radius))
    
    plt.hist(d,bins=100)
    
    # start_time = time.time()
    # correlations = calculate_correlation(position_pairs1,position_pairs2,velocity_pairs1,velocity_pairs2,position_pairs1.shape[0])
    # print(f"Elapsed time: {time.time()-start_time} seconds")
    
    # plt.plot(correlations[:,0],correlations[:,1],'.')
    
    return correlations

def find_contact_cluster(nodes_over_time):
    
    # num_rods = parsed_info['num_rods']                    
    # nodes_at_a_time = nodes_over_time[frame,:]
    # nodes_at_a_time_matrix = nodes_at_a_time.reshape(num_rods,-1)
    # position_pairs1,position_pairs2,i,j = create_curve_pairs(nodes_at_a_time_matrix)
    # d = all_distances_between_curves2(position_pairs1,position_pairs2)
    # i_contact = i[d < 2.1*parsed_info['rod_radius']]
    # j_contact = j[d < 2.1*parsed_info['rod_radius']]        
    # rods_in_contact = np.unique(np.hstack([i_contact,j_contact]))
    # rods_not_in_contact = np.setdiff1d(np.arange(num_rods),rods_in_contact)
        
    return 0

def cluster_analysis(pth):
    print(pth)
    
    parsed_info = parse_filename(pth)
    dta = np.loadtxt(pth,delimiter=',')
    start_column=1
    max_rows=1000000
    row_skip=100
    zoom = 1
    
    assert(len(dta.shape) >1)
    
    parsed_info = parse_filename(pth)
    
    num_rods = parsed_info["num_rods"]
    nodes_over_time, timepoints, num_vertices = import_from_dismech_hook(pth,num_rods,start_col = start_column, max_rows = max_rows, row_skip=row_skip)            
    
    def get_pairwise_distances(curves):
        pairs1,pairs2,i,j = create_curve_pairs(curves)
        d = all_distances_between_curves2(pairs1,pairs2)
        return d,i,j
    
    start_time = time.time()
    rod_radius = parsed_info['rod_radius']
    
    num_frames = nodes_over_time.shape[0]
    cluster_size_over_time = np.zeros(num_frames)
    
    logfiledir = f'/Users/yeonsu/Data/analysis/{parsed_info["batch_id"]}/{parsed_info["date_time"]}'
    if not os.path.exists(logfiledir):
        os.makedirs(logfiledir) 
    
    cluster_size_logfile = f'{logfiledir}/{parsed_info["sim_id"]}_largest_cluster.txt'
    with open(cluster_size_logfile,'w') as f:
    
        for frame in range(num_frames):
            d,idx,jdx = get_pairwise_distances(nodes_over_time[frame,:].reshape(num_rods,-1))
            
            d = np.array(d)
            idx = np.array(idx)
            jdx = np.array(jdx)    
            adj_values = np.where(d < 2.1*rod_radius)
                
            idx_adj = idx[adj_values]
            jdx_adj = jdx[adj_values]
            input_list = []    
            for ii in range(idx_adj.shape[0]):
                input_list.append((idx_adj[ii],jdx_adj[ii]))
            
            graph = nx.Graph(input_list)    
            n_conncomp = nx.number_connected_components(graph)
            
            largest_cc = []
            if n_conncomp > 0:
                largest_cc = max(nx.connected_components(graph), key=len)            
                cluster_size_over_time[frame] = len(largest_cc)
                
            f.write(f'{timepoints[frame]},')
            for rod in largest_cc:
                f.write(f'{rod},')
            f.write('\n')
            
    # print(f"Elapsed time: {time.time()-start_time}")
    
    fig,ax = plt.subplots(figsize=(4,3))
    ax.plot(timepoints,cluster_size_over_time)
    ax.set_xlabel('Time (sec)')
    ax.set_ylabel('Number of rods in the cluster')
    
    figure_outdir = f"/Users/yeonsu/Figures/{parsed_info['batch_id']}/{parsed_info['date_time']}"
    if not os.path.exists(figure_outdir):
        os.makedirs(figure_outdir)
    plt.savefig(f'{figure_outdir}/{parsed_info["sim_id"]}_cluster_size.png',dpi=300)
    
    return cluster_size_over_time,largest_cc
    
def main():
    
    # pth = '/Users/yeonsu/Documents/GitHub/entanglement-optimization/DataFromCluster/Jesse_20240506-0123/**/*.csv'
    # analyze_batch_data(pth)
    
    
    return 1

def seg_len(seg):
    return np.sum(np.sqrt(np.sum(np.diff(seg,axis=0)**2,axis=1)))
    
def last_frame_analysis(last_frame):
    new_cl = []
    for i in range(last_frame.shape[0]):
        rr = last_frame[i,:].reshape(-1,3)
        rr_len = seg_len(rr)
        if np.isnan(rr).any():            
            pass
        if rr_len < 1.5:
            new_cl.append(rr)
            
    flattend_again = []
    for rr in new_cl:
        flattend_again.append(rr.flatten())
    
    flattend_again = np.array(flattend_again)
    return flattend_again


def check_last_frame():
    dta = np.loadtxt(pth,delimiter=',',max_rows=10000)
    print(f'Number of rows: {dta.shape[0]}')
    print(f'Number of columns: {dta.shape[1]}')
    
    num_rods = (dta.shape[1] - 1)//30
    
    last_frame = dta[-1,1:].reshape(num_rods,-1)
    
    to_export = last_frame_analysis(last_frame)
    N = to_export.shape[0]
    
    # nan check
    fig,ax=plt.subplots(subplot_kw={'projection':'3d'})
    min_z = 1e10
    for i in range(N):
        rr = to_export[i,:].reshape(-1,3)
        ax.plot(rr[:,0],rr[:,1],rr[:,2])
        
        if np.min(rr[:,2]) < min_z:
            min_z = np.min(rr[:,2])
            i_z = i
        
    
    deleted = np.delete(to_export,i_z,axis=0)
    deleted.shape
    
    fig,ax=plt.subplots(subplot_kw={'projection':'3d'})
    for i in range(N-1):
        rr = deleted[i,:].reshape(-1,3)
        ax.plot(rr[:,0],rr[:,1],rr[:,2])
        
    np.savetxt(f'prunedCenterlines-N{N-1}-AR200-Scale1.txt',deleted)   
    
def do_visualize():
    
    def create_connectivity(lengths):
        edges = []
        start = 0

        for length in lengths:
            for i in range(start, start + length - 1):
                edges.append([i, i+1])
            start += length

        return np.array(edges)        
    
    edges = create_connectivity(lengths)

    all_nodes = nodes.reshape(-1,3)
    # permute 213
    all_nodes = all_nodes[:,[0,2,1]]
    
    import polyscope as ps
    ps.init()    
    # ps.register_point_cloud("rod1",rod1)
    ps_net = ps.register_curve_network("my network", all_nodes,edges)
    ps.show()   
# %%
def main():
    pth = '/Users/yeonsu/Data/from-cluster/NonIntersectingBox-N250-AR50-Scale1-mu0.20-visc0.00-amp0.00_allLog_20240528-002014.csv'
    parsed_info = parse_filename(pth)
    start_column = 1
    max_rows = 10000000
    row_skip = 1
    for key in parsed_info.keys():
        print(f'{key}: {parsed_info[key]}')
    
    file_id,surfix,num_rods = parse_path_string(pth)
    time_line, node_list, contact_list = import_all_log(pth)
        
    centerlines = node_list[-1]
    centerlines = centerlines.reshape(num_rods,-1,3)
    
    point = np.array([0,0,0])
    R = 0.1
    h = 0.01
    rod_diameter = 0.01
    rod_length = 1
    
    get_local_fields_at_a_point(centerlines, point, R, h, rod_diameter, rod_length)
    
def compute3DFields(centerlines):
    
    R_omega = (1*1/50)**0.5
    h_omega = R_omega/10
    n_field,phi_field,e_field,S_field,cx,cy,cz = get_local_fields_over_domain(centerlines, R_omega,h_omega, 1, 1/50.)
    
    import polyscope as ps
    ps.init()

    dims = (n_field.shape[0], n_field.shape[1], n_field.shape[2])
    bound_low = (cx[0], cz[0], cy[0])
    bound_high = (cx[-1], cz[-1], cy[-1])

    ps_grid = ps.register_volume_grid("sample grid", dims, bound_low, bound_high)
    ps_grid.add_scalar_quantity("n_field", e_field, defined_on='nodes', enabled=True)
    
    ps.show()
    
def testLk():
    p1 = np.array([-100.,0,0])
    p2 = np.array([100.,0,0])
    p3 = np.array([0,-100.,1.])
    p4 = np.array([0,100.,1.])
    
    lk = compute_linking_number(p1,p2,p3,p4)
    print(f'Linking number: {lk}')
    fig,ax=plt.subplots(subplot_kw={'projection':'3d'})
    ax.plot([p1[0],p2[0]],[p1[1],p2[1]],[p1[2],p2[2]],'k')
    ax.plot([p3[0],p4[0]],[p3[1],p4[1]],[p3[2],p4[2]],'r')
    ax.axis('equal')
    
def plot_closest_points(contact_entry,curr_nodes):
    contact_ij = curr_force_all_info[:,4:6].astype(int)
    contact_ij_next_frame = next_force_all_info[:,4:6].astype(int)            
    graph = nx.Graph()
    graph.add_nodes_from(range(len(curr_nodes)))
    graph.add_edges_from(contact_ij)
    
    graph[10]
    
    fig,ax=plt.subplots(1,1,figsize=(10,10),subplot_kw={'projection':'3d'})
    hub_rod_label = 10
    hub_rod = curr_nodes[hub_rod_label]
    hub_rod_next = next_nodes[hub_rod_label]
    ax.plot(hub_rod[:,0],hub_rod[:,1],hub_rod[:,2],'k',linewidth=1)
    ax.plot(hub_rod_next[:,0],hub_rod_next[:,1],hub_rod_next[:,2],'r',linewidth=1)
    
    scale_factor = 100
    velocity_of_hub_rod = hub_rod_next - hub_rod
    velocity_of_hub_rod *= scale_factor
    ax.quiver(hub_rod[0,0],hub_rod[0,1],hub_rod[0,2],velocity_of_hub_rod[0,0],velocity_of_hub_rod[0,1],velocity_of_hub_rod[0,2],color='b')
    
    neighbors_of_hub = list(graph[hub_rod_label])
    for neighbors in neighbors_of_hub:
        rod = curr_nodes[neighbors]
        rod_next = next_nodes[neighbors]
        rod_velolcity = rod_next - rod
        rod_velolcity *= scale_factor
        ax.plot(rod[:,0],rod[:,1],rod[:,2],'r',linewidth=0.2)
        ax.plot(rod_next[:,0],rod_next[:,1],rod_next[:,2],'r',linewidth=0.2)
        ax.quiver(rod[:,0],rod[:,1],rod[:,2],rod_velolcity[:,0],rod_velolcity[:,1],rod_velolcity[:,2],color='b')
    ax.view_init(0,0)
    plt.show()
    
def get_closest_points(contact_entry,curr_nodes):
    rodlabel_i = int(contact_entry[4])
    rodlabel_j = int(contact_entry[5])
    
    ind_i1 = int(contact_entry[0])
    ind_i2 = int(contact_entry[2])
    ind_j1 = int(contact_entry[1])
    ind_j2 = int(contact_entry[3])
    
    x_i1 = curr_nodes[rodlabel_i][ind_i1]
    x_i2 = curr_nodes[rodlabel_i][ind_i2]
    x_j1 = curr_nodes[rodlabel_j][ind_j1]
    x_j2 = curr_nodes[rodlabel_j][ind_j2]
    
    t,u,d1,d2,d12 = lumelsky_dist_vec(x_i1,x_i2,x_j1,x_j2)
    popt_i = x_i1 + d1*t
    popt_j = x_j1 + d2*u
    dvec = d1*t - d2*u - d12
    
    return popt_i,popt_j,dvec,x_i1,x_i2,x_j1,x_j2
    
def analyze_csv_file(data_path,skip_frames):
    
    
    log_string = ''
    file_id,surfix,num_rods,AR,datetime_string = parse_path_string(data_path)
    time_line, node_list, contact_list = import_all_log(data_path,max_rows=100000)

    time_line = np.array(time_line)
    time_line = time_line[time_line <= 10]
    node_list = node_list[:len(time_line)]
    contact_list = contact_list[:len(time_line)]

    time_line = time_line[1:]
    node_list = node_list[1:]
    contact_list = contact_list[1:]

    print(f'Size of time_line: {len(time_line)}')
    print(f'Number of rods: {num_rods}')

    log_string = log_string + f'Number of rods: {num_rods}\n'
    log_string = log_string + f'Number of time points: {len(time_line)}\n'

    total_number_of_contacts = np.zeros(len(time_line))
    total_force_sum = np.zeros(len(time_line))

    last_frame = len(time_line)-1
    print(f'Last frame: {last_frame}')
    
    initial_nodes = node_list[0].reshape((-1,10,3))        
    timeline_checkout = range(0,len(time_line)-1,skip_frames)
    avg_velocities_over_time = np.zeros( len(timeline_checkout) )
    centroid_velocities_over_time = np.zeros( len(timeline_checkout) )
    avg_contact_displacement_over_time = np.zeros( len(timeline_checkout) )
    avg_initial_centroid_displacement_over_time = np.zeros( len(timeline_checkout) )        
    fraction_of_nodes_in_largest_cluster_over_time = np.zeros( len(timeline_checkout) )
    lk_mat_over_time = []
    
    fF = filamentFields.filamentFields([],[])
    
    for i_frame,frame in enumerate(timeline_checkout):
        
        curr_nodes = node_list[frame].reshape((-1,10,3))
        next_nodes = node_list[frame+1].reshape((-1,10,3))
        curr_force_all_info = contact_list[frame].reshape(-1,18)
        next_force_all_info = contact_list[frame+1].reshape(-1,18)
        
        initial_node_displacement = curr_nodes - initial_nodes
        avg_initial_centroid_displacement =  np.mean(np.linalg.norm(np.mean(initial_node_displacement,axis=1),axis=1))
        avg_initial_centroid_displacement_over_time[i_frame] = avg_initial_centroid_displacement
        
        contact_displacement_list = np.zeros(len(curr_force_all_info))
        
        contact_ij = curr_force_all_info[:,4:6].astype(int)
        # contact_ij_next_frame = next_force_all_info[:,4:6].astype(int)            
        graph = nx.Graph()
        graph.add_nodes_from(range(len(curr_nodes)))
        graph.add_edges_from(contact_ij)
        clusters = list(nx.connected_components(graph))
        
        # largest clusters
        largest_cluster = max(clusters,key=len)
        fraction_of_nodes_in_largest_cluster = len(largest_cluster)/len(curr_nodes)
        fraction_of_nodes_in_largest_cluster_over_time[i_frame] = fraction_of_nodes_in_largest_cluster
        
        fF.update_filament_nodes_list(curr_nodes)
        fF.precompute(1000) # arbitrarily large number
        fF.compute_filament_linking_matrix()
        lk_mat = fF.return_filament_linking_matrix()        
        lk_mat_over_time.append(lk_mat)
        
        for i_,contact_entry in enumerate(curr_force_all_info):
            popt_i,popt_j,dvec,x_i1,x_i2,x_j1,x_j2 = get_closest_points(contact_entry,curr_nodes)
            popt_i_next,popt_j_next,dvec_next,x_i1_next,x_i2_next,x_j1_next,x_j2_next = get_closest_points(contact_entry,next_nodes)
            contact_displacement = dvec_next - dvec
            contact_displacement_norm = np.linalg.norm(contact_displacement)
            contact_displacement_list[i_] = contact_displacement_norm
            
        average_contact_displacement = np.mean(contact_displacement_list)
        avg_contact_displacement_over_time[i_frame] = average_contact_displacement
        
        # node velocities
        rod_velocities = np.zeros((num_rods,10,3))
        for i_rod in range(0,num_rods,1):
            curr_rod = curr_nodes[i_rod]
            next_rod = next_nodes[i_rod]
            rod_velocities[i_rod] = next_rod - curr_rod
            
        avg_velocity_at_the_frame = np.mean(np.linalg.norm(rod_velocities.reshape(-1,3),axis=1))
        avg_velocities_over_time[i_frame] = avg_velocity_at_the_frame
        
        centroid_velocities_at_the_frame = np.mean(rod_velocities,axis=1)
        centroid_velocities_over_time[i_frame] = np.mean(np.linalg.norm(centroid_velocities_at_the_frame,axis=1))
        
    output_dict = { 'data_path': data_path,
                    'num_rods': num_rods,
                    'AR': AR,
                    'actual_timeline': time_line[timeline_checkout],
                    'avg_velocities_over_time': avg_velocities_over_time,
                    'centroid_velocities_over_time': centroid_velocities_over_time,
                    'avg_contact_displacement_over_time': avg_contact_displacement_over_time,
                    'avg_initial_centroid_displacement_over_time': avg_initial_centroid_displacement_over_time,
                    'lk_mat_over_time': lk_mat_over_time,
                    'fraction_of_nodes_in_largest_cluster_over_time': fraction_of_nodes_in_largest_cluster_over_time}
    
    return output_dict
    
# %%
if __name__ == '__main__':
    import pickle
    import filamentFields
    from distances import lumelsky_dist_vec
    from pathlib import Path
    
    parent_folders = []
    parent_folders.append(Path('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/HangModelo3'))
    # parent_folders.append(Path('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/HangModelo1'))
    # parent_folders.append(Path('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/Modelo1_SlowExcitation'))    
    
    # parent_folders.append(Path('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/Modelo1_FineExcitation'))
    # parent_folders.append(Path('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/Modelo3_FineExcitation'))
    # parent_folders.append(Path('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/Modelo2_FineExcitation'))
    
    output_root = '/Users/yeonsu/Dropbox (Harvard University)/Data/PrunedData/rod-sim-pnas-revision'
    analysis_id = 'Micromechanics-HangModelo3'
    
    output_path = f'{output_root}/{analysis_id}'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    output_dict_list_repeated = []
    for parent_folder in parent_folders:
        pathlist = [str(x) for x in parent_folder.iterdir() if x.is_dir()]
        
        ARs = []
        for pth in pathlist:
            search_result = re.search(r'N(\d+)_AR(\d+)',pth)
            ARs.append(int(search_result.group(2)))

        pathlist = [x for _,x in sorted(zip(ARs,pathlist))]
        ARs = sorted(ARs)
        
        # find csv file
        
        data_path_list = []
        for pth in pathlist:        
            data_path = None
            for file in Path(pth).rglob('*.csv'):
                if str(file.stem).endswith('lastFrame'):
                    continue        
                data_path = file
                data_path_list.append(data_path)
                break
        
        skip_frames = 30
        
        ##### function implementation below
        output_dict_list = []
        for data_path in data_path_list:
            output_dict = analyze_csv_file(data_path,skip_frames)
            output_dict_list.append(output_dict)
            
            
        output_dict_list_repeated.append(output_dict_list)
        
        with open(f'{output_path}/output_dict_list_repeated.pkl','wb') as f:
            pickle.dump(output_dict_list_repeated,f)
        
    
    
