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

jax.config.update("jax_enable_x64", True)

def find_general_contacts(q,rod_radius):
    # q is an one dimensional array, having multiple node points.    
    return 1
    

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
    
    fig,ax=plt.subplots(figsize=(4,3))
    ax.plot(timepoints,cluster_size_list)
    ax.set_xlabel('Time')
    ax.set_ylabel('Number of rods in the cluster')
    
    outdir = f'/Users/yeonsu/Figures/{parsed_info["batch_id"]}/{parsed_info["date_time"]}'
    if not os.path.exists(outdir):
        os.makedirs(outdir)    
    plt.savefig(f'{outdir}/{parsed_info["sim_id"]}_cluster_size.png',dpi=300)
    
    dataout = np.vstack([timepoints,cluster_size_list]).T
    np.savetxt(f'{logfiledir}/{parsed_info["sim_id"]}_cluster_size.txt',dataout)
    
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
        # plot_many_curves(nodes_over_time[frame,:],num_rods,ax)
        print(frame)
        curves = nodes_over_time[frame,:].reshape(parsed_info['num_rods'],-1)
        pairs1,pairs2,i,j = create_curve_pairs(curves)
        d = all_distances_between_curves2(pairs1,pairs2)
        rods_in_contact = np.unique(np.vstack([i[d < rod_radius*2.05], j[d < rod_radius*2.05]]))
        rods_not_in_contact = np.setdiff1d(np.arange(num_rods),rods_in_contact)
        
        nodes_at_a_time_matrix = nodes_over_time[frame,:].reshape(num_rods,-1)
        if len(rods_in_contact) > 0:
            plot_many_curves(nodes_at_a_time_matrix[rods_in_contact,:].flatten(),len(rods_in_contact),ax,params={'color':'k','alpha':0.2})
        if len(rods_not_in_contact) > 0:
            plot_many_curves(nodes_at_a_time_matrix[rods_not_in_contact,:].flatten(),len(rods_not_in_contact),ax)
        cluster_size_list.append(len(rods_in_contact))
    
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
    max_rows=5000
    row_skip=100
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
    
def main():
    
    # pth = '/Users/yeonsu/Documents/GitHub/entanglement-optimization/DataFromCluster/Jesse_20240506-0123/**/*.csv'
    # analyze_batch_data(pth)
    
    
    return 1

if __name__ == '__main__':
    batch_root = '/Users/yeonsu/Data/Nacho,/20240506-2217'
    pth = glob.glob(f'{batch_root}/**/*.csv',recursive=True)[0]
    
    # pth = '/Users/yeonsu/Data/Nacho,/20240506-2217/N200_AR200_mu1.0_visc0.0_amp10.0/EntangledRelaxedPackingHook-N200-AR200-Scale1-mu1.00-visc0.00-amp10.0_node_20240506-221710.csv'
    # pth = '/Users/yeonsu/Data/Nacho,/20240506-2217/N200_AR500_mu0.0_visc0.0_amp10.0/EntangledRelaxedPackingHook-N200-AR500-Scale1-mu0.00-visc0.00-amp10.0_node_20240506-221710.csv'    
    
    for pth in glob.glob(f'{batch_root}/**/*.csv',recursive=True):
        analyze_single_data(pth)
        break
    
    # pth = '/Users/yeonsu/Data/from-cluster/EntangledRelaxedPackingHook-N300-AR100-Scale1-mu3.00-visc0.00-amp10.0_node_20240506-151401.csv'
    # analyze_single_data(pth)  