import numpy as np
import jax.numpy as jnp
import jax

from jax import jit
from potentials import create_pairs, all_pairwise_distances, dist_lin_seg, all_distances_between_curves2
from matplotlib import pyplot as plt

from visualizations import set_3d_plot, plot_edges, plot_many_curves
from data_io import read_data, import_from_dismech, import_from_dismech_hook
from transforms import q_to_u, q_to_x, x_to_rpairs, x_to_epairs,vert_to_edge

import matplotlib.animation as animation
import sys
import os

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
    
def postprocess():
    sim_id = 'EntangledRelaxedPackingXYZ_node_20240502-220459'
    
    root_dir = '/Users/yeonsu/Data/from-cluster'
    pth = f'{root_dir}/{sim_id}.csv'
    num_rods = 100
    
    from data_io import import_from_dismech, import_from_dismech_hook
    nodes_over_time, timepoints = import_from_dismech_hook(pth,num_rods,start_col = 1,max_rows = 1000000,row_skip=10)
    
    print(f"Shape of nodes_over_time: {nodes_over_time.shape}")
    print(f"Final time point: {timepoints[-1]} sec")
    idx = -1
    
    from visualizations import plot_many_curves,set_3d_plot
    fig,ax = set_3d_plot()
    plot_many_curves(nodes_over_time[idx,:],num_rods,ax)       
    
    # export last dofs
    export_dir = '/Users/yeonsu/Data/export'
    data_to_export = np.reshape(nodes_over_time[idx,:],(-1,30))
    np.savetxt(f'{export_dir}/{sim_id}_last_nodes.txt',data_to_export)    
    plt.show()
    
    
    # q_pairs = create_pairs(jnp.reshape(nodes_over_time[idx,:],(-1,5)))    
    from potentials import create_pairs2, all_distnaces_between_curves2
    
    curves = nodes_over_time[idx,:]
    
    reshaped = curves.reshape(-1,3*10)
    pairs = create_pairs2(reshaped,reshaped)
    pairs1 = pairs[:,:30]
    pairs2 = pairs[:,30:]
    
    d = all_distnaces_between_curves2(pairs1,pairs2)
    
    rod_radius = 2
    print(f"Min distance: {jnp.min(d)}")
    print(f"Number of contacts: {jnp.count_nonzero(d < 2*rod_radius*1.001)}")
    
    fig,ax=set_3d_plot()
    params = {}
    # spatial_data = spatial_data.reshape((-1,num_rods,num_vertices,3))
    
    plot_many_curves(nodes_over_time[0,:],num_rods,params=params,ax=ax)
    
    def update(frame):
        ax.clear()
        print(f"frame: {frame}")        
        plot_many_curves(nodes_over_time[frame,:],num_rods,params=params,ax=ax)
        # plt.axis([-100,100,-100,100,-100,100])
        # axis for 3D plot
        # ax.set_xlim(-100,100)
        # ax.set_ylim(-100,100)
        # ax.set_zlim(-100,100)
        return ax
    
    ani = animation.FuncAnimation(fig=fig, func=update, frames=np.arange(0,nodes_over_time.shape[0],10), interval=30, )
    
    FFwriter = animation.FFMpegWriter(fps=10)
    ani.save(f'/Users/yeonsu/Videos/{sim_id}.mp4', writer = FFwriter)
    
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
        return pairs1, pairs2

def inspect_dismech_nodes(pth,zoom,start_column=1,max_rows=100000,row_skip=1,visualize=0):
    sim_id,num_rods,rod_radius,AR,rod_length,note,batch_id = parse_filename(pth)
    
    nodes_over_time, timepoints, num_vertices = import_from_dismech_hook(pth,num_rods,start_col = start_column, max_rows = max_rows, row_skip=row_skip)    
    dta = np.loadtxt(pth,delimiter=',',max_rows=1)
    
    print(f"Shape of nodes_over_time: {nodes_over_time.shape}")
    print(f"Final time point: {timepoints[-1]} sec")
    print(f"Number of vertices: {num_vertices}")
    
    idx = -1
    # visualize last frame
    if visualize:    
        fig,ax = set_3d_plot()
        plot_many_curves(nodes_over_time[idx,:],num_rods,ax)
        plot_edges(vert_to_edge(dta[1:start_column].reshape(-1,3)),ax=ax,params={'color':'black','alpha':0.5})
        plt.axis('equal')
        plt.show()
    
    # export last dofs
    export_dir = '/Users/yeonsu/Data/export'
    data_to_export = np.reshape(nodes_over_time[idx,:],(-1,3*num_vertices))    
    np.savetxt(f'{export_dir}/{sim_id}_last_nodes.txt',data_to_export)
    
    curves = nodes_over_time[0,:].reshape(num_rods,-1) # curves are num_rods x num_vertices x 3 array
    pairs1,pairs2 = create_curve_pairs(curves)
    d = all_distances_between_curves2(pairs1,pairs2)
        
    print(f"Min distance: {jnp.min(d)}")
    print(f"Number of contacts: {jnp.count_nonzero(d < 2*rod_radius*1.05)}")
    # log file
    if not os.path.exists(f'/Users/yeonsu/Data/analysis/{batch_id}'):
        os.makedirs(f'/Users/yeonsu/Data/analysis/{batch_id}')
    logfilepath = f'/Users/yeonsu/Data/analysis/{batch_id}/{note}_log.txt'
    
    with open(logfilepath,'w') as f:
        f.write(f"File path: {pth}\n")        
        f.write(f"Simulation ID: {sim_id}\n")
        f.write(f"Num. rods: {num_rods}\n")
        f.write(f"Rod radius: {rod_radius}\n")
        f.write(f"Rod length: {rod_length}\n")
        f.write(f"Aspect ratio: {AR}\n")
        f.write(f"Number of contacts: {jnp.count_nonzero(d < 2*rod_radius*1.05)}\n")
        f.write(f"Min distance: {jnp.min(d)}\n")
    
    # save animation
    create_animation(batch_id,sim_id,nodes_over_time,timepoints,zoom,row_skip,num_rods,dta,start_column,rod_length,note)
    
def create_animation(batch_id,sim_id,nodes_over_time,timepoints,zoom,row_skip,num_rods,dta,start_column,rod_length,note):
    fig,ax=set_3d_plot()
    params = {}    
    plot_many_curves(nodes_over_time[0,:],num_rods,params=params,ax=ax)
    plot_edges(vert_to_edge(dta[1:start_column].reshape(-1,3)),ax=ax,params={'color':'black','alpha':0.5})
    # title
    
    def update(frame):
        ax.clear()
        print(f"frame: {frame}")
        plot_many_curves(nodes_over_time[frame,:],num_rods,params=params,ax=ax)
        plot_edges(vert_to_edge(dta[1:start_column].reshape(-1,3)),ax=ax,params={'color':'black','alpha':0.5})
        # (f't={timepoints[frame]}')
        # text
        ax.set_title(f'{note}',fontsize=10)
        ax.text2D(0.05, 0.95, f't={timepoints[frame]}', transform=ax.transAxes)
        
        # plot_edges(vert_to_edge(dta[1:start_column].reshape(-1,3)),ax=ax,params={'color':'black','alpha':0.5})
        # plt.axis([-100,100,-100,100,-100,100])
        # axis for 3D plot
        ax.set_xlim(-rod_length/zoom,rod_length/zoom)
        ax.set_ylim(-rod_length/zoom,rod_length/zoom)
        ax.set_zlim(-rod_length/zoom,rod_length/zoom)
        # ax.view_init(elev=0, azim=90)
        return ax
    
    ani = animation.FuncAnimation(fig=fig, func=update, frames=np.arange(0,nodes_over_time.shape[0],1), interval=30, )    
    FFwriter = animation.FFMpegWriter(fps=10)
    # mkdir
    
    if not os.path.exists(f'/Users/yeonsu/Videos/{batch_id}'):
        os.makedirs(f'/Users/yeonsu/Videos/{batch_id}')
    ani.save(f'/Users/yeonsu/Videos/{batch_id}/{sim_id}_zoom{zoom}_skip{row_skip}.mp4', writer = FFwriter)
    
def parse_filename(pth):
    sim_id = pth.split('/')[-1].split('.csv')[0]
    note = pth.split('/')[-2]
    batch_id = pth.split('/')[-3] # TO DO: make it more general
    
    tmp = pth.split('_node')[0]
    tmp = tmp.split('/')[-1].split('.csv')[0].split('-')
    tmp = tmp[1:]
    num_rods = [int(i.split('N')[-1]) for i in tmp if 'N' in i][0]
    AR = [float(i.split('AR')[-1]) for i in tmp if 'AR' in i][0]
    rod_length = [float(i.split('Scale')[-1]) for i in tmp if 'Scale' in i][0]
        
    rod_radius = rod_length/AR/2
    
    return sim_id,num_rods,rod_radius,AR,rod_length,note, batch_id

def analyze_single_data(pth):    
    sim_id,num_rods,rod_radius,AR,rod_length,note, batch_id = parse_filename(pth)
    print(f"Simulation ID: {sim_id}")
    print(f"Num. rods: {num_rods}")
    print(f"Rod radius: {rod_radius}")
    print(f"Rod length: {rod_length}")
    print(f"Aspect ratio: {AR}")
    
    dta = np.loadtxt(pth,delimiter=',')
    start_column=1
    max_rows=100000
    row_skip=100
    zoom = 1
    
    if len(dta.shape) == 1:
        inspect_packing(pth)
    elif len(dta.shape) == 2:
        inspect_dismech_nodes(pth,zoom,start_column=start_column,max_rows=max_rows,row_skip=row_skip)
    
def main():
    pth = '/Users/yeonsu/Documents/GitHub/entanglement-optimization/DataFromCluster/20240505-1516/**/*.csv'        
    
    for fname in glob.glob(pth):
        sim_id,num_rods,rod_radius,AR,rod_length,note, batch_id = parse_filename(fname)
        analyze_single_data(fname)        
        break
    
    print('done')
    
    # orientational order
    return 1

if __name__ == '__main__':
    main()