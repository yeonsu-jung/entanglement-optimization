import numpy as np
import jax.numpy as jnp
import jax
from potentials import create_pairs, all_pairwise_distances
from matplotlib import pyplot as plt

from visualizations import plot_many_rods, plot_contacts, set_3d_plot
from data_io import read_data, import_from_dismech

from transforms import q_to_u, q_to_x

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


def main():
    pth = '/Users/yeonsu/Data/from-cluster/20240425-215943_node_20240426-150758.csv'
    filepart = pth.split('/')[-1].split('.')[0]    
    num_rods = 100
    curves, timepoints = import_from_dismech(pth,num_rods)
    
    last_curve = curves[-1,:]
    last_curve = last_curve.reshape((-1,30))
    
    print(last_curve)
    export_dir = '/Users/yeonsu/Data/export'
    np.savetxt(f'{export_dir}/{filepart}_last_nodes.txt',last_curve)
    
    from visualizations import plot_many_curves
    
        
    print(last_curve)
    export_dir = '/Users/yeonsu/Data/export'
    np.savetxt(f'{export_dir}/{filepart}_last_nodes.txt',last_curve)
    
    from visualizations import plot_many_curves    
    
    nodes_at_a_time = curves[-1,:]
    print(nodes_at_a_time.shape)
    num_vertices = curves.shape[1]//3//num_rods
    
    nodes_in_matrix = nodes_at_a_time.reshape((num_rods,-1,3))
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    plot_many_curves(nodes_in_matrix,ax)
    plt.show()
    
    # np.savetxt('/Users/yeonsu/Data/export/last_curve.txt',last_curve)
    
    # curve = curves[0,:]
    # num_vertices = curve.shape[0]//3//num_rods    
    # print(num_vertices)    
    # S = calculate_oreintational_order(q)
    
    return 1

def find_curve_contact(all_nodes,num_rods,rod_radius):
    
    return 1

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
    sim_id = '20240426-215217_node_20240427-014524'
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

if __name__ == '__main__':
    # distance_check()
    main3()