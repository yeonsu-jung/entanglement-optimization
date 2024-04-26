import numpy as np
import jax.numpy as jnp
import jax
from potentials import create_pairs, all_pairwise_distances
from matplotlib import pyplot as plt

from visualizations import plot_many_rods, plot_contacts, set_3d_plot
from data_io import read_data, import_from_dismech

from transforms import q_to_u, q_to_x

jax.config.update("jax_enable_x64", True)

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
    pth = '/Users/yeonsu/Data/from-cluster/20240425-215943_node_20240426-014535.csv'
    filepart = pth.split('/')[-1].split('.')[0]    
    num_rods = 100
    curves, timepoints = import_from_dismech(pth,num_rods)
    
    last_curve = curves[-1,:]
    last_curve = last_curve.reshape((-1,30))
    
    print(last_curve)
    export_dir = '/Users/yeonsu/Data/export'
    np.savetxt(f'{export_dir}/{filepart}_last_nodes.txt',last_curve)
    
    # np.savetxt('/Users/yeonsu/Data/export/last_curve.txt',last_curve)
    
    # curve = curves[0,:]
    # num_vertices = curve.shape[0]//3//num_rods
    
    # print(num_vertices)
    
    # S = calculate_oreintational_order(q)
    
    return 1

def find_curve_contact(all_nodes,num_rods,rod_radius):
    
    return 1


if __name__ == '__main__':
    main()
    