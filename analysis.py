import numpy as np
import jax.numpy as jnp
import jax
from potentials import create_pairs, all_pairwise_distances
from matplotlib import pyplot as plt
from visualizations import plot_many_rods, plot_contacts, set_3d_plot

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
    # make it faster?
    for i in range(num_rods):
        # nnz
        contact_degrees[i] = np.count_nonzero(contacts[:,0] == i) + np.count_nonzero(contacts[:,1] == i)
        neighbors.append(np.concatenate([contacts[contacts[:,0] == i,1], contacts[contacts[:,1] == i,0]]))
    
    print('Number of contacts: ', num_contacts)
    print('Average number of contacts per rod: ', avg_num_contacts_per_rod)
    print('Avg. contact degrees: ', np.mean(contact_degrees))
    
    fig,ax = set_3d_plot()
    plot_contacts(q,0,neighbors[0])
    
    return 1

    
def read_data(pth):
    q = np.loadtxt(pth)
    q = jnp.array(q,dtype=jnp.float64)
    return q

if __name__ == '__main__':
    pth = '/Users/yeonsu/Data/entangled_rods_N300_relaxed_22-04-2024_00-36-18.txt'
    q = read_data(pth)    
    rod_radius = 0.08 # TO DO: read from file
    
    contacts = find_contacts(q,rod_radius)
    