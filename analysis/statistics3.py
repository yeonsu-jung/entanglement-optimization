import pickle
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import jax.numpy as jnp
from potentials import acn_over_ij

def get_clusters(contact_ij,num_rods):
    import networkx as nx
    G = nx.Graph()
    G.add_nodes_from(range(num_rods))
    G.add_edges_from(contact_ij)
    clusters = list(nx.connected_components(G))
    num_clusters = len(clusters)
    cluster_sizes = [len(cluster) for cluster in clusters]
    cluster_sizes = np.array(cluster_sizes)
    max_cluster_size = np.max(cluster_sizes)
    return clusters, num_clusters, cluster_sizes, max_cluster_size

# Data class
class SingleKickData:
    def __init__(self, dt_string, AR, num_rods, kick_amplitude, friction_coefficient, data_path):
        self.dt_string = dt_string
        self.AR = AR
        self.num_rods = num_rods
        self.kick_amplitude = kick_amplitude
        self.friction_coefficient = friction_coefficient
        self.data_path = data_path

    def __repr__(self):
        return f"dt_string: {self.dt_string}, AR: {self.AR}, num_rods: {self.num_rods}, kick_amplitude: {self.kick_amplitude}, friction_coefficient: {self.friction_coefficient}\n"
    
def analyze_alpha500():
    t_u = np.sqrt( np.sqrt(2) - 1) / 2
    # Load the list of experiments
    with open('higher_data_list.pkl','rb') as f:
        higher_data_list = pickle.load(f)

    # Flatten the list if it is nested
    all_data_list = []
    for each_random_key_data in higher_data_list:
        for dta in each_random_key_data:
            all_data_list.append(dta)
   
    # Choose the data matching our criteria
    AR = 500
    chosen_data = []
    for dta in all_data_list:
        if dta.AR == AR and dta.kick_amplitude == 1. and dta.num_rods == 500 and dta.friction_coefficient == 0.2:
            chosen_data.append(dta)

    # Process each chosen experiment and compute metrics for every time frame
    global_data_list = []
    for dta in chosen_data:
        # data = loadmat(dta.data_path)
        data = loadmat(dta.data_path, simplify_cells=True)
        time_line = data['time_line']
        node_list = data['node_list']
        velocity_list = data['velocity_list']
        contacts_list = data['contact_list'].copy()
        # contacts_list = np.array(contacts_list)

        rad_gyr_list = []
        num_contacts_list = []
        entanglement_list = []
        largest_cluster_list = []
        for i in range(len(node_list)):
            nodes_at_frame = node_list[i]
            # In case contacts_list is empty
            contacts_at_frame = contacts_list[i] if ~np.any(np.isnan(contacts_list[i])) else []
            
            # Reshape to get rod endpoints (assuming shape is (2*num_rods,) per rod)
            xyz = nodes_at_frame.reshape(-1, 6)
            centroids = (xyz[:, :3] + xyz[:, 3:]) / 2
            global_centroid = np.mean(centroids, axis=0)
            moment_arm = centroids - global_centroid
            rad_gyr = np.mean(np.linalg.norm(moment_arm, axis=1))
            rad_gyr_list.append(rad_gyr)
            num_contacts_list.append(len(contacts_at_frame))
            
            # Compute entanglement metric
            r1 = xyz[:, :3]
            r2 = xyz[:, 3:]
            num_rods = r1.shape[0]
            i_indices, j_indices = jnp.triu_indices(num_rods, k=1)
            pairwise_acn = acn_over_ij(r1, r2, i_indices, j_indices)
            entanglement = np.sum(np.abs(pairwise_acn)) / (num_rods*(num_rods-1)/2)
            entanglement_list.append(entanglement)

            # Compute largest cluster size
            if len(contacts_at_frame) == 0:
                contact_ij = []
                max_cluster_size = 0
                largest_cluster_list.append(0)
            else:
                contact_ij = contacts_at_frame.reshape(-1,8)[:,:2].astype(int)
                max_cluster_size = get_clusters(contact_ij,num_rods)[-1]
                largest_cluster_list.append(max_cluster_size/num_rods)        

        local_dataset = {
            'AR': AR,
            'time_line': time_line,
            'data_obj': dta,
            'node_list': node_list,
            'velocity_list': velocity_list,
            'contact_list': contacts_list,
            'rad_gyr_list': rad_gyr_list,
            'num_contacts_list': num_contacts_list,
            'entanglement_list': entanglement_list,
            'largest_cluster_list': largest_cluster_list
        }
        global_data_list.append(local_dataset)

    # --- Average data across experiments ---
    # Assuming all experiments have the same time_line length & points:
    experiments = global_data_list
    time_line = experiments[0]['time_line']

    # Stack the metrics from each experiment (each row corresponds to one experiment)
    rad_gyr_array = np.array([exp['rad_gyr_list'] for exp in experiments])
    num_contacts_array = np.array([exp['num_contacts_list'] for exp in experiments])
    entanglement_array = np.array([exp['entanglement_list'] for exp in experiments])

    # Compute the mean and standard deviation across experiments (axis=0)
    rad_gyr_avg = np.mean(rad_gyr_array, axis=0)
    rad_gyr_std = np.std(rad_gyr_array, axis=0)
    num_contacts_avg = np.mean(num_contacts_array, axis=0)
    num_contacts_std = np.std(num_contacts_array, axis=0)
    entanglement_avg = np.mean(entanglement_array, axis=0)
    entanglement_std = np.std(entanglement_array, axis=0)

    # --- Plot the averaged data ---
    plt.figure(figsize=(8, 6))
    
    # Plot averaged Radius of Gyration with error band
    plt.subplot(4, 1, 1)
    plt.plot(time_line/t_u, rad_gyr_avg, label='Avg RMS dist.', color='b')
    plt.fill_between(time_line/t_u, rad_gyr_avg - rad_gyr_std, rad_gyr_avg + rad_gyr_std, alpha=0.3)
    plt.ylabel('Root mean sqare distance, $R_g$')
    plt.legend()
    
    # Plot averaged Number of Contacts with error band
    plt.subplot(4, 1, 2)
    plt.plot(time_line/t_u, num_contacts_avg, label='Avg Number of Contacts', color='g')
    plt.fill_between(time_line/t_u, num_contacts_avg - num_contacts_std, num_contacts_avg + num_contacts_std, color='g', alpha=0.3)
    plt.ylabel(r'Number of Contacts, $c$')
    plt.legend()
    
    # Plot averaged Entanglement with error band
    plt.subplot(4, 1, 3)
    plt.plot(time_line/t_u, entanglement_avg, label='Avg Entanglement', color='r')
    plt.plot(time_line/t_u, entanglement_array.T)
    plt.fill_between(time_line/t_u, entanglement_avg - entanglement_std, entanglement_avg + entanglement_std, color='r', alpha=0.3)
    # plt.xlabel(r'Normalized time, $t/t_u$')
    plt.ylabel(r'Normalized entanglement, $e$')
    plt.legend()

    # Plot largest cluster size
    plt.subplot(4, 1, 4)
    largest_cluster_array = np.array([exp['largest_cluster_list'] for exp in experiments])
    largest_cluster_avg = np.mean(largest_cluster_array, axis=0)
    largest_cluster_std = np.std(largest_cluster_array, axis=0)
    plt.plot(time_line/t_u, largest_cluster_avg, label='Avg Largest Cluster Size', color='m')
    plt.fill_between(time_line/t_u, largest_cluster_avg - largest_cluster_std, largest_cluster_avg + largest_cluster_std, color='m', alpha=0.3)
    plt.ylabel(r'Normalized largest cluster size')
    plt.xlabel(r'Normalized time, $t/t_u$')
    plt.legend()
    
    plt.suptitle(f'AR: {AR}, num_rods: {num_rods}, kick_amplitude: 1, friction: 0.1')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('alpha500.png')
    plt.show()

if __name__ == '__main__':
    analyze_alpha500()
