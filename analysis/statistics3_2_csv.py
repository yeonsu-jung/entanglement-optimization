import pickle
import numpy as np
import pandas as pd
from scipy.io import loadmat
import matplotlib.pyplot as plt
import jax.numpy as jnp
from potentials import acn_over_ij

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
        return (f"dt_string: {self.dt_string}, AR: {self.AR}, num_rods: {self.num_rods}, "
                f"kick_amplitude: {self.kick_amplitude}, friction_coefficient: {self.friction_coefficient}\n")
    
def analyze_alpha50_to_dataframe():
    # Load the list of experiments
    with open('higher_data_list.pkl','rb') as f:
        higher_data_list = pickle.load(f)

    # Flatten the list if it is nested
    all_data_list = []
    for each_random_key_data in higher_data_list:
        for dta in each_random_key_data:
            all_data_list.append(dta)
   
    # Choose the data matching our criteria
    # AR = 500
    # chosen_data = []
    # for dta in all_data_list:
    #     if dta.AR == AR and dta.kick_amplitude == 0.1 and dta.num_rods == 500 and dta.friction_coefficient == 0.1:
    #         chosen_data.append(dta)
    chosen_data = all_data_list

    # Process each chosen experiment and compute metrics for every time frame
    global_data_list = []
    for dta in chosen_data:
        data = loadmat(dta.data_path, simplify_cells=True)
        time_line = data['time_line']
        node_list = data['node_list']
        velocity_list = data['velocity_list']
        contacts_list = data['contact_list']

        rad_gyr_list = []
        num_contacts_list = []
        entanglement_list = []
        # largest_cluster_size_list is defined but not computed in the current code.
        # It could be computed if you have a method for that.
        largest_cluster_size_list = []  # Placeholder if needed
        
        for i in range(len(node_list)):
            nodes_at_frame = node_list[i]
            # If contacts_list is empty, assign an empty list
            contacts_at_frame = contacts_list[i] if np.any(~np.isnan(contacts_list[i])) else []
            
            # Reshape to get rod endpoints (each row: [x1,y1,z1, x2,y2,z2])
            xyz = nodes_at_frame.reshape(-1, 6)
            centroids = (xyz[:, :3] + xyz[:, 3:]) / 2
            global_centroid = np.mean(centroids, axis=0)
            moment_arm = centroids - global_centroid
            radius_of_gyration = np.mean(np.linalg.norm(moment_arm, axis=1))
            rad_gyr_list.append(radius_of_gyration)
            num_contacts_list.append(len(contacts_at_frame))
            
            # Compute entanglement metric over rod pairs
            r1 = xyz[:, :3]
            r2 = xyz[:, 3:]
            num_rods = r1.shape[0]
            i_indices, j_indices = jnp.triu_indices(num_rods, k=1)
            pairwise_acn = acn_over_ij(r1, r2, i_indices, j_indices)
            entanglement = np.sum(np.abs(pairwise_acn)) / (num_rods*(num_rods-1)/2)
            entanglement_list.append(entanglement)
        
        local_dataset = {
            'AR': dta.AR,
            'time_line': time_line,
            'data_obj': dta,
            'node_list': node_list,
            'velocity_list': velocity_list,
            'contact_list': contacts_list,
            'rad_gyr_list': rad_gyr_list,
            'num_contacts_list': num_contacts_list, 
            'largest_cluster_size_list': largest_cluster_size_list,  # Placeholder
            'entanglement_list': entanglement_list
        }
        global_data_list.append(local_dataset)

    # Build a Pandas DataFrame containing every data point.
    # Each row corresponds to one time step in one experiment.
    records = []
    for exp in global_data_list:
        meta = exp['data_obj']  # metadata stored in the data object
        for idx, t in enumerate(exp['time_line']):
            record = {
                'dt_string': meta.dt_string,
                'AR': meta.AR,
                'num_rods': meta.num_rods,
                'kick_amplitude': meta.kick_amplitude,
                'friction_coefficient': meta.friction_coefficient,
                'time': t,
                'rad_gyr': exp['rad_gyr_list'][idx],
                'num_contacts': exp['num_contacts_list'][idx],
                'entanglement': exp['entanglement_list'][idx],
                # 'largest_cluster_size': exp['largest_cluster_size_list'][idx]  # Uncomment if computed
            }
            records.append(record)

    df = pd.DataFrame(records)
    return df

if __name__ == '__main__':
    df = analyze_alpha50_to_dataframe()
    print(df.head())
    # Optionally, save the DataFrame for later use:
    df.to_csv('experiment_data.csv', index=False)
