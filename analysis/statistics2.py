import pickle
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import jax.numpy as jnp
from potentials import acn_over_ij

# with open(f'higher_data_list.pkl','wb') as f:
#     pickle.dump(higher_data_list,f)
# data class
class SingleKickData:
    def __init__(self, dt_string, AR, num_rods, kick_amplitude, friction_coefficient, data_path):
        self.dt_string = dt_string
        self.AR = AR
        self.num_rods = num_rods
        self.kick_amplitude = kick_amplitude
        self.friction_coefficient = friction_coefficient
        self.data_path = data_path

    # print function
    def __repr__(self):
        return f"dt_string: {self.dt_string}, AR: {self.AR}, num_rods: {self.num_rods}, kick_amplitude: {self.kick_amplitude}, friction_coefficient: {self.friction_coefficient}\n"
    
def analyze_alpha50():
    # this dataset has three repeated experiments
    # i want to get the average of the three experiments
    # for
    # 1. radius of gyration
    # 2. number of contacts
    # 3. largest cluster size
    # 4. entanglement

    with open(f'higher_data_list.pkl','rb') as f:
        higher_data_list = pickle.load(f)

    # unpack higher_data_list
    all_data_list = []
    for each_random_key_data in higher_data_list:
        for dta in each_random_key_data:
            all_data_list.append(dta)
   
    AR = 50
    chosen_data = []
    for dta in all_data_list:
        if dta.AR == AR and dta.kick_amplitude == 1. and dta.num_rods == 200 and dta.friction_coefficient == 0.1:
            chosen_data.append(dta)


    
    for dta in chosen_data:

        global_data_list = []
        for dta in chosen_data:
            data = loadmat(dta.data_path,simplify_cells=True)
            
            time_line = data['time_line']
            node_list = data['node_list']
            velocity_list = data['velocity_list']
            contacts_list = data['contact_list']

            nodes_at_last_frame = node_list[-1]
            xyz = nodes_at_last_frame.reshape(-1,6)

            rad_gyr_list = []
            num_contacts_list = []
            largest_cluster_size_list = []
            entanglement_list = []
            for i in range(len(node_list)):
                nodes_at_frame = node_list[i]

                if len(contacts_list) == 0:
                    contacts_at_frame = []
                else:
                    contacts_at_frame = contacts_list[i]

                xyz = nodes_at_frame.reshape(-1,6)
                centroids = (xyz[:,:3]+xyz[:,3:])/2
                global_centroid = np.mean(centroids,axis=0)
                moment_arm = centroids - global_centroid
                num_contacts_list.append(len(contacts_at_frame))
                radius_of_gyration = np.mean(np.linalg.norm(moment_arm,axis=1))
                rad_gyr_list.append(radius_of_gyration)

                r1 = nodes_at_frame.reshape(-1,6)[:,:3]
                r2 = nodes_at_frame.reshape(-1,6)[:,3:]
                num_rods = r1.shape[0]                
                i_indices, j_indices = jnp.triu_indices(num_rods, k=1)
                pairwise_acn = acn_over_ij(r1, r2, i_indices, j_indices)
                entanglement_list.append(np.sum( np.abs(pairwise_acn) )/(num_rods*(num_rods-1)/2))

            local_dataset = {
                'AR': AR,
                'time_line': time_line,
                'data_obj': dta,
                'node_list': node_list,
                'velocity_list': velocity_list,
                'contact_list': contacts_list,
                'rad_gyr_list': rad_gyr_list,
                'num_contacts_list': num_contacts_list,
                'largest_cluster_size_list': largest_cluster_size_list,
                'entanglement_list': entanglement_list
            }
            global_data_list.append(local_dataset)

        # sort global_data_list by friction_coefficient
        global_data_list = sorted(global_data_list,key=lambda x: x['data_obj'].friction_coefficient)



    for dta in global_data_list:
        # print(dta.keys())
        # print(dta['AR'])
        # print(dta['time_line'].shape)
        # for each friction coefficient, plot the averaged radius of gyration
        if dta['data_obj'].friction_coefficient == 0.1:
            print(dta['data_obj'].friction_coefficient)

        if dta['data_obj'].friction_coefficient == 0.1:            
            plt.plot(dta['time_line'],dta['rad_gyr_list'],label=f'{dta["data_obj"].friction_coefficient}')
            plt.legend()
            plt.title(f'AR: {dta["AR"]}, num_rods: {dta["data_obj"].num_rods}, kick_amplitude: {dta["data_obj"].kick_amplitude}')
            plt.xlabel('Time')
            plt.ylabel('Radius of Gyration')
        plt.show()


    
        



if __name__ == '__main__':
    analyze_alpha50()