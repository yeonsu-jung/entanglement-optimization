import numpy as np
import re
from data_io import save_as_mat
from pathlib import Path
from matplotlib import pyplot as plt
from scipy.io import loadmat
import jax.numpy as jnp
from potentials import dist_lin_seg_over_ij, acn_over_ij

def find_csv_file(folder_path):
    possible_paths = []
    for pth in folder_path.glob('**/*.csv'):
        if 'lastFrame' in str(pth):
            continue
        else:
            possible_paths.append(pth)    
    if len(possible_paths) == 0:
        print('No csv files found in the folder')
        exit()
    elif len(possible_paths) > 1:
        print('Multiple csv files found in the folder')
        # find heaviest file
        max_size = 0
        for pth in possible_paths:
            size = os.path.getsize(pth)
            if size > max_size:
                max_size = size
                heaviest_file = pth
        possible_paths = [heaviest_file]
        
    pth = str(possible_paths[0])
    return pth


def parse_pathname(pathname):
    dt_string = re.search(r'(\d{8}-\d{4})',pathname).group(1)
    AR = float(re.search('AR(\d+)',pathname).group(1))
    num_rods = int(re.search('N(\d+)',pathname).group(1))    
    kick_amplitude = float(re.search('Kick(\d+.\d+)',pathname).group(1))
    friction_info = re.search('Friction(\d+.\d+)',pathname)

    key_info = re.search('RandomKeys_(\d+),(\d+),(\d+),(\d+)',pathname)
    if key_info:
        random_keys = key_info.group(1)
        random_keys += ',' + key_info.group(2)
        random_keys += ',' + key_info.group(3)
    else:
        random_keys = '3,1,2,N/A'

    if friction_info:
        friction_coefficient = float(friction_info.group(1))
    else:
        friction_coefficient = 0.4

    # if a string contains a substring
    if "NoFriction" in str(pathname):
        friction_coefficient = 0

    return dt_string, AR, num_rods, kick_amplitude, random_keys, friction_coefficient

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


def get_data(pth):
    dt_string, AR, num_rods, kick_amplitude, random_keys, friction_coefficient = parse_pathname(str(pth))
    csv_file_path = find_csv_file(pth)
    mat_file_path = csv_file_path.replace('.csv','.mat')
    if not Path(mat_file_path).exists():
        save_as_mat(csv_file_path,max_rows=100000)
    else:
        print(f'{mat_file_path} already exists')
    return SingleKickData(dt_string, AR, num_rods, kick_amplitude, friction_coefficient, mat_file_path)

def main():

    pth = '/Users/yeonsu/GitHub/dismech-rods-main/runs/20250216-1347_COMPILE__RUN_RandomKeys_5,7,9,72_Kick1.00_Friction0.00_N500_AR0500_k1e4'
    dta = get_data(Path(pth))
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
    pairwise_dist_list = []
    entanglement_list = []

    import time

    start = time.time()
    for i in range(len(node_list)):

        nodes_at_frame = node_list[i]
        contacts_at_frame = contacts_list[i]
        xyz = nodes_at_frame.reshape(-1,6)
        centroids = (xyz[:,:3]+xyz[:,3:])/2
        global_centroid = np.mean(centroids,axis=0)
        moment_arm = centroids - global_centroid
        num_contacts_list.append(len(contacts_at_frame))
        radius_of_gyration = np.mean(np.linalg.norm(moment_arm,axis=1))
        rad_gyr_list.append(radius_of_gyration)

        rr = nodes_at_frame.reshape(-1,6)
        r1 = rr[:,:3]
        r2 = rr[:,3:]
        N = r1.shape[0]
        
        i_indices, j_indices = jnp.triu_indices(N, k=1)
        pairwise_dist_list.append(dist_lin_seg_over_ij(r1, r2, i_indices, j_indices))

        pairwise_acn = acn_over_ij(r1, r2, i_indices, j_indices)
        num_rods = r1.shape[0]
        entanglement_list.append(np.sum( np.abs(pairwise_acn) )/(num_rods*(num_rods-1)/2))



        if i % 100 == 0:
            print(f'{i}th frame processed')
            end = time.time()
            print(f'{end-start} seconds elapsed')

    end = time.time()
    print(f'{end-start} seconds elapsed')

    # plt.plot(time_line,rad_gyr_list)
    fig = plt.figure(figsize=(2.5,2))
    plt.plot(time_line,entanglement_list,label='entanglement over time (N = 500)')

    xx = np.linspace(0.001,1,100)
    yy = np.arctan(1/(xx*np.sqrt(8+16*xx**2)))/np.pi
    plt.plot(xx,yy, label='analytical solution (d_ij only increase)')

    crossover = np.sqrt(np.sqrt(2) - 1)/2
    plt.axvline(x=crossover, color='r', linestyle='--', label='crossover point')
    plt.axhline(y=0.25, color='g', linestyle='--', label='0.5')

    plt.xlabel('time (sec)')
    plt.ylabel('entanglement')
    # legend at outside of the plot
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig('junkyard/entanglement.svg',bbox_inches='tight')

    plt.show()
    
    return 0
    
if __name__ == '__main__':
    main()