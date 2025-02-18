import numpy as np
import jax.numpy as jnp

from data_io import import_all_log, save_as_mat
from scipy.io import loadmat
import matplotlib.pyplot as plt
from potentials import acn_over_ij, dist_lin_seg_over_ij

import os
import re
from pathlib import Path

import time
import pickle
import pandas as pd

def create_folder(pathname):
    if not os.path.exists(pathname):
        os.makedirs(pathname)

# data class
class SingleKickData:
    def __init__(self, random_keys,dt_string, AR, num_rods, kick_amplitude, friction_coefficient, data_path):
        self.random_keys = random_keys
        self.dt_string = dt_string
        self.AR = AR
        self.num_rods = num_rods
        self.kick_amplitude = kick_amplitude
        self.friction_coefficient = friction_coefficient
        self.data_path = data_path

    # print function
    def __repr__(self):
        return f"dt_string: {self.dt_string}, AR: {self.AR}, num_rods: {self.num_rods}, kick_amplitude: {self.kick_amplitude}, friction_coefficient: {self.friction_coefficient}\n"

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

    return random_keys, dt_string, AR, num_rods, kick_amplitude, random_keys, friction_coefficient

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

def csv_to_dict(alllog_pth, start_row=0,max_rows = 100000000,skip_rows=1):
    with open(alllog_pth) as f:
        lines = f.readlines()
    lines = lines[start_row:max_rows:skip_rows]
        
    time_line = []
    node_list = []
    velocity_list = []
    force_list = []
    contact_list = []
    box_contact_list = []

    for i,line in enumerate(lines):
        if line.startswith('Time'):
            time_line.append(float(line.split('Time: ')[-1].rstrip('\n')))
            
        if line.startswith('Node'):
            next_line = lines[i+1]                       
            node_list.append(np.array([float(x) for x in next_line.split(',')]))

        if line.startswith('Velocity'):
            next_line = lines[i+1]                       
            velocity_list.append(np.array([float(x) for x in next_line.split(',')]))
            
        if line.startswith('Force'):
            next_line = lines[i+1]
            if next_line == "\n":
                force_list.append(np.array([]))
            else:
                force_list.append(np.array([float(x) for x in next_line.split(',') if x != '\n']))
                
        if line.startswith('Contact'):
            next_line = lines[i+1]
            if next_line == "\n":
                contact_list.append(np.array([]))
            else:
                contact_list.append(np.array([float(x) for x in next_line.split(',') if x != '\n']))
                
        if line.startswith('Box'):
            next_line = lines[i+1]
            if next_line == "\n":
                box_contact_list.append(np.array([]))
            else:
                box_contact_list.append(np.array([float(x) for x in next_line.split(',') if x != '\n']))

    time_line = np.array(time_line)
    node_list = np.array(node_list)
    velocity_list = np.array(velocity_list)

    force_list = np.array(force_list, dtype=object)
    contact_list = np.array(contact_list, dtype=object)
    box_contact_list = np.array(box_contact_list, dtype=object)

    data = {'time_line':time_line, 'node_list':node_list, 'velocity_list':velocity_list, 'force_list':force_list, 'contact_list':contact_list, 'box_contact_list':box_contact_list}
    return data

def explore_folder(folder_path):

    pathlist = list(folder_path.glob("*RUN*"))
    data_list = []
    for pth in pathlist:
        random_keys, dt_string, AR, num_rods, kick_amplitude, random_keys, friction_coefficient = parse_pathname(str(pth))
        # if AR == 500 and kick_amplitude == 0.1:
        print(f"random keys: {random_keys}, dt_string: {dt_string}, AR: {AR}, num_rods: {num_rods}, kick_amplitude: {kick_amplitude}, friction_coefficient: {friction_coefficient}")
        csv_file_path = find_csv_file(pth)        
        data_list.append(SingleKickData(random_keys, dt_string, AR, num_rods, kick_amplitude, friction_coefficient, csv_file_path))

    # chosen_data = []
    # for dta in all_data_list:
    #     if dta.AR == AR and dta.kick_amplitude == .1 and dta.num_rods == 500 and dta.friction_coefficient == 0.2:
    #         chosen_data.append(dta)

    return data_list

def analyze_rms(nodes_at_ith_frame):

    xyz = nodes_at_ith_frame.reshape(-1, 6)
    centroids = (xyz[:, :3] + xyz[:, 3:]) / 2
    global_centroid = jnp.mean(centroids, axis=0)
    moment_arm = centroids - global_centroid
    rad_gyr = jnp.mean(jnp.linalg.norm(moment_arm, axis=1))

    return rad_gyr

def analyze_pairwise_dist_entanglement(nodes_at_ith_frame,num_rods):
    i_indices, j_indices = jnp.triu_indices(num_rods, k=1)
    r1 = nodes_at_ith_frame.reshape(-1,6)[:,:3]
    r2 = nodes_at_ith_frame.reshape(-1,6)[:,3:]
    pairwise_acn = acn_over_ij(r1,r2, i_indices, j_indices)
    pairwise_dist = dist_lin_seg_over_ij(r1,r2, i_indices, j_indices)
    dist = jnp.min(pairwise_dist)
    entanglement = jnp.sum(jnp.abs(pairwise_acn)) / (num_rods*(num_rods-1)/2)
    return dist,entanglement,pairwise_dist, pairwise_acn

def analyze_fraction_of_the_largest_cluster(contacts_at_ith_frame,num_rods):
    contact_ij = contacts_at_ith_frame.reshape(-1,8)[:,:2].astype(int)
    clusters, num_clusters, cluster_sizes, max_cluster_size = get_clusters(contact_ij,num_rods)
    fraction_of_the_largest_cluster = max_cluster_size / num_rods
    num_contacts = len(contact_ij)
    return fraction_of_the_largest_cluster, num_contacts, clusters

def postprocessing_and_caching(pathlist,filename):
    
    num_frames = 100
    chosen_data = []
    for pth in pathlist:        
        chosen_data.extend(explore_folder(Path(pth)))

    # main loop

    num_datasets = len(chosen_data)
    
    # "register" (or initialize) the dataframe
    random_key_data = []
    AR_data = np.zeros(num_datasets)
    num_rods_data = np.zeros(num_datasets)
    kick_amplitude_data = np.zeros(num_datasets)
    friction_coefficient_data = np.zeros(num_datasets)

    data = csv_to_dict(chosen_data[0].data_path)
    time_line = data['time_line']
    skip = time_line.shape[0] // (num_frames)
    real_num_frames = len(range(0,time_line.shape[0],skip))

    time_line_data    = np.zeros((num_datasets,real_num_frames))
    rad_gyr_data      = np.zeros((num_datasets,real_num_frames))
    entanglement_data = np.zeros((num_datasets,real_num_frames))
    fraction_tlc_data = np.zeros((num_datasets,real_num_frames))
    num_contacts_data = np.zeros((num_datasets,real_num_frames))

    for i_dataset,dta in enumerate(chosen_data):
            
        data = csv_to_dict(dta.data_path)
        time_line = data['time_line']
        node_list = data['node_list']
        velocity_list = data['velocity_list']
        contacts_list = data['contact_list']

        # store the results
        random_key_data.append(dta.random_keys)
        AR_data[i_dataset] = dta.AR
        num_rods_data[i_dataset] = dta.num_rods
        kick_amplitude_data[i_dataset] = dta.kick_amplitude
        friction_coefficient_data[i_dataset] = dta.friction_coefficient
        
        skip = time_line.shape[0] // (num_frames-1)

        time_line_data[i_dataset,:] = time_line[::skip]
        k = 0
        for i in range(0,time_line.shape[0],skip):
            time_at_ith_frame = time_line[i]
            nodes_at_ith_frame = node_list[i]
            velocities_at_ith_frame = velocity_list[i]
            contacts_at_ith_frame = contacts_list[i]

            # vectorization...?
            
            rad_gyr = analyze_rms(nodes_at_ith_frame)
            dist,entanglement,pairwise_dist,pairwise_acn = analyze_pairwise_dist_entanglement(nodes_at_ith_frame, dta.num_rods) # 
            fraction_tlc, num_contacts, clusters = analyze_fraction_of_the_largest_cluster(contacts_at_ith_frame,dta.num_rods)
            
            rad_gyr_data[i_dataset,k] = rad_gyr
            entanglement_data[i_dataset,k] = entanglement
            fraction_tlc_data[i_dataset,k] = fraction_tlc
            num_contacts_data[i_dataset,k] = num_contacts
            k = k + 1

    # create dataframe
    # df = pd.DataFrame(data = {'AR': AR_data.T,
    #                           'num_rods': num_rods_data.T,
    #                           'kick_amplitude': kick_amplitude_data.T,
    #                           'friction_coefficient': friction_coefficient_data.T,
    #                           'rad_gyr': rad_gyr_data,
    #                           'entanglement': entanglement_data, 
    #                           'fraction_of_the_largest_cluster': fraction_tlc_data,
    #                           'num_contacts': num_contacts_data})
    
    # create dataframe with each time-series stored as a list in a cell
    df = pd.DataFrame({
        'random_keys': random_key_data,
        'AR': AR_data,
        'num_rods': num_rods_data,
        'kick_amplitude': kick_amplitude_data,
        'friction_coefficient': friction_coefficient_data,
        'time_line': [list(row) for row in time_line_data],
        'rad_gyr': [list(row) for row in rad_gyr_data],
        'entanglement': [list(row) for row in entanglement_data],
        'fraction_of_the_largest_cluster': [list(row) for row in fraction_tlc_data],
        'num_contacts': [list(row) for row in num_contacts_data]
    })
    
    # save the dataframe
    create_folder('dataframe_results')
    df.to_pickle(f'dataframe_results/{filename}.pkl')

    print('done')

            

    # output here: big square dataframe.
    # columns: AR, num_rods, kick_amplitude, friction_coefficient, time, rad_gyr (num_time_points,), entanglement (num_time_points,), fraction_of_the_largest_cluster (num_time_points,)


def test():
    filename = '29,19,70_New'
    num_frames = 50

    pth = '/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/29,19,70_New'
    data_list = explore_folder(Path(pth))

    num_datasets = len(data_list)
    
    # "register" (or initialize) the dataframe
    random_key_data = []
    AR_data = np.zeros(num_datasets)
    num_rods_data = np.zeros(num_datasets)
    kick_amplitude_data = np.zeros(num_datasets)
    friction_coefficient_data = np.zeros(num_datasets)

    data = csv_to_dict(data_list[0].data_path)
    time_line = data['time_line']
    skip = time_line.shape[0] // (num_frames)
    real_num_frames = len(range(0,time_line.shape[0],skip))

    time_line_data    = np.zeros((num_datasets,real_num_frames))
    rad_gyr_data      = np.zeros((num_datasets,real_num_frames))
    entanglement_data = np.zeros((num_datasets,real_num_frames))
    fraction_tlc_data = np.zeros((num_datasets,real_num_frames))
    num_contacts_data = np.zeros((num_datasets,real_num_frames))

    for i_dataset,dta in enumerate(data_list):
            
        data = csv_to_dict(dta.data_path)
        time_line = data['time_line']
        node_list = data['node_list']
        velocity_list = data['velocity_list']
        contacts_list = data['contact_list']

        # store the results
        random_key_data.append(dta.random_keys)
        AR_data[i_dataset] = dta.AR
        num_rods_data[i_dataset] = dta.num_rods
        kick_amplitude_data[i_dataset] = dta.kick_amplitude
        friction_coefficient_data[i_dataset] = dta.friction_coefficient
        
        skip = time_line.shape[0] // (num_frames-1)
        time_line_data[i_dataset,:] = time_line[::skip]
        k = 0
        for i in range(0,time_line.shape[0],skip):
            time_at_ith_frame = time_line[i]
            nodes_at_ith_frame = node_list[i]
            velocities_at_ith_frame = velocity_list[i]
            contacts_at_ith_frame = contacts_list[i]

            # vectorization...?
            
            rad_gyr = analyze_rms(nodes_at_ith_frame)
            dist,entanglement,pairwise_dist,pairwise_acn = analyze_pairwise_dist_entanglement(nodes_at_ith_frame, dta.num_rods) # 
            fraction_tlc, num_contacts, clusters = analyze_fraction_of_the_largest_cluster(contacts_at_ith_frame,dta.num_rods)
            
            rad_gyr_data[i_dataset,k] = rad_gyr
            entanglement_data[i_dataset,k] = entanglement
            fraction_tlc_data[i_dataset,k] = fraction_tlc
            num_contacts_data[i_dataset,k] = num_contacts
            k = k + 1

    df = pd.DataFrame({
        'random_keys': random_key_data,
        'AR': AR_data,
        'num_rods': num_rods_data,
        'kick_amplitude': kick_amplitude_data,
        'friction_coefficient': friction_coefficient_data,
        'time_line': [list(row) for row in time_line_data],
        'rad_gyr': [list(row) for row in rad_gyr_data],
        'entanglement': [list(row) for row in entanglement_data],
        'fraction_of_the_largest_cluster': [list(row) for row in fraction_tlc_data],
        'num_contacts': [list(row) for row in num_contacts_data]
    })
    
    # save the dataframe
    create_folder('dataframe_results')
    df.to_pickle(f'dataframe_results/{filename}.pkl')

    all_friction_coefficients = df['friction_coefficient']

    for friction_coefficient in all_friction_coefficients:

        local_df = df[ (df['friction_coefficient'] == friction_coefficient) ]
        time_line = local_df['time_line']

        f = local_df['fraction_of_the_largest_cluster']
        f = np.array(f.tolist())

        avg = np.mean(f,axis=0)
        std = np.std(f,axis=0)

        plt.plot(time_line,avg)
        plt.fill_between(time_line, avg-std, avg+std, alpha=0.3)
        
    plt.show()
            
def analyze_over_friction_coefficient(df,analysis_id):
    
    # pickup AR = 100 and average
    t_u = np.sqrt( np.sqrt(2) - 1) / 2
    all_mu = df['friction_coefficient']
    all_mu = np.unique(all_mu)
    num_mus = len(all_mu)

    all_AR = df['AR']
    all_AR = np.unique(all_AR)
    num_ARs = len(all_AR)
    # colors = plt.cm.viridis(np.linspace(0,1,num_mus))
    colors = plt.cm.tab20(np.linspace(0,1,num_mus))
    
    for AR in all_AR:
        plt.figure(figsize=(2.5,2))
        for i_,mu in enumerate(all_mu):

            local_df = df[ (df['AR'] == AR) & (df['friction_coefficient'] == mu) ]
            time_line = np.array(local_df['time_line'].tolist())[0].flatten()
            
            y_val = local_df['entanglement']
            y_val = np.array(y_val.tolist())
            
            avg = np.mean(y_val,axis=0)
            std = np.std(y_val,axis=0)
            
            plt.plot(time_line/t_u,avg,label=f'${mu}$',color=colors[i_])
            plt.fill_between(time_line/t_u, avg-std, avg+std, alpha=0.3,color=colors[i_])

        plt.xlabel(r'Normalized time, $t/t_u$')
        plt.ylabel(r'Normalized entanglement, $\tilde{e}$')
        plt.legend(title='$\mu$',loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig(f'junkyard/{analysis_id}_entanglement_AR{AR}.svg',bbox_inches='tight')
        plt.show()
        plt.close('all')

    for AR in all_AR:
        plt.figure(figsize=(2.5,2))
        for i_,mu in enumerate(all_mu):

            local_df = df[ (df['AR'] == AR) & (df['friction_coefficient'] == mu) ]
            time_line = np.array(local_df['time_line'].tolist())[0].flatten()
            
            y_val = local_df['fraction_of_the_largest_cluster']
            y_val = np.array(y_val.tolist())
            
            avg = np.mean(y_val,axis=0)
            std = np.std(y_val,axis=0)
            
            plt.plot(time_line/t_u,avg,label=f'${mu}$',color=colors[i_])
            plt.fill_between(time_line/t_u, avg-std, avg+std, alpha=0.3,color=colors[i_])

        plt.xlabel(r'Normalized time, $t/t_u$')
        plt.ylabel(r'Fraction of the largest cluster, $f$')
        plt.legend(title='$\mu$',loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig(f'junkyard/{analysis_id}_fraction_AR{AR}.svg',bbox_inches='tight')
        plt.show()
        plt.close('all')

def analyze_over_AR(df,analysis_id):

    t_u = np.sqrt( np.sqrt(2) - 1) / 2
    all_mu = df['friction_coefficient']
    all_mu = np.unique(all_mu)
    num_mus = len(all_mu)

    all_AR = df['AR']
    all_AR = np.unique(all_AR)
    num_ARs = len(all_AR)
    # colors = plt.cm.viridis(np.linspace(0,1,num_mus))
    colors = plt.cm.tab20(np.linspace(0,1,num_mus))
    
    plt.figure(figsize=(2.5,2))
    # plot final entanglement as a function of friction coefficient
    
    for i_AR, AR in enumerate(all_AR):
        y_val = []
        y_err = []
        for i_,mu in enumerate(all_mu):

            local_df = df[ (df['AR'] == AR) & (df['friction_coefficient'] == mu) ]
            
            entanglement = np.array(local_df['entanglement'].tolist())[:,-1]
            avg= np.mean(entanglement)
            std = np.std(entanglement)

            y_val.append(avg)
            y_err.append(std)

        plt.errorbar(all_mu,y_val,yerr=y_err,fmt='o-',label=f'AR={AR}')
    plt.legend(title='Aspect ratio',loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel(r'Friction coefficient, $\mu$')
    plt.ylabel(r'Final entanglement, $\tilde{e}$')
    plt.savefig(f'junkyard/{analysis_id}_final_entanglement_vs_mu.svg',bbox_inches='tight')

    plt.figure(figsize=(2.5,2))
    for i_AR, AR in enumerate(all_AR):
        y_val = []
        y_err = []
        for i_,mu in enumerate(all_mu):

            local_df = df[ (df['AR'] == AR) & (df['friction_coefficient'] == mu) ]
            
            fraction_of_the_largest_cluster = np.array(local_df['fraction_of_the_largest_cluster'].tolist())[:,-1]
            avg= np.mean(fraction_of_the_largest_cluster)
            std = np.std(fraction_of_the_largest_cluster)

            y_val.append(avg)
            y_err.append(std)

        plt.errorbar(all_mu,y_val,yerr=y_err,fmt='o-',label=fr'${AR}$')
        # logscale
        plt.yscale('log')

    plt.legend(title=r'$\alpha$',loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel(r'Friction coefficient, $\mu$')
    plt.ylabel(r'Final largest cluster fraction, $f$')
    plt.savefig(f'junkyard/{analysis_id}_final_fraction_vs_mu.svg',bbox_inches='tight')


    

    return 0


if __name__ == '__main__':

    # test()

    pathlist = []
    # pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/29,19,70_NewK1e5')
    # pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/6,7,8_K1e5')
    # pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/37,178,56_K1e5')
    # pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/919,468,568_K1e5')

    # analysis_id = 'N200_NewK1e5'
    # pathlist.append(Path('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/37,178,56,72_Kick0.10'))
    # pathlist.append(Path('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/6,7,8,72_Kick0.10'))
    # pathlist.append(Path('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/919,461,568,72_Kick0.10'))

    analysis_id = 'N500_NewK1e5'
    pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/919,461,568_N500_Final')
    pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/6,7,8_N500_Final')
    pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/37,178,56_N500_Final')    
    
    postprocessing_and_caching(pathlist,analysis_id)

    # pth = '/Users/yeonsu/GitHub/entanglement-optimization/dataframe_results/N200_NewK1e5.pkl'
    pth = '/Users/yeonsu/GitHub/entanglement-optimization/dataframe_results/N500_NewK1e5.pkl'
    df = pd.read_pickle(pth)
    analysis_id = pth.split('/')[-1].split('.')[0]
    print(analysis_id)

    analyze_over_friction_coefficient(df,analysis_id)
    analyze_over_AR(df,analysis_id)

    print('done')

