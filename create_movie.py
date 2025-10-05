# %%
# todo: goal is to combine three datasets

import re
from pathlib import Path
import numpy as np
from data_io import import_all_log, save_as_mat
from scipy.io import loadmat
from matplotlib import pyplot as plt
from transforms import x_to_q
from potentials import total_effective_potential
import time
from analysis_functions import create_folder
import pickle
import os

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
# %%
# meta folder's'
meta_folders = []
# meta_folders.append(Path('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/919,461,568') )# for all AR
# meta_folders.append(Path('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/37,178,56') )# for all AR
# meta_folders.append(Path('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/6,7,8') )# for all AR

# meta_folders.append(Path('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/52,33,20_kick0.01'))

# meta_folders.append(Path('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/46,15,99'))
# meta_folders.append(Path('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/32,0,98'))
# meta_folders.append(Path('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/29,19,70_'))

# meta_folders.append(Path('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/37,178,56_N500'))
# meta_folders.append(Path('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/6,7,8_N500'))
# meta_folders.append(Path('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/919,461,568_N500'))

# meta_folders.append(Path('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/919,461,568,72_Kick0.10'))
# meta_folders.append(Path('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/6,7,8,72_Kick0.10'))
# meta_folders.append(Path('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/37,178,56,72_Kick0.10'))

# meta_folder = Path('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/919,461,568,72_Kick0.10')
meta_folder = Path('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/919,461,568,72_Kick0.10')
meta_folders.append(meta_folder)


# %%
# get directory list
# pathlist = [str(p) for p in pathlist if p.is_dir()]

higher_data_list = []
for meta_folder in meta_folders:
    pathlist = list(meta_folder.glob("*RUN*"))
    data_list = []
    for pth in pathlist:
        dt_string, AR, num_rods, kick_amplitude, random_keys, friction_coefficient = parse_pathname(str(pth))
        csv_file_path = find_csv_file(pth)
        mat_file_path = csv_file_path.replace('.csv','.mat')
        save_as_mat(csv_file_path,max_rows=100000)
        if not Path(mat_file_path).exists():
            save_as_mat(csv_file_path,max_rows=100000)
        else:
            print(f'{mat_file_path} already exists')
        data_list.append(SingleKickData(dt_string, AR, num_rods, kick_amplitude, friction_coefficient, mat_file_path))
    higher_data_list.append(data_list)

with open(f'higher_data_list.pkl','wb') as f:
    pickle.dump(higher_data_list,f)
# %%

# def analyze_a_single_metafolder(meta_folder):
#     pathlist = list(meta_folder.glob("*RUN*"))
#     data_list = []
#     for pth in pathlist:
#         dt_string, AR, num_rods, kick_amplitude, random_keys, friction_coefficient = parse_pathname(str(pth))
#         csv_file_path = find_csv_file(pth)
#         mat_file_path = csv_file_path.replace('.csv','.mat')
#         # save_as_mat(csv_file_path,max_rows=100000)
#         if not Path(mat_file_path).exists():
#             save_as_mat(csv_file_path,max_rows=100000)
#         else:
#             print(f'{mat_file_path} already exists')
#         data_list.append(SingleKickData(dt_string, AR, num_rods, kick_amplitude, friction_coefficient, mat_file_path))
#     print(data_list)

#     AR = 500
#     AR_data = []
#     AR_entanglement_data = []

#     for AR in [50,100,150,200,300,500]:
#     # for AR in [50]:


#         # chosen_data = []
#         # for dta in data_list:
#         #     # if dta.AR == AR and (0.1 <= dta.friction_coefficient < 0.21) and dta.kick_amplitude == 0.01:
#         #     # if dta.AR == AR and dta.kick_amplitude == 1. and dta.num_rods == 500:
#         #     if dta.AR == AR and dta.kick_amplitude == 1. and dta.num_rods == 200:

#         #     # if dta.kick_amplitude == 0.01 and dta.num_rods == 200:
#         #     # if dta.AR == AR and (dta.friction_coefficient > 0.2) and dta.kick_amplitude == 0.01 and dta.num_rods == 200:
#         #         # data = loadmat(dta.data_path,simplify_cells=True)
#         #         kick_amplitude = dta.kick_amplitude
#         #         chosen_data.append(dta)
#         #         print(dta)

#         chosen_data = data_list

#         global_data_list = []
#         for dta in chosen_data:
#             data = loadmat(dta.data_path,simplify_cells=True)
            
#             time_line = data['time_line']
#             node_list = data['node_list']
#             velocity_list = data['velocity_list']
#             contacts_list = data['contact_list']

#             nodes_at_last_frame = node_list[-1]
#             xyz = nodes_at_last_frame.reshape(-1,6)

#             rad_gyr_list = []
#             num_contacts_list = []
#             largest_cluster_size_list = []
#             for i in range(len(node_list)):
#                 nodes_at_frame = node_list[i]

#                 if len(contacts_list) == 0:
#                     contacts_at_frame = []
#                 else:
#                     contacts_at_frame = contacts_list[i]

#                 xyz = nodes_at_frame.reshape(-1,6)
#                 centroids = (xyz[:,:3]+xyz[:,3:])/2
#                 global_centroid = np.mean(centroids,axis=0)
#                 moment_arm = centroids - global_centroid
#                 num_contacts_list.append(len(contacts_at_frame))
#                 radius_of_gyration = np.mean(np.linalg.norm(moment_arm,axis=1))
#                 rad_gyr_list.append(radius_of_gyration)

#             local_dataset = {
#                 'AR': AR,
#                 'time_line': time_line,
#                 'data_obj': dta,
#                 'node_list': node_list,
#                 'velocity_list': velocity_list,
#                 'contact_list': contacts_list,
#                 'rad_gyr_list': rad_gyr_list,
#                 'num_contacts_list': num_contacts_list,
#             }
#             global_data_list.append(local_dataset)

#         # sort global_data_list by friction_coefficient
#         global_data_list = sorted(global_data_list,key=lambda x: x['data_obj'].friction_coefficient)
#         AR_data.append(global_data_list)

#         friction_coefficient_list = [local_dataset['data_obj'].friction_coefficient for local_dataset in global_data_list]

#         _trimmed = []
#         _trimmed = global_data_list

#         global_entanglement_over_time = {}
#         time_skip = 1

#         for local_dataset in _trimmed:
#             time_line = local_dataset['time_line']
#             nodes_list = local_dataset['node_list']
#             contact_list = local_dataset['contact_list']
#             velocity_list = local_dataset['velocity_list']
#             num_contacts_list = local_dataset['num_contacts_list']

#             AR = local_dataset['AR']
#             rod_radius = 1/AR/2
#             rod_length = 1
#             rod_density = 1000
#             rod_volume = np.pi*rod_radius**2*rod_length
#             rod_mass = rod_density*rod_volume
#             rod_inertia = rod_mass*(rod_length**2/12 + rod_radius**2/4)

#             centroid_velocity_over_time = []
#             kinetic_energy_over_time = []
#             mean_velocity_over_time = []
#             entanglement_over_time = []
#             largest_cluster_size_over_time = []
#             num_contacts_over_time = []

#             start = time.time()
#             for i in range(0,len(nodes_list),time_skip):
#                 x = nodes_list[i].reshape(-1,6)
#                 q = x_to_q(x)
#                 e = -total_effective_potential(q)
#                 entanglement_over_time.append(e)

#                 v = velocity_list[i].reshape(-1,6)
#                 v1 = v[:,:3]
#                 v2 = v[:,3:]

#                 centroid_velocity = (v1+v2)/2
#                 centroid_velocity_over_time.append(centroid_velocity)

#                 linear_kinetic_energy = 0.5*rod_mass*np.sum(np.linalg.norm(centroid_velocity,axis=1)**2)

#                 v1r = v1 - centroid_velocity
#                 v2r = v2 - centroid_velocity

#                 moment_arm_for_omega = (x[:,3:]-x[:,:3])/2
#                 omega = np.cross(moment_arm_for_omega,v1r)
#                 angular_kinetic_energy = 0.5*np.sum(omega**2*rod_inertia)
#                 non_rigid_body_velocity = v.reshape(-1,6) - np.tile(centroid_velocity,(1,2))
                
#                 mean_velocity = np.mean(np.linalg.norm(non_rigid_body_velocity,axis=1))
#                 mean_velocity_over_time.append(mean_velocity)
#                 # kinetic_energy = 0.5*np.sum(np.linalg.norm(v,axis=1)**2)
#                 # kinetic_energy = 0.5*np.sum(v**2)
#                 kinetic_energy = linear_kinetic_energy + angular_kinetic_energy
#                 kinetic_energy_over_time.append(kinetic_energy)

#                 if len(contact_list) == 0:
#                     contacts_at_frame = []
#                     num_contacts_over_time.append(0)
#                 else:
#                     contacts_at_frame = contact_list[i]
#                     num_contacts_over_time.append(contacts_at_frame.reshape(-1,8).shape[0])

#                 if len(contacts_at_frame) == 0:
#                     largest_cluster_size_over_time.append(0)
#                 else:            
#                     contact_ij = contacts_at_frame.reshape(-1,8)[:,:2].astype(int)
#                     largest_cluster_size_over_time.append(get_clusters(contact_ij,num_rods)[-1])

#             print(f'{local_dataset["data_obj"].friction_coefficient} took {time.time()-start:.2f} seconds')
#             skipped_time_line = time_line[::time_skip]
            
#             local_data = {
#                 'time_line': skipped_time_line,
#                 'entanglement': entanglement_over_time,
#                 'mean_velocity': mean_velocity_over_time,
#                 'kinetic_energy': kinetic_energy_over_time,
#                 'centroid_velocity': centroid_velocity_over_time,
#                 'num_contacts': num_contacts_over_time,
#                 'largest_cluster_size': largest_cluster_size_over_time,
#             }
#             global_entanglement_over_time[local_dataset['data_obj'].friction_coefficient] = local_data
#         AR_entanglement_data.append(global_entanglement_over_time)

#     subfolder_name = f'Random{random_keys}'
#     figure_output_folder = f"/Users/yeonsu/Figures/{subfolder_name}"
#     create_folder(figure_output_folder)

#     t_u = np.sqrt(np.sqrt(2)-1)/2

#     fig,ax=plt.subplots(figsize=(2.5,2))
#     for local_dataset in _trimmed:
#         print(local_dataset['data_obj'].friction_coefficient)
#         time_line = global_entanglement_over_time[local_dataset['data_obj'].friction_coefficient]['time_line']
#         f = global_entanglement_over_time[local_dataset['data_obj'].friction_coefficient]['largest_cluster_size']
#         f = np.array(f)/(local_dataset['data_obj'].num_rods)

#         ax.plot(time_line/t_u,f,'-',label=f'{local_dataset["data_obj"].friction_coefficient}')

#     plt.xlabel('$t/t_u$')
#     plt.ylabel('$f$')
#     # plt.savefig(f'/Users/yeonsu/Figures/Random{random_keys}/FractionOfClusterOverTime_AR{AR}.png',dpi=300,bbox_inches='tight')
#     plt.savefig(f'/Users/yeonsu/Figures/Random{random_keys}/FractionOfClusterOverTime_AR{AR}.svg',bbox_inches='tight')
#     plt.legend(title='$\\mu$',loc='upper right')

#     fig,ax=plt.subplots(figsize=(2.5,2))
#     for local_dataset in _trimmed:
#         time_line = global_entanglement_over_time[local_dataset['data_obj'].friction_coefficient]['time_line']
#         kinetic_energy = global_entanglement_over_time[local_dataset['data_obj'].friction_coefficient]['kinetic_energy']
#         ax.loglog(time_line/t_u,kinetic_energy,'-',label=f'{local_dataset["data_obj"].friction_coefficient}')

#     ax.axhline(kinetic_energy[0],linestyle='--',color='k')

#     plt.xlabel('$t/t_u$')
#     plt.ylabel('Kinetic energy, $K$ (?)')
#     plt.legend(title='$\\mu$',loc='upper right')
#     # plt.savefig(f'/Users/yeonsu/Figures/Random{random_keys}/KineticEnergyOverTime_AR{AR}.png',dpi=300,bbox_inches='tight')
#     plt.savefig(f'/Users/yeonsu/Figures/Random{random_keys}/KineticEnergyOverTime_AR{AR}.svg',bbox_inches='tight')
        
#     fig,ax=plt.subplots(figsize=(2.5,2))
#     for local_dataset in _trimmed[:1]:

#         time_line = global_entanglement_over_time[local_dataset['data_obj'].friction_coefficient]['time_line']

#         centroid_velocity = global_entanglement_over_time[local_dataset['data_obj'].friction_coefficient]['centroid_velocity']
#         centroid_velocity = np.array(centroid_velocity)
#         centroid_velocity.shape
#         centroid_velocity_mag = np.linalg.norm(centroid_velocity,axis=2)
#         ax.plot(time_line/t_u,centroid_velocity_mag,'-',label=f'{local_dataset["data_obj"].friction_coefficient}')

#     plt.xlabel('$t/t_u$ (sec)')
#     plt.ylabel('Mean relative velocity, $v$ (L/sec)')

#     create_folder(f'/Users/yeonsu/Figures/Random{random_keys}/')

#     fig,ax=plt.subplots(figsize=(2.5,2))
#     for local_dataset in _trimmed:
#         time_line = global_entanglement_over_time[local_dataset['data_obj'].friction_coefficient]['time_line']
#         entanglement = global_entanglement_over_time[local_dataset['data_obj'].friction_coefficient]['entanglement']
#         entanglement = np.array(entanglement)/(num_rods*(num_rods-1)/2)
#         ax.loglog(time_line/t_u,entanglement,label=f'{local_dataset["data_obj"].friction_coefficient}')

#     ax.axhline(0.5,linestyle='--',color='k')
#     ax.set_xlabel(r'$t/t_u$')
#     ax.set_ylabel(r'Norm. entanglement, $\tilde{e}$')
#     # ax.legend(title='$\\mu$',loc='upper right')
#     ax.legend(title='$\mu$',loc='center left', bbox_to_anchor=(1, 0.5))
#     plt.savefig(f'/Users/yeonsu/Figures/Random{random_keys}/EntanglementOverTimeLogLog_AR{AR}.svg',bbox_inches='tight')

#     fig,ax=plt.subplots(figsize=(2.5,2))
#     for local_dataset in _trimmed:
#         time_line = global_entanglement_over_time[local_dataset['data_obj'].friction_coefficient]['time_line']
#         entanglement = global_entanglement_over_time[local_dataset['data_obj'].friction_coefficient]['entanglement']
#         entanglement = np.array(entanglement)/(num_rods*(num_rods-1)/2)
#         ax.plot(time_line/t_u,entanglement,label=f'{local_dataset["data_obj"].friction_coefficient}')

#     ax.axhline(0.5,linestyle='--',color='k')
#     ax.set_xlabel(r'$t/t_u$')
#     ax.set_ylabel(r'Norm. entanglement, $\tilde{e}$')
#     ax.legend(title='$\\mu$',loc='upper right')
#     plt.savefig(f'/Users/yeonsu/Figures/Random{random_keys}/EntanglementOverTimeLinear_AR{AR}.svg',bbox_inches='tight')
#     # plt.savefig(f'/Users/yeonsu/Figures/Random{random_keys}/EntanglementOverTimeLinear_AR{AR}.png',dpi=300,bbox_inches='tight')


#     def quadratic_fit(x,a,b):
#         return a*x**b
#     from scipy.optimize import curve_fit

#     fig,ax=plt.subplots(figsize=(2.5,2))
#     for local_dataset in _trimmed:
#         time_line = global_entanglement_over_time[local_dataset['data_obj'].friction_coefficient]['time_line']
#         entanglement = global_entanglement_over_time[local_dataset['data_obj'].friction_coefficient]['entanglement']
#         entanglement = np.array(entanglement)/(num_rods*(num_rods-1)/2)

#         # popt, pcov = curve_fit(quadratic_fit, time_line, entanglement, p0=[0.5,-1.5])
#         ax.loglog(time_line/t_u,entanglement,label=f'{local_dataset["data_obj"].friction_coefficient}')
#         # ax.loglog(time_line,quadratic_fit(time_line,*popt),'--',color='k')

#     # ax.axhline(0.5,linestyle='--',color='k')
#     ax.set_xlabel(r'$t/t_u$')
#     ax.set_ylabel(r'Norm. entanglement, $\tilde{e}$')
#     # ax.legend(title='$\\mu$',loc='upper right')
#     plt.savefig(f'/Users/yeonsu/Figures/Random{random_keys}/EntanglementOverTimeLogLogPower_AR{AR}.svg',bbox_inches='tight')

#     fig,ax=plt.subplots(figsize=(2.5,2))
#     final_radius_of_gyration = []
#     for ix, local_dataset in enumerate(_trimmed):
#         time_line = local_dataset['time_line']
#         num_contacts_list = local_dataset['num_contacts_list']
#         fric_coef = local_dataset['data_obj'].friction_coefficient    
#         num_rods = local_dataset['data_obj'].num_rods

#         num_contacts_list = np.array(num_contacts_list)
#         num_contacts_list = num_contacts_list/(num_rods)

#         ax.plot(time_line/t_u,num_contacts_list,label=f'{local_dataset["data_obj"].friction_coefficient}')
#         final_radius_of_gyration.append(rad_gyr_list[-1])

#     ax.set_xlabel(r'$t/t_u$')
#     ax.set_ylabel(r'No. contacts per rod')
#     # small legend
#     ax.legend(title='$\\mu$',loc='upper right',fontsize=7)
#     # plt.savefig(f'{figure_output_folder}/RadiusOfGyrationOverTime_AR{AR}.png',dpi=300,bbox_inches='tight')
#     plt.savefig(f'{figure_output_folder}/NumContactsOverTime_AR{AR}.svg',bbox_inches='tight')

#     fig,axs=plt.subplots(2,1,figsize=(4,5))
#     final_radius_of_gyration = []
#     for ix, local_dataset in enumerate(_trimmed):
#         time_line = local_dataset['time_line']
#         rad_gyr_list = local_dataset['rad_gyr_list']
#         num_contacts_list = local_dataset['num_contacts_list']
#         fric_coef = local_dataset['data_obj'].friction_coefficient
        
#         axs[0].plot(time_line/t_u,rad_gyr_list,'-',label=f'{local_dataset["data_obj"].friction_coefficient}')
#         axs[1].plot(time_line/t_u,num_contacts_list)

#         final_radius_of_gyration.append(rad_gyr_list[-1])

#     # xx = np.linspace(0,15,100)
#     # yy = np.sqrt(1+200*xx**2)
#     # axs[0].plot(xx,yy,'k--',linewidth=2)

#     axs[1].set_xlabel('$t/t_u$')
#     axs[1].set_ylabel('Number of contacts')
#     axs[0].set_ylabel('Radius of gyration, $R_g$')
#     # small legend
#     axs[0].legend(title='$\\mu$',loc='upper right',fontsize=7)
#     # plt.savefig(f'{figure_output_folder}/RadiusOfGyrationOverTime_AR{AR}.png',dpi=300,bbox_inches='tight')
#     plt.savefig(f'{figure_output_folder}/RadiusOfGyrationOverTime_AR{AR}.svg',bbox_inches='tight')

    
#     fig,ax=plt.subplots(figsize=(2.5,2))
#     for global_data_list in AR_data:
        
#         fric_coeff = [local_dataset['data_obj'].friction_coefficient for local_dataset in global_data_list]
#         fric_coeff = np.array(fric_coeff)

#         final_radius_of_gyration = []
#         for ix, local_dataset in enumerate(global_data_list):
#             time_line = local_dataset['time_line']
#             rad_gyr_list = local_dataset['rad_gyr_list']
#             num_contacts_list = local_dataset['num_contacts_list']
#             fric_coef = local_dataset['data_obj'].friction_coefficient
#             final_radius_of_gyration.append(rad_gyr_list[-1])
        
#         plt.plot(fric_coeff,final_radius_of_gyration,'o-',label=f'AR={local_dataset["AR"]}')
#         plt.xlabel('Friction coefficient, $\\mu$')
#         plt.ylabel('Radius of gyration, $R_g$')

#     plt.legend(fontsize=8)
#     plt.savefig(f'{figure_output_folder}/RadiusOfGyration_all_AR_kick{kick_amplitude}.svg',bbox_inches='tight')
    
#     # fraction of clusters
#     import networkx as nx

#     fig,ax=plt.subplots(figsize=(2.5,2))
#     for global_data_list in AR_data:
        
#         fric_coeff = [local_dataset['data_obj'].friction_coefficient for local_dataset in global_data_list]
#         fric_coeff = np.array(fric_coeff)

#         final_cluster_fraction = []
#         for ix, local_dataset in enumerate(global_data_list):
#             time_line = local_dataset['time_line']
#             num_contacts_list = local_dataset['num_contacts_list']
#             fric_coef = local_dataset['data_obj'].friction_coefficient
#             num_contacts_list = np.array(num_contacts_list)
#             num_contacts_list = num_contacts_list/(local_dataset['data_obj'].num_rods)

#             contact_list = local_dataset['contact_list']

#             if len(contact_list) == 0:
#                 contact_ij = []
#                 max_cluster_size = 0
#                 final_cluster_fraction.append(0)
#             else:
#                 contact_ij = contact_list[-1].reshape(-1,8)[:,:2].astype(int)
#                 max_cluster_size = get_clusters(contact_ij,num_rods)[-1]
#                 final_cluster_fraction.append(max_cluster_size/num_rods)        
        
#         plt.plot(fric_coeff,final_cluster_fraction,'o-',label=f'AR={local_dataset["AR"]}')
#         plt.xlabel('Friction coefficient, $\\mu$')
#         plt.ylabel('Fraction of clusters, $f$')

#     plt.legend(fontsize=8)
#     plt.savefig(f'{figure_output_folder}/FractionOfClusters_all_AR_kick{kick_amplitude}.svg',bbox_inches='tight')

#     fig,ax=plt.subplots(figsize=(2.5,2))
#     for global_data_list in AR_data:
        
#         fric_coeff = [local_dataset['data_obj'].friction_coefficient for local_dataset in global_data_list]
#         fric_coeff = np.array(fric_coeff)

#         final_radius_of_gyration = []
#         final_num_contacts = []
#         for ix, local_dataset in enumerate(global_data_list):
#             time_line = local_dataset['time_line']
#             rad_gyr_list = local_dataset['rad_gyr_list']
#             num_contacts_list = local_dataset['num_contacts_list']
#             fric_coef = local_dataset['data_obj'].friction_coefficient
#             final_radius_of_gyration.append(rad_gyr_list[-1])
#             final_num_contacts.append(num_contacts_list[-1])
        
#         plt.plot(fric_coeff,final_num_contacts,'o-',label=f'AR={local_dataset["AR"]}')
#         plt.xlabel('Friction coefficient, $\\mu$')
#         plt.ylabel('Final num contacts, $N_c$')

#     # plt.legend(fontsize=8)
#     # plt.savefig(f'{figure_output_folder}/NumContacts_all_AR_kick{kick_amplitude}.png',dpi=300,bbox_inches='tight')
#     plt.savefig(f'{figure_output_folder}/NumContacts_all_AR_kick{kick_amplitude}.svg',bbox_inches='tight')

#     # AR_entanglement_data
#     fig,ax=plt.subplots(figsize=(2.5,2))
#     for global_data_list,global_entanglement_over_time in zip(AR_data,AR_entanglement_data):
        
#         fric_coeff = [local_dataset['data_obj'].friction_coefficient for local_dataset in global_data_list]
#         fric_coeff = np.array(fric_coeff)

#         final_entanglement = []

#         for local_dataset in global_data_list:
#             time_line = global_entanglement_over_time[local_dataset['data_obj'].friction_coefficient]['time_line']
#             entanglement = global_entanglement_over_time[local_dataset['data_obj'].friction_coefficient]['entanglement']
#             entanglement = np.array(entanglement)/(num_rods*(num_rods-1)/2)
#             final_entanglement.append(entanglement[-1])
#             # ax.loglog(time_line,entanglement,label=f'{local_dataset["data_obj"].friction_coefficient}')
        
#         plt.plot(fric_coeff,final_entanglement,'o-',label=f'AR={local_dataset["AR"]}')
#         plt.xlabel('Friction coefficient, $\\mu$')
#         plt.ylabel(r'Final entanglement, $\tilde{e}$')
#         # break
#     ax.set_ylim([0,0.5])
#     # plt.legend(fontsize=8)
#     plt.savefig(f'{figure_output_folder}/Entanglement_all_AR_kick{kick_amplitude}.svg',bbox_inches='tight')

#     fig,ax=plt.subplots(figsize=(2.5,2))
#     for local_dataset in _trimmed:
#         time_line = global_entanglement_over_time[local_dataset['data_obj'].friction_coefficient]['time_line']
#         mean_velocity = global_entanglement_over_time[local_dataset['data_obj'].friction_coefficient]['mean_velocity']
#         ax.plot(time_line,mean_velocity,'o-',label=f'{local_dataset["data_obj"].friction_coefficient}')
#     plt.xlabel('Time, $t$ (sec)')
#     plt.ylabel('Mean relative velocity, $v$ (L/sec)')
#     plt.savefig(f'{figure_output_folder}/MeanVelocityOverTime_AR{AR}.png',dpi=300,bbox_inches='tight')

#     fig,ax=plt.subplots(figsize=(2.5,2))
#     for local_dataset in _trimmed:
#         time_line = global_entanglement_over_time[local_dataset['data_obj'].friction_coefficient]['time_line']
#         mean_velocity = global_entanglement_over_time[local_dataset['data_obj'].friction_coefficient]['mean_velocity']
#         ax.plot(time_line,mean_velocity,'o-',label=f'{local_dataset["data_obj"].friction_coefficient}')
#     plt.xlabel('Time, $t$ (sec)')
#     plt.ylabel('Mean relative velocity, $v$ (L/sec)')

#     fig,ax=plt.subplots(figsize=(2.5,2))
#     for global_data_list,global_entanglement_over_time in zip(AR_data,AR_entanglement_data):
        
#         fric_coeff = [local_dataset['data_obj'].friction_coefficient for local_dataset in global_data_list]
#         fric_coeff = np.array(fric_coeff)

#         final_entanglement = []

#         for local_dataset in global_data_list:
#             time_line = global_entanglement_over_time[local_dataset['data_obj'].friction_coefficient]['time_line']
#             entanglement = global_entanglement_over_time[local_dataset['data_obj'].friction_coefficient]['entanglement']
#             entanglement = np.array(entanglement)/(num_rods*(num_rods-1)/2)
#             ax.plot(time_line/t_u,entanglement,label=f'{local_dataset["data_obj"].friction_coefficient}')

#             # break
#             # final_entanglement.append(entanglement[-1])
#             # ax.loglog(time_line,entanglement,label=f'{local_dataset["data_obj"].friction_coefficient}')
#     plt.legend()
#         # plt.plot(fric_coeff,final_entanglement,'o-',label=f'AR={local_dataset["AR"]}')
#         # plt.xlabel('Friction coefficient, $\\mu$')
#         # plt.ylabel(r'Final entanglement, $\tilde{e}$')

#     # entanglement over time for different AR
#     fig,ax=plt.subplots(figsize=(2.5,2))
#     for global_data_list,global_entanglement_over_time in zip(AR_data,AR_entanglement_data):
#         for local_dataset in global_data_list:

#             time_line = global_entanglement_over_time[local_dataset['data_obj'].friction_coefficient]['time_line']
#             entanglement = global_entanglement_over_time[local_dataset['data_obj'].friction_coefficient]['entanglement']
#             entanglement = np.array(entanglement)/(num_rods*(num_rods-1)/2)
#             ax.plot(time_line,entanglement,label=f'{local_dataset["data_obj"].friction_coefficient}')

#     plt.xlabel(r'$t/t_u$')
#     plt.ylabel(r'Norm. entanglement, $\tilde{e}$')
#     plt.legend(title='$\mu$',loc='upper right')
#     plt.savefig(f'{figure_output_folder}/EntanglementOverTime_AR{AR}.png',dpi=300,bbox_inches='tight')
#     plt.savefig(f'{figure_output_folder}/EntanglementOverTime_AR{AR}.svg',dpi=300,bbox_inches='tight')

#     fig,ax=plt.subplots(figsize=(2.5,2))
#     for global_data_list,global_entanglement_over_time in zip(AR_data,AR_entanglement_data):
#         for local_dataset in global_data_list:

#             time_line = global_entanglement_over_time[local_dataset['data_obj'].friction_coefficient]['time_line']
#             entanglement = global_entanglement_over_time[local_dataset['data_obj'].friction_coefficient]['entanglement']
#             entanglement = np.array(entanglement)/(num_rods*(num_rods-1)/2)
#             ax.loglog(time_line,entanglement,label=f'{local_dataset["data_obj"].friction_coefficient}')

#     plt.xlabel(r'$t/t_u$')
#     plt.ylabel(r'Norm. entanglement, $\tilde{e}$')
#     plt.legend(title='$\mu$',loc='center left', bbox_to_anchor=(1, 0.5))
#     plt.savefig(f'{figure_output_folder}/EntanglementOverTimeLogLog_AR{AR}.png',dpi=300,bbox_inches='tight')
#     plt.savefig(f'{figure_output_folder}/EntanglementOverTimeLogLog_AR{AR}.svg',dpi=300,bbox_inches='tight')

#     plt.close('all')
#     fig,ax=plt.subplots(figsize=(2.5,2))
#     for global_data_list,global_entanglement_over_time in zip(AR_data,AR_entanglement_data):
#         for local_dataset in global_data_list:

#             time_line = global_entanglement_over_time[local_dataset['data_obj'].friction_coefficient]['time_line']
#             num_contacts = global_entanglement_over_time[local_dataset['data_obj'].friction_coefficient]['num_contacts']
#             num_contacts = np.array(num_contacts)
            
#             ax.loglog(time_line,num_contacts,label=f'{local_dataset["data_obj"].friction_coefficient}')

#     plt.xlabel(r'$t/t_u$')
#     plt.ylabel(r'Norm. entanglement, $\tilde{e}$')
#     plt.legend(title='$\mu$',loc='center left', bbox_to_anchor=(1, 0.5))
#     plt.savefig(f'{figure_output_folder}/NumContactsOverTimeLogLog_AR{AR}.png',dpi=300,bbox_inches='tight')
#     plt.savefig(f'{figure_output_folder}/NumContactsOverTimeLogLog_AR{AR}.svg',dpi=300,bbox_inches='tight')


#     # fric_coeff = [local_dataset['data_obj'].friction_coefficient for local_dataset in global_data_list]
#     # fric_coeff = np.array(fric_coeff)

#     # fig,ax=plt.subplots(figsize=(2.5,2))
#     # plt.plot(fric_coeff,final_radius_of_gyration,'o-')
#     # plt.xlabel('Friction coefficient, $\\mu$')
#     # plt.ylabel('Radius of gyration, $R_g$')

#     # # x_fit = np.linspace(0.1,0.2,100)
#     # # y_fit = 0.0001*x_fit**(-4)
#     # # plt.loglog(x_fit,y_fit,'--',color='k')

#     # # y_fit = 0.04*x_fit**(-1)
#     # # plt.loglog(x_fit,y_fit,'--',color='k')

#     # if not os.path.exists(figure_output_folder):
#     #     os.makedirs(figure_output_folder)
#     # # plt.savefig(f'{figure_output_folder}/RadiusOfGyration_AR{AR}.png',dpi=300,bbox_inches='tight')
#     # plt.savefig(f'{figure_output_folder}/RadiusOfGyration_AR{AR}.svg',bbox_inches='tight')

#     dta = chosen_data[2]
#     data = loadmat(dta.data_path,simplify_cells=True)
#     time_line = data['time_line']
#     node_list = data['node_list']
#     velocity_list = data['velocity_list']
#     contacts_list = data['contact_list']
#     friction_coefficient = dta.friction_coefficient
#     num_rods = dta.num_rods
#     density = 1000


#     file_id = f'N{num_rods}-AR{int(AR):04d}-Kick{kick_amplitude}-Friction{friction_coefficient}-Density{density}'
#     output_path = f'/Users/yeonsu/Videos/{subfolder_name}/{file_id}'

#     import os
#     if not os.path.exists(output_path):
#         os.makedirs(output_path)

#     import polyscope as ps
#     from visualizations import prep_for_polyscope

#     rod_diameter = 1/AR
#     # rod_diameter = 1/500
#     ps.init()

#     _t = 0
#     a_list_of_curves = node_list[_t].reshape(num_rods,-1,3)
#     nodes,edges,edge_colors = prep_for_polyscope(a_list_of_curves,num_rods)
#     min_z = np.min(nodes[:,2])
#     # ps.set_ground_plane_height_factor(-min_z)
                
#     ps_curves = ps.register_curve_network("filaments",nodes,edges)
#     ps_curves.add_color_quantity("edge_colors",edge_colors,defined_on='edges',enabled=True)
#     ps_curves.set_radius(rod_diameter/2,relative=False)

#     ps.set_up_dir("z_up")
#     ps.screenshot('temp.png',transparent_bg=False)
#     # ps.show()
#     # exit()
#     num_files_already = len(list(Path(output_path).glob('frame_*')))
#     print(f'Number of frames: {num_files_already}')

#     skip_factor = 10
#     for i, a_list_of_curves in enumerate(node_list[num_files_already::skip_factor]):
#         a_list_of_curves = a_list_of_curves.reshape(num_rods,-1,3)
#         # num_rods = len(a_list_of_curves)
#         nodes = np.vstack(a_list_of_curves)
#         ps_curves.update_node_positions(nodes)
#         file_path = f'{output_path}/frame_{i+num_files_already:04d}.png'
#         ps.screenshot(file_path)

#     import subprocess
#     subprocess.run(['ffmpeg', '-framerate', '10', '-i', f'{output_path}/frame_%04d.png', '-r', '30', '-pix_fmt', 'yuv420p', f'{output_path}/output.mp4', '-y'])

#     return data_list
# # %%

# higher_data_list = []
# for meta_folder in meta_folders:
#     higher_data_list.append(analyze_a_single_metafolder(meta_folder))

# # dump
# import pickle

# with open(f'higher_data_list.pkl','wb') as f:
#     pickle.dump(higher_data_list,f)
# %%
