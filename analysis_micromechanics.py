# %%
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
from data_io import import_all_log, parse_path_string
from analysis import get_curr_force_essentials
import re
import k3d
from analysis import process_contact_data
from visualizations import plot_contacts
import networkx as nx
from distances import lumelsky_dist_vec   

import filamentFields
# %%


def analyze_a_path(pth):
    data_path = None
    for file in Path(pth).rglob('*.csv'):
        if str(file.stem).endswith('lastFrame'):
            continue        
        data_path = file
        break

    log_string = ''

    file_id,surfix,num_rods,AR,datetime_string = parse_path_string(data_path)
    time_line, node_list, contact_list = import_all_log(data_path,max_rows=100000)

    time_line = np.array(time_line)
    time_line = time_line[time_line <= 10]
    node_list = node_list[:len(time_line)]
    contact_list = contact_list[:len(time_line)]

    time_line = time_line[1:]
    node_list = node_list[1:]
    contact_list = contact_list[1:]

    print(f'Size of time_line: {len(time_line)}')
    print(f'Number of rods: {num_rods}')

    log_string = log_string + f'Number of rods: {num_rods}\n'
    log_string = log_string + f'Number of time points: {len(time_line)}\n'

    total_number_of_contacts = np.zeros(len(time_line))
    total_force_sum = np.zeros(len(time_line))

    last_frame = len(time_line)-1
    print(f'Last frame: {last_frame}')

    avg_velocities_over_time = np.zeros(len(time_line)-1)
    centroid_velocities_over_time = np.zeros(len(time_line)-1)
    avg_contact_displacement_over_time = np.zeros(len(time_line)-1)
    avg_initial_centroid_displacement_over_time = np.zeros(len(time_line)-1)
    total_entanglement_over_time = np.zeros(len(time_line)-1)

    # correlation with the initial data!
    initial_nodes = node_list[0].reshape((-1,10,3))
    
    # fF = filamentFields.filamentFields([],[])

    for frame in range(0,len(time_line)-1,1):
        curr_nodes = node_list[frame].reshape((-1,10,3))
        next_nodes = node_list[frame+1].reshape((-1,10,3))
        curr_force_all_info = contact_list[frame].reshape(-1,18)
        next_force_all_info = contact_list[frame+1].reshape(-1,18)        
        
        R_omega = 200
        fF.update_filament_nodes_list(curr_nodes)
        fF.precompute(R_omega)
        
        total_entanglement_over_time[i_frame] = fF.return_total_entanglement()
        
        contact_ij = curr_force_all_info[:,4:6].astype(int)
        graph = nx.Graph()
        graph.add_nodes_from(range(len(curr_nodes)))
        graph.add_edges_from(contact_ij)
        
        initial_node_displacement = curr_nodes - initial_nodes
        avg_initial_centroid_displacement =  np.mean(np.linalg.norm(np.mean(initial_node_displacement,axis=1),axis=1))
        avg_initial_centroid_displacement_over_time[frame] = avg_initial_centroid_displacement

        # plt.plot(avg_initial_centroid_displacement_over_time)
        # plt.xlabel('Frame')
        # plt.ylabel('Average initial centroid displacement (mm)')

        # fig,ax=plt.subplots(1,1,figsize=(10,10),subplot_kw={'projection':'3d'})
        # rod_label = 130
        # ax.plot(curr_nodes[rod_label][:,0],curr_nodes[rod_label][:,1],curr_nodes[rod_label][:,2],'k',linewidth=1)
        # ax.plot(initial_nodes[rod_label][:,0],initial_nodes[rod_label][:,1],initial_nodes[rod_label][:,2],'r',linewidth=1)    
        
        # for i_rod,rod in enumerate(range(len(curr_nodes))):
        #     neighbors = list(graph[i_rod])
        #     for neighbor in neighbors:
        #         rod = curr_nodes[i_rod]
        #         rod_next = next_nodes[i_rod]
        #         rod_velolcity = rod_next - rod
        #     break
        
        contact_displacement_list = np.zeros(len(curr_force_all_info))
        for i_,contact_entry in enumerate(curr_force_all_info):
            popt_i,popt_j,dvec,x_i1,x_i2,x_j1,x_j2 = get_closest_points(contact_entry,curr_nodes)
            popt_i_next,popt_j_next,dvec_next,x_i1_next,x_i2_next,x_j1_next,x_j2_next = get_closest_points(contact_entry,next_nodes)
            contact_displacement = dvec_next - dvec
            contact_displacement_norm = np.linalg.norm(contact_displacement)
            contact_displacement_list[i_] = contact_displacement_norm
            
        average_contact_displacement = np.mean(contact_displacement_list)
        avg_contact_displacement_over_time[frame] = average_contact_displacement
        
            
            
# %%
output_root = '/Users/yeonsu/Dropbox (Harvard University)/Data/PrunedData/rod-sim-pnas-revision'

# Entangle
protocol_id = 'Micromechanics-HeavyHangModelos'

output_path = f'{output_root}/{protocol_id}'
if not os.path.exists(output_path):
    os.makedirs(output_path)

# %%
parent_folders = []
parent_folders.append(Path('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/HeavyHangModelo1'))
# parent_folders.append(Path('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/Modelo3_FineExcitation'))
# parent_folders.append(Path('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/Modelo2_FineExcitation'))
# pathlist is subdirs, not including itself

avg_velocities_wrt_AR_repeated = []
avg_centroid_velocity_wrt_AR_repeated = []
avg_contact_displacement_wrt_AR_repeated = []
avg_initial_centroid_displacement_wrt_AR_repeated = []
initial_centroid_disp_over_time_wrt_AR_repeated = []
total_entanglement_over_time_wrt_AR_repeated = []

skip_frames = 249
for parent_folder in parent_folders:
    pathlist = [str(x) for x in parent_folder.iterdir() if x.is_dir()]

    ARs = []
    for pth in pathlist:
        search_result = re.search(r'N(\d+)_AR(\d+)',pth)
        ARs.append(int(search_result.group(2)))

    pathlist = [x for _,x in sorted(zip(ARs,pathlist))]
    ARs = sorted(ARs)
    
    avg_velocities_wrt_AR = []
    avg_centroid_velocity_wrt_AR = []
    avg_contact_displacement_wrt_AR = []
    avg_initial_centroid_displacement_wrt_AR = []
    initial_centroid_disp_over_time_wrt_AR = []
    total_entanglement_over_time_wrt_AR = []
    
    for pth in pathlist:
        # if os.path.exists(f'{output_path}/avg_contact_displacement_wrt_AR.pkl'):
        #     exit()

        # if not os.path.exists(f'{output_path}/avg_contact_displacement_wrt_AR.pkl'):
        #     pass
        
        # find csv file
        data_path = None
        for file in Path(pth).rglob('*.csv'):
            if str(file.stem).endswith('lastFrame'):
                continue        
            data_path = file
            break
        
        log_string = ''
        
        file_id,surfix,num_rods,AR,datetime_string = parse_path_string(data_path)
        time_line, node_list, contact_list = import_all_log(data_path,max_rows=100000)

        time_line = np.array(time_line)
        time_line = time_line[time_line <= 10]
        node_list = node_list[:len(time_line)]
        contact_list = contact_list[:len(time_line)]

        time_line = time_line[1:]
        node_list = node_list[1:]
        contact_list = contact_list[1:]

        print(f'Size of time_line: {len(time_line)}')
        print(f'Number of rods: {num_rods}')

        log_string = log_string + f'Number of rods: {num_rods}\n'
        log_string = log_string + f'Number of time points: {len(time_line)}\n'

        total_number_of_contacts = np.zeros(len(time_line))
        total_force_sum = np.zeros(len(time_line))

        last_frame = len(time_line)-1
        print(f'Last frame: {last_frame}')
        
        initial_nodes = node_list[0].reshape((-1,10,3))
        timeline_checkout = range(0,len(time_line)-1,skip_frames)
        avg_velocities_over_time = np.zeros( len(timeline_checkout) )
        centroid_velocities_over_time = np.zeros( len(timeline_checkout) )
        avg_contact_displacement_over_time = np.zeros( len(timeline_checkout) )
        avg_initial_centroid_displacement_over_time = np.zeros( len(timeline_checkout) )
        total_entanglement_over_time = np.zeros( len(timeline_checkout) )
        
        # fF = filamentFields.filamentFields([],[])
        
        for i_frame,frame in enumerate(timeline_checkout):
            
            curr_nodes = node_list[frame].reshape((-1,10,3))
            next_nodes = node_list[frame+1].reshape((-1,10,3))
            curr_force_all_info = contact_list[frame].reshape(-1,18)
            next_force_all_info = contact_list[frame+1].reshape(-1,18)
            
            initial_node_displacement = curr_nodes - initial_nodes
            avg_initial_centroid_displacement =  np.mean(np.linalg.norm(np.mean(initial_node_displacement,axis=1),axis=1))
            avg_initial_centroid_displacement_over_time[i_frame] = avg_initial_centroid_displacement
            
            contact_displacement_list = np.zeros(len(curr_force_all_info))
            
            contact_ij = curr_force_all_info[:,4:6].astype(int)
            contact_ij_next_frame = next_force_all_info[:,4:6].astype(int)            
            graph = nx.Graph()
            graph.add_nodes_from(range(len(curr_nodes)))
            graph.add_edges_from(contact_ij)
            
            
    
            # fF.update_filament_nodes_list(curr_nodes)
            # R_omega = 10000
            # fF.compute_total_linking_matrix()
            # lk_mat = fF.return_total_linking_matrix()
            
            # total_entanglement_over_time[i_frame] = fF.return_total_entanglement()
            
            # for i_,contact_entry in enumerate(curr_force_all_info):
            #     popt_i,popt_j,dvec,x_i1,x_i2,x_j1,x_j2 = get_closest_points(contact_entry,curr_nodes)
            #     popt_i_next,popt_j_next,dvec_next,x_i1_next,x_i2_next,x_j1_next,x_j2_next = get_closest_points(contact_entry,next_nodes)
            #     contact_displacement = dvec_next - dvec
            #     contact_displacement_norm = np.linalg.norm(contact_displacement)
            #     contact_displacement_list[i_] = contact_displacement_norm
                
            average_contact_displacement = np.mean(contact_displacement_list)
            avg_contact_displacement_over_time[i_frame] = average_contact_displacement
            
            # node velocities
            rod_velocities = np.zeros((num_rods,10,3))
            for i_rod in range(0,num_rods,1):
                curr_rod = curr_nodes[i_rod]
                next_rod = next_nodes[i_rod]
                rod_velocities[i_rod] = next_rod - curr_rod
                
            avg_velocity_at_the_frame = np.mean(np.linalg.norm(rod_velocities.reshape(-1,3),axis=1))
            avg_velocities_over_time[i_frame] = avg_velocity_at_the_frame
            
            centroid_velocities_at_the_frame = np.mean(rod_velocities,axis=1)
            centroid_velocities_over_time[i_frame] = np.mean(np.linalg.norm(centroid_velocities_at_the_frame,axis=1))
            
        avg_velocities_wrt_AR.append(np.mean(avg_velocities_over_time*0.05/1e-5/100))
        avg_centroid_velocity_wrt_AR.append(np.mean(centroid_velocities_over_time*0.05/1e-5/100))
        avg_contact_displacement_wrt_AR.append(np.mean(avg_contact_displacement_over_time*0.05/1e-5/100))
        avg_initial_centroid_displacement_wrt_AR.append(np.mean(avg_initial_centroid_displacement_over_time*0.05/1e-5/100))
        initial_centroid_disp_over_time_wrt_AR.append(avg_initial_centroid_displacement_over_time*0.05/1e-5/100)        
        total_entanglement_over_time_wrt_AR.append(total_entanglement_over_time)
    
    avg_velocities_wrt_AR_repeated.append(avg_velocities_wrt_AR)
    avg_centroid_velocity_wrt_AR_repeated.append(avg_centroid_velocity_wrt_AR)
    avg_contact_displacement_wrt_AR_repeated.append(avg_contact_displacement_wrt_AR)
    avg_initial_centroid_displacement_wrt_AR_repeated.append(avg_initial_centroid_displacement_wrt_AR)
    initial_centroid_disp_over_time_wrt_AR_repeated.append(initial_centroid_disp_over_time_wrt_AR)
    total_entanglement_over_time_wrt_AR_repeated.append(total_entanglement_over_time_wrt_AR)
    
# %% save data
import pickle
with open(f'{output_path}/avg_velocities_wrt_AR.pkl','wb') as f:
    pickle.dump(avg_velocities_wrt_AR_repeated,f)
    
with open(f'{output_path}/avg_centroid_velocity_wrt_AR.pkl','wb') as f:
    pickle.dump(avg_centroid_velocity_wrt_AR_repeated,f)
    
with open(f'{output_path}/avg_contact_displacement_wrt_AR.pkl','wb') as f:
    pickle.dump(avg_contact_displacement_wrt_AR_repeated,f)
    
with open(f'{output_path}/avg_initial_centroid_displacement_wrt_AR.pkl','wb') as f:
    pickle.dump(avg_initial_centroid_displacement_wrt_AR_repeated,f)
    
with open(f'{output_path}/initial_centroid_disp_over_time_wrt_AR.pkl','wb') as f:
    pickle.dump(initial_centroid_disp_over_time_wrt_AR_repeated,f)
    
with open(f'{output_path}/total_entanglement_over_time_wrt_AR.pkl','wb') as f:
    pickle.dump(total_entanglement_over_time_wrt_AR_repeated,f)
# %%
for i_,dta in enumerate(total_entanglement_over_time_wrt_AR[::2]):
    dta0 = dta[0]
    plt.plot(dta-dta0,'o-',label=f'AR={ARs[i_]}')

plt.legend()

# %%
for i_,dta in enumerate(initial_centroid_disp_over_time_wrt_AR):
    plt.plot(time_line[timeline_checkout],dta,'o-',label=f'AR={ARs[i_]}')
plt.xlabel('Time (s)')
plt.ylabel('Centroid displacement (mm)')
plt.legend()

# %%
for i_,dta in enumerate(initial_centroid_disp_over_time_wrt_AR):
    plt.loglog(dta,'o-',label=f'AR={ARs[i_]}')
plt.legend()
# %%
plt.loglog(initial_centroid_disp_over_time_wrt_AR[0],'o-')
plt.loglog(initial_centroid_disp_over_time_wrt_AR[5],'o-')
plt.loglog(initial_centroid_disp_over_time_wrt_AR[-2],'o-')
plt.loglog(initial_centroid_disp_over_time_wrt_AR[-1],'o-')
    
# %%



plt.plot(time_line[timeline_checkout],avg_initial_centroid_displacement_over_time)

# %%
plt.plot(time_line[timeline_checkout],centroid_velocities_over_time)

# %%
for dta in avg_initial_centroid_displacement_wrt_AR_repeated:
    plt.loglog(ARs,dta,'o-')
# %%

# %%
mean_dta = np.mean(avg_initial_centroid_displacement_wrt_AR_repeated,axis=0)
plt.loglog(ARs,mean_dta,'o-')

# %%
# fit power law
from scipy.optimize import curve_fit
def power_law(x,a,b):
    return a*x**b

popt,pcov = curve_fit(power_law,ARs,mean_dta)
plt.errorbar(ARs,mean_dta,yerr=np.std(avg_initial_centroid_displacement_wrt_AR_repeated,axis=0)*0.05/1e-5/100,fmt='o',label='Data')
plt.loglog(ARs,power_law(ARs,*popt),'r-',label=f'Fit: {popt[0]:.2e}x^{popt[1]:.2e}')
plt.legend()



# %%
for dta in avg_contact_displacement_wrt_AR_repeated:
    plt.loglog(ARs,dta,'o-')
    
# %%
avg_dta = np.mean(avg_contact_displacement_wrt_AR_repeated,axis=0)*0.05/1e-5/100
# %%
# fit power law
from scipy.optimize import curve_fit
def power_law(x,a,b):
    return a*x**b

popt,pcov = curve_fit(power_law,ARs,avg_dta)
plt.errorbar(ARs,avg_dta,yerr=np.std(avg_contact_displacement_wrt_AR_repeated,axis=0)*0.05/1e-5/100,fmt='o')
plt.loglog(ARs,power_law(ARs,*popt),'r-')

plt.xlabel('Aspect ratio')
plt.ylabel('Average contact displacement (mm)')

# %%
rod_packing_topology = np.zeros((num_rods,num_rods))
for _i in range(num_rods):
    rod_i = curr_nodes[_i]
    for _j in range(num_rods):
        rod_j = curr_nodes[_j]
        
# %%
# pathlist = []

# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/StudyPhaseDiagram/20240609-1051_RUN_TickleEEModelo1_N0125_AR025_a100')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/StudyPhaseDiagram/20240609-1052_RUN_TickleEEModelo1_N0500_AR100_a100')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/StudyPhaseDiagram/20240609-1052_RUN_TickleEEModelo1_N1500_AR300_a100')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/StudyPhaseDiagram/20240613-1743_RUN_TickleModelo1_N0500_AR100_a02')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/StudyPhaseDiagram/20240613-1743_RUN_TickleModelo1_N0500_AR100_a05')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/StudyPhaseDiagram/20240613-1743_RUN_TickleModelo1_N0500_AR100_a10')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/StudyPhaseDiagram/20240613-1743_RUN_TickleModelo1_N0500_AR100_a25')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/StudyPhaseDiagram/20240613-1743_RUN_TickleModelo1_N0500_AR100_a50')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/StudyPhaseDiagram/20240613-1746_RUN_TickleModelo1_N0125_AR025_a02')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/StudyPhaseDiagram/20240613-1746_RUN_TickleModelo1_N0125_AR025_a05')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/StudyPhaseDiagram/20240613-1746_RUN_TickleModelo1_N0125_AR025_a10')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/StudyPhaseDiagram/20240613-1746_RUN_TickleModelo1_N0125_AR025_a25')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/StudyPhaseDiagram/20240613-1746_RUN_TickleModelo1_N0125_AR025_a50')

# # %%
# avg_velocities_wrt_AR_repeated = []
# avg_centroid_velocity_wrt_AR_repeated = []
# avg_contact_displacement_wrt_AR_repeated = []
# avg_initial_centroid_displacement_wrt_AR_repeated = []
# initial_centroid_disp_over_time_wrt_AR_repeated = []
# total_entanglement_over_time_wrt_AR_repeated = []

# skip_frames = 5
# for pth in pathlist:
#     # if os.path.exists(f'{output_path}/avg_contact_displacement_wrt_AR.pkl'):
#     #     exit()

#     # if not os.path.exists(f'{output_path}/avg_contact_displacement_wrt_AR.pkl'):
#     #     pass
    
#     # find csv file
#     data_path = None
#     for file in Path(pth).rglob('*.csv'):
#         if str(file.stem).endswith('lastFrame'):
#             continue        
#         data_path = file
#         break
    
#     log_string = ''
    
#     file_id,surfix,num_rods,AR,datetime_string = parse_path_string(data_path)
#     time_line, node_list, contact_list = import_all_log(data_path,max_rows=100000)

#     time_line = np.array(time_line)
#     time_line = time_line[time_line <= 10]
#     node_list = node_list[:len(time_line)]
#     contact_list = contact_list[:len(time_line)]

#     time_line = time_line[1:]
#     node_list = node_list[1:]
#     contact_list = contact_list[1:]

#     print(f'Size of time_line: {len(time_line)}')
#     print(f'Number of rods: {num_rods}')

#     log_string = log_string + f'Number of rods: {num_rods}\n'
#     log_string = log_string + f'Number of time points: {len(time_line)}\n'

#     total_number_of_contacts = np.zeros(len(time_line))
#     total_force_sum = np.zeros(len(time_line))

#     last_frame = len(time_line)-1
#     print(f'Last frame: {last_frame}')
    
#     initial_nodes = node_list[0].reshape((-1,10,3))        
#     timeline_checkout = range(0,len(time_line)-1,skip_frames)
#     avg_velocities_over_time = np.zeros( len(timeline_checkout) )
#     centroid_velocities_over_time = np.zeros( len(timeline_checkout) )
#     avg_contact_displacement_over_time = np.zeros( len(timeline_checkout) )
#     avg_initial_centroid_displacement_over_time = np.zeros( len(timeline_checkout) )        
#     total_entanglement_over_time = np.zeros( len(timeline_checkout) )
    
#     fF = filamentFields.filamentFields([],[])
    
#     for i_frame,frame in enumerate(timeline_checkout):
        
#         curr_nodes = node_list[frame].reshape((-1,10,3))
#         next_nodes = node_list[frame+1].reshape((-1,10,3))
#         curr_force_all_info = contact_list[frame].reshape(-1,18)
#         next_force_all_info = contact_list[frame+1].reshape(-1,18)
        
#         initial_node_displacement = curr_nodes - initial_nodes
#         avg_initial_centroid_displacement =  np.mean(np.linalg.norm(np.mean(initial_node_displacement,axis=1),axis=1))
#         avg_initial_centroid_displacement_over_time[i_frame] = avg_initial_centroid_displacement
        
#         contact_displacement_list = np.zeros(len(curr_force_all_info))
        
#         contact_ij = curr_force_all_info[:,4:6].astype(int)
#         contact_ij_next_frame = next_force_all_info[:,4:6].astype(int)            
#         graph = nx.Graph()
#         graph.add_nodes_from(range(len(curr_nodes)))
#         graph.add_edges_from(contact_ij)
        
#         graph[0]

#         fF.update_filament_nodes_list(curr_nodes)
#         R_omega = 10000
#         fF.compute_total_linking_matrix()
#         lk_mat = fF.return_total_linking_matrix()
        
#         lk_mat[0,15]
        
#         total_entanglement_over_time[i_frame] = fF.return_total_entanglement()
        
#         for i_,contact_entry in enumerate(curr_force_all_info):
#             popt_i,popt_j,dvec,x_i1,x_i2,x_j1,x_j2 = get_closest_points(contact_entry,curr_nodes)
#             popt_i_next,popt_j_next,dvec_next,x_i1_next,x_i2_next,x_j1_next,x_j2_next = get_closest_points(contact_entry,next_nodes)
#             contact_displacement = dvec_next - dvec
#             contact_displacement_norm = np.linalg.norm(contact_displacement)
#             contact_displacement_list[i_] = contact_displacement_norm
            
#         average_contact_displacement = np.mean(contact_displacement_list)
#         avg_contact_displacement_over_time[i_frame] = average_contact_displacement
        
#         # node velocities
#         rod_velocities = np.zeros((num_rods,10,3))
#         for i_rod in range(0,num_rods,1):
#             curr_rod = curr_nodes[i_rod]
#             next_rod = next_nodes[i_rod]
#             rod_velocities[i_rod] = next_rod - curr_rod
            
#         avg_velocity_at_the_frame = np.mean(np.linalg.norm(rod_velocities.reshape(-1,3),axis=1))
#         avg_velocities_over_time[i_frame] = avg_velocity_at_the_frame
        
#         centroid_velocities_at_the_frame = np.mean(rod_velocities,axis=1)
#         centroid_velocities_over_time[i_frame] = np.mean(np.linalg.norm(centroid_velocities_at_the_frame,axis=1))
        
#     avg_velocities_wrt_AR.append(np.mean(avg_velocities_over_time*0.05/1e-5/100))
#     avg_centroid_velocity_wrt_AR.append(np.mean(centroid_velocities_over_time*0.05/1e-5/100))
#     avg_contact_displacement_wrt_AR.append(np.mean(avg_contact_displacement_over_time*0.05/1e-5/100))
#     avg_initial_centroid_displacement_wrt_AR.append(np.mean(avg_initial_centroid_displacement_over_time*0.05/1e-5/100))
#     initial_centroid_disp_over_time_wrt_AR.append(avg_initial_centroid_displacement_over_time*0.05/1e-5/100)        
#     total_entanglement_over_time_wrt_AR.append(total_entanglement_over_time)

# avg_velocities_wrt_AR_repeated.append(avg_velocities_wrt_AR)
# avg_centroid_velocity_wrt_AR_repeated.append(avg_centroid_velocity_wrt_AR)
# avg_contact_displacement_wrt_AR_repeated.append(avg_contact_displacement_wrt_AR)
# avg_initial_centroid_displacement_wrt_AR_repeated.append(avg_initial_centroid_displacement_wrt_AR)
# initial_centroid_disp_over_time_wrt_AR_repeated.append(initial_centroid_disp_over_time_wrt_AR)
# total_entanglement_over_time_wrt_AR_repeated.append(total_entanglement_over_time_wrt_AR)
# # %%
