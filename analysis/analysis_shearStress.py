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
    
# %%
output_root = '/Users/yeonsu/Dropbox (Harvard University)/Data/PrunedData/rod-sim-pnas-revision'

# Entangle
protocol_id = 'PerturbCarrotCake5_fine'

pathlist = []
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/CarrotCake5_FineExcitation/20240607-1808_RUN_PerturbEECarrotCake5_N125_AR25_g0.5')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/CarrotCake5_FineExcitation/20240607-1809_RUN_PerturbEECarrotCake5_N250_AR50_g0.5')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/CarrotCake5_FineExcitation/20240607-1809_RUN_PerturbEECarrotCake5_N375_AR75_g0.5')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/CarrotCake5_FineExcitation/20240608-1901_RUN_PerturbEECarrotCake5_N620_AR124_g0.5')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/CarrotCake5_FineExcitation/20240607-1823_RUN_PerturbEECarrotCake5_N1500_AR300_g0.5')

# %%
for pth in pathlist:
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
    for frame in range(-1,len(time_line),1):
        curr_nodes = node_list[frame].reshape((-1,10,3))
        curr_force_all_info = contact_list[frame].reshape(-1,18)
        break
    break
        
        # curr_force_essentials = get_curr_force_essentials(curr_force_all_info,curr_nodes)        
        # total_number_of_contacts[frame] = len(curr_force_essentials)
        # total_force_sum[frame] = np.sum(np.linalg.norm(curr_force_essentials[:,3:6],axis=1))
# %%
# correlation?
frame = 1
curr_nodes = node_list[frame].reshape((-1,10,3))
next_nodes = node_list[frame+1].reshape((-1,10,3))
curr_force_all_info = contact_list[frame].reshape(-1,18)
next_force_all_info = contact_list[frame+1].reshape(-1,18)

contact_ij = curr_force_all_info[:,4:6].astype(int)
contact_ij_next_frame = next_force_all_info[:,4:6].astype(int)

import networkx as nx

graph = nx.Graph()
graph.add_nodes_from(range(len(curr_nodes)))
graph.add_edges_from(contact_ij)

graph_next_frame = nx.Graph()
graph_next_frame.add_nodes_from(range(len(next_nodes)))
graph_next_frame.add_edges_from(contact_ij_next_frame)

hub_rod_label = 10
hub_rod = curr_nodes[hub_rod_label]

fig,ax=plt.subplots(1,1,figsize=(10,10),subplot_kw={'projection':'3d'})
ax.plot(hub_rod[:,0],hub_rod[:,1],hub_rod[:,2],'k',linewidth=0.2)


for neighbors in list(graph[hub_rod_label]):
    rod = curr_nodes[neighbors]
    next_rod = next_nodes[neighbors]
    
    ax.plot(rod[:,0],rod[:,1],rod[:,2],'k',linewidth=0.2)
    vel = (next_rod - rod)*100
    ax.quiver(rod[:,0],rod[:,1],rod[:,2],vel[:,0],vel[:,1],vel[:,2],color='b')
    
ax.view_init(0,0)
# %%
fig,ax=plt.subplots(1,1,figsize=(10,10),subplot_kw={'projection':'3d'})
for rod in curr_nodes:
    ax.plot(rod[:,0],rod[:,1],rod[:,2],'k',linewidth=0.2)
    
for rod in next_nodes:
    ax.plot(rod[:,0],rod[:,1],rod[:,2],'r',linewidth=0.2)
ax.view_init(0,0)
# ax.set_ylim([-0.5,0.5])
ax.axis('equal')

num_rods = len(curr_nodes)
# node velocities
rod_velocities = np.zeros((num_rods,10,3))
for i_rod in range(0,num_rods,1):
    curr_rod = curr_nodes[i_rod]
    next_rod = next_nodes[i_rod]    
    rod_velocities[i_rod] = next_rod - curr_rod
    
# %%
rod_velocities.shape

np.linalg.norm(np.mean(np.mean(rod_velocities,axis=0),axis=0))

# %%
np.linalg.norm(rod_velocities)


# %%
arrow_scale_factor = 100

fig,ax=plt.subplots(1,1,figsize=(10,10),subplot_kw={'projection':'3d'})
for rod in curr_nodes:
    ax.plot(rod[:,0],rod[:,1],rod[:,2],'k',linewidth=0.2)
ax.view_init(0,0)
# %%
sigma = np.zeros((3,3))
for query_index in range(0,len(curr_force_all_info),1):
    single_contact_info = curr_force_all_info[query_index]
    contact_info = process_contact_data(single_contact_info,curr_nodes)
    
    rodlabel_i = contact_info['rod_i']
    rodlabel_j = contact_info['rod_j']
    contact_force_i = contact_info['contact_force_i']
    contact_force_j = contact_info['contact_force_j']
    
    centroid_i = np.mean(curr_nodes[rodlabel_i],axis=0)
    centroid_j = np.mean(curr_nodes[rodlabel_j],axis=0)
    
    # make it symmetric?
    sigma += (centroid_j - centroid_i)*contact_force_i[:,None]
    
    # plot_contacts(contact_info,arrow_scale_factor,ax)


    
# %%


fig,ax=plt.subplots(1,1,figsize=(10,10),subplot_kw={'projection':'3d'})
for i_rod in range(0,num_rods,1):
    rod_i = curr_nodes[i_rod]
    rod_j = next_nodes[i_rod]
    
    ax.plot(rod_i[:,0],rod_i[:,1],rod_i[:,2],'k',linewidth=0.2)
    ax.plot(rod_j[:,0],rod_j[:,1],rod_j[:,2],'r',linewidth=0.2)
    
    centroid_i = np.mean(rod_i,axis=0)
    centroid_j = np.mean(rod_j,axis=0)
    
    u_i = centroid_j - centroid_i
    u_i = u_i*100
    
    ax.quiver(centroid_i[0],centroid_i[1],centroid_i[2],u_i[0],u_i[1],u_i[2],color='b')
    
ax.view_init(0,0)
    
    

# %% using k3d
# arrow_scale_factor = 100
# plot = k3d.plot()
# for rod in curr_nodes:
#     plot += k3d.line(rod,shader='mesh',width=0.01)

# for query_index in range(0,len(curr_force_all_info),1):
#     single_contact_info = curr_force_all_info[query_index]
#     contact_info = process_contact_data(single_contact_info,curr_nodes)
#     plot += k3d.points(contact_info['contact_point_i'],point_size=0.01)
#     plot += k3d.points(contact_info['contact_point_j'],point_size=0.01)
#     plot += k3d.vectors(contact_info['contact_point_i'],contact_info['log_contact_force_i']/arrow_scale_factor)
#     plot += k3d.vectors(contact_info['contact_point_j'],contact_info['log_contact_force_j']/arrow_scale_factor)

# plot.display()

# %%
