# %%
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import glob
import re
import pickle
import networkx as nx

from data_io import import_all_log, parse_path_string    
    
# %%

output_root = '/Users/yeonsu/Dropbox (Harvard University)/Data/PrunedData/rod-sim-pnas-revision'

# Jostle
protocol_id = 'HangEECarrotCake5'

pathlist = []
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/HangEECarrotCake5/20240604-0050_RUN_HangEECarrotCake5_N125_AR25')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/HangEECarrotCake5/20240604-0110_RUN_HangEECarrotCake5_N250_AR50')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/HangEECarrotCake5/20240604-0110_RUN_HangEECarrotCake5_N375_AR75')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/HangEECarrotCake5/20240604-0050_RUN_HangEECarrotCake5_N500_AR100')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/HangEECarrotCake5/20240604-0110_RUN_HangEECarrotCake5_N625_AR125')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/HangEECarrotCake5/20240604-0110_RUN_HangEECarrotCake5_N1000_AR200')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/HangEECarrotCake5/20240604-0110_RUN_HangEECarrotCake5_N1500_AR300')

# for pth in pathlist:    
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
    
#     print(f'Size of time_line: {len(time_line)}')
#     print(f'Number of rods: {num_rods}')
    
#     log_string = log_string + f'Number of rods: {num_rods}\n'
#     log_string = log_string + f'Number of time points: {len(time_line)}\n'
    
#     output_path = f'{output_root}/{protocol_id}/{file_id}_{surfix}'
#     if not os.path.exists(output_path):
#         os.makedirs(output_path)
#         start_point = 0
#     else:
#         start_point = len(glob.glob(f'{output_path}/*.png'))        
        
#     nodes_in_matrix = node_list[0].reshape((-1,30))
#     packing_center = np.mean(np.mean(nodes_in_matrix.reshape((-1,10,3)),axis=1),axis=0)
    
    
#     fig,ax=plt.subplots(subplot_kw={'projection':'3d'})
#     locked_nodes = []
#     for rr in nodes_in_matrix.reshape((-1,10,3)):
#         I = np.linalg.norm(rr - packing_center,axis=1) < 0.1
#         # ax.plot(rr[I,0],rr[I,1],rr[I,2],'k-')
#         locked_nodes.append(rr[I,:])
        
#     print(f'Number of locked nodes: {len(locked_nodes)}')
    
#     for i in range(start_point,len(node_list),1):
#         nodes_in_matrix = node_list[i].reshape((-1,30))
#         for node in nodes_in_matrix:
#             rr = node.reshape((-1,3))
#             ax.plot(rr[:,0],rr[:,1],rr[:,2],alpha=0.1)
            
#         for rr in locked_nodes:
#             ax.plot(rr[:,0],rr[:,1],rr[:,2],'k-',linewidth=2)
            
#         ax.set_xlim(-2,2)
#         ax.set_ylim(-2,2)
#         ax.set_zlim(-2,2)
#         ax.view_init(elev=0,azim=0)
#         ax.text(1,1,1,f'time: {time_line[i]}')
#         plt.tight_layout(pad=0)
        
#         plt.savefig(f'{output_path}/frames_{i:04d}.png', dpi=300, bbox_inches='tight', pad_inches=0)
#         ax.clear()
        
#     with open(f'{output_path}/log.txt','w') as f:
#         f.write(log_string)
        
# %%
fig,ax=plt.subplots()

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
    
    print(f'Size of time_line: {len(time_line)}')
    print(f'Number of rods: {num_rods}')
    
    log_string = log_string + f'Number of rods: {num_rods}\n'
    log_string = log_string + f'Number of time points: {len(time_line)}\n'
    
    output_path = f'{output_root}/{protocol_id}/{file_id}_{surfix}'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        start_point = 0
    else:
        start_point = len(glob.glob(f'{output_path}/*.png'))        
    start_point = 0
    nodes_in_matrix = node_list[0].reshape((-1,30))
    packing_center = np.mean(np.mean(nodes_in_matrix.reshape((-1,10,3)),axis=1),axis=0)
    
    # locked_nodes = []
    # for rr in nodes_in_matrix.reshape((-1,10,3)):
    #     I = np.linalg.norm(rr - packing_center,axis=1) < 0.1
    #     locked_nodes.append(rr[I,:])
    
    cluster_size = np.zeros((len(node_list)-1),dtype=int)
    
    for i in range(1,len(node_list),1):
        nodes_in_matrix = node_list[i].reshape((-1,30))
        
        curr_contact_list = contact_list[i]
        curr_contact_list = curr_contact_list.reshape(-1,18)
        
        graph = nx.Graph()
        graph.add_nodes_from(range(num_rods))
        graph.add_edges_from(curr_contact_list[:,4:6])
        conn_comp = list(nx.connected_components(graph))        
        cl_size = [len(cc) for cc in conn_comp]        
        cluster_size[i-1] = max(cl_size)
        
    ax.plot(time_line[1:],cluster_size/num_rods,label=f'AR = {AR}, N = {num_rods}')

plt.legend()
plt.xlabel('Time (sec)')
plt.ylabel('Fraction of largest cluster')
    

# at edges between (idx1, idx3) and (idx2, idx4)
# (4 * idx1 + e1, contact_gradient[e1], idx5);
# (4 * idx3 + e1, contact_gradient[e1 + 3], idx5);
# (4 * idx2 + e1, contact_gradient[e1 + 6], idx6);
# (4 * idx4 + e1, contact_gradient[e1 + 9], idx6);