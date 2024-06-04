# %%
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import glob
import re
import pickle

from data_io import import_all_log, parse_path_string    
    
# %%

output_root = '/Users/yeonsu/Dropbox (Harvard University)/Data/PrunedData/rod-sim-pnas-revision'

# Jostle
protocol_id = 'HangEntangledCarrotCake5'

pathlist = []
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/JostleCarrotCake5/20240531-2228_RUN_JostleCarrotCake5_N0125_AR025_g0.5')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/JostleCarrotCake5/20240531-2228_RUN_JostleCarrotCake5_N0250_AR050_g0.5')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/JostleCarrotCake5/20240531-2228_RUN_JostleCarrotCake5_N0375_AR075_g0.5')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/JostleCarrotCake5/20240531-2228_RUN_JostleCarrotCake5_N0500_AR100_g0.5')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/JostleCarrotCake5/20240531-2228_RUN_JostleCarrotCake5_N0625_AR125_g0.5')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/JostleCarrotCake5/20240531-2228_RUN_JostleCarrotCake5_N1000_AR200_g0.5')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/JostleCarrotCake5/20240531-2228_RUN_JostleCarrotCake5_N1500_AR300_g0.5')

pathlist.append('/Users/yeonsu/Data/from_cluster/20240604-0050_RUN_HangEECarrotCake5_N125_AR25')
# /Users/yeonsu/Data/from_cluster/NonIntersectingBox-N375-AR75-Scale1-mu0.20-visc0.00-amp0.00_allLog_20240604-011053.csv

# pathlist.append('/Users/yeonsu/GitHub/dismech-rods-main/runs/20240603-2216_RUN_N500_AR100/log_files')

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
        
    nodes_in_matrix = node_list[0].reshape((-1,30))
    packing_center = np.mean(np.mean(nodes_in_matrix.reshape((-1,10,3)),axis=1),axis=0)
    
    
    fig,ax=plt.subplots(subplot_kw={'projection':'3d'})
    locked_nodes = []
    for rr in nodes_in_matrix.reshape((-1,10,3)):
        I = np.linalg.norm(rr - packing_center,axis=1) < 0.1
        # ax.plot(rr[I,0],rr[I,1],rr[I,2],'k-')
        locked_nodes.append(rr[I,:])
        
    print(f'Number of locked nodes: {len(locked_nodes)}')
    
    for i in range(start_point,len(node_list),1):
        nodes_in_matrix = node_list[i].reshape((-1,30))
        for node in nodes_in_matrix:
            rr = node.reshape((-1,3))
            ax.plot(rr[:,0],rr[:,1],rr[:,2],alpha=0.1)
        for rr in locked_nodes:
            ax.plot(rr[:,0],rr[:,1],rr[:,2],'k-',linewidth=2)
            
        ax.set_xlim(-2,2)
        ax.set_ylim(-2,2)
        ax.set_zlim(-2,2)
        ax.view_init(elev=0,azim=0)
        ax.text(1,1,1,f'time: {time_line[i]}')
        plt.tight_layout(pad=0)
        
        plt.savefig(f'{output_path}/frames_{i:04d}.png', dpi=300, bbox_inches='tight', pad_inches=0)
        ax.clear()
        
    with open(f'{output_path}/log.txt','w') as f:
        f.write(log_string)