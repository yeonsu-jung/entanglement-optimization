# %%
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import glob
import re

from data_io import import_all_log, parse_path_string    
    

# %%
# Perturb
# pathlist = []
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PerturbCarrotCake5/20240602-0259_RUN_PerturbEECarrotCake5_N125_AR25_g0.5')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PerturbCarrotCake5/20240602-0259_RUN_PerturbEECarrotCake5_N250_AR50_g0.5')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PerturbCarrotCake5/20240602-0259_RUN_PerturbEECarrotCake5_N375_AR75_g0.5')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PerturbCarrotCake5/20240602-0259_RUN_PerturbEECarrotCake5_N500_AR100_g0.5')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PerturbCarrotCake5/20240602-0259_RUN_PerturbEECarrotCake5_N625_AR125_g0.5')

# %%
# Jostle
pathlist = []
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/JostleCarrotCake5/20240531-2228_RUN_JostleCarrotCake5_N0125_AR025_g0.5')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/JostleCarrotCake5/20240531-2228_RUN_JostleCarrotCake5_N0250_AR050_g0.5')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/JostleCarrotCake5/20240531-2228_RUN_JostleCarrotCake5_N0375_AR075_g0.5')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/JostleCarrotCake5/20240531-2228_RUN_JostleCarrotCake5_N0500_AR100_g0.5')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/JostleCarrotCake5/20240531-2228_RUN_JostleCarrotCake5_N0625_AR125_g0.5')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/JostleCarrotCake5/20240531-2228_RUN_JostleCarrotCake5_N1000_AR200_g0.5')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/JostleCarrotCake5/20240531-2228_RUN_JostleCarrotCake5_N1500_AR300_g0.5')

for pth in pathlist:
    
    log_string = ''
    
    file_id,surfix,num_rods,AR,datetime_string = parse_path_string(alllog_pth)
    time_line, node_list, contact_list = import_all_log(alllog_pth,max_rows=100000)
    
    print(f'Size of time_line: {len(time_line)}')
    print(f'Number of rods: {num_rods}')
    
    log_string = log_string + f'Number of rods: {num_rods}\n'
    log_string = log_string + f'Number of time points: {len(time_line)}\n'
    
    output_path = f'/Users/yeonsu/Videos/{protocol_id}/{file_id}_{surfix}'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        start_point = 0
        
    fig,ax=plt.subplots(subplot_kw={'projection':'3d'})
    for i in range(start_point,len(node_list),1):
        nodes_in_matrix = node_list[i].reshape((-1,30))
        for node in nodes_in_matrix:
            rr = node.reshape((-1,3))
            ax.plot(rr[:,0],rr[:,1],rr[:,2])
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












# %%



# %%


# Formation of entanglement
node_contact_data_folder = '/Users/yeonsu/Dropbox (Harvard University)/Data/analysis-data/EntangleCarrotCake5'
field_analysis_folder = '/Users/yeonsu/Dropbox (Harvard University)/Data/analysis-data/EatEntangledCarrotCake5'

field_analysis_folder = Path(field_analysis_folder)

data_folder_list = []
for pth in field_analysis_folder.iterdir():    
    if str(pth.stem).startswith('.'):
        continue
    if pth.is_dir():
        data_folder_list.append(pth)
        print(f'Found data folder: {pth.stem}')
    
# %%
for data_folder in data_folder_list:
    # choose heavist folder
    data_dict = {}    
    
    for folders in data_folder.iterdir():
        # check size of the folder
        possible_files = []
        for files in folders.glob('**/all_fields_over_time.npz'):
            possible_files.append(files)
            
        if len(possible_files) == 0:
            print(f'No file found in {folders.stem}')
            continue
        
        if len(possible_files) == 1:
            pass
        
        if len(possible_files) > 1:
            print(f'Found multiple files in {folders.stem}')
            
            max_size = os.path.getsize(possible_files[0])
            heaviest_file = None
            for fpth in possible_files:
                size = os.path.getsize(fpth)
                if size > max_size:
                    heaviest_file = fpth
            
            possible_files = []
            possible_files.append(heaviest_file)
        
        re.search(data_folder.stem,'')
        
        data_dict['file_path'] = possible_files[0]
        data_dict['folder'] = folders


# %%

    
