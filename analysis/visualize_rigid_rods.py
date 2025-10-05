# %%
import numpy as np
from visualizations import prep_for_polyscope
from pathlib import Path
from data_io import import_all_log
import os


# %%
pathlist = []

pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PerturbRigidModelo1/20240722-0331_RUN_PerturbCalmRigidModelo1_N1250_AR250_freq10')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PerturbRigidModelo1/20240722-0331_RUN_PerturbCalmRigidModelo1_N0125_AR025_freq10')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PerturbRigidModelo1/20240722-0331_RUN_PerturbCalmRigidModelo1_N0250_AR050_freq10')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PerturbRigidModelo1/20240722-0331_RUN_PerturbCalmRigidModelo1_N0300_AR060_freq10')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PerturbRigidModelo1/20240722-0331_RUN_PerturbCalmRigidModelo1_N0350_AR070_freq10')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PerturbRigidModelo1/20240722-0331_RUN_PerturbCalmRigidModelo1_N0375_AR075_freq10')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PerturbRigidModelo1/20240722-0331_RUN_PerturbCalmRigidModelo1_N0400_AR080_freq10')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PerturbRigidModelo1/20240722-0331_RUN_PerturbCalmRigidModelo1_N0450_AR090_freq10')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PerturbRigidModelo1/20240722-0331_RUN_PerturbCalmRigidModelo1_N0500_AR100_freq10')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PerturbRigidModelo1/20240722-0331_RUN_PerturbCalmRigidModelo1_N0525_AR105_freq10')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PerturbRigidModelo1/20240722-0331_RUN_PerturbCalmRigidModelo1_N0550_AR110_freq10')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PerturbRigidModelo1/20240722-0331_RUN_PerturbCalmRigidModelo1_N0575_AR115_freq10')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PerturbRigidModelo1/20240722-0331_RUN_PerturbCalmRigidModelo1_N0600_AR120_freq10')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PerturbRigidModelo1/20240722-0331_RUN_PerturbCalmRigidModelo1_N0625_AR125_freq10')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PerturbRigidModelo1/20240722-0331_RUN_PerturbCalmRigidModelo1_N0750_AR150_freq10')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PerturbRigidModelo1/20240722-0331_RUN_PerturbCalmRigidModelo1_N0875_AR175_freq10')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PerturbRigidModelo1/20240722-0331_RUN_PerturbCalmRigidModelo1_N1000_AR200_freq10')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PerturbRigidModelo1/20240722-0331_RUN_PerturbCalmRigidModelo1_N1500_AR300_freq10')

from pathlib import Path
    
    
# %%
import re

data_dict = {}
data_pathlist = []
for folders in pathlist:
    folders = Path(folders)
    # check size of the folder
    possible_files = []
    for files in folders.glob('**/*.csv'):
        possible_files.append(files)
        
    if len(possible_files) == 0:
        print(f'No file found in {folders.stem}')
        continue
    
    if len(possible_files) == 1:
        pass
    
    if len(possible_files) > 1:
        print(f'Found multiple files in {folders.stem}')            
        for fpth in possible_files:
            print(f'{fpth.stem}')
        
        max_size = os.path.getsize(possible_files[0])
        heaviest_file = possible_files[0]
        for fpth in possible_files:
            size = os.path.getsize(fpth)
            if size > max_size:
                heaviest_file = fpth
        
        possible_files = []
        possible_files.append(heaviest_file)
    
    # re.search(data_folder.stem,'')
    
    data_dict['file_path'] = possible_files[0]
    data_dict['folder'] = folders
    data_pathlist.append(possible_files[0])
            

# %%
for pth in data_pathlist:
    print(pth)    
    search_result = re.search('N(\d+)-AR(\d+)',Path(pth).stem)
    num_rods = int(search_result.group(1))
    AR = int(search_result.group(2))
    
    time_line, node_list, contact_list = import_all_log(pth,max_rows=100)
    
    folder_path = Path(pth).parent
    subfolder_name = 'Inbox'
    pth = str(pth)
    file_id = pth.split('/')[-1].split('.')[0].split('_allLog_')[0]
    surfix = pth.split('.')[-2].split('allLog_')[-1]
    file_id = f'{file_id}'  
    
    output_path = f'/Users/yeonsu/Videos/{subfolder_name}/{file_id}_{surfix}'
    import os

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    rod_diameter = 1/AR
    
    import polyscope as ps
    ps.init()
    a_list_of_curves = node_list[-1].reshape(num_rods,-1,3)
    nodes,edges,edge_colors = prep_for_polyscope(a_list_of_curves,num_rods)
    ps_curves = ps.register_curve_network("filaments",nodes,edges)
    ps_curves.add_color_quantity("edge_colors",edge_colors,defined_on='edges',enabled=True)
    ps_curves.set_radius(rod_diameter/2,relative=False)
    ps.set_up_dir("z_up")
    file_path = f'{output_path}/frame_{0:04d}.png'
    ps.screenshot(file_path)
    
    time_line, node_list, contact_list = import_all_log(pth,max_rows=10000000)
    skip_factor = 10
    for i,a_list_of_curves in enumerate(node_list[::skip_factor]):
        a_list_of_curves = a_list_of_curves.reshape(num_rods,-1,3)
        nodes = np.vstack(a_list_of_curves)
        ps_curves.update_node_positions(nodes) 
        file_path = f'{output_path}/frame_{i:04d}.png'
        ps.screenshot(file_path)
        