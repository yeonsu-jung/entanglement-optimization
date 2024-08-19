# %%
import numpy as np
from visualizations import prep_for_polyscope
from pathlib import Path
from data_io import import_all_log
import os


# %%
pathlist = []

# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/20240711-0441_COMPILE_gripper_2')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/20240711-0104_COMPILE_gripper_1')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/20240711-2058_COMPILE_metal_nest')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/20240711-1331_COMPILE_fric_worm_5')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/20240711-1330_COMPILE_fric_worm_4')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/20240711-1329_COMPILE_fric_worm_3')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/20240711-1326_COMPILE_fric_worm_2')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/20240711-1321_COMPILE_fric_worm_1')
# pathlist.append(Path('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/curved_examples'))
import re

# data_pathlist = []
# for data_folder in pathlist:
    # choose heavist folder


# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/curved_examples/20240711-0104_COMPILE_gripper_1')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/curved_examples/20240711-1326_COMPILE_fric_worm_2')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/curved_examples/20240711-1329_COMPILE_fric_worm_3')
pathlist = []
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/GripperPerturbation/20240715-0135_RUN_gripper_1_20x_scale_relaxed.txt')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/GripperPerturbation/20240715-0308_RUN_gripper_2_20x_scale_relaxed.txt')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/WormPerturbation/20240714-1524_RUN_worm_1_20x_scale_relaxed.txt')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/WormPerturbation/20240714-1524_RUN_worm_2_20x_scale_relaxed.txt')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/WormPerturbation/20240714-1524_RUN_worm_3_20x_scale_relaxed.txt')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/WormPerturbation/20240714-1524_RUN_worm_4_20x_scale_relaxed.txt')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/WormPerturbation/20240715-0426_COMPILE_Worm5Perturbation')

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
    time_line, node_list, contact_list = import_all_log(pth,max_rows=100)
    
    folder_path = Path(pth).parent
    subfolder_name = 'Inbox'
    pth = str(pth)
    file_id = pth.split('/')[-1].split('.')[0].split('_allLog_')[0]
    surfix = pth.split('.')[-2].split('allLog_')[-1]
    file_id = f'{file_id}'

    # worm 1
    if 'gripper' in pth:
        num_rods = 12
        rod_diameter = 0.004
    elif 'worm' in pth:
        num_rods = 12
        rod_diameter = 0.25*2
        if 'worm_3' in pth:
            num_rods = 12
        if 'worm_4' in pth:
            num_rods = 12
        if 'worm_4' in pth:
            num_rods = 13
        if 'worm_5' in pth:
            num_rods = 12
            
    elif 'metal' in pth:
        num_rods = node_list[0].shape[0]//20//3
        rod_diameter = 0.025
    num_rods = 12
    output_path = f'/Users/yeonsu/Videos/{subfolder_name}/{file_id}_{surfix}'
    import os

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    rod_diameter = 0.25*20/1000
    # rod_diameter = 0.002*20/2
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
    skip_factor = 50
    for i,a_list_of_curves in enumerate(node_list[::skip_factor]):
        a_list_of_curves = a_list_of_curves.reshape(num_rods,-1,3)
        nodes = np.vstack(a_list_of_curves)
        ps_curves.update_node_positions(nodes) 
        file_path = f'{output_path}/frame_{i:04d}.png'
        ps.screenshot(file_path)
        