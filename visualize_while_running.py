# %%
import numpy as np
from visualizations import prep_for_polyscope
from pathlib import Path
from data_io import import_all_log, import_from_dismech


# %%
pth = '/Users/yeonsu/GitHub/dismech-rods-main/runs/20240902-1720_RUN_single_rod_still_AR300/log_files/SingleRod-N1-AR300-Scale1-mu0.50-visc0.00-amp10.0_allLog_20240902-172048.csv'
time_line, node_list, contact_list = import_all_log(pth,max_rows=10000000)

import re
search_result = re.search('N(\d+)-AR(\d+)',Path(pth).stem)
num_rods = int(search_result.group(1))
AR = int(search_result.group(2))
rod_diameter = 1/AR

# pth = '/Users/yeonsu/GitHub/dismech-rods-main/runs/20240813-0135_COMPILE_AR20_2/log_files/node_20240813-013602.csv'
# node_list, time_line= import_from_dismech(pth,500)
# num_rods = 500
# AR = 20
# rod_diameter = 1/AR
# %%
folder_path = Path(pth).parent
subfolder_name = 'Inbox'

file_id = pth.split('/')[-1].split('.')[0].split('_allLog_')[0]
surfix = pth.split('.')[-2].split('allLog_')[-1]
file_id = f'{file_id}'

# worm 1
if 'gripper' in pth:
    num_rods = 12
    rod_diameter = 0.002
elif 'worm' in pth:
    num_rods = node_list[0].shape[0]//25//3
    rod_diameter = 0.25
elif 'metal' in pth:
    num_rods = node_list[0].shape[0]//10//3
    rod_diameter = 0.025
    
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
    
output_path = f'/Users/yeonsu/Videos/{subfolder_name}/{file_id}_{surfix}'
import os

if not os.path.exists(output_path):
    os.makedirs(output_path)

# live_view_with_polyscope(a_list_of_curves)
len(node_list)
# %%
# from matplotlib import pyplot as plt
# fig,ax=plt.subplots(subplot_kw={'projection':'3d'})
# for i in range(num_rods):
#     r1 = a_list_of_curves[i,0]
#     r2 = a_list_of_curves[i,-1]
#     ax.plot([r1[0],r2[0]],[r1[1],r2[1]],[r1[2],r2[2]])
# %%

print(num_rods)
import polyscope as ps
ps.init()
# num_rods = 500

# num_rods = 815
a_list_of_curves = node_list[0].reshape(num_rods,-1,3)
# num_rods = len(a_list_of_curves)
nodes,edges,edge_colors = prep_for_polyscope(a_list_of_curves,num_rods)

min_z = np.min(nodes[:,2])
# ps.set_ground_plane_height_factor(-min_z)
               
ps_curves = ps.register_curve_network("filaments",nodes,edges)
ps_curves.add_color_quantity("edge_colors",edge_colors,defined_on='edges',enabled=True)
ps_curves.set_radius(rod_diameter/2,relative=False)
# 0.25*20/1000
# ps_curves.set_radius(0.25*20/1000/2,relative=False)
# ps_curves.set_radius(0.002*20/2,relative=False)
# ps_curves.set_radius(0.0035*300*300/2000/1000*20/2)
ps.set_up_dir("z_up")
file_path = f'{output_path}/frame_{0:04d}.png'
ps.screenshot(file_path)
# %%
# time_line, node_list, contact_list = import_all_log(pth,max_rows=1000000)


# check how many files are in output_path
# from pathlib import Path
# files = list(Path(output_path).glob('*.png'))
# num_files = len(files)


# %%
skip_factor = 1
for i,a_list_of_curves in enumerate(node_list[::skip_factor]):
    a_list_of_curves = a_list_of_curves.reshape(num_rods,-1,3)
    # num_rods = len(a_list_of_curves)
    nodes = np.vstack(a_list_of_curves)
    ps_curves.update_node_positions(nodes)
    file_path = f'{output_path}/frame_{i:04d}.png'
    ps.screenshot(file_path)
    
# %%
