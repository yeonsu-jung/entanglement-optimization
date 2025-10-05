# %%
import numpy as np
from visualizations import prep_for_polyscope
from pathlib import Path
from data_io import import_all_log, import_from_dismech


# %%
pth="/Users/yeonsu/GitHub/dismech-rods-main/runs/20250124-0117_COMPILE_/log_files/capsules-mu0.20-amp0.10_allLog_20250124-011711.csv"
time_line, node_list, velocity_list, force_list, contact_list, box_contact_list = import_all_log(pth,max_rows=10000000)

import re
# search_result = re.search('N(\d+)-AR(\d+)',Path(pth).stem)
# num_rods = int(search_result.group(1))
# AR = int(search_result.group(2))
# rod_diameter = 1/AR*1.5
# node_list[0].shape[0]/3

num_center_points = 2
num_rods = 12
rod_diameter = 0.01



# %%
xyz = node_list[0].reshape(num_rods,-1,3)
max_z = np.max((xyz[:,:,2]))
min_z = np.min((xyz[:,:,2]))

max_t = time_line[-1]
final_strain = 0.95
strain_rate = (final_strain - 1)/max_t

# %%
folder_path = Path(pth).parent
subfolder_name = 'Inbox2'

file_id = pth.split('/')[-1].split('.')[0].split('_allLog_')[0]
surfix = pth.split('.')[-2].split('allLog_')[-1]
file_id = f'{file_id}'
    
output_path = f'/Users/yeonsu/Videos/{subfolder_name}/{file_id}_{surfix}'
import os

if not os.path.exists(output_path):
    os.makedirs(output_path)

import polyscope as ps
ps.init()


_t = 0
a_list_of_curves = node_list[_t].reshape(num_rods,-1,3)
nodes,edges,edge_colors = prep_for_polyscope(a_list_of_curves,num_rods)

min_z = np.min(nodes[:,2])
# ps.set_ground_plane_height_factor(-min_z)
               
ps_curves = ps.register_curve_network("filaments",nodes,edges)
ps_curves.add_color_quantity("edge_colors",edge_colors,defined_on='edges',enabled=True)
ps_curves.set_radius(rod_diameter/2,relative=False)

plane_size = 2

ps_bottom_plane = ps.register_surface_mesh("bottom_plane",np.array([[-plane_size/2,-plane_size/2,min_z],[plane_size/2,-plane_size/2,min_z],[plane_size/2,plane_size/2,min_z],[-plane_size/2,plane_size/2,min_z]]),np.array([[0,1,2],[0,2,3]]))
ps_top_plane = ps.register_surface_mesh("top_plane",np.array([[-plane_size/2,-plane_size/2,max_z],[plane_size/2,-plane_size/2,max_z],[plane_size/2,plane_size/2,max_z],[-plane_size/2,plane_size/2,max_z]]),np.array([[0,1,2],[0,2,3]]))
updated_top_position = (max_z + rod_diameter/2*1.05)*(1 + strain_rate*time_line[_t])
ps_top_plane.update_vertex_positions(np.array([[-plane_size/2,-plane_size/2,updated_top_position],[plane_size/2,-plane_size/2,updated_top_position],[plane_size/2,plane_size/2,updated_top_position],[-plane_size/2,plane_size/2,updated_top_position]]))

ps_top_plane.set_enabled(False)
ps_bottom_plane.set_enabled(False)

ps.set_up_dir("x_up")
ps.look_at([0,-10,3.5],[0,10,3.5])

ps.screenshot('temp.png')
# ps.show()
# exit()
# %%
num_files_already = len(list(Path(output_path).glob('frame_*')))
print(f'Number of frames: {len(node_list)}')

# %%
skip_factor = 10
for i,a_list_of_curves in enumerate(node_list[num_files_already::skip_factor]):
    ps.look_at([0,-10,3.5],[0,10,3.5])

    a_list_of_curves = a_list_of_curves.reshape(num_rods,-1,3)
    # num_rods = len(a_list_of_curves)
    nodes = np.vstack(a_list_of_curves)
    ps_curves.update_node_positions(nodes)
    # updated_top_position = (max_z + rod_diameter/2*1.05)*(1 + strain_rate*time_line[(i+num_files_already)*skip_factor])
    # ps_top_plane.update_vertex_positions(np.array([[-plane_size/2,-plane_size/2,updated_top_position],[plane_size/2,-plane_size/2,updated_top_position],[plane_size/2,plane_size/2,updated_top_position],[-plane_size/2,plane_size/2,updated_top_position]]))
    file_path = f'{output_path}/frame_{i+num_files_already:04d}.png'
    ps.screenshot(file_path)
    
# %%
# ps.show()
import subprocess
subprocess.run(['ffmpeg', '-framerate', '10', '-i', f'{output_path}/frame_%04d.png', '-r', '30', '-pix_fmt', 'yuv420p', f'{output_path}/output.mp4'])
# %%
