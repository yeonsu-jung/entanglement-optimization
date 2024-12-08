# %%
import numpy as np
from visualizations import prep_for_polyscope
from pathlib import Path
from data_io import import_all_log


# %%
pth = '/Users/yeonsu/GitHub/dismech-rods-main/runs/20240711-1929_COMPILE_test/log_files/metal_nest_allLog_20240711-192931.csv'
# pth = '/Users/yeonsu/GitHub/dismech-rods-main/runs/20240711-1954_COMPILE_test/log_files/metal_nest_allLog_20240711-195417.csv'
time_line, node_list, contact_list = import_all_log(pth,max_rows=100)
len(node_list)
# %%
folder_path = Path(pth).parent
subfolder_name = 'Inbox'

file_id = pth.split('/')[-1].split('.')[0].split('_allLog_')[-2]
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
    num_rods = node_list[0].shape[0]//20//3
    rod_diameter = 0.025
    
output_path = f'/Users/yeonsu/Videos/{subfolder_name}/{file_id}_{surfix}'
import os

if not os.path.exists(output_path):
    os.makedirs(output_path)

# live_view_with_polyscope(a_list_of_curves)
len(node_list)
# %%

initial_list_of_curves = node_list[0].reshape(-1,20,3)
xlim = [np.min(initial_list_of_curves[:,:,0]),np.max(initial_list_of_curves[:,:,0])]
ylim = [np.min(initial_list_of_curves[:,:,1]),np.max(initial_list_of_curves[:,:,1])]
zlim = [np.min(initial_list_of_curves[:,:,2]),np.max(initial_list_of_curves[:,:,2])]


# %%
from matplotlib import pyplot as plt
error_list = []
for nodes in node_list:
    a_list_of_curves = nodes.reshape(-1,20,3)
    cnt = 0
    where = []
    for i,seg in enumerate(a_list_of_curves):
        # if seg is out of bounds
        if (np.min(seg[:,0]) < xlim[0]) or (np.max(seg[:,0]) > xlim[1]):
            continue
        if (np.min(seg[:,1]) < ylim[0]) or (np.max(seg[:,1]) > ylim[1]):
            continue
        if (np.min(seg[:,2]) < zlim[0]) or (np.max(seg[:,2]) > zlim[1]):
            continue
        if np.any(np.isnan(seg)):
            continue
        where.append(i)
        cnt += 1
    error_list.append(cnt)
# %%
plt.plot(error_list)

# %%
nodes = node_list[5]
a_list_of_curves = nodes.reshape(-1,20,3)
cnt = 0
where = []
for i,seg in enumerate(a_list_of_curves):
    # if seg is out of bounds
    if (np.min(seg[:,0]) < xlim[0]) or (np.max(seg[:,0]) > xlim[1]):
        continue
    if (np.min(seg[:,1]) < ylim[0]) or (np.max(seg[:,1]) > ylim[1]):
        continue
    if (np.min(seg[:,2]) < zlim[0]) or (np.max(seg[:,2]) > zlim[1]):
        continue
    if np.any(np.isnan(seg)):
        continue
    where.append(i)
# %%
a_list_of_curves = node_list[0].reshape(-1,20,3)
troubles = np.delete(a_list_of_curves,where,axis=0)
troubles_flat = [r.flatten() for r in troubles]
    
with open(f'/Users/yeonsu/GitHub/dismech-rods-main/data/curved_samples/metal_nest_troubles.txt','w') as f:
    # full precision
    for r in troubles_flat:
        f.write(' '.join([str(x) for x in r])+'\n')        
        
# %%
# %%
import polyscope as ps
ps.init()
a_list_of_curves = new_node_list[0].reshape(-1,20,3)
num_rods = len(a_list_of_curves)
nodes,edges,edge_colors = prep_for_polyscope(a_list_of_curves,num_rods)
ps_curves = ps.register_curve_network("filaments",nodes,edges)
ps_curves.add_color_quantity("edge_colors",edge_colors,defined_on='edges',enabled=True)
ps.set_up_dir("z_up")
file_path = f'{output_path}/frame_{0:04d}.png'
ps.screenshot(file_path)
# %%
# time_line, node_list, contact_list = import_all_log(pth,max_rows=1000000)
skip_factor = 1
for i,a_list_of_curves in enumerate(new_node_list[::skip_factor]):
    a_list_of_curves = a_list_of_curves.reshape(-1,20,3)
    num_rods = len(a_list_of_curves)
    nodes = np.vstack(a_list_of_curves)
    ps_curves.update_node_positions(nodes) 
    file_path = f'{output_path}/frame_{i:04d}.png'
    ps.screenshot(file_path)
    
# %%
pth = '/Users/yeonsu/GitHub/dismech-rods-main/runs/20240711-1929_COMPILE_test/log_files/metal_nest_pruned.txt'
flat_node_list = [r.flatten() for r in new_node_list[-1]]

with open(f'/Users/yeonsu/GitHub/dismech-rods-main/data/curved_samples/metal_nest_pruned.txt','w') as f:
    # full precision
    for r in flat_node_list:
        f.write(' '.join([str(x) for x in r])+'\n')