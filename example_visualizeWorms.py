
# %%
from data_io import import_from_dismech
from transforms import q_to_x
import polyscope as ps
import numpy as np

# %%
pth = '/Users/yeonsu/GitHub/dismech-rods-main/runs/20240708-1807_COMPILE_test/log_files/active_entanglement/node_20240708-180744.csv'

from pathlib import Path

identifer = Path(pth).stem

# %%

dta = np.loadtxt(pth,delimiter=',')
# fig,ax=plt.subplots(subplot_kw={'projection':'3d'})
# num_nodes_each_rod = 100
# num_rods = (dta.shape[1] - 1)//num_nodes_each_rod//3
# print(num_rods)
# %%

rod_diameter = 0.002
num_rods = 1
spatial_data,timepoints = import_from_dismech(pth,num_rods)
# spatial_data.shape
num_nodes_each_rod = spatial_data.shape[1]//3
# %%
num_time_points = spatial_data.shape[0]
x0 = spatial_data[0]
nodes = x0.reshape(-1,3)
edges = np.array([[i, i + 1] for i in range(len(nodes) - 1) if i % num_nodes_each_rod != num_nodes_each_rod - 1])

# %%
# x0 = np.loadtxt('/Users/yeonsu/GitHub/entanglement-optimization/vertices_n1.txt',delimiter=',')
# rod_diameter = 0.015
# num_nodes_each_rod = x0.shape[0]
# x0.reshape(num_rods,-1).shape
# nodes = x0.reshape(-1,3)
# num_nodes_each_rod = 2
# edges = np.array([[i, i + 1] for i in range(len(nodes) - 1) if i % num_nodes_each_rod != num_nodes_each_rod - 1])
# %%
colors = np.array([
    [76, 153, 204],   # light blue
    [204, 76, 153],   # pinkish red
    [76, 204, 153],   # mint green
    [153, 204, 76],   # light olive green
    [204, 153, 76],   # goldenrod
    [153, 76, 204],   # medium purple
    [204, 76, 102],   # crimson
    [76, 204, 204],   # cyan
    [204, 204, 76],   # sunflower yellow
    [102, 76, 204]    # indigo
])


ps.init()

ps.set_SSAA_factor(3)
ps.set_navigation_style("free")

# ps.set_ground_plane_mode("none") 
# ps.set_ground_plane_mode("shadow_only")  # set +Z as up direction
ps.set_ground_plane_height_factor(0.5) # adjust the plane height
ps.set_shadow_darkness(0.1)              # lighter shadows

nodes = x0.reshape(-1,3)
edges = np.array([[i, i + 1] for i in range(len(nodes) - 1) if i % num_nodes_each_rod != num_nodes_each_rod - 1])
# edges[:10]
ps_all_nodes = ps.register_curve_network("all_nodes", nodes, edges, enabled=True)
ps_all_nodes.set_radius(rod_diameter, relative=True)
vals_edge = np.ones((len(edges),3))

num_edges_in_a_rod = num_nodes_each_rod-1
for i in range(num_rods):
    vals_edge[i*num_edges_in_a_rod:(i+1)*num_edges_in_a_rod] = colors[i%10]/255

ps_all_nodes.add_color_quantity(f"rod_colors", vals_edge, defined_on='edges', enabled=True)
# ps.look_at((-0.5,-0.5,-0.5),(0,0,0))
ps.set_up_dir("z_up")
ps.set_front_dir("x_front")
# ps.look_at((-0.5,-0.5,-0.5),(0,0,0))
# ps.show()
ps.screenshot('temp.png',transparent_bg=False)

# %%
# ps.set_front_dir("x_front")
ps.set_front_dir("y_front")

output_path = f'/Users/yeonsu/Videos/TestActiveWorms/{identifer}'
import os
if not os.path.exists(output_path):
    os.makedirs(output_path,exist_ok=True)
    
num_existed_files = len(os.listdir(output_path))

dta = np.loadtxt(pth,delimiter=',')
spatial_data,timepoints = import_from_dismech(pth,num_rods)

spatial_data = spatial_data[::10]
num_time_points = spatial_data.shape[0]
num_snapshots = num_time_points

from matplotlib import pyplot as plt
for i in range(num_existed_files, num_snapshots):
    x = spatial_data[i]
    ps_all_nodes.update_node_positions(x.reshape(-1,3))
    ps.screenshot(f'{output_path}/q_{i:04d}.png',transparent_bg=False)
    
    img = plt.imread(f'{output_path}/q_{i:04d}.png')
    if any(np.array(img.shape[:2]) % 2 != 0):
        if img.shape[0] % 2 != 0:
            img = np.pad(img,((0,1),(0,0),(0,0)),mode='edge')
        if img.shape[1] % 2 != 0:
            img = np.pad(img,((0,0),(0,1),(0,0)),mode='edge')
        plt.imsave(f'{output_path}/q_{i:04d}.png',img)
    
    
# %%
# ps.show()
# %%
