
# %%
from data_io import import_from_dismech
from transforms import q_to_x
import polyscope as ps
import numpy as np

# %%
pth = '/Users/yeonsu/Data/Flynn,/20240507-1613/N300_AR20_mu0.2_visc0.0_amp1.0/Entrel-N300-AR20-Scale1-mu0.20-visc0.00-amp1.00_node_20240507-161344.csv'

import re
search_result = re.search(r'N(\d+)_AR(\d+)',pth)
num_rods = int(search_result.group(1))
AR = int(search_result.group(2))
# num_rods = 300
# AR = 300
rod_diameter = 1/AR
spatial_data,timepoints = import_from_dismech(pth,num_rods)
# %%
num_time_points = spatial_data.shape[0]
x0 = spatial_data[0]
num_nodes_each_rod = 10
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
ps.set_ground_plane_mode("shadow_only")  # set +Z as up direction
ps.set_ground_plane_height_factor(-0.25) # adjust the plane height
ps.set_shadow_darkness(0.1)              # lighter shadows

nodes = x0.reshape(-1,3)
edges = np.array([[i, i + 1] for i in range(len(nodes) - 1) if i % num_nodes_each_rod != num_nodes_each_rod - 1])
# edges[:10]
ps_all_nodes = ps.register_curve_network("all_nodes", nodes, edges, enabled=True)
ps_all_nodes.set_radius(rod_diameter, relative=False)
vals_edge = np.ones((len(edges),3))

num_edges_in_a_rod = num_nodes_each_rod-1
for i in range(num_rods):
    vals_edge[i*num_edges_in_a_rod:(i+1)*num_edges_in_a_rod] = colors[i%10]/255

ps_all_nodes.add_color_quantity(f"rod_colors", vals_edge, defined_on='edges', enabled=True)
ps.look_at((-3.0,-3.0,-3.0),(0,0,0))
# ps.show()
ps.screenshot('temp.png',transparent_bg=False)

# %%
output_path = f'/Users/yeonsu/Videos/Flynn/N{num_rods}_AR{AR}'
import os
os.makedirs(output_path,exist_ok=True)

num_snapshots = num_time_points
for i in range(1, num_snapshots, 10):
    x = spatial_data[i]    

    ps_all_nodes.update_node_positions(x.reshape(-1,3))    
    ps.screenshot(f'{output_path}/q_{i:04d}.png',transparent_bg=False)
# %%
