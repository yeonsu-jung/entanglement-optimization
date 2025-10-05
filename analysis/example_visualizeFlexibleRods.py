
# %%
from data_io import import_from_dismech
from transforms import q_to_x
import polyscope as ps
import numpy as np
import re


pth = '/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/20240629-1608_RUN_EntangleSoftModelo1_N1000_AR200/NonIntersectingBox-N1000-AR200-Scale1-mu0.20-visc0.00-amp0.00_allLog_20240629-160918.csv'

from data_io import import_all_log
time_line, node_list, contact_list = import_all_log(pth,max_rows=1000000)

# %%
num_nodes_each_rod = 20
num_rods = len(node_list[0])//num_nodes_each_rod//3

from matplotlib import pyplot as plt
fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
for i in range(num_rods):
    rr = node_list[0][i*num_nodes_each_rod*3:(i+1)*num_nodes_each_rod*3].reshape(-1,3)
    ax.plot(rr[:,0],rr[:,1],rr[:,2])
    
# dta = np.loadtxt(pth,delimiter=',')
# num_nodes_each_rod = 20
# num_rods = (dta.shape[1] - 1)//num_nodes_each_rod//3

# %%
search_result = re.search(r'N(\d+)[-_]AR(\d+)',pth)
num_rods = int(search_result.group(1))
AR = int(search_result.group(2))
rod_diameter = 1/AR
# %%
num_time_points = len(time_line)
x0 = node_list[0]
x0.shape



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
ps_all_nodes.set_radius(rod_diameter/4, relative=True)
vals_edge = np.ones((len(edges),3))

num_edges_in_a_rod = num_nodes_each_rod-1
for i in range(num_rods):
    vals_edge[i*num_edges_in_a_rod:(i+1)*num_edges_in_a_rod] = colors[i%10]/255

ps_all_nodes.add_color_quantity(f"rod_colors", vals_edge, defined_on='edges', enabled=True)
ps.look_at((-0.5,-0.5,-0.5),(0,0,0))
ps.set_up_dir("z_up")
# ps.show()
ps.screenshot('temp.png',transparent_bg=False)

# %%
output_path = f'/Users/yeonsu/Videos/TestFlexibleRods'
import os
if not os.path.exists(output_path):
    os.makedirs(output_path,exist_ok=True)
    
num_existed_files = len(os.listdir(output_path))

time_line, node_list, contact_list = import_all_log(pth,max_rows=1000000)
num_time_points = len(time_line)
num_snapshots = num_time_points
for i in range(num_existed_files, num_snapshots):
    x = node_list[i]

    ps_all_nodes.update_node_positions(x.reshape(-1,3))    
    ps.screenshot(f'{output_path}/q_{i:04d}.png',transparent_bg=False)
# %%

