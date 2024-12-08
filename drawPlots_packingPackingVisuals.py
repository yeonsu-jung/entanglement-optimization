# %%
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import glob
import re
import pickle
import networkx as nx
import polyscope as ps

from data_io import import_all_log, parse_path_string    
from scipy.optimize import curve_fit
def power_law(x,a,b):
    return a*x**b
# %%
pathlist = []
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/20240617-1509_RUN_FlipModelo1_N1500_AR300')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/20240619-0111_RUN_FlipModelo1_N500_AR100')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/20240619-0111_RUN_FlipModelo1_N250_AR050')


# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/CalmModelo1/20240619-2353_RUN_CalmEEModelo1_N0125_AR025')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/20240620-1407_RUN_PerturbCalmEEModelo1_N0125_AR025_g0.5_freq100')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/Modelo1_SlowExcitation/20240611-1241_RUN_WeakPerturbEEModelo1_N0125_AR025')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/Modelo1_FineExcitation/20240609-1052_RUN_PerturbEEModelo1_N1500_AR300')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/EntangledModelo1/20240609-0108_RUN_EntangleModelo1_N1500_AR300')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/EntangledModelo1/20240609-0108_RUN_EntangleModelo1_N0125_AR025')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/CalmModelo1/20240619-2353_RUN_CalmEEModelo1_N0125_AR025')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/CalmModelo1/20240620-0039_RUN_CalmEEModelo1_N0250_AR050')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/CalmModelo1/20240620-0039_RUN_CalmEEModelo1_N0300_AR060')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/CalmModelo1/20240620-0039_RUN_CalmEEModelo1_N0350_AR070')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/CalmModelo1/20240620-0039_RUN_CalmEEModelo1_N0375_AR075')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/CalmModelo1/20240620-0039_RUN_CalmEEModelo1_N0400_AR080')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/CalmModelo1/20240620-0039_RUN_CalmEEModelo1_N0450_AR090')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/CalmModelo1/20240620-0039_RUN_CalmEEModelo1_N0500_AR100')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/CalmModelo1/20240620-0039_RUN_CalmEEModelo1_N0525_AR105')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/CalmModelo1/20240620-0039_RUN_CalmEEModelo1_N0550_AR110')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/CalmModelo1/20240620-0039_RUN_CalmEEModelo1_N0575_AR115')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/CalmModelo1/20240620-0039_RUN_CalmEEModelo1_N0600_AR120')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/CalmModelo1/20240620-0039_RUN_CalmEEModelo1_N0625_AR125')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/CalmModelo1/20240620-0039_RUN_CalmEEModelo1_N0750_AR150')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/CalmModelo1/20240620-0039_RUN_CalmEEModelo1_N0875_AR175')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/CalmModelo1/20240620-0039_RUN_CalmEEModelo1_N1000_AR200')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/CalmModelo1/20240620-0039_RUN_CalmEEModelo1_N1250_AR250')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/CalmModelo1/20240620-0039_RUN_CalmEEModelo1_N1500_AR300')

# %%


class data_container:
    def __init__(self,dataphat,max_rows=100000):
        self.path = Path(dataphat)
        out = parse_path_string(self.path)
        # self.file_id,self.surfix,self.num_rods,self.AR,self.datetime_string
        self.file_id = out[0]
        self.surfix = out[1]
        self.num_rods = out[2]
        self.AR = out[3]
        self.datetime_string = out[4]
        self.time_line, self.node_list, self.contact_list = import_all_log(self.path,max_rows=max_rows)

max_rows = 1000000
data_container_list = []
for pth in pathlist:
    # find csv file
    data_path = None
    for file in Path(pth).rglob('*.csv'):
        if str(file.stem).endswith('lastFrame'):
            continue
        data_container_list.append(data_container(file,max_rows=max_rows))
        break
    
    
# %%
num_rods = data_container_list[0].num_rods
AR = data_container_list[0].AR
exp_id = data_container_list[0].path.parent.stem
dt_id = data_container_list[0].datetime_string

rod_diameter = 1/AR

all_node = data_container_list[0].node_list[0]
all_node = all_node.reshape(num_rods,30)
# %%
final_node = data_container_list[0].node_list[-1].reshape(num_rods,10,3)
initial_node = data_container_list[0].node_list[0].reshape(num_rods,10,3)

# %%

# %%
nodes_all_along = data_container_list[0].node_list

nodes_all_along = np.array(nodes_all_along)
nodes_all_along = nodes_all_along.reshape(-1,3)
bounding_box_all_along = np.array([[np.min(nodes_all_along[:,i]),np.max(nodes_all_along[:,i])] for i in range(3)])
bounding_box_all_along = np.array([[-0.5,0.5],
                                   [-0.5,0.5],
                                   [-1,1.5]])
# %%
initial_node = data_container_list[0].node_list[0]
initial_node = initial_node.reshape(num_rods,10,3)
# %%


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
# %%
num_nodes_each_rod = 10
packing_center = np.mean(np.mean(initial_node,axis=1),axis=0)
locked_nodes = []
locked_edges = [] # edge_ids for locked edges

locked_rod_labels = []
# Packing center: 
# get edge_id for locked edges
for i_,rr in enumerate(initial_node.reshape((-1,10,3))):
    
    assert(rr.shape == (10,3))
    for j_ in range(9):
        I = np.linalg.norm(rr[j_] - packing_center) < 0.1
        J = np.linalg.norm(rr[j_+1] - packing_center) < 0.1
        
        if I and J:
            locked_rod_labels.append(i_)
            locked_nodes.append(rr[j_])
            locked_edge_id = i_*9 + j_            
            locked_edges.append(locked_edge_id)
            
    
locked_rod_labels = np.unique(locked_rod_labels)
locked_rods = initial_node[locked_rod_labels]
locked_rods = locked_rods.reshape(-1,3)
locked_rods_edges = np.array([[i, i + 1] for i in range(len(locked_rods) - 1) if i % num_nodes_each_rod != num_nodes_each_rod - 1])



nodes = all_node.reshape(-1,3)
cen = np.mean(nodes,axis=0)
nodes = nodes - cen
locked_rods = locked_rods - cen



edges = np.array([[i, i + 1] for i in range(len(nodes) - 1) if i % num_nodes_each_rod != num_nodes_each_rod - 1])

bounding_box = bounding_box_all_along - np.vstack((cen,cen)).transpose()
# Define the box vertices from the bounding box
box_vertices = np.array([
    [bounding_box[0, 0], bounding_box[1, 0], bounding_box[2, 0]],
    [bounding_box[0, 1], bounding_box[1, 0], bounding_box[2, 0]],
    [bounding_box[0, 1], bounding_box[1, 1], bounding_box[2, 0]],
    [bounding_box[0, 0], bounding_box[1, 1], bounding_box[2, 0]],
    [bounding_box[0, 0], bounding_box[1, 0], bounding_box[2, 1]],
    [bounding_box[0, 1], bounding_box[1, 0], bounding_box[2, 1]],
    [bounding_box[0, 1], bounding_box[1, 1], bounding_box[2, 1]],
    [bounding_box[0, 0], bounding_box[1, 1], bounding_box[2, 1]]
])

# Define the box edges
box_edges = np.array([
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 0],
    [4, 5],
    [5, 6],
    [6, 7],
    [7, 4],
    [0, 4],
    [1, 5],
    [2, 6],
    [3, 7]
])

# hook
hook_length = 2
# hook_nodes: from cen + [0,hook_length,0] to cen
hook_nodes = np.array([cen + [0,hook_length,0],cen]) - cen
hook_edges = np.array([[0,1]])

locked_nodes = locked_nodes - cen
locked_edges = np.array([[i, i + 1] for i in range(len(locked_nodes) - 1)])
# %%

# Example spherical coordinates for nodes
phi = np.linspace(0, np.pi, 10)
theta = np.linspace(0, 2 * np.pi, 20)
phi, theta = np.meshgrid(phi, theta)
phi = phi.flatten()
theta = theta.flatten()

sphere_radius = 0.1
# Convert spherical coordinates to Cartesian coordinates
x = sphere_radius*np.sin(phi) * np.cos(theta)
y = sphere_radius*np.sin(phi) * np.sin(theta)
z = sphere_radius*np.cos(phi)
sphere_nodes = np.vstack((x, y, z)).T

# Define edges (example: connecting each node to its neighbor)
sphere_edges = []
num_phi = 10
num_theta = 20
for i in range(num_theta):
    for j in range(num_phi - 1):
        sphere_edges.append([i * num_phi + j, i * num_phi + (j + 1)])
    sphere_edges.append([i * num_phi, (i + 1) * num_phi - 1])

sphere_edges = np.array(sphere_edges)

# Register the curve network


# %%


# %%

ps.init()
ps.set_SSAA_factor(3)
ps.set_navigation_style("free")

# ps.set_ground_plane_mode("none") 
ps.set_ground_plane_mode("shadow_only")  # set +Z as up direction
ps.set_ground_plane_height_factor(-0.25) # adjust the plane height
ps.set_shadow_darkness(0.1)              # lighter shadows
ps.set_view_projection_mode("perspective")
# ps.set_transparency_mode('simple')

ps_all_nodes = ps.register_curve_network("all_nodes", nodes, edges, enabled=True)
ps_all_nodes.set_radius(rod_diameter/2*2,relative=False)

ps_locked_rods = ps.register_curve_network("locked_rods", locked_rods, locked_rods_edges, color=(0,0,0), enabled=False)
ps_locked_rods.set_radius(rod_diameter/2*2,relative=False) # radius is relative to a scene length scale by default

ps_locekd_nodes = ps.register_curve_network("locked_nodes", locked_nodes, locked_edges,color=(0,0,0), enabled=False)
ps_locekd_nodes.set_radius(rod_diameter/2,relative=False) # radius is relative to a scene length scale by default

ps_hook = ps.register_curve_network("hook", hook_nodes, hook_edges, enabled=False)
ps_hook.set_radius(0.01,relative=False)  # Set the radius of the hook edges
ps_hook.set_color((0.8, 0.8, 0.8))  # Set the color of the hook edges

# Add color to edges
num_edges_in_a_rod = num_nodes_each_rod-1
vals_edge = np.ones((len(edges),3))
for i in range(num_rods):
    vals_edge[i*num_edges_in_a_rod:(i+1)*num_edges_in_a_rod] = colors[i%10]/255
    
ps_all_nodes.add_color_quantity(f"rod_colors", vals_edge, defined_on='edges', enabled=True)

# add sphere
ps_sphere = ps.register_curve_network("spherical network", sphere_nodes, sphere_edges)
ps_sphere.set_radius(0.01,relative=False)
ps_sphere.set_color((0.8, 0.8, 0.8))
# ps_sphere = ps.register_point_cloud("sphere", np.array([[0,0,0]]), enabled=False)
# ps_sphere: triangulated sphere

# Register the box as a curve network
ps_box = ps.register_curve_network("box", box_vertices, box_edges,enabled=False)
ps_box.set_radius(0.01, relative=False)  # Set the radius of the box edges
ps_box.set_color((0.8, 0.8, 0.8))  # Set the color of the box edges

ps_all_nodes.set_material("clay")
ps.look_at((-5., 0., 2.), (0., 0., 2.))

ps_all_nodes.set_transparency(0.2)
ps_sphere.set_transparency(1.)
# ps_locekd_nodes.set_transparency(2.)
# alpha = ps_all_nodes.get_transparency()
# ps_locekd_nodes.get_transparency()

ps_box.set_enabled(False)
ps_all_nodes.set_enabled(False)
ps_locked_rods.set_enabled(True)
ps_locekd_nodes.set_enabled(False)
ps_hook.set_enabled(False)
ps_sphere.set_enabled(True)

ps.set_screenshot_extension(".png");
ps.screenshot('temp.png',transparent_bg=False)
# ps.look_at((4.,3.,3.),(0.,0.,1.))
# %%
root_dir = f'/Users/yeonsu/Dropbox (Harvard University)/Data/PrunedData/rod-sim-pnas-revision/visuals'
output_dir = f'{root_dir}/N{num_rods}_AR_{AR}_{exp_id}'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
# %%
# scene 1: all nodes
# for 10 frames
last_frame = 0
period = 60
final_frame_here = last_frame + period

ps_box.set_enabled(False)
ps_all_nodes.set_enabled(True)
ps_locked_rods.set_enabled(False)
ps_locekd_nodes.set_enabled(False)
ps_hook.set_enabled(False)
ps_sphere.set_enabled(False)    

all_node = data_container_list[0].node_list[0]
all_node = all_node.reshape(num_rods,30)
nodes = all_node.reshape(-1,3)
nodes = nodes - cen

ps_all_nodes.set_transparency(1.)
ps_all_nodes.update_node_positions(nodes)
    
for t in range(last_frame,final_frame_here):    
    
    screenshot_path = f'{output_dir}/rod_{num_rods}_AR_{AR}_{t:04d}.png'
    ps.screenshot(screenshot_path,transparent_bg=False)    
    
last_frame = final_frame_here


period = 60
final_frame_here = last_frame + period

ps_box.set_enabled(False)
ps_all_nodes.set_enabled(False)
ps_locked_rods.set_enabled(True)
ps_locekd_nodes.set_enabled(False)
ps_hook.set_enabled(True)
ps_sphere.set_enabled(True)
    
for t in range(last_frame,final_frame_here):
    
    # all_node = data_container_list[0].node_list[0]
    # all_node = all_node.reshape(num_rods,30)
    # nodes = all_node.reshape(-1,3)
    # nodes = nodes - cen
    
    # ps_all_nodes.set_transparency(1.)
    # ps_all_nodes.update_node_positions(nodes)
    
    screenshot_path = f'{output_dir}/rod_{num_rods}_AR_{AR}_{t:04d}.png'
    ps.screenshot(screenshot_path,transparent_bg=False)    

last_frame = final_frame_here

period = len(data_container_list[0].time_line)
final_frame_here = last_frame + period

ps_box.set_enabled(False)
ps_all_nodes.set_enabled(True)
ps_locked_rods.set_enabled(False)
ps_locekd_nodes.set_enabled(False)
ps_hook.set_enabled(True)
ps_sphere.set_enabled(False)
    
for i_,t in enumerate(range(last_frame,final_frame_here)):
    
    all_node = data_container_list[0].node_list[i_]
    all_node = all_node.reshape(num_rods,30)
    nodes = all_node.reshape(-1,3)
    nodes = nodes - cen
    
    ps_all_nodes.set_transparency(1.)
    ps_all_nodes.update_node_positions(nodes)
    
    screenshot_path = f'{output_dir}/rod_{num_rods}_AR_{AR}_{t:04d}.png'
    ps.screenshot(screenshot_path,transparent_bg=False)    
# %%
last_frame = final_frame_here
# %%
ps_all_nodes.get_radius()

# %%


screenshot_path = f'{output_dir}/temp.png'
ps.screenshot(screenshot_path,transparent_bg=False)
# %%
# ps.show()
