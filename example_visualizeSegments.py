# %%
from pathlib import Path
import pickle
import numpy as np


rod_data_root_dir = Path('/Users/yeonsu/Data/steel-rods-xray-data')
segments_file_path = rod_data_root_dir / 'alpha200_epsilon00' / 'pruned_segments.pkl'

with open(segments_file_path, 'rb') as f:
    pruned_segments = pickle.load(f)

# %%
segments = pruned_segments

global_centroid = np.mean(np.vstack(segments),axis=0)
local_segments = []
for i,segment in enumerate(segments):    
    if np.any(np.linalg.norm(segment - global_centroid,axis=1) < 100):
        local_segments.append(segment)
        
segments = local_segments
print(f'Number of segments (sampled): {len(segments)}')
# %%
from Segments import Segments

segm = Segments(pruned_segments)
segm.initialize_filament_processing()
next_round = segm.end_to_end_clustering(number_of_endpoint_averaging=30,dist_threshold=100,align_threshold=0.3)
# %%


# i_max = np.argmax(segm.cluster_size_list)
i_max = np.argsort(segm.cluster_size_list)[-10]
print(f'Cluster size: {segm.cluster_size_list[i_max]}')
cc_max = segm.end_to_end_cluster[i_max]
individual_segments = []
for i_ in cc_max:
    if i_ % 2 == 1:
        continue
    individual_segments.append(segm.segments[i_//2])
    
# %%

nodes = np.vstack(individual_segments)
edges = []
last_i = 0


# colors = np.array([
#     [76, 153, 204],   # light blue
#     [204, 76, 153],   # pinkish red
#     [76, 204, 153],   # mint green
#     [153, 204, 76],   # light olive green
#     [204, 153, 76],   # goldenrod
#     [153, 76, 204],   # medium purple
#     [204, 76, 102],   # crimson
#     [76, 204, 204],   # cyan
#     [204, 204, 76],   # sunflower yellow
#     [102, 76, 204]    # indigo
# ])

colors = np.array([
    [31 , 119 , 180 ],  # Blue
    [255 , 127 , 14 ],  # Orange
    [44 , 160 , 44 ],   # Green
    [214 , 39 , 40 ],   # Red
    [148 , 103 , 189 ], # Purple
    [140 , 86 , 75 ],   # Brown
    [227 , 119 , 194 ], # Pink
    [127 , 127 , 127 ], # Gray
    [188 , 189 , 34 ],  # Olive
    [23 , 190 , 207 ]   # Cyan
])


vals_edge = []
last_i = 0
for i in range(len(individual_segments)):
    segment = individual_segments[i]    
    num_nodes = len(segment)
    edges.append([(last_i+i, last_i+i + 1) for i in range(len(segment) - 1)])
    
    # colors[i%10]/255
    # repeat
    clr = colors[i%10]/255
    vals_edge.append(np.tile(clr,(num_nodes-1,1)))
    last_i += num_nodes
    
vals_edge = np.vstack(vals_edge)
edges = np.vstack(edges)
edges[:100]

# %%

import polyscope as ps


# %%
ps.init()

ps.set_SSAA_factor(3)
# ps.set_navigation_style("free")

ps.set_ground_plane_mode("none") 
ps.set_ground_plane_mode("shadow_only")  # set +Z as up direction
ps.set_ground_plane_height_factor(-0.25) # adjust the plane height
ps.set_shadow_darkness(0.1)              # lighter shadows

# edges[:10]
rod_diameter = 3

# edges = np.array([[i, i + 1] for i in range(len(nodes) - 1) if i % num_nodes_each_rod != num_nodes_each_rod - 1])
# edges = np.array([[i, i + 1] for i in range(len(nodes) - 1)])
ps_all_nodes = ps.register_curve_network("all_nodes", nodes, edges, enabled=True)

ps_all_nodes.set_radius(rod_diameter, relative=False)
# ps_all_nodes.set_color([1,0,0])

ps_all_nodes.add_color_quantity(f"rod_colors", vals_edge, defined_on='edges', enabled=True)
ps.look_at((-1000.0,-1000.0,1000.0),(500,500,500))
ps.show()
# ps.screenshot('temp.png',transparent_bg=False)
# %%
