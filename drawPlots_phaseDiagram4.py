# %%
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import glob
import re
import pickle
import networkx as nx

from data_io import import_all_log, parse_path_string    
from scipy.optimize import curve_fit
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})
output_dir = '/Users/yeonsu/Dropbox (Harvard University)/Data/PrunedData/rod-sim-pnas-revision/entanglement'


# %%
class data_container:
    def __init__(self,dataphat,start_row=0,max_rows=100000,skip_rows=1):
        self.path = Path(dataphat)
        out = parse_path_string(self.path)
        # self.file_id,self.surfix,self.num_rods,self.AR,self.datetime_string
        self.file_id = out[0]
        self.surfix = out[1]
        self.num_rods = out[2]
        self.AR = out[3]
        self.datetime_string = out[4]
        self.time_line, self.node_list, self.contact_list = import_all_log(self.path,start_row=start_row,max_rows=max_rows,skip_rows=skip_rows)


root_dir = Path('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PhaseDiagramStudy-Modelo1')


pathlist = []
for pth in root_dir.rglob('*.csv'):
    if str(pth.stem).endswith('lastFrame'):
        continue
    pathlist.append(pth.parent)

# Read data
# max_rows = 1000000
# data_container_list = []

# # (g, freq)
# data_container_dict = {}

# for pth in pathlist:
#     # find csv file
#     data_path = None
#     for file in Path(pth).rglob('*.csv'):
#         if str(file.stem).endswith('lastFrame'):
#             continue
        
#         # data_container_list.append(data_container(file,max_rows=max_rows))
#         pth = str(pth)
#         exp_id = pth.split('/')[-1]
#         search_result = re.search(r'N(\d+)[-_]AR(\d+)_g(\d+(\.\d+)?)_freq(\d+(\.\d+)?)', exp_id)
        
#         N = int(search_result.group(1))
#         AR = int(search_result.group(2))
#         g = float(search_result.group(3))
#         freq = float(search_result.group(5))
        
#         data_container_list.append(data_container(file,start_row=480,max_rows=10000))
        
#         print(N,AR,g,freq)
        
        
# 
# with open('data_container_list.pkl','wb') as f:
#     pickle.dump(data_container_list,f)
    
# %%
with open('data_container_list.pkl','rb') as f:
    data_container_list = pickle.load(f)



# %%
# np.where([data_container_list[i].AR == 100 for i in range(len(data_container_list))])
i_ = 0

data_entry = data_container_list[i_]

num_rods = data_entry.num_rods
tt = data_entry.time_line
vv = data_entry.contact_list[-1].reshape(-1,18)

contact_ij = vv[:,4:6].astype(int)
# contact_ij_next_frame = next_force_all_info[:,4:6].astype(int)            
curr_nodes = data_entry.node_list[-1]

graph = nx.Graph()
graph.add_nodes_from(range(len(curr_nodes)))
graph.add_edges_from(contact_ij)
clusters = list(nx.connected_components(graph))
len(clusters)

# largest clusters
largest_cluster = max(clusters,key=len)
f = len(largest_cluster)/num_rods

print(f'f = {f}')
print(f'AR = {data_entry.AR}')
print(f'{len(contact_ij)}')
# %%
nodes_in_shape = curr_nodes.reshape(-1,10,3)

fig,ax=plt.subplots(subplot_kw={'projection': '3d'})
for rr in nodes_in_shape:
    ax.plot(rr[:,0],rr[:,1],rr[:,2],alpha=0.2)
# # %%
# fig,ax=plt.subplots(subplot_kw={'projection': '3d'})
for i_ in largest_cluster:
    ax.plot(nodes_in_shape[i_][:,0],nodes_in_shape[i_][:,1],nodes_in_shape[i_][:,2],'k')
ax.axis('equal')
# %%
rods_in_contact = np.array(list(largest_cluster))
rods_lost = np.array([i for i in range(num_rods) if i not in rods_in_contact])

node_in_the_cluster = nodes_in_shape[rods_in_contact,:]

cluster_nodes_for_ps = node_in_the_cluster.reshape(-1,3)
cluster_edges_for_ps = []
num_vertices_in_rod = 10
num_edges_in_rod = num_vertices_in_rod - 1

for i in range(0,len(cluster_nodes_for_ps),num_vertices_in_rod):
    cluster_edges_for_ps += [[i+j,i+j+1] for j in range(num_edges_in_rod)]
cluster_edges_for_ps = np.array(cluster_edges_for_ps)

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


rod_diameter = 1/data_entry.AR
import polyscope as ps
ps.init()

nodes_for_ps = curr_nodes.reshape(-1,3)

nodes_for_ps = nodes_in_shape[rods_lost,:].reshape(-1,3)
num_vertices_in_rod = 10
num_edges_in_rod = num_vertices_in_rod - 1

edges_for_ps = []
for i in range(0,len(nodes_for_ps),num_vertices_in_rod):
    edges_for_ps += [[i+j,i+j+1] for j in range(num_edges_in_rod)]
edges_for_ps = np.array(edges_for_ps)

vals_edge = np.ones((len(edges_for_ps),3))
for i in range(num_rods):
    vals_edge[i*9:(i+1)*9] = colors[i%10]/255

ps_all_nodes = ps.register_curve_network("all_nodes", nodes_for_ps, edges_for_ps, enabled=True)
ps_all_nodes.set_radius(rod_diameter/2*2,relative=False)
ps_all_nodes.add_color_quantity(f"rod_colors", vals_edge, defined_on='edges', enabled=True)
# ps_all_nodes.set_color(plt.cm.coolwarm(0))

ps_cluster_nodes = ps.register_curve_network("cluster_nodes", cluster_nodes_for_ps, cluster_edges_for_ps, enabled=True)
ps_cluster_nodes.set_radius(rod_diameter/2*2,relative=False)
ps_cluster_nodes.set_color((0,0,0))

bounding_box = np.array([[-2.4,2.4],[-2.4,2.4],[-1,1.5]])
floor_vertices = np.array([
        [bounding_box[0, 0], bounding_box[1, 0], bounding_box[2, 0]],
        [bounding_box[0, 1], bounding_box[1, 0], bounding_box[2, 0]],
        [bounding_box[0, 1], bounding_box[1, 1], bounding_box[2, 0]],
        [bounding_box[0, 0], bounding_box[1, 1], bounding_box[2, 0]]])    

floor_edges = np.array([[0,1],[1,2],[2,3],[3,0]])

ps_floor = ps.register_curve_network("floor", floor_vertices, floor_edges, enabled=True)
ps_floor.set_color((0.8, 0.8, 0.8))
ps_floor.set_radius(0.02,relative=False)

ps.set_up_dir('z_up')
ps.look_at([0,0,5],[0,0,0])


ps.show()