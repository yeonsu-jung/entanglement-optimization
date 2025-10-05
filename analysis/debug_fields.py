# %%
import numpy as np
import matplotlib.pyplot as plt
import importlib
import filamentFields
importlib.reload(filamentFields)

from fields import *

filaments = []
for i in range(10):
    x = np.cumsum(np.random.randn(100))
    y = np.cumsum(np.random.randn(100))
    z = np.cumsum(np.random.randn(100))
    x = np.convolve(x, np.ones(5)/5, mode='valid')
    y = np.convolve(y, np.ones(5)/5, mode='valid')
    z = np.convolve(z, np.ones(5)/5, mode='valid')        
    filaments.append(np.vstack([x,y,z]).T)
    

    
fig,ax=plt.subplots(subplot_kw={'projection':'3d'})
for r in filaments:
    ax.plot(r[:,0],r[:,1],r[:,2])

# %%
fF = filamentFields.filamentFields(filaments)
all_nodes = fF.return_all_nodes()

query_points = np.array([0,0,0])
R_omega = 2
local_edges = fF.analyzeLocalVolume(query_points, R_omega, 0.01)
# %%
fig,ax=plt.subplots(subplot_kw={'projection':'3d'})
for i in range(len(local_edges)):
    edge = local_edges[i]
    p1 = edge[:3]
    p2 = edge[3:]
    ax.plot([p1[0],p2[0]],[p1[1],p2[1]],[p1[2],p2[2]])
# %%
# my sampling
filament_edges_list = fF.return_filament_nodes_list()
all_edges = fF.return_all_edges()
len(filament_edges_list)
fig,ax=plt.subplots(subplot_kw={'projection':'3d'})
for edge in all_edges:
    p1 = edge[:3]
    p2 = edge[3:]
    
    if (np.linalg.norm(p1 - query_points) < R_omega ) & (np.linalg.norm(p2 - query_points) < R_omega):        
        ax.plot([p1[0],p2[0]],[p1[1],p2[1]],[p1[2],p2[2]])
# %%
_,labels,edges_all_in_one = get_edges_labels_from_centerlines(filaments)

I_local = sample_edges_locally_and_return_indices(edges_all_in_one,query_points,R_omega)
unique_labels_in_sphere = np.unique(labels[I_local])
local_edges = edges_all_in_one[I_local,:]
local_labels = labels[I_local]
def collect_local_edges(edges,labels,unique_labels_in_sphere):
    ee_list = []
    for lb in unique_labels_in_sphere:
        # no, if you do this, then you don't do cropping.
        # or should we do that?        
        # I = labels == lb
        # ee_list.append(edges[I,:])
        
        I = labels == lb
        ee_list.append(edges[I,:])
        
    return ee_list

ee_list = collect_local_edges(local_edges,local_labels,unique_labels_in_sphere)

total_edges = np.vstack(ee_list)
edge_length = np.linalg.norm(total_edges[:,3:6] - total_edges[:,:3],axis=1)                
# %%
fig,ax=plt.subplots(subplot_kw={'projection':'3d'})
for edge in total_edges:
    p1 = edge[:3]
    p2 = edge[3:]
    ax.plot([p1[0],p2[0]],[p1[1],p2[1]],[p1[2],p2[2]])

# %%












# %%
num_rods = 100
def linspace3(start, stop, num):
    return np.vstack([np.linspace(start[i], stop[i], num) for i in range(3)]).T 

single_edge = linspace3([0,0,0],[0,0,1],10)

mg = np.meshgrid(np.linspace(0,1,num_rods),np.linspace(0,1,num_rods))
mg = [mg[0].flatten(),mg[1].flatten()]
aligned_filaments = []
for i in range(num_rods**2):
    translator = np.array([mg[0][i],mg[1][i],0]).T
    aligned_filaments.append(single_edge + translator)
    
# fig,ax=plt.subplots(subplot_kw={'projection':'3d'})
# for i in range(100):
#     ax.plot(aligned_filaments[i][:,0],aligned_filaments[i][:,1],aligned_filaments[i][:,2])
    
# %%
fF = filamentFields.filamentFields(aligned_filaments)
# %%
fF.analyzeLocalVolume(np.array([0,0,0]), 1,0.01)



# %%
# %%
import numpy as np
from matplotlib import pyplot as plt
import re
import os
from data_io import import_all_log
from fields import get_local_fields_at_a_point

def sample_nodes_locally(nodes,center,R):
    I1 = np.linalg.norm((nodes[:,:3] - center), axis=1) < R
    I2 = np.linalg.norm((nodes[:,3:6] - center), axis=1) < R
    I_sphere_segments = I1 & I2    
    return nodes[I_sphere_segments]

def label_nodes(nodes):
    labels = []
    for i,nodeset in enumerate(nodes):
        n = nodeset.shape[0]
        lb = np.ones(n,dtype=np.int64)*i        
        labels.append(lb)
        
    return np.hstack(np.array(labels))

def rowwise_norm(A):
    return np.sqrt(np.sum(A**2,axis=1))

def get_bbox(curves):
    x_min = np.min(curves[:,:,0])
    x_max = np.max(curves[:,:,0])
    y_min = np.min(curves[:,:,1])
    y_max = np.max(curves[:,:,1])
    z_min = np.min(curves[:,:,2])
    z_max = np.max(curves[:,:,2])
    return [x_min, x_max, y_min, y_max, z_min, z_max]

def parse_path_string(pth):    
    filename = pth.split('/')[-1]        
    file_id = filename.split('-mu')[0]
    
    surfix_match = re.search(r'\d{8}-\d{6}', filename)
    surfix = surfix_match.group(0) if surfix_match else None
    
    num_rods_match = re.search(r'-N(\d+)-', filename)
    num_rods = int(num_rods_match.group(1)) if num_rods_match else None
    
    AR_match = re.search(r'-AR(\d+)-', filename)
    AR = int(AR_match.group(1)) if AR_match else None
    
    return file_id, surfix, num_rods, AR


alllog_pth = '/Users/yeonsu/Data/from-cluster/NonIntersectingBox-N1000-AR200-Scale1-mu0.20-visc0.00-amp0.00_allLog_20240527-193121.csv'
protocol_id = 'CarrotCake2'
file_id,surfix,num_rods,AR = parse_path_string(alllog_pth)
time_line, node_list, contact_list = import_all_log(alllog_pth,max_rows=10000)


output_folder = f'/Users/yeonsu/Videos/{protocol_id}/{file_id}_twoPlots_highResolution/'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
else:
    print(f'Folder already exists: {output_folder}')
    

print(f'Size of time_line: {len(time_line)}')
print(f'Number of rods: {num_rods}')
print(f'Aspect ratio: {AR}')
# %%
rod_length = 1
rod_diameter = rod_length/AR
R_omega = np.sqrt(2*rod_length*rod_diameter)
# %%
mid_y = 0
num_grids = 30
xlim = [-0.5,0.5]
zlim = [-1,1]

mg = np.meshgrid(np.linspace(xlim[0],xlim[1],num_grids),mid_y,np.linspace(zlim[0],zlim[1],num_grids))
sampling_points = np.array([mg[0].flatten(),mg[1].flatten(),mg[2].flatten()]).T

import importlib
import fields
importlib.reload(fields)
from fields import get_local_fields_at_a_point

import time
start = time.time()
import filamentFields
# %%

start = time.time()
frame = 200
curves = node_list[frame]
curves = curves.reshape(num_rods, -1, 3)
# interpolate curves, 3 times more points
curves = np.array([np.vstack([np.linspace(curve[i], curve[i+1], 4)[:-1] for i in range(len(curve)-1)]) for curve in curves])

fF = filamentFields.filamentFields(curves)

nf = np.zeros(len(sampling_points))
vf = np.zeros(len(sampling_points))
Sf = np.zeros(len(sampling_points))
ef = np.zeros(len(sampling_points))

for iterator,sampling_point in enumerate(sampling_points):
    local_volume = fF.analyzeLocalVolume(sampling_point, R_omega*1.5, rod_diameter)
    nf[iterator] = fF.return_number_of_labels()
    vf[iterator] = fF.return_volume_fraction()
    Sf[iterator] = fF.return_orientational_order_parameter()
    ef[iterator] = fF.return_entanglement()
    

ef[nf == 0] = 0
e_fields_img = ef.reshape(num_grids,num_grids)
e_fields_img = np.flipud(e_fields_img.T)

fig, axs = plt.subplots(1, 1, figsize=(12, 6))
fig.colorbar(axs.imshow(e_fields_img, extent=[xlim[0], xlim[1], zlim[0], zlim[1]],vmin=0,vmax=120), ax=axs)
# %%
fig,ax=plt.subplots(subplot_kw={'projection':'3d'})
for curve in local_volume:
    ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], alpha=1)
# %%
nf.max()
# %%
all_nodes_from_fF = fF.return_all_nodes()
all_edges_from_fF = fF.return_all_edges()

node_labels = fF.return_node_labels()
edge_labels = fF.return_edge_labels()
# %%
edge_labels[:10]
# %%
all_nodes_from_Python = np.vstack(curves)
from fields import get_edges_labels_from_centerlines
_,edge_labels,all_edges_from_Python = get_edges_labels_from_centerlines(curves)
# %%
np.abs(all_nodes_from_fF - all_nodes_from_Python).sum()
np.abs(all_edges_from_fF - all_edges_from_Python).sum()

# %%
nf2 = np.zeros(len(sampling_points))
vf2 = np.zeros(len(sampling_points))
Sf2 = np.zeros(len(sampling_points))
ef2 = np.zeros(len(sampling_points))

curves = node_list[frame]
curves = curves.reshape(num_rods, -1, 3)
for iterator,sampling_point in enumerate(sampling_points):
    result = get_local_fields_at_a_point(curves, sampling_point, R_omega, rod_diameter, visualize=False)
    nf2[iterator] = result[0]
    vf2[iterator] = result[1]
    Sf2[iterator] = result[2]
    ef2[iterator] = result[3]

ef2[nf2 == 0] = 0
e_fields_img2 = ef2.reshape(num_grids,num_grids)
e_fields_img2 = np.flipud(e_fields_img2.T)

fig, axs = plt.subplots(1, 1, figsize=(12, 6))
fig.colorbar(axs.imshow(e_fields_img2, extent=[xlim[0], xlim[1], zlim[0], zlim[1]],vmin=0,vmax=120), ax=axs)
