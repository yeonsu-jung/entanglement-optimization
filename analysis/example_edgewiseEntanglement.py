# %%
import numpy as np
from matplotlib import pyplot as plt
import re
import os
from data_io import import_all_log,parse_path_string
from fields import get_local_fields_at_a_point


alllog_pth = '/Users/yeonsu/Data/from-cluster/NonIntersectingBox-N1000-AR200-Scale1-mu0.20-visc0.00-amp0.00_allLog_20240527-193121.csv'
protocol_id = 'CarrotCake2'
file_id,surfix,num_rods,AR = parse_path_string(alllog_pth)
time_line, node_list, contact_list = import_all_log(alllog_pth,max_rows=100)

# %%
import importlib
import fields
importlib.reload(fields)
from fields import get_edges_labels_from_centerlines

centerlines = node_list[10]
centerlines = centerlines.reshape(num_rods,-1,3)
edges,labels,all_edges = get_edges_labels_from_centerlines(centerlines)
# %%
import time


from fields import compute_linking_number_for_edges
from numba import jit as njit

@njit(nopython=True)
def compute_edge_wise_entanglement(all_edges,labels):
    num_all_edges = all_edges.shape[0]
    entanglement_matrix = np.full((num_all_edges,num_all_edges),np.nan)
    for idx in range(num_all_edges):        
        edge1 = all_edges[idx]
        
        for jdx in range(idx+1,num_all_edges):
            if labels[idx] == labels[jdx]:
                continue
                        
            edge2 = all_edges[jdx]            
            entanglement_matrix[idx,jdx] = compute_linking_number_for_edges(edge1,edge2)
        
    return entanglement_matrix

start = time.time()
ent_mat = compute_edge_wise_entanglement(all_edges,labels)
print(f'Elapsed time: {time.time()-start:.2f} seconds')
# %%
rod_diameter = 1/AR
R_omega = np.sqrt(2*rod_diameter)
I0 = np.abs(all_edges[:,1]) < R_omega
np.count_nonzero(I0)
# %%
meta_local_edges = all_edges[I0]
meta_local_labels = labels[I0]
# %%
start = time.time()
ent_mat = compute_edge_wise_entanglement(meta_local_edges,meta_local_labels)
print(f'Elapsed time: {time.time()-start:.2f} seconds')
# %%
I1 = np.linalg.norm(meta_local_edges[:,:3] - np.array([0,0,0]),axis=1) < R_omega
I2 = np.linalg.norm(meta_local_edges[:,3:] - np.array([0,0,0]),axis=1) < R_omega
I = I1 & I2

np.where(I)

# %% 
labeled_segments_in_sphere = meta_local_edges[I]
labels_in_sphere = meta_local_labels[I]
unique_labels_in_sphere = np.unique(labels_in_sphere)

ee_list = []
for lb in unique_labels_in_sphere:                    
    I_ee = labels_in_sphere == lb
    ee_list.append(labeled_segments_in_sphere[I_ee,:])
    
from fields import compute_local_lk
lk_mat = compute_local_lk(ee_list)

np.sum(np.abs(lk_mat[np.triu_indices(lk_mat.shape[0],k=1)]))




# %%
def get_local_lk_from_precomputed(edge_ids,ent_mat):
    num_local_edges = len(edge_ids)
    local_lk = np.full((num_local_edges,num_local_edges),np.nan)
    for idx in range(num_local_edges):
        for jdx in range(idx+1,num_local_edges):
            local_lk[idx,jdx] = ent_mat[edge_ids[idx],edge_ids[jdx]]
            
    return local_lk

local_lk_mat = get_local_lk_from_precomputed(np.where(I)[0],ent_mat)

np.sum(np.abs(local_lk_mat[~np.isnan(local_lk_mat)]))
