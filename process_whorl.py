# %%
%matplotlib qt
from matplotlib import pyplot as plt
import numpy as np

def seg_len(seg):
    return np.sum(np.sqrt(np.sum(np.diff(seg,axis=0)**2,axis=1)))
# %%
# columns
edges = np.loadtxt('/Users/yeonsu/GitHub/entanglement-optimization/vert_to_edge.csv',delimiter=',')
# %%
# get vertices from edges
vertices = []
added = []
for i in range(len(edges)-1):
    v1 = edges[i,:3]
    v2 = edges[i,3:]
    v3 = edges[i+1,:3]
    
    if np.all(v2 == v3):
        added.append(v1)
    else:
        vertices.append(np.array(added))
        added = []
# %%
len(vertices)

# %%

fig,ax=plt.subplots(subplot_kw={'projection':'3d'})
for v in vertices:
    v = np.array(v)
    ax.plot(v[:,0],v[:,1],v[:,2],'.-')
    
# %%
# vert = vertices[0]
# plt.close('all')
# fig,ax=plt.subplots(subplot_kw={'projection':'3d'})
# ax.plot(vert[:,0],vert[:,1],vert[:,2],'.-')

# fig,ax=plt.subplots()
# ax.plot(edge_lengths,'.-')

clean_segments = []
for vert in vertices:
    edge_lengths = np.linalg.norm(np.diff(vert, axis=0), axis=1)
    # cut by edge length
    # if edge length is greater than 500 then cut
    added = []
    for i, edge_len in enumerate(edge_lengths):
        added.append(vert[i])
        if edge_len >= 500:
            clean_segments.append(np.array(added))
            added = []
    # Append the remaining segment
    added.append(vert[-1])  # Add the last vertex of the segment
    if added:  # Check if there are remaining elements to add
        clean_segments.append(np.array(added))

    
# %%
len(clean_segments)
plt.close('all')
fig,ax=plt.subplots(subplot_kw={'projection':'3d'})
for v in clean_segments:
    v = np.array(v)
    ax.plot(v[:,0],v[:,1],v[:,2],'.-')
# %%
# sanity check
final_edge_lengths = []
for vert in clean_segments:
    edge_lengths = np.linalg.norm(np.diff(vert, axis=0), axis=1)
    final_edge_lengths.extend(edge_lengths)
# %%
from scipy.io import savemat

save_folder = 'segmenting_whorl'
layered = []
layered = np.zeros((len(clean_segments),1),dtype=object)
for i in range(len(layered)):
    layered[i] = [clean_segments[i].astype(np.uint16)]
# %%
savemat(f'{save_folder}/clean_segments.mat',{'clean_segments':layered},do_compression=True)

# %%
plt.close('all')
plt.hist(final_edge_lengths, bins=100)
# %%
end_points = np.array([[seg[0],seg[-1]] for seg in clean_segments])
end_points = end_points.reshape(-1,3)
# end_points = end_points.reshape(-1,6)

plt.close('all')
fig,ax=plt.subplots(subplot_kw={'projection':'3d'})
for v in clean_segments:
    clr = np.random.rand(3)
    v = np.array(v)
    ax.plot(v[:,0],v[:,1],v[:,2],'.-',color=clr)
    
    ep1 = v[0]
    ep2 = v[-1]
    # use same color
    
    ax.plot([ep1[0],ep2[0]],[ep1[1],ep2[1]],[ep1[2],ep2[2]],'o',color=clr)

# %%
from numba import jit
@jit(nopython=True)
def pdist2(rr1,rr2):
    n = rr1.shape[0]
    m = rr2.shape[0]
    dist_matrix = np.zeros((n,m))
    for i in range(n):
        for j in range(m):
            dist_matrix[i,j] = np.linalg.norm(rr1[i] - rr2[j])
    return dist_matrix

dist_mat = pdist2(end_points,end_points)
# %%
dist_mat[np.diag_indices_from(dist_mat)] = np.inf
np.argwhere(dist_mat < 300)

# 10 and 34
i = 28//2
j = 56//2

seg_i = clean_segments[i]
seg_j = clean_segments[j]

plt.close('all')
fig,ax=plt.subplots(subplot_kw={'projection':'3d'})
ax.plot(seg_i[:,0],seg_i[:,1],seg_i[:,2],'.-')
ax.plot(seg_j[:,0],seg_j[:,1],seg_j[:,2],'.-')

# %%


# %%
import os
import Segments
segm = Segments.Segments(clean_segments)
segm.initialize_filament_processing()

next_round = segm.end_to_end_clustering(number_of_endpoint_averaging=5,dist_threshold=300,align_threshold=0.1)

os.makedirs(save_folder,exist_ok=True)
tracker = 0
for _i in range(5):
    new_segm = Segments.Segments(next_round)
    new_segm.initialize_filament_processing()
    next_round = new_segm.end_to_end_clustering_cpp(number_of_endpoint_averaging=5,dist_threshold=100,align_threshold=0.3)
    
    plt.close('all')
    new_segm.plot_length_histogram()
    
    plt.savefig(f'{save_folder}/length_histogram_{tracker}.png')
    tracker += 1
    plt.close('all')
    
# %%
plt.close('all')
fig,ax=plt.subplots(subplot_kw={'projection':'3d'})
for v in next_round[:1]:
    v = np.array(v)
    ax.plot(v[:,0],v[:,1],v[:,2],'.-')