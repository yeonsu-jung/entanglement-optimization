# %%
%matplotlib qt
from matplotlib import pyplot as plt
import filamentprocessing
from fitting import prep_svd_cylinder, fit_rod
from pathlib import Path
import pickle
from example_prune import inspect_segments
import numpy as np
import networkx as nx
from clustering import find_connected_components, explode_local_cluster
import os
from distances import lumelsky_dist_vec
import filamentprocessing

import jax
import jax.numpy as jnp

import pickle

from scipy.special import comb
from scipy.spatial.distance import cdist
from scipy.interpolate import make_interp_spline
from scipy.optimize import minimize

def prune_mst(mst):
    import heapq

    # Track the degree of each node
    node_degrees = {node: 0 for node in mst.nodes}
    pruned_edges = []
    mandatory_edges = []
    added_edges = set()

    # Enforce the mandatory connections for even i
    for i in range(0, max(mst.nodes), 2):
        if i in mst.nodes and i+1 in mst.nodes:
            if mst.has_edge(i, i+1):
                mandatory_edges.append((i, i+1, mst[i][i+1]['weight']))
                node_degrees[i] += 1
                node_degrees[i+1] += 1
                added_edges.add((i, i+1))

    # Priority queue for edges sorted by weight
    edge_queue = []
    for u, v, data in mst.edges(data=True):
        if (u, v) not in added_edges and (v, u) not in added_edges:
            heapq.heappush(edge_queue, (data['weight'], u, v))

    # Union-Find data structure for tracking connected components
    parent = {node: node for node in mst.nodes}

    def find(node):
        if parent[node] != node:
            parent[node] = find(parent[node])
        return parent[node]

    def union(node1, node2):
        root1 = find(node1)
        root2 = find(node2)
        if root1 != root2:
            parent[root2] = root1

    # Add mandatory edges to pruned graph
    for u, v, weight in mandatory_edges:
        pruned_edges.append((u, v, {'weight': weight}))
        union(u, v)

    # Add remaining edges while respecting degree constraints and maintaining connectivity
    while edge_queue:
        weight, u, v = heapq.heappop(edge_queue)
        if node_degrees[u] < 2 and node_degrees[v] < 2 and find(u) != find(v):
            pruned_edges.append((u, v, {'weight': weight}))
            node_degrees[u] += 1
            node_degrees[v] += 1
            union(u, v)

    # Create a new graph from the pruned edges
    pruned_graph = nx.Graph()
    pruned_graph.add_nodes_from(mst.nodes)
    pruned_graph.add_edges_from((u, v, data) for u, v, data in pruned_edges)
    
    return pruned_graph

def remove_branches(subgraph2,degree=2):
    nodes_to_remove = [node for node in subgraph2.nodes() if subgraph2.degree(node) > degree]
    subgraph2.remove_nodes_from(nodes_to_remove)
    conn_comp = list(nx.connected_components(subgraph2))
    return conn_comp,nodes_to_remove

def sort_curve(rr):
    centroid = np.mean(rr,axis=0)
    rr_centered = rr - centroid        
    _,_, V = np.linalg.svd(rr_centered, full_matrices=False)
    v1 = V[0,:]
    orientation = v1 * np.sign(np.sum(v1 * (rr_centered[-1, :] - rr_centered[0, :])))
    slist = np.dot((rr - centroid), orientation)
    sorted_indices = np.argsort(slist)
    return centroid + rr_centered[sorted_indices]



# %%
def reorder_points(xyz, th, start_index):
    N = xyz.shape[0]
    I = [start_index]

    for i in range(1, N):
        dist = np.sqrt(np.sum((xyz - xyz[I[-1], :]) ** 2, axis=1))
        I_sorted = np.argsort(dist)

        j = 1
        while True:
            if I_sorted[j] not in I:
                if dist[I_sorted[j]] > th:
                    break

                I.append(I_sorted[j])
                break
            else:
                j += 1

    xyz2 = xyz[I, :]
    return xyz2, I

# %%
import jax.numpy as jnp
from jax import vmap, jit, lax

from scipy.io import loadmat

# dataobj = loadmat('/Users/yeonsu/Dropbox (Harvard University)/Data/prunedGrosbeakScan/segments.mat')
# dataobj = loadmat('/Users/yeonsu/Dropbox (Harvard University)/Data/prunedMetalNest/segments.mat')
dataobj = loadmat('/Users/yeonsu/Dropbox (Harvard University)/Data/steel-rods-xray-data/alpha200_epsilon00/segments.mat')


segments = dataobj['segments']
segments = [np.array(seg[0]) for seg in segments]
# %%
len(segments)
# %%
cen = np.mean(np.vstack(segments),axis=0)
plt.close('all')
fig,ax=plt.subplots(1,1,figsize=(10,10),subplot_kw={'projection':'3d'})
for i in np.random.choice(len(segments),1000):
    ax.plot(segments[i][:,0],segments[i][:,1],segments[i][:,2])
    
# %%
plt.close('all')
fig,ax=plt.subplots(1,1,figsize=(10,10),subplot_kw={'projection':'3d'})
for seg in segments:
    
    if np.all( np.linalg.norm(seg - cen) < 800):
        ax.plot(seg[:,0],seg[:,1],seg[:,2])
        
# %%
def seg_len(seg):
    return np.sum(np.sqrt(np.sum(np.diff(seg,axis=0)**2,axis=1)))

length_list = []
for seg in segments:
    length_list.append(seg_len(seg))
    
# %%
log_bins = np.logspace(0,4,100)
fig,ax=plt.subplots(1,1,figsize=(10,10))
ax.hist(length_list,bins=log_bins)
ax.set_xscale('log')
ax.set_yscale('log')

# %%
plt.close('all')
fig,ax=plt.subplots(1,1,figsize=(10,10),subplot_kw={'projection':'3d'})
for i_ in np.where(np.array(length_list) < 20)[0]:
    ax.plot(segments[i_][:,0],segments[i_][:,1],segments[i_][:,2])
    
# %%
plt.close('all')
fig,ax=plt.subplots(1,1,figsize=(10,10),subplot_kw={'projection':'3d'})
for i_ in np.where(np.array(length_list) > 20)[0]:
    if np.all( np.linalg.norm(segments[i_] - cen) < 800):
        ax.plot(segments[i_][:,0],segments[i_][:,1],segments[i_][:,2])
        
# %%
long_segments = [np.array(seg).astype(np.float64) for seg in segments if seg_len(seg) > 20]
    
def quick_tangent(rr):
    cen = np.mean(rr,axis=0)
    rr_centered = rr - cen
    _,_,V = np.linalg.svd(rr_centered,full_matrices=False)
    v1 = V[0,:]
    
    return v1


end_tangents = np.zeros((len(long_segments),6))
for i_ in range(len(long_segments)):
    rr = long_segments[i_]
    # first 20 points; TO DO: get input for this
    rr_first = rr[:20]
    # last 20 points
    rr_last = rr[-20:]
    
    first_to_last = rr[-1] - rr[0]
    
    
    tan1 = quick_tangent(rr_first)
    tan1 = -tan1 * np.sign(np.dot(tan1,first_to_last))
    tan2 = quick_tangent(rr_last)
    tan2 = tan2 * np.sign(np.dot(tan2,first_to_last))
    
    end_tangents[i_,:3] = tan1
    end_tangents[i_,3:] = tan2
    
# %%
scale = 100
plt.close('all')
fig,ax=plt.subplots(1,1,figsize=(10,10),subplot_kw={'projection':'3d'})
for i_ in range(1000):
    rr = long_segments[i_]
    tan1 = end_tangents[i_,:3]
    tan2 = end_tangents[i_,3:]
    
    ax.plot(rr[:,0],rr[:,1],rr[:,2],linewidth=1)
    ax.quiver(rr[0,0],rr[0,1],rr[0,2],scale*tan1[0],scale*tan1[1],scale*tan1[2],color='r')
    ax.quiver(rr[-1,0],rr[-1,1],rr[-1,2],scale*tan2[0],scale*tan2[1],scale*tan2[2],color='r')
    ax.axis('equal')
    
# %%

@jax.jit
def calculate_distances(cyl_i, cyl_j):
    d1 = jnp.linalg.norm(cyl_i[:3] - cyl_j[:3])
    d2 = jnp.linalg.norm(cyl_i[3:6] - cyl_j[3:6])
    d3 = jnp.linalg.norm(cyl_i[:3] - cyl_j[3:6])
    d4 = jnp.linalg.norm(cyl_i[3:6] - cyl_j[:3])
    return jnp.min(jnp.array([d1, d2, d3, d4]))

@jax.jit
def calculate_e2e_dist_mat(end_points):
    n = len(end_points)
    indices = jnp.tril_indices(n, -1)
    distances = jax.vmap(lambda i, j: calculate_distances(end_points[i], end_points[j]))(indices[0], indices[1])
    e2e_distance = jnp.zeros((n, n))
    e2e_distance = e2e_distance.at[indices].set(distances)
    return e2e_distance

# %%
def pairwise_alignment(p_i, p_j, tan_i, tan_j):
    tan_i1 = tan_i[:3]
    tan_i2 = tan_i[3:]
    tan_j1 = tan_j[:3]
    tan_j2 = tan_j[3:]
    
    p_i1 = p_i[:3]
    p_i2 = p_i[3:]
    p_j1 = p_j[:3]
    p_j2 = p_j[3:]
    
    d1 = jnp.linalg.norm(p_i1 - p_j1)
    d2 = jnp.linalg.norm(p_i2 - p_j2)
    d3 = jnp.linalg.norm(p_i1 - p_j2)
    d4 = jnp.linalg.norm(p_i2 - p_j1)
    
    dvec = jnp.array([d1, d2, d3, d4])    
    i_min = jnp.argmin(dvec)
    
    dvec = jnp.where(i_min == 0, p_i1 - p_j1,
                     jnp.where(i_min == 1, p_i2 - p_j2,
                               jnp.where(i_min == 2, p_i1 - p_j2,
                                         p_i2 - p_j1)))
    tan_i = jnp.where(i_min == 0, tan_i1,
                      jnp.where(i_min == 1, tan_i2,
                                jnp.where(i_min == 2, tan_i1,
                                          tan_i2)))
    tan_j = jnp.where(i_min == 0, tan_j1,
                      jnp.where(i_min == 1, tan_j2,
                                jnp.where(i_min == 2, tan_j2,
                                          tan_j1)))
    
    dvec = dvec / jnp.linalg.norm(dvec)
    alignment = (jnp.linalg.norm(jnp.cross(dvec, tan_i)) + jnp.linalg.norm(jnp.cross(dvec, tan_j))) / 2
    return alignment

vectorized_pairwise_alignment = vmap(vmap(pairwise_alignment, in_axes=(None, 0, None, 0)), in_axes=(0, None, 0, None))


# %%

end_points = np.zeros((len(long_segments), 6))
for i_ in range(len(long_segments)):
    rr = long_segments[i_]
    end_points[i_, :3] = rr[0]
    end_points[i_, 3:] = rr[-1]
# %%
import time
start = time.time()
e2e_dist_mat = calculate_e2e_dist_mat(end_points)
print(f'Elapsed time: {time.time() - start}')
# %%
e2e_dist_mat = np.array(e2e_dist_mat)
e2e_dist_mat = e2e_dist_mat + e2e_dist_mat.T
e2e_dist_mat[np.diag_indices(len(long_segments))] = np.inf
# %%
close_ij = np.where(e2e_dist_mat < 5)[0]


# %%
# start = time.time()
# e2e_alig_mat = vectorized_pairwise_alignment(end_points, end_points, end_tangents, end_tangents)
# print(f'Elapsed time: {time.time() - start}')

# %%
e2e_dist_mat = np.array(e2e_dist_mat)
e2e_alig_mat = np.array(e2e_alig_mat)

e2e_dist_mat = e2e_dist_mat + e2e_dist_mat.T
e2e_alig_mat = e2e_alig_mat + e2e_alig_mat.T

e2e_dist_mat[np.diag_indices(len(long_segments))] = np.inf
e2e_alig_mat[np.diag_indices(len(long_segments))] = np.inf

# %%
i_ = 1
e2e_neighbors = np.where(e2e_dist_mat[i_] < 100)[0]
fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
ax.plot(long_segments[i_][:,0],long_segments[i_][:,1],long_segments[i_][:,2],linewidth=1)
for nb in e2e_neighbors:
    ax.plot(long_segments[nb][:,0],long_segments[nb][:,1],long_segments[nb][:,2],linewidth=1)

ax.axis('equal')

# %%
# end data
end_points2 = end_points.copy()
end_points2 = end_points2.reshape(-1,3)

end_tangents2 = end_tangents.copy()
end_tangents2 = end_tangents2.reshape(-1,3)

# %%
# now each entry in end_poitns2 and end_tangents2 is considered as an independent one
# but we need to retrieve the connectivity and properties like length at some point

@jax.jit
def calculate_euclidean_distances(point1, point2):    
    return jnp.linalg.norm(point1 - point2)

@jax.jit
def calculate_dist_mat(points):
    n = points.shape[0]
    indices = jnp.tril_indices(n, -1)
    distances = jax.vmap(lambda i, j: calculate_euclidean_distances(points[i], points[j]))(indices[0], indices[1])
    dist_mat = jnp.zeros((n, n))
    dist_mat = dist_mat.at[indices].set(distances)
    return dist_mat

def pairwise_alignment_for_ends(p_i, p_j, tan_i, tan_j):
    dvec = p_j - p_i
    dvec = dvec / jnp.linalg.norm(dvec)
    # i to j
    
    alignment = (jnp.linalg.norm(jnp.cross(dvec, tan_i)) + jnp.linalg.norm(jnp.cross(dvec, tan_j))) / 2
    
    return alignment

vectorized_pairwise_alignment_for_ends = vmap(vmap(pairwise_alignment_for_ends, in_axes=(None, 0, None, 0)), in_axes=(0, None, 0, None))
# %%
end_dist_mat = calculate_dist_mat(end_points2)
end_alig_mat = vectorized_pairwise_alignment_for_ends(end_points2, end_points2, end_tangents2, end_tangents2)

end_dist_mat = np.array(end_dist_mat)
end_dist_mat = end_dist_mat + end_dist_mat.T
end_dist_mat[np.diag_indices(len(end_points2))] = np.inf
# Set distances between adjacent indices to infinity
for i in range(len(end_dist_mat) - 1):
    if i % 2 == 1:
        continue
    end_dist_mat[i, i + 1] = -1000
    end_dist_mat[i + 1, i] = -1000
    
end_alig_mat = np.array(end_alig_mat)
end_alig_mat = end_alig_mat + end_alig_mat.T
end_alig_mat[np.diag_indices(len(end_points2))] = np.inf
# Set distances between adjacent indices to infinity
for i in range(len(end_alig_mat) - 1):
    if i % 2 == 1:
        continue
    end_alig_mat[i, i + 1] = -1000
    end_alig_mat[i + 1, i] = -1000

# %%
# end_alig_mat[4756,4757]
end_dist_mat[4756,4757]
# %%
mask = (end_dist_mat < 5) & (end_alig_mat < 1)
e2e_graph = nx.Graph()
e2e_graph.add_nodes_from(range(len(long_segments)*2))

edges_indices = np.array(np.where(mask)).T
weights = end_alig_mat[mask]
weighted_edges = [(i, j, w) for (i, j), w in zip(edges_indices, weights)]
e2e_graph.add_weighted_edges_from(weighted_edges)

# e2e_graph.add_edges_from(np.array(np.where(mask)).T)

e2e_clusters = list(nx.connected_components(e2e_graph))

cluster_size_list = [len(x) for x in e2e_clusters]
print(f'Number of end points: {len(long_segments)*2}')
print(f'Number of connected components: {len(e2e_clusters)}')
print(f'Max. cluster size {np.max(cluster_size_list)} at {np.argmax(cluster_size_list)}')
# %%
import time
start = time.time()
mst = nx.minimum_spanning_tree(e2e_graph)
print(f'Elapsed time: {time.time() - start}')

# %%
pruned_graph = prune_mst(mst)
# %%


# %%
conn_comp = list(nx.connected_components(pruned_graph))
len(conn_comp)

plt.close('all')
fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
for i_,cc in enumerate(conn_comp):
    if i_ % 2 == 0:
        continue
    clr = np.random.rand(3)
    for j_ in cc:
        if j_ % 2 == 0:
            continue
        ax.plot(long_segments[j_//2][:,0],long_segments[j_//2][:,1],long_segments[j_//2][:,2],linewidth=1,color=clr)
        
# %%
cc = list(conn_comp[0])
subgraph = pruned_graph.subgraph(cc)
degrees = np.array([subgraph.degree(node) for node in subgraph.nodes])

segmented = []
for cc in conn_comp:
    cc = list(cc)
    subgraph = pruned_graph.subgraph(cc)
    eps = [node for node in subgraph.nodes if subgraph.degree[node] == 1]

    if len(eps) != 2:
        raise ValueError("The graph does not have exactly two endpoints.")

    # Find the shortest path between the two endpoints
    path = nx.shortest_path(subgraph, source=eps[0], target=eps[1])    
    straight_curve = []
    for i_ in path[::2]:
        if i_ % 2 == 0:
            straight_curve.append(long_segments[i_//2])
        elif i_ % 2 == 1: 
            straight_curve.append(long_segments[i_//2][::-1])            
    straight_curve = np.vstack(straight_curve)
    segmented.append(straight_curve)
    
# %%
cen = np.mean(np.vstack(long_segments),axis=0)
plt.close('all')
fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
for seg in segmented:
    if np.all( np.linalg.norm(seg - cen,axis=1) < 500 ):
        ax.plot(seg[:,0],seg[:,1],seg[:,2],linewidth=1)
# %%
size_list = [len(seg) for seg in segmented]
np.mean(size_list)
for i_,seg in enumerate(segmented):
    segmented[i_] = seg[::10]

# %%
import filamentFields

fF = filamentFields.filamentFields([],[])
# %%
R_omega = 200
fF.update_filament_nodes_list(segmented)
fF.precompute(R_omega)
# %%
all_segments = np.vstack(segmented)
xlim = [np.min(all_segments[:,0]),np.max(all_segments[:,0])]
ylim = [np.min(all_segments[:,1]),np.max(all_segments[:,1])]
zlim = [np.min(all_segments[:,2]),np.max(all_segments[:,2])]
num_grids = 30
    
mg = np.meshgrid(np.linspace(xlim[0],xlim[1],num_grids),np.linspace(-ylim[0],ylim[1],num_grids),np.linspace(zlim[0],zlim[1],num_grids))
sampling_points = np.array([mg[0].flatten(),mg[1].flatten(),mg[2].flatten()]).T
    
total_entanglement = fF.return_total_entanglement()    
rod_diameter = 6;

n_fields = np.zeros(len(sampling_points))
phi_fields = np.zeros(len(sampling_points))
S_fields = np.zeros(len(sampling_points))
Q_fields = np.zeros((len(sampling_points),9))
e_fields = np.zeros(len(sampling_points))
c_fields = np.zeros(len(sampling_points))
f_fields = np.zeros(len(sampling_points))

for iterator,sampling_point in enumerate(sampling_points):
    fF.analyze_local_volume_from_precomputed(sampling_point, R_omega, rod_diameter)
    n_fields[iterator] = fF.return_number_of_labels()
    phi_fields[iterator] = fF.return_volume_fraction()
    S_fields[iterator] = fF.return_orientational_order_parameter()
    e_fields[iterator] = fF.return_entanglement()
    Q_fields[iterator] = fF.return_local_Q_tensor()
    
# %%
e_fields_vol = e_fields.reshape(num_grids,num_grids,num_grids)
e_fields_maxProj = np.max(e_fields_vol,axis=0)
e_fields_img = np.flipud(e_fields_maxProj.T)

fig, axs = plt.subplots(1, 1, figsize=(12, 6))
fig.colorbar(axs.imshow(e_fields_img, extent=[xlim[0], xlim[1], zlim[0], zlim[1]],vmin=0,vmax=np.nanmax(e_fields_vol)), ax=axs)
# %%
np.nanstd(e_fields)/np.nanmean(e_fields)
np.nanstd(phi_fields)/np.nanmean(phi_fields)
# %%
# save to mat file
from scipy.io import savemat
savemat('e_fields.mat',{'e_fields':e_fields_vol})

# %%
straight_curve = np.vstack(straight_curve)
plt.close('all')
fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
ax.plot(straight_curve[:,0],straight_curve[:,1],straight_curve[:,2],linewidth=1)
ep1 = end_points2[eps[0]]
ep2 = end_points2[eps[1]]
ax.plot(ep1[0],ep1[1],ep1[2],'o',markersize=5)
ax.plot(ep2[0],ep2[1],ep2[2],'o',markersize=5)

# %%
plt.close('all')
fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
for i_ in cc:
    if i_ % 2 == 1:
        continue
    ax.plot(long_segments[i_//2][:,0],long_segments[i_//2][:,1],long_segments[i_//2][:,2],linewidth=1)



# %%
conn_comp = list(nx.connected_components(e2e_graph))
cen = np.mean(np.vstack(long_segments),axis=0)
plt.close('all')

fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
for cc in conn_comp:
    
    tmp = np.array(list(cc))
    tmp = tmp[tmp % 2 == 0]
    if len(tmp) == 0:
        continue
    joined = np.vstack([long_segments[i//2] for i in tmp])
    joined = sort_curve(joined)
    joined,_ = reorder_points(joined,1.e4,0)
    
    if seg_len(joined) < 50:
        continue
    
    if np.any( np.linalg.norm(joined - cen,axis=1) > 200):
        continue
    
    ax.plot(joined[:,0],joined[:,1],joined[:,2],linewidth=1) 
    
# %%
# i_max = np.argmax(cluster_size_list)
i_max = np.argsort(cluster_size_list)[-2]
cc = e2e_clusters[i_max]
plt.close('all')
fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
scale=10
for i_ in cc:
    # plot each end    
    if i_ % 2 == 0:
        clr = np.random.rand(3)
        ax.plot(long_segments[i_//2][:,0],long_segments[i_//2][:,1],long_segments[i_//2][:,2],linewidth=1,color=clr)
        
    ax.plot(end_points2[i_][0],end_points2[i_][1],end_points2[i_][2],'o',markersize=2,color=clr)
ax.axis('equal')
# %%
# minimum spanning tree
subgraph = e2e_graph.subgraph(cc)
mst = nx.minimum_spanning_tree(subgraph)
# %%
plt.close('all')
fig, ax=plt.subplots(1,1,figsize=(10,10))
nx.draw(mst,ax=ax,with_labels=False,node_size=3)
degrees = np.array([mst.degree(node) for node in mst.nodes()])

# remove edges with degree > 2
subgraph2 = mst.copy()
conn_comp,removed_nodes = remove_branches(subgraph2,degree=2)
# %%


# Display or use the pruned graph
degrees = np.array([pruned_graph.degree(node) for node in pruned_graph.nodes])
# %%
for node in pruned_graph.nodes():
    print(node)
# %% For example, visualize it
plt.close('all')
nx.draw(pruned_graph, with_labels=True)

    
# %%
conn_cozmp = list(nx.connected_components(pruned_graph))
plt.close('all')
fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
for i_ in conn_comp:
    clr = np.random.rand(3)
    for j_ in i_:
        if j_ % 2 == 0:
            continue
        ax.plot(long_segments[j_//2][:,0],long_segments[j_//2][:,1],long_segments[j_//2][:,2],linewidth=1,color=clr)
    
ax.axis('equal')
    
# %%
def join_curves(a_list_of_curves):
    return

plt.close('all')
fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
for cc in conn_comp:
    a_list_of_curves = []
    for i_ in cc:
        if i_ % 2 == 0:
            a_list_of_curves.append(long_segments[i_//2])

    merged_curve = join_curves(a_list_of_curves)
    ax.plot(merged_curve[:,0],merged_curve[:,1],merged_curve[:,2],linewidth=1)
    
    


# %%


# %%
plt.close('all')
fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
for cc in conn_comp:
    # joined = np.vstack([long_segments[i] for i in cc])
    # joined = sort_curve(joined)
    # joined,_ = reorder_points(joined,1.e4,0)    
    # ax.plot(joined[:,0],joined[:,1],joined[:,2],linewidth=1)
    clr = np.random.rand(3)
    for i_ in cc:
        ax.plot(long_segments[i_][:,0],long_segments[i_][:,1],long_segments[i_][:,2],linewidth=1,color=clr)
    
ax.axis('equal')
ax.axis('off')


# %%
fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
for cc in e2e_clusters:
    joined = np.vstack([long_segments[i] for i in cc])
    joined,_ = reorder_points(joined,1.e4,0)
    if seg_len(joined) > 10:
        ax.plot(joined[:,0],joined[:,1],joined[:,2],linewidth=0.2)

# %%
length_list = []
error_list = []
for i_ in e2e_clusters:
    joined = np.vstack([long_segments[i] for i in i_])
    joined,_ = reorder_points(joined,1.e4,0)
    length_list.append(seg_len(joined))
    fit_result = fit_rod(joined,0.00001,10000)
    error_list.append(fit_result['err'])
    
# %%
plt.close('all')
log_bins = np.logspace(1,3,100)
fig,ax=plt.subplots(1,1)
ax.hist(length_list,bins=log_bins)
ax.set_xscale('log')
# %%
plt.close('all')
log_bins = np.logspace(-2,1,100)
fig,ax=plt.subplots(1,1)
ax.hist(error_list,bins=log_bins)
# %%
i_max = np.argmax(length_list)
i_max = np.argsort(length_list)[-125]
cc = e2e_clusters[i_max]
plt.close('all')
fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
scale=10
for i_ in cc:
    ax.plot(long_segments[i_][:,0],long_segments[i_][:,1],long_segments[i_][:,2],linewidth=1)
    ax.quiver(long_segments[i_][0,0],long_segments[i_][0,1],long_segments[i_][0,2],scale*end_tangents[i_,0],scale*end_tangents[i_,1],scale*end_tangents[i_,2],color='r')
    ax.quiver(long_segments[i_][-1,0],long_segments[i_][-1,1],long_segments[i_][-1,2],scale*end_tangents[i_,3],scale*end_tangents[i_,4],scale*end_tangents[i_,5],color='r')

# %%

# supposed to be simple

class Segments:
    def __init__(self, segments):
        # segments must be a list of numpy arrays
        # also, numpy array whose shape is M x N x 3 where M is the number of segments and N is the number of points
        # because each segment can have different number of points, each entry might have been padded with nans
        
        # the motivation of this class is to provide a simple interface to keep partitions of segments
        # upon partitioning, it naturally creates a 'subjective' map which maps the global index to the local index
        # e.g., the information of partitioning is stored in such a way (i) which segment belongs to which partition
        # and (ii) the local index of the segment in the partition
        
        # to track those partition information and to provide a simple interface, this class is created
        
        # especially, when multiple partitioning (or clustering) needs to be done, we should keep track of
        # the sequence of partitioning.
        
        
        
        self.segments = segments
        
        
        
        
        