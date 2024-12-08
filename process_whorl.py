# %%
%matplotlib qt
import os
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
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
    
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
dist_mat[np.tril_indices_from(dist_mat)] = np.inf
dist_mat[np.diag_indices_from(dist_mat)] = np.inf

for i in range(0,len(clean_segments)*2,2):
    dist_mat[i,i+1] = -1
# %%
ij = np.argwhere(dist_mat < 500)
new_ij = []
for i,j in ij:
    new_ij.append([i,j])

ij = np.array(new_ij)

# 10 and 34
# i = 28//2
# j = 56//2
for i,j in ij:
    seg_i = clean_segments[i//2]
    seg_j = clean_segments[j//2]
    
    # connection curvature
    end_points_i = end_points[i]
    end_points_j = end_points[j]
    if len(seg_i) < 2 or len(seg_j) < 2:
        continue
        
    elif i%2 == 0:
        end_tangent_i = seg_i[0] - seg_i[1]
    else:
        end_tangent_i = seg_i[-1] - seg_i[-2]
        
    if j%2 == 0:
        end_tangent_j = seg_i[1] - seg_i[0]
    else:
        end_tangent_j = seg_i[-2] - seg_i[-1]
        
    end_tangent_i /= np.linalg.norm(end_tangent_i)
    end_tangent_j /= np.linalg.norm(end_tangent_j)
    
    connecting_tangent = end_points_j - end_points_i
    connecting_tangent /= np.linalg.norm(connecting_tangent)
    
    curvature_ic = 2*np.cross(connecting_tangent,end_tangent_i)/(1+np.sum(connecting_tangent*end_tangent_i))
    curvature_jc = 2*np.cross(connecting_tangent,end_tangent_j)/(1+np.sum(connecting_tangent*end_tangent_j))
    
    connecting_curvature = (np.linalg.norm(curvature_ic)+np.linalg.norm(curvature_jc))/2    

    plt.close('all')
    fig,ax=plt.subplots(subplot_kw={'projection':'3d'})
    ax.plot(seg_i[:,0],seg_i[:,1],seg_i[:,2],'.-')
    ax.plot(seg_j[:,0],seg_j[:,1],seg_j[:,2],'.-')
    ax.quiver(end_points_i[0],end_points_i[1],end_points_i[2],end_tangent_i[0],end_tangent_i[1],end_tangent_i[2],color='r',length=1000)
    ax.quiver(end_points_j[0],end_points_j[1],end_points_j[2],end_tangent_j[0],end_tangent_j[1],end_tangent_j[2],color='b',length=1000)
    ax.quiver(end_points_i[0],end_points_i[1],end_points_i[2],connecting_tangent[0],connecting_tangent[1],connecting_tangent[2],color='g',length=1000)
    ax.set_title(f'Connecting curvature: {connecting_curvature}')
    # plt.savefig(f'{save_folder}/segment_{i}_{j}.png')
    plt.show()
    
    


# %%
import networkx as nx
G = nx.Graph()
G.add_nodes_from(range(len(clean_segments)))
G.add_weighted_edges_from([(i,j,dist_mat[i,j]) for i,j in ij])
degrees = [len(list(G.neighbors(i))) for i in range(len(clean_segments))]
np.argsort(degrees)

plt.close('all')
fig,ax=plt.subplots(subplot_kw={'projection':'3d'})
seg_i = clean_segments[i//2]
for j in list(G[i]):
    seg_j = clean_segments[j//2]
    ax.plot(seg_i[:,0],seg_i[:,1],seg_i[:,2],'.-')
    ax.plot(seg_j[:,0],seg_j[:,1],seg_j[:,2],'.-')
    plt.savefig(f'{save_folder}/segment_{i}_{j}.png')

# %%

def prune_graph(a_graph):
    import heapq
    # Track the degree of each node
    node_degrees = {node: 0 for node in a_graph.nodes}
    pruned_edges = []
    mandatory_edges = []
    added_edges = set()
    
    i = 0
    a_graph.has_edge(i, i+1)
    
    for i in range(0,max(a_graph.nodes),2):
        if a_graph.has_edge(i,i+1):
            continue
        else:
            print(f'Edge {i} - {i+1} does not exist')
    
    # Enforce the mandatory connections for even i
    for i in range(0, max(a_graph.nodes), 2):
        if i in a_graph.nodes and i+1 in a_graph.nodes:
            if a_graph.has_edge(i, i+1):
                mandatory_edges.append((i, i+1, a_graph[i][i+1]['weight']))
                node_degrees[i] += 1
                node_degrees[i+1] += 1
                added_edges.add((i, i+1))

    # Priority queue for edges sorted by weight
    edge_queue = []
    for u, v, data in a_graph.edges(data=True):
        if (u, v) not in added_edges and (v, u) not in added_edges:
            heapq.heappush(edge_queue, (data['weight'], u, v))

    # Union-Find data structure for tracking connected components
    parent = {node: node for node in a_graph.nodes}

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
            
    # Ensure each connected component is a path
    def is_path(graph):
        for component in nx.connected_components(graph):
            if sum(1 for node in component if graph.degree[node] == 1) != 2:
                return False
        return True

    pruned_graph = nx.Graph()
    pruned_graph.add_nodes_from(a_graph.nodes)
    pruned_graph.add_edges_from((u, v, data) for u, v, data in pruned_edges)
    
    # if not is_path(pruned_graph):
        # raise ValueError("The pruned graph is not a path graph in all connected components")
    
    return pruned_graph


pruned_graph = prune_graph(G)
# %%
cc = list(nx.connected_components(pruned_graph))
plt.close('all')
fig,ax=plt.subplots(subplot_kw={'projection':'3d'})
for js in cc:
    clr = np.random.rand(3)
    for j in js:
        seg = clean_segments[j//2]
        ax.plot(seg[:,0],seg[:,1],seg[:,2],'.-',color=clr)

len(ij)
np.unique(pruned_graph.nodes)

# %%
import Segments
segm = Segments.Segments(clean_segments)
segm.initialize_filament_processing()
next_round = segm.end_to_end_clustering(number_of_endpoint_averaging=2,dist_threshold=1000,align_threshold=1)

cc = segm.end_to_end_cluster
max_cc = max(cc,key=len)

# %%
plt.close('all')
fig,ax=plt.subplots(subplot_kw={'projection':'3d'})
for j in max_cc:
    if j%2 == 1:
        continue
    seg = clean_segments[j//2]
    ax.plot(seg[:,0],seg[:,1],seg[:,2],'.-')
    plt.savefig(f'{save_folder}/max_cc_{j}.png')

# %%
plt.close('all')
fig,ax=plt.subplots(subplot_kw={'projection':'3d'})
for i in range(len(cc)):
    js = cc[i]
    for j in js:        
        seg = clean_segments[j//2]
        ax.plot(seg[:,0],seg[:,1],seg[:,2],'-')
        


# %%
tracker = 0
dist_threshold = 500
dist_threshold_inc = 2
for _i in range(50):
    dist_threshold += dist_threshold_inc
    
    new_segm = Segments.Segments(next_round)
    new_segm.initialize_filament_processing()
    next_round = new_segm.end_to_end_clustering_cpp(number_of_endpoint_averaging=2,dist_threshold=dist_threshold,align_threshold=0.3)
    plt.close('all')
    new_segm.plot_length_histogram()
    plt.savefig(f'{save_folder}/length_histogram_{tracker}.png')
    plt.close('all')

    tracker += 1
    
# %%
plt.close('all')
fig,ax=plt.subplots(subplot_kw={'projection':'3d'})
for v in next_round[:2]:
    v = np.array(v)
    ax.plot(v[:,0],v[:,1],v[:,2],'.-')