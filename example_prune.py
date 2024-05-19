from pathlib import Path
from scipy.io import loadmat
import pickle
import numpy as np
from example_PhysicalRodRelaxation import prep_svd_cylinder
import networkx as nx
from visualizations import set_3d_plot,plot_single_rod
from sklearn_extra.cluster import KMedoids
from fitting import fit_rod
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt

from example_PhysicalRodRelaxation import lumelsky_dist_vec
from visualizations import plot_centerline_with_container
import itertools
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from scipy.ndimage import uniform_filter1d

import time

def explode_local_cluster(a_cluster):
    # print(segment_length_list[a_cluster].sum()/650)
    # len(a_cluster)

    subgraph = nx.Graph()
    subgraph.add_nodes_from(range(len(a_cluster)))

    local_dist_mat = get_local_dist_mat(segments,a_cluster)
    local_i,local_j = np.triu_indices(len(a_cluster),1)
    # local_dist = local_dist_mat[np.triu_indices(len(a_cluster),1)]

    subgraph.remove_edges_from(subgraph.edges())
    subgraph.add_weighted_edges_from(zip(local_i,local_j,local_dist_mat[np.triu_indices(len(a_cluster),1)]))
    # first make sure a node has two edges
    edges_to_remove = find_edges_to_remove_by_weight(subgraph)
    subgraph.remove_edges_from(edges_to_remove)
    # minimum spanning tree
    mst = nx.minimum_spanning_tree(subgraph, weight='weight',algorithm='boruvka')
    mst_deg = np.array(list(mst.degree()))[:,1]

    # remove hubs
    hubs = np.where(mst_deg > 2)[0]
    mst.remove_nodes_from(hubs)
    subclusters = print_connected_components(mst)
    
    for i,subcluster in enumerate(subclusters):
        subclusters[i] = np.sort(np.array(list(subcluster)).astype(int))
    
    return subclusters

def get_local_dist_mat(segments,a_cluster):
    num_elements = len(a_cluster)
    local_dist_mat = np.full((num_elements,num_elements),np.inf)    
    for i in range(num_elements):
        segment_i = segments[a_cluster[i]]
        p1,p2 = segment_i[0],segment_i[-1]
        for j in range(i+1,num_elements):
            segment_j = segments[a_cluster[j]]
            q1,q2 = segment_j[0],segment_j[-1]
            t,u,d1,d2,d12=lumelsky_dist_vec(p1,p2,q1,q2)
            
            # tol = 1e-6
            # if ( t < tol) or (t > 1-tol) or (u < tol) or (u > 1-tol):
            #     local_dist_mat[i,j] = 1e4
            #     continue
            
            vec = d1*t - d2*u - d12
            popt1 = segment_i[0]+t*d1
            popt1_opp = segment_i[0]+(1-t)*d1
            popt2 = segment_j[0]+u*d2
            popt2_opp = segment_j[0]+(1-u)*d2
            dist = np.linalg.norm(vec)
            local_dist_mat[i,j] = dist
            
    local_dist_mat = np.minimum(local_dist_mat,local_dist_mat.T)
    local_dist_mat[local_dist_mat == np.inf] = 1e4
    return local_dist_mat


def print_connected_components(a_graph):
    clusters = list(nx.connected_components(a_graph))
    for i,cluster in enumerate(clusters):
        clusters[i] = np.sort(np.array(list(cluster)).astype(int))
        
    return clusters

def find_edges_to_remove_by_weight(a_graph):
    min_weight_edges = {}
    
    for node in a_graph.nodes():
        edges = list(a_graph.edges(node, data='weight'))
        if edges:
            # Find the two minimum weight edges for each node
            sorted_edges = sorted(edges, key=lambda x: x[2])
            min_edges = sorted_edges[:2]  # Take the minimum and next minimum edges

            # Store these edges ensuring each node is only considered once per edge
            for edge in min_edges:
                node1, node2, weight = edge
                if node1 not in min_weight_edges:
                    min_weight_edges[node1] = []
                if node2 not in min_weight_edges:
                    min_weight_edges[node2] = []
                if len(min_weight_edges[node1]) < 2:
                    min_weight_edges[node1].append(edge)
                if len(min_weight_edges[node2]) < 2:
                    min_weight_edges[node2].append(edge)

    # Collect edges to keep
    edges_to_keep = {tuple(edge[:2]) for edges in min_weight_edges.values() for edge in edges}

    # Collect edges to remove
    edges_to_remove = set(a_graph.edges()) - edges_to_keep

    return edges_to_remove



def minimum_spanning_forest(graph):
    msf = nx.Graph()
    for component in nx.connected_components(graph):
        subgraph = graph.subgraph(component)
        mst = nx.minimum_spanning_tree(subgraph, weight='weight')
        msf = nx.compose(msf, mst)
    return msf


linearity_threshold = 0.5
radius_curvature_threshold = 500


    
rod_data_root_dir = Path('/Users/yeonsu/Data/steel-rods-xray-data')
segments_file_path = rod_data_root_dir / 'alpha200_epsilon00' / 'segments.mat'
adjacency_file_path = rod_data_root_dir / 'alpha200_epsilon00' / 'adjacency_distance_scale0p98_threshold0p3_ij_score.pkl'

mat_obj = loadmat(segments_file_path)
segments = mat_obj['segments']
segments = [seg[0] for seg in segments]
N_segments = len(segments)

for i,segment in enumerate(segments):
    segments[i] = np.array(segment,dtype=np.float64)

# print(f'Staircase clustering: {segments_file_path.parent}')
print(f'Number of segments: {N_segments}')

pickle_in = open(adjacency_file_path,'rb')
adjij = pickle.load(pickle_in)
svd_cylinders,centroids,orientations, = prep_svd_cylinder(segments,scale_factor=0.98)
segment_length_list = np.zeros(len(segments))
for i,seg in enumerate(segments):
    segment_length_list[i] = np.sum(np.sqrt(np.sum(np.diff(seg,axis=0)**2,axis=1)))

#adjij: i,j,score,dist,t,u
adjij = np.array(adjij)
ijs = adjij[:,0:2].astype(int)
align_score = adjij[:,2]
dist_score = adjij[:,3]

svd_cylinders,centroids,orientations, = prep_svd_cylinder(segments,scale_factor=0.98)
distance_threshold = 50
alignment_threshold = 0.05

initial_maks = (align_score < alignment_threshold) & (dist_score < distance_threshold)

Graph0 = nx.Graph()
Graph0.add_nodes_from(range(N_segments))
Graph0.add_weighted_edges_from(zip(ijs[initial_maks,0],ijs[initial_maks,1],dist_score[initial_maks]))



clusters = print_connected_components(Graph0)

# check if common elements exist between clusters
if np.hstack(clusters).size != len(np.unique(np.hstack(clusters))):
    print('Common elements exist between clusters')
        
cluster_size_list = [len(x) for x in clusters]
max_cluster_size = np.max(cluster_size_list)
print(f'Number of clusters: {len(clusters)}')
print(f'Max cluster size: {max_cluster_size}')

cluster_length_list = np.zeros(len(clusters))
for i in range(len(clusters)):
    cluster_length_list[i] = segment_length_list[clusters[i]].sum()/650
np.argsort(cluster_length_list)[::-1]

subcluster_error_threshold = 1
subcluster_length_threshold = 200

good_clusters = []
for i_cluster, a_cluster in enumerate(clusters):
    subclusters = explode_local_cluster(a_cluster)        
    # check quality
    subcluster_error_list = np.zeros(len(subclusters))
    subcluster_length_list = np.zeros(len(subclusters))
    for i_subcluster,subcluster in enumerate(subclusters):
        joined = np.vstack([segments[ a_cluster[iii] ] for iii in subcluster])
        fit_result = fit_rod(joined,linearity_threshold=0.0001,radius_curvature_threshold=100000)
        subcluster_error_list[i_subcluster] = fit_result['err']
        # rec = fit_result['rec']
        subcluster_length_list[i_subcluster] = segment_length_list[ a_cluster[subcluster] ].sum()
    
    certification = (subcluster_error_list < subcluster_error_threshold) & (subcluster_length_list > subcluster_length_threshold)
    
    for i_cert in np.where(certification)[0]:
        good_clusters.append([a_cluster[i] for i in subclusters[i_cert]])
    

# fig,ax = set_3d_plot()
# for i_gcl in np.random.choice(len(good_clusters),1000):
#     gcl = good_clusters[i_gcl]
#     joined = np.vstack([segments[i] for i in gcl])
#     plot_single_rod(joined,ax=ax)

good_nodes = np.hstack(good_clusters)
not_yet_nodes = np.setdiff1d(range(N_segments),good_nodes)
for i_ny in not_yet_nodes:
    rr = segments[i_ny]
    fit_result = fit_rod(rr,linearity_threshold=0.0001,radius_curvature_threshold=100000)
    rr_len = np.sum(np.sqrt(np.sum(np.diff(rr,axis=0)**2,axis=1)))
    
    if fit_result['err'] < 1 and rr_len > 200:
        good_clusters.append(np.array([i_ny]))
        
good_nodes = np.hstack(good_clusters)
not_yet_nodes = np.setdiff1d(range(N_segments),good_nodes)

print(f'Number of good clusters: ', len(good_clusters))
print(f'Number of not yet nodes: ', len(not_yet_nodes))


def sort_curve(rr):
    centroid = np.mean(rr,axis=0)
    rr_centered = rr - centroid        
    _,_, V = np.linalg.svd(rr_centered, full_matrices=False)
    v1 = V[0,:]
    orientation = v1 * np.sign(np.sum(v1 * (rr_centered[-1, :] - rr_centered[0, :])))
    slist = np.dot((rr - centroid), orientation)
    sorted_indices = np.argsort(slist)
    return centroid + rr_centered[sorted_indices]

good_segments = []
for i_gcl in range(len(good_clusters)):
    gcl = good_clusters[i_gcl]
    joined = np.vstack([segments[i] for i in gcl])
    good_segments.append(sort_curve(joined))
    
not_yet_segments = []
for i_ny in range(len(not_yet_nodes)):
    rr = segments[not_yet_nodes[i_ny]]
    not_yet_segments.append(rr)
    
len(not_yet_nodes)

segments_next_round = good_segments + not_yet_segments
len(segments_next_round)


from example_get_adjij import calculate_alignment_adjacency_numba


svd_cylinders,centroids,orientations = prep_svd_cylinder(segments_next_round,scale_factor=0.98)

start = time.time()
adjij = calculate_alignment_adjacency_numba(svd_cylinders,orientations,threshold=0.15)
print(f'Elapsed time: {time.time()-start} sec')

pickle_in = open('adjacency_distance_scale0p98_threshold0p3_ij_score_next.pkl','wb')
pickle.dump(adjij,pickle_in)

# below is an attempt, but wouldn't work

fig,ax=set_3d_plot()
rr_len_list = np.zeros((3000,2))
k = 0
for i_node in np.random.choice(len(not_yet_nodes),3000):
    rr = segments[ not_yet_nodes[i_node] ]
    rr_len = np.sum(np.sqrt(np.sum(np.diff(rr,axis=0)**2,axis=1)))
    rr_len_list[k,:] = [i_node,rr_len]
    k = k + 1
    plot_single_rod(rr,ax=ax)

i_list = rr_len_list[:,0].astype(int)
rlen_list = rr_len_list[:,1]

np.where(rlen_list>300)

i_node = i_list[2298]
rr = segments[i_node]
print(np.sum(np.sqrt(np.sum(np.diff(rr,axis=0)**2,axis=1))))

fig,ax=set_3d_plot()
plot_single_rod(rr,ax=ax)
ax.axis('equal')
fit_result = fit_rod(rr,linearity_threshold=0.0001,radius_curvature_threshold=100000)
print(fit_result['err'])
    
    
    

num_good_clusters = len(good_clusters)
good_clusters_length_list = np.zeros(num_good_clusters)
for i_gcl in range(num_good_clusters):
    gcl = good_clusters[i_gcl]
    joined = np.vstack([segments[i] for i in gcl])
    
    fit_result = fit_rod(joined,linearity_threshold=0.0001,radius_curvature_threshold=100000)
    good_clusters_length_list[i_gcl] = np.sum(np.sqrt(np.sum(np.diff(fit_result['rec'],axis=0)**2,axis=1)))

fig,ax = plt.subplots(1,1)
plt.hist(good_clusters_length_list,bins=100)
plt.savefig('/Users/yeonsu/Figures/good_clusters_length_list.png')





Graph0.number_of_nodes()


distance_threshold = 50
alignment_threshold = 0.05

initial_maks = (align_score < 0.05) & (dist_score < 200)

Graph1 = nx.Graph()
Graph1.add_nodes_from(range(N_segments))
Graph1.add_weighted_edges_from(zip(ijs[initial_maks,0],ijs[initial_maks,1],dist_score[initial_maks]))
Graph1.remove_nodes_from(good_nodes)

clusters_after = print_connected_components(Graph1)

# check if common elements exist between clusters_after
if np.hstack(clusters_after).size != len(np.unique(np.hstack(clusters_after))):
    print('Common elements exist between clusters_after')
        
cluster_size_list = [len(x) for x in clusters_after]
max_cluster_size = np.max(cluster_size_list)
print(f'Number of clusters_after: {len(clusters_after)}')
print(f'Max cluster size: {max_cluster_size}')


np.argmax(cluster_size_list)

cluster = clusters_after[np.argmax(cluster_size_list)]

fig,ax=set_3d_plot()
for i in range(len(cluster)):
    plot_single_rod(segments[cluster[i]],ax=ax)

