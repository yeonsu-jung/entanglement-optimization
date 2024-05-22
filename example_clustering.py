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

def inspect_clustering(good_cl,segments):
    fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
    for i_gcl in range(len(good_cl)):
        gcl = good_cl[i_gcl]
        joined = np.vstack([segments[i] for i in gcl])    
        ax.plot(joined[:,0],joined[:,1],joined[:,2],linewidth=0.5)
        
    joined_segment_length_list = np.zeros(len(good_cl))
    for i_gcl in range(len(good_cl)):
        gcl = good_cl[i_gcl]
        joined = np.vstack([segments[i] for i in gcl])
        joined = sort_curve(joined)
        joined_segment_length_list[i_gcl] = seg_len(joined)
        
    fig,ax=plt.subplots(1,1)
    ax.hist(joined_segment_length_list,bins=100)
    return joined_segment_length_list



def check_overlap(good_cl,already_cl):
    no_overlap = []    
    for i_gcl in range(len(good_cl)):
        gcl = good_clusters[i_gcl]
        if np.isin(gcl,already_cl).any():
            tmp_nodes = gcl[np.isin(gcl,already_cl)]
            
            print(f'Cluster {i_gcl} has already been clustered')
            print(f'Nodes: {tmp_nodes}')
            
            print(f'Corresponding cluster: ', gcl)
            continue
        else:
            no_overlap.append(gcl)
            
    return no_overlap






















linearity_threshold = 0.5
radius_curvature_threshold = 500
already_clustered = []
    
rod_data_root_dir = Path('/Users/yeonsu/Data/steel-rods-xray-data')
segments_file_path = rod_data_root_dir / 'alpha200_epsilon00' / 'pruned_segments.pkl'
connectivity_file_path = rod_data_root_dir / 'alpha200_epsilon00' / 'ijscores.pkl'

if os.path.exists(segments_file_path):
    segments = pickle.load(open(segments_file_path,'rb'))    
    # for i,segment in enumerate(segments):
    #     segments[i] = np.array(segment,dtype=np.float64)
else:
    print(f'File not found: {segments_file_path}')
    pass
    # pickle_in = open(segments_file_path,'rb')
    # segments = pickle.load(pickle_in)
    exit

if os.path.exists(connectivity_file_path):
    ij_scores = pickle.load(open(connectivity_file_path,'rb'))
    pass
else:
    # svd_cylinders,centroids,orientations, = prep_svd_cylinder(segments,scale_factor=0.98)
    pass
    fp = filamentprocessing.FilamentProcessing(segments,2500,0.15,0.99)

    ij = np.array(fp.get_svd_ij())
    scores = np.array(fp.get_svd_scores())

    pickle_out = open(connectivity_file_path,'wb')
    ij_scores = np.hstack([ij,scores])
    pickle.dump(ij_scores,pickle_out)

N_segments = len(segments)
# segments_length_list,segments_error_list = inspect_segments(segments) # turn off this if you are sure about the segments

distance_threshold = 50
alignment_threshold = 0.05
fitting_threshold = 1
ij = ij_scores[:,:2]
scores = ij_scores[:,2:]

dist_score = scores[:,0]
align_score = scores[:,1]
fit_score = scores[:,2]
initial_mask = (align_score < alignment_threshold) & (dist_score < distance_threshold)

Graph0 = nx.Graph()
Graph0.add_nodes_from(range(N_segments))
Graph0.add_weighted_edges_from(zip(ij[initial_mask,0],ij[initial_mask,1],dist_score[initial_mask]))    
clusters = find_connected_components(Graph0)
cluster_size_list = [len(x) for x in clusters]
max_cluster_size = np.max(cluster_size_list)
print(f'Number of clusters: {len(clusters)}')
print(f'Max cluster size: {max_cluster_size}')

subcluster_error_threshold = 1
subcluster_length_threshold = 600



import importlib
import clustering
importlib.reload(clustering)
from clustering import subclustering_by_mst


good_clusters,not_yet_nodes = subclustering_by_mst(clusters,segments,subcluster_error_threshold,subcluster_length_threshold,subcluster_length_tolerance=50)

def sort_curve(rr):
    centroid = np.mean(rr,axis=0)
    rr_centered = rr - centroid        
    _,_, V = np.linalg.svd(rr_centered, full_matrices=False)
    v1 = V[0,:]
    orientation = v1 * np.sign(np.sum(v1 * (rr_centered[-1, :] - rr_centered[0, :])))
    slist = np.dot((rr - centroid), orientation)
    sorted_indices = np.argsort(slist)
    return centroid + rr_centered[sorted_indices]

def seg_len(seg):
    return np.sum(np.sqrt(np.sum(np.diff(seg,axis=0)**2,axis=1)))

joined_segment_length_list = np.zeros(len(good_clusters))
for i_gcl in range(len(good_clusters)):
    gcl = good_clusters[i_gcl]
    joined = np.vstack([segments[i] for i in gcl])
    joined = sort_curve(joined)
    joined_segment_length_list[i_gcl] = seg_len(joined)

fig,ax=plt.subplots(1,1)
ax.hist(joined_segment_length_list,bins=100)

fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
for i_gcl in range(len(good_clusters)):
    gcl = good_clusters[i_gcl]
    joined = np.vstack([segments[i] for i in gcl])    
    ax.plot(joined[:,0],joined[:,1],joined[:,2],linewidth=0.5)
    









already_clustered.extend(good_clusters)

alreday_clustered_nodes = np.concatenate(already_clustered)
print(f'Number of already clustered nodes: {len(alreday_clustered_nodes)}')














distance_threshold = 200
alignment_threshold = 0.05
fitting_threshold = 2
ij = ij_scores[:,:2]
scores = ij_scores[:,2:]

dist_score = scores[:,0]
align_score = scores[:,1]
fit_score = scores[:,2]
initial_mask = (align_score < alignment_threshold) & (dist_score < distance_threshold)

Graph0 = nx.Graph()
Graph0.add_nodes_from(range(N_segments))
Graph0.add_weighted_edges_from(zip(ij[initial_mask,0],ij[initial_mask,1],dist_score[initial_mask]))    
Graph0.remove_nodes_from(alreday_clustered_nodes)

clusters = find_connected_components(Graph0)
cluster_size_list = [len(x) for x in clusters]
max_cluster_size = np.max(cluster_size_list)
print(f'Number of clusters: {len(clusters)}')
print(f'Max cluster size: {max_cluster_size}')

np.argmax(cluster_size_list)

cc = clusters[1973]
fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
for i in range(len(cc)):
    ax.plot(segments[cc[i]][:,0],segments[cc[i]][:,1],segments[cc[i]][:,2],linewidth=0.5)
ax.axis('equal')

good_clusters,not_yet_nodes = subclustering_by_mst(clusters,segments,subcluster_error_threshold,subcluster_length_threshold,subcluster_length_tolerance=50)



jsll = inspect_clustering(good_clusters,segments)

    
no_overlap = check_overlap(good_clusters,alreday_clustered_nodes)










already_clustered.extend(no_overlap)

alreday_clustered_nodes = np.concatenate(already_clustered)
print(f'Number of already clustered nodes: {len(alreday_clustered_nodes)}')










distance_threshold = 200
alignment_threshold = 0.05
fitting_threshold = 2
ij = ij_scores[:,:2]
scores = ij_scores[:,2:]

dist_score = scores[:,0]
align_score = scores[:,1]
fit_score = scores[:,2]
initial_mask = (align_score < alignment_threshold) & (dist_score < distance_threshold)

Graph0 = nx.Graph()
Graph0.add_nodes_from(range(N_segments))
Graph0.add_weighted_edges_from(zip(ij[initial_mask,0],ij[initial_mask,1],dist_score[initial_mask]))    
Graph0.remove_nodes_from(alreday_clustered_nodes)

clusters = find_connected_components(Graph0)
cluster_size_list = [len(x) for x in clusters]
max_cluster_size = np.max(cluster_size_list)
print(f'Number of clusters: {len(clusters)}')
print(f'Max cluster size: {max_cluster_size}')

good_clusters,not_yet_nodes = subclustering_by_mst(clusters,segments,subcluster_error_threshold,subcluster_length_threshold,subcluster_length_tolerance=50)

jsll = inspect_clustering(good_clusters,segments)






