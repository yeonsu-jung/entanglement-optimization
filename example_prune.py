from pathlib import Path
from scipy.io import loadmat
import pickle
import numpy as np
from fitting import prep_svd_cylinder
import networkx as nx
from visualizations import set_3d_plot,plot_single_rod
from sklearn_extra.cluster import KMedoids
from fitting import fit_rod
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt

from distances import lumelsky_dist_vec
from visualizations import plot_centerline_with_container
import itertools
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from scipy.ndimage import uniform_filter1d

from clustering import find_connected_components,explode_local_cluster
import time
from fitting import fit_rod

def edge_lengths(curve):
    return (np.sqrt(np.sum(np.diff(curve,axis=0)**2,axis=1)))

def break_segments(segs):
    new_segments = []
    for seg in segs:
        edge_len = edge_lengths(seg)
        grph = nx.Graph()
        grph.add_nodes_from(range(len(seg)))

        for i in range(len(seg)-1):
            if edge_len[i] <= np.sqrt(3):
                grph.add_edge(i,i+1)
            
        clusters = list(nx.connected_components(grph))
        for i,cluster in enumerate(clusters):
            if len(cluster) == 1:                
                continue
            rr = np.array(seg,dtype=np.float64)            
            new_segments.append(rr[list(cluster)])
        
    return new_segments

def inspect_segments(segments):
    N_segments = len(segments)
    segments_length_list = np.zeros(N_segments)
    for i,seg in enumerate(segments):
        segments_length_list[i] = np.sum(np.sqrt(np.sum(np.diff(seg,axis=0)**2,axis=1)))   
        
        
    fig,ax=plt.subplots(1,1)
    ax.hist(segments_length_list,bins=100)
    ax.set_xlim([0,1000])
    
    from fitting import fit_rod

    segments_error_list = np.zeros(N_segments)
    for i,seg in enumerate(segments):
        rr = np.array(seg,dtype=np.float64)
        fit_result = fit_rod(rr,0.00001,10000)
        segments_error_list[i] = fit_result['err']
        
    fig,ax=plt.subplots(1,1)
    ax.hist(segments_error_list,bins=100)
        
    print(f'Maximum segment length: {np.max(segments_length_list)} at index {np.argmax(segments_length_list)}')
    print(f'Maximum segment error: {np.max(segments_error_list)} at index {np.argmax(segments_error_list)}')
    
    return segments_length_list,segments_error_list

    fig,ax=plt.subplots(1,1)
    ax.hist(segments_length_list,bins=100)
    ax.set_xlim([0,1000])
    
    from fitting import fit_rod

    segments_error_list = np.zeros(N_segments)
    for i,seg in enumerate(segments):
        rr = np.array(seg,dtype=np.float64)
        fit_result = fit_rod(rr,0.00001,10000)
        segments_error_list[i] = fit_result['err']
        
    fig,ax=plt.subplots(1,1)
    ax.hist(segments_error_list,bins=100)
        
    print(f'Maximum segment length: {np.max(segments_length_list)} at index {np.argmax(segments_length_list)}')
    print(f'Maximum segment error: {np.max(segments_error_list)} at index {np.argmax(segments_error_list)}')
    
    return segments_length_list,segments_error_list

def sort_curve(rr):
    centroid = np.mean(rr,axis=0)
    rr_centered = rr - centroid        
    _,_, V = np.linalg.svd(rr_centered, full_matrices=False)
    v1 = V[0,:]
    orientation = v1 * np.sign(np.sum(v1 * (rr_centered[-1, :] - rr_centered[0, :])))
    slist = np.dot((rr - centroid), orientation)
    sorted_indices = np.argsort(slist)
    return centroid + rr_centered[sorted_indices]
    
def initial_prune_segments(segments):
    for i,seg in enumerate(segments):
        seg = np.unique(seg,axis=0)
        segments[i] = sort_curve(seg)
        
    return segments

def minimum_spanning_forest(graph):
    msf = nx.Graph()
    for component in nx.connected_components(graph):
        subgraph = graph.subgraph(component)
        mst = nx.minimum_spanning_tree(subgraph, weight='weight')
        msf = nx.compose(msf, mst)
    return msf

def curvature_of_polygonal_curve(nodes):
    tan2 = nodes[2:,:] - nodes[1:-1,:]    
    tan1 = nodes[1:-1,:] - nodes[:-2,:]
    
    nom = np.linalg.norm(2*np.cross(tan1,tan2,axis=1),axis=1)
    den = np.sum(tan1*tan2,axis=1)
    # curvature = np.sum(nom/den)
    return nom/den

def break_curved_rods(seg,curvature_threshold):
    curvature = curvature_of_polygonal_curve(seg)
    break_points = np.where(np.abs(curvature)>curvature_threshold)[0]
    if len(break_points)==0:
        return [seg]
    else:
        segs = []
        start_idx = 0
        for bp in break_points:
            segs.append(seg[start_idx:bp+1])
            start_idx = bp
        segs.append(seg[start_idx:])
        return segs

if __name__ == '__main__':
    linearity_threshold = 0.5
    radius_curvature_threshold = 500
        
    rod_data_root_dir = Path('/Users/yeonsu/Data/steel-rods-xray-data')
    segments_file_path = rod_data_root_dir / 'alpha200_epsilon00' / 'segments.mat'
    # adjacency_file_path = rod_data_root_dir / 'alpha200_epsilon00' / 'adjacency_distance_scale0p98_threshold0p3_ij_score.pkl'

    mat_obj = loadmat(segments_file_path)
    segments = mat_obj['segments']
    segments = [seg[0] for seg in segments]
    N_segments = len(segments)

    for i,segment in enumerate(segments):
        segments[i] = np.array(segment,dtype=np.float64)
        
    segments = initial_prune_segments(segments)
    segments_length_list,segments_error_list = inspect_segments(segments)
    new_segments = break_segments(segments)

    for i,seg in enumerate(new_segments):
        new_segments[i] = sort_curve(seg)
        
    segments_length_list,segments_error_list = inspect_segments(new_segments)

    fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
    broken_segments_list = []
    for i in np.where(segments_error_list>1)[0]:
        rr = new_segments[i]
        broken_pieces = break_curved_rods(rr,10)    
        for bp in broken_pieces:
            ax.plot(bp[:,0],bp[:,1],bp[:,2])    
            broken_segments_list.append(bp)
            

    # delete segments_error_list>1
    new_segments2 = [seg for i,seg in enumerate(new_segments) if segments_error_list[i]<=1]
    new_segments2 = new_segments2 + broken_segments_list
    
    
    
    # print(f'Staircase clustering: {segments_file_path.parent}')
    print(f'Number of segments: {len(new_segments2)}')
    
    segments_length_list,segments_error_list = inspect_segments(new_segments2)
    
    segments = new_segments2
    
    # rod_data_root_dir = Path('/Users/yeonsu/Data/steel-rods-xray-data')
    segments_file_path = rod_data_root_dir / 'alpha200_epsilon00' / 'segments.mat'
    
    pickle_out = open(segments_file_path.parent / 'pruned_segments.pkl','wb')
    pickle.dump(segments,pickle_out)
    
    




# below is an attempt, but wouldn't work

# fig,ax=set_3d_plot()
# rr_len_list = np.zeros((3000,2))
# k = 0
# for i_node in np.random.choice(len(not_yet_nodes),3000):
#     rr = segments[ not_yet_nodes[i_node] ]
#     rr_len = np.sum(np.sqrt(np.sum(np.diff(rr,axis=0)**2,axis=1)))
#     rr_len_list[k,:] = [i_node,rr_len]
#     k = k + 1
#     plot_single_rod(rr,ax=ax)

# i_list = rr_len_list[:,0].astype(int)
# rlen_list = rr_len_list[:,1]

# np.where(rlen_list>300)

# i_node = i_list[2298]
# rr = segments[i_node]
# print(np.sum(np.sqrt(np.sum(np.diff(rr,axis=0)**2,axis=1))))

# fig,ax=set_3d_plot()
# plot_single_rod(rr,ax=ax)
# ax.axis('equal')
# fit_result = fit_rod(rr,linearity_threshold=0.0001,radius_curvature_threshold=100000)
# print(fit_result['err'])
    
    
    

# num_good_clusters = len(good_clusters)
# good_clusters_length_list = np.zeros(num_good_clusters)
# for i_gcl in range(num_good_clusters):
#     gcl = good_clusters[i_gcl]
#     joined = np.vstack([segments[i] for i in gcl])
    
#     fit_result = fit_rod(joined,linearity_threshold=0.0001,radius_curvature_threshold=100000)
#     good_clusters_length_list[i_gcl] = np.sum(np.sqrt(np.sum(np.diff(fit_result['rec'],axis=0)**2,axis=1)))

# fig,ax = plt.subplots(1,1)
# plt.hist(good_clusters_length_list,bins=100)
# plt.savefig('/Users/yeonsu/Figures/good_clusters_length_list.png')





# Graph0.number_of_nodes()


# distance_threshold = 50
# alignment_threshold = 0.05

# initial_maks = (align_score < 0.05) & (dist_score < 200)

# Graph1 = nx.Graph()
# Graph1.add_nodes_from(range(N_segments))
# Graph1.add_weighted_edges_from(zip(ijs[initial_maks,0],ijs[initial_maks,1],dist_score[initial_maks]))
# Graph1.remove_nodes_from(good_nodes)

# clusters_after = print_connected_components(Graph1)

# # check if common elements exist between clusters_after
# if np.hstack(clusters_after).size != len(np.unique(np.hstack(clusters_after))):
#     print('Common elements exist between clusters_after')
        
# cluster_size_list = [len(x) for x in clusters_after]
# max_cluster_size = np.max(cluster_size_list)
# print(f'Number of clusters_after: {len(clusters_after)}')
# print(f'Max cluster size: {max_cluster_size}')


# np.argmax(cluster_size_list)

# cluster = clusters_after[np.argmax(cluster_size_list)]

# fig,ax=set_3d_plot()
# for i in range(len(cluster)):
#     plot_single_rod(segments[cluster[i]],ax=ax)

