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
    
    
def prune_segments_by_curvature():           
    rod_data_root_dir = Path('/Users/yeonsu/Data/steel-rods-xray-data')
    segments_file_path = rod_data_root_dir / 'alpha200_epsilon00' / 'segments.mat'

    mat_obj = loadmat(segments_file_path)
    segments = mat_obj['segments']
    segments = [seg[0] for seg in segments]
    segments2 = segments.copy()

    for i,segment in enumerate(segments2):
        seg = np.array(segment,dtype=np.float64)
        seg = np.unique(seg,axis=0)
        seg = sort_curve(seg)
        segments2[i] = seg
    segments2 = break_segments(segments2)

    for i,seg in enumerate(segments2):
        segments2[i] = sort_curve(seg)
        
    segments_length_list,segments_error_list = inspect_segments(segments2)

    # fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
    broken_segments_list = []
    for i in np.where(segments_error_list>1)[0]:
        rr = segments2[i]
        broken_pieces = break_curved_rods(rr,10)    
        for bp in broken_pieces:
            # ax.plot(bp[:,0],bp[:,1],bp[:,2])    
            broken_segments_list.append(bp)
            

    # delete segments_error_list>1
    segments2 = [seg for i,seg in enumerate(segments2) if segments_error_list[i]<=1]
    segments2 = segments2 + broken_segments_list
    
    # print(f'Staircase clustering: {segments_file_path.parent}')
    print(f'Number of segments: {len(segments2)}')
    
    segments_length_list,segments_error_list = inspect_segments(segments2)    
    
    import filamentprocessing    
    fp = filamentprocessing.FilamentProcessing(segments,2500,0.3,0.99)
    
    ij = fp.get_svd_ij()
    scores = fp.get_svd_scores()    

    # fp.calculate_filament_distance_matrix(1000,0.3)    
    # ij = fp.get_ij()
    # scores = fp.get_scores()    

    ij = np.array(ij)
    scores = np.array(scores)

    dist_score = scores[:,0]
    align_score = scores[:,1]
    fit_score = scores[:,2]
    mask = align_score < 0.2
    
    import networkx as nx
    
    graph = nx.Graph()
    graph.add_nodes_from(range(len(segments2)))
    graph.add_edges_from(zip(ij[mask,0],ij[mask,1]))
    
    connected_component_list = list(nx.connected_components(graph))
    connected_component_list = [list(x) for x in connected_component_list]
    for i in range(len(connected_component_list)):
        connected_component_list[i] = sorted(connected_component_list[i])
        
    cluster_size_list = [len(x) for x in connected_component_list]
    max_cluster_size = np.max(cluster_size_list)
    print(f'Max cluster size: {max_cluster_size}')
    print(f'Number of clusters: {len(connected_component_list)}')
    
    def seg_len(seg):
        return np.sum(np.sqrt(np.sum(np.diff(seg,axis=0)**2,axis=1)))

    length_sum_list = np.zeros(len(connected_component_list))
    number_list = np.zeros(len(connected_component_list))        
    
    k = 0
    for cc in connected_component_list:
        
        rr_len_sum = 0
        for i in cc:
            rr = segments2[i]
            rr_len = seg_len(rr)
            rr_len_sum += rr_len            
        
        length_sum_list[k] = rr_len_sum
        number_list[k] = len(cc)
        k += 1
        
    np.count_nonzero( (length_sum_list < 50) & (number_list == 1) )
    
    to_remove = np.where( (length_sum_list < 50) & (number_list == 1) )[0]    
    # to_remove = np.where( (length_sum_list < 50) & (number_list == 1) & (length_sum_list > 30))[0]
    
    plt.close('all')
    fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
    for i in to_remove:
        for j in connected_component_list[i]:
            rr = segments2[j]
            ax.plot(rr[:,0],rr[:,1],rr[:,2],linewidth=1)
            # ax.text(rr[0,0],rr[0,1],rr[0,2],str(j))
            
    
            
    ax.axis('equal')    
    
    # plt.close('all')
    rr = segments2[1464]
    fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
    ax.plot(rr[:,0],rr[:,1],rr[:,2])
    ax.axis('equal')
    ax.view_init(0,0)
    
    # from scipy.io import savemat
    # savemat(rod_data_root_dir / 'alpha200_epsilon00' / 'connected_components.mat',{'connected_component_list':connected_component_list})
    
    
    return segments2
    
    
if __name__ == '__main__':
    segments2 = prune_segments_by_curvature()
    