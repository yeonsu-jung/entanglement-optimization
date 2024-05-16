from scipy.spatial import distance
from scipy.cluster import hierarchy
import numpy as np
from pathlib import Path
import networkx as nx
import pickle
from fitting import fit_rod_error, fit_rod
from data_io import load_xray_data
from visualizations import set_3d_plot, plot_single_rod
from matplotlib import pyplot as plt


def check_centerlines(centerlines):
    unpacked = np.vstack(centerlines)
    labels = np.zeros(unpacked.shape[0],dtype=np.int64)
    start_idx = 0
    for i,cl in enumerate(centerlines):
        end_idx = start_idx + cl.shape[0]
        labels[start_idx:end_idx] = i
        start_idx = end_idx
        
    unq,ind,inv,cnt = np.unique(unpacked,axis=0,return_counts=True,return_inverse=True,return_index=True)
    nonoverlap_labels = np.unique(labels[(cnt == 1)[inv]])
    fig,ax = set_3d_plot()
    for lb in nonoverlap_labels:
        rr = centerlines[lb]
        plot_single_rod(rr,'-',ax=ax)

if __name__ == '__main__':
    pth = Path('/Users/yeonsu/Documents/GitHub/entanglement-optimization/xray_raw_data/alpha200_epsilon00/centerlines.mat')
    centerlines,_ = load_xray_data(pth)    
    for i,cl in enumerate(centerlines):
        centerlines[i] = np.unique(cl,axis=0)
    
    pickle_in = open('duplicate_groups.pkl','rb')
    duplicate_groups = pickle.load(pickle_in)
    pickle_in = open('group_labels.pkl','rb')
    group_labels = pickle.load(pickle_in)
    
    i = 489
    glb = group_labels[i]
    group_vertices = np.vstack([centerlines[lb] for lb in glb])
    group_unq,group_count = np.unique(group_vertices,axis=0,return_counts=True)
        
    # Step 1: Calculate the pairwise distance matrix
    dist_matrix = distance.pdist(group_unq, 'euclidean')
    square_dist_matrix = distance.squareform(dist_matrix)

    # Step 2: Perform hierarchical clustering
    linked = hierarchy.linkage(dist_matrix, 'single')

    # Step 3: Plot the dendrogram to help decide the cut-off
    plt.figure(figsize=(10, 7))
    dendrogram = hierarchy.dendrogram(linked)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Sample index')
    plt.ylabel('Distance')
    plt.show()
    
    fig,ax = set_3d_plot()
    plot_single_rod(group_unq,'.',ax=ax,markersize=0.5)
    
    print