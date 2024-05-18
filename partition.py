from typing import List, Set, Tuple

def partition(collection: List[int], k: int, enemies: List[Tuple[int, int]]) -> List[List[Set[int]]]:
    if len(collection) == 1:
        return [[set(collection)]]

    first = collection[0]
    rest = collection[1:]

    # Recursive call
    rest_parts = partition(rest, k, enemies)

    parts = []

    # For each smaller partition...
    for rest_part in rest_parts:
        # Insert `first` in each of the subpartition's subsets
        for j in range(len(rest_part)):
            new_subset = rest_part[j] | {first}
            new_part = rest_part[:j] + [new_subset] + rest_part[j+1:]
            
            if len(new_part) <= k:
                parts.append(new_part)

        # Put `first` in its own subset
        new_part = [{first}] + rest_part
        if len(new_part) <= k:
            parts.append(new_part)

    # Filter out partitions with enemies in the same subset
    enemy_sets = [set(enemy) for enemy in enemies]
    filtered_parts = []
    for part in parts:
        valid = True
        for subset in part:
            for enemy_set in enemy_sets:
                if enemy_set <= subset:  # Check if the enemy_set is a subset of the subset
                    valid = False
                    break
            if not valid:
                break
        if valid:
            filtered_parts.append(part)

    return filtered_parts

if __name__ == '__main__':    
    from pathlib import Path
    from scipy.io import loadmat
    import pickle
    import numpy as np
    from example_PhysicalRodRelaxation import prep_svd_cylinder
    import networkx as nx
    from visualizations import set_3d_plot,plot_single_rod
    from sklearn_extra.cluster import KMedoids
    from fitting import fit_rod
    
    distance_threshold = 200
    alignment_threshold = 0.1

    first_stair = 0.05
    last_stair = 0.06
    num_stairs = 1
    
    max_nodes_each_cluster = 500
    
    
    rod_data_root_dir = Path('/Users/yeonsu/Data/steel-rods-xray-data')
    segments_file_path = rod_data_root_dir / 'alpha200_epsilon00' / 'segments.mat'
    adjacency_file_path = rod_data_root_dir / 'alpha200_epsilon00' / 'adjacency_distance_scale0p98_threshold0p3_ij_score.pkl'
    
    mat_obj = loadmat(segments_file_path)
    segments = mat_obj['segments']
    segments = [seg[0] for seg in segments]
    N_segments = len(segments)
    
    print(f'Staircase clustering: {segments_file_path.parent}')
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
    
    error_criterion = 1
    length_criterion = 300
    stairs = np.linspace(0.001,0.15,30)
    distance_thresholds = np.linspace(10,300,30)
    max_cluster_size_criterion = 25
    
    G = nx.Graph()
    G.add_nodes_from(range(len(segments))) # play with edges only
    list_of_cluster_stats = []
    good_clusters = []
    good_cluster_nodes = []
    
    print(f'Distance threshold: {distance_threshold}')
    
    
    
    G.add_edges_from(ijs[(align_score < alignment_threshold) & (dist_score < distance_threshold)])
    G.remove_nodes_from(good_cluster_nodes)
    G.number_of_nodes()
    connected_component_list = list(nx.connected_components(G))
    connected_component_list = [list(x) for x in connected_component_list]
    for i in range(len(connected_component_list)):
        connected_component_list[i] = sorted(connected_component_list[i])
        
    cluster_size_list = [len(x) for x in connected_component_list]
    max_cluster_size = np.max(cluster_size_list)
    print(f'Max cluster size: {max_cluster_size}')
    
    cluster_length_list = np.zeros(len(connected_component_list))
    for i in range(len(connected_component_list)):
        cluster_length_list[i] = segment_length_list[connected_component_list[i]].sum()/650    
    np.argsort(cluster_length_list)[-100:-90]
    
    i_cluster = 96
    a_cluster = connected_component_list[i_cluster]    
    print(segment_length_list[a_cluster].sum()/650)
    len(a_cluster)
    
    fig,ax=set_3d_plot()
    for i in range(len(a_cluster)):
        plot_single_rod(segments[a_cluster[i]],'-',ax=ax)
    ax.axis('equal')
    
    local_dist_mat = np.full((len(a_cluster),len(a_cluster)),np.inf)
    for i in range(len(a_cluster)):
        global_i = a_cluster[i]
        mask_i = ijs[:,0] == global_i
        for j in range(i+1,len(a_cluster)):
            global_j = a_cluster[j]
            mask_j = ijs[:,1] == global_j
            d = dist_score[(mask_i & mask_j)]
            if d.size > 0:
                local_dist_mat[i,j] = d
                print(i,j,d)
            
    tmp = local_dist_mat[np.triu_indices(len(a_cluster),k=1)]
    np.count_nonzero(tmp > 1e10)
    

    
    def local_clustering_by_kmedoids(segments,a_cluster,N_rods_estimate,align_connectivity):
        fitting_score_over_trials = np.full(N_rods_estimate+1,np.inf)
        for i in range(1,N_rods_estimate+1):
            km = KMedoids(n_clusters=i, random_state=0).fit(align_connectivity)
            cluster_fit_errors = []
            cluster_lengths = []
            
            for ii in range(km.labels_.max() + 1):
                idx = np.where(km.labels_ == ii)[0]
                joined = np.vstack([segments[j] for j in idx])
                res = fit_rod(joined)
                cluster_fit_errors.append(res['err']**5)
                cluster_lengths.append(res['len']**2)
            
            fitting_score_over_trials[i] = np.sum(cluster_fit_errors)/np.mean(cluster_lengths)
        
        j_min = np.argmin(fitting_score_over_trials)
        km = KMedoids(n_clusters=j_min, random_state=0).fit(align_connectivity)
        
        return km,fitting_score_over_trials[j_min]
    
            
        
            
    
    
    
    
    print()

