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
    from sklearn.cluster import KMeans
    from matplotlib import pyplot as plt
    
    from example_PhysicalRodRelaxation import lumelsky_dist_vec
    from visualizations import plot_centerline_with_container
    
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
    
    i_cluster = 538
    a_cluster = connected_component_list[i_cluster]    
    print(segment_length_list[a_cluster].sum()/650)
    len(a_cluster)
    
    fig,ax=set_3d_plot()
    for i in range(len(a_cluster)):
        plot_single_rod(segments[a_cluster[i]],'-',ax=ax)
    ax.axis('equal')
    
    local_align_mat = np.full((len(a_cluster),len(a_cluster)),np.inf)
    local_dist_mat = np.full((len(a_cluster),len(a_cluster)),np.inf)
    local_ferr_mat = np.full((len(a_cluster),len(a_cluster)),np.inf)
    local_fcurv_mat = np.full((len(a_cluster),len(a_cluster)),np.inf)
    
    svd_cylinders,centroids,orientations, = prep_svd_cylinder(segments,scale_factor=0.12)
    
    from scipy.spatial import ConvexHull, convex_hull_plot_2d
    num_a_cluster = len(a_cluster)
    points = np.zeros((num_a_cluster*2,3))
    for i in range(len(a_cluster)):
        segment_i = segments[a_cluster[i]]
        points[i] = segment_i[0]
        points[i+num_a_cluster] = segment_i[-1]
            
    hull = ConvexHull(points)
    fig,ax=set_3d_plot()
    for i in range(len(a_cluster)):
        plot_single_rod(segments[a_cluster[i]],'-',ax=ax)
    # ax.axis('equal')
    # ax.plot(points[:,0],points[:,1],points[:,2],'o')
    for simplex in hull.simplices:
        ax.plot(points[simplex, 0], points[simplex, 1], points[simplex, 2], 'k-')
    
    end_segments = []
    
    plt.close()
    fig,ax=set_3d_plot()
    for i in range(len(a_cluster)):
        plot_single_rod(segments[a_cluster[i]],'-',ax=ax)
    for vert in hull.vertices:
        ax.plot(points[vert, 0], points[vert, 1], points[vert, 2], 'o')
    
    enemies = []    
    for i in range(len(a_cluster)):
        segment_i = segments[a_cluster[i]]
        cylinder_i = svd_cylinders[a_cluster[i]]
        orientation1 = orientations[a_cluster[i]]
        
        for j in range(i+1,len(a_cluster)):
            segment_j = segments[a_cluster[j]]
            cylinder_j = svd_cylinders[a_cluster[j]]
            orientation2 = orientations[a_cluster[j]]
            
            joined = np.vstack([segment_i,segment_j])
            fit_result = fit_rod(joined,linearity_threshold=0.1,radius_curvature_threshold=500)
            local_ferr_mat[i,j] = fit_result['err']
            local_fcurv_mat[i,j] = 1/fit_result['radius']
            
            t,u,d1,d2,d12 = lumelsky_dist_vec(cylinder_i[0:3],cylinder_i[3:6],cylinder_j[0:3],cylinder_j[3:6])
            # end-facing condition
            vec = d1*t - d2*u - d12
            dist = np.linalg.norm(vec)
            vec /= dist
            align_score = (np.linalg.norm(np.cross(vec,orientation1)) + np.linalg.norm(np.cross(vec,orientation2)))/2
            # align_score = 1/(np.abs(np.dot(vec,orientation1)) + np.abs(np.dot(vec,orientation2)))**2
            local_align_mat[i,j] = align_score
            local_dist_mat[i,j] = dist

    local_align_mat = np.minimum(local_align_mat,local_align_mat.T)
    local_align_mat[local_align_mat == np.inf] = 1e4
    local_align_mat[np.diag_indices(len(a_cluster))] = 0
    
    local_dist_mat = np.minimum(local_dist_mat,local_dist_mat.T)
    local_dist_mat[local_dist_mat == np.inf] = 1e4
    local_dist_mat[np.diag_indices(len(a_cluster))] = 0
    
    local_ferr_mat = np.minimum(local_ferr_mat,local_ferr_mat.T)
    local_ferr_mat[local_ferr_mat == np.inf] = 1e4
    local_ferr_mat[np.diag_indices(len(a_cluster))] = 0
    
    local_align_ijs = np.where(local_align_mat < 0.05)
    local_align_ijs = np.where((local_dist_mat < 95) & (local_align_mat < 0.01))
    
    
    local_align_ijs = np.where((local_ferr_mat < 2) & (local_align_mat < 0.01))
    S = nx.Graph()
    S.add_nodes_from(range(len(a_cluster)))
    S.add_edges_from(zip(local_align_ijs[0],local_align_ijs[1]))
    conncomp_S = list(nx.connected_components(S))
    
    subcluster_length_list = np.zeros(len(conncomp_S))
    for i in range(len(conncomp_S)):
        idx = np.array(list(conncomp_S[i]))
        subcluster_length_list[i] = np.sum([segment_length_list[ a_cluster[j] ] for j in idx])
    
    
    fig,ax=set_3d_plot()
    for i in range(len(conncomp_S)):
        idx = list(conncomp_S[i])
        joined = np.vstack([segments[ a_cluster[j] ] for j in idx])
        clr = np.random.rand(3)
        if len(idx)>1:
            for each_rod in idx:
                plot_single_rod(segments[a_cluster[each_rod]],'-',ax=ax,color=clr,markersize=2)
        else:
            for each_rod in idx:
                plot_single_rod(segments[a_cluster[each_rod]],'-',ax=ax,color='black',alpha=0.5)
    ax.axis('equal')
    
    
    for i in range(len(conncomp_S)):
        idx = np.array(list(conncomp_S[i]))
        joined = np.vstack([segments[ a_cluster[j] ] for j in idx])
        fit_result = fit_rod(joined)
        subcluster_err = fit_result['err']
        subcluster_len = 1
        
    
    
    
    fig,ax=set_3d_plot()
    for i in range(len(a_cluster)):
        cylinder_i = svd_cylinders[a_cluster[i]]
        orientation1 = orientations[a_cluster[i]]
        for j in range(i+1,len(a_cluster)):
            cylinder_j = svd_cylinders[a_cluster[j]]
            orientation2 = orientations[a_cluster[j]]
            
            t,u,d1,d2,d12 = lumelsky_dist_vec(cylinder_i[0:3],cylinder_i[3:6],cylinder_j[0:3],cylinder_j[3:6])
            vec = d1*t - d2*u - d12
            align_score = (np.linalg.norm(np.cross(vec,orientation1)) + np.linalg.norm(np.cross(vec,orientation2)))/2
            # align_score = 1/(np.abs(np.dot(vec,orientation1)) + np.abs(np.dot(vec,orientation2)))**2
            
            dist = np.linalg.norm(vec)
            plot_centerline_with_container(segments,svd_cylinders,a_cluster[i],ax)
            plot_centerline_with_container(segments,svd_cylinders,a_cluster[j],ax)
            
            # plot_single_rod(segments[a_cluster[i]],'-',ax=ax)
            # plot_single_rod(segments[a_cluster[j]],'-',ax=ax)
            
            ax.text(0.5*(segments[a_cluster[i]][0,0] + segments[a_cluster[j]][0,0]),\
                    0.5*(segments[a_cluster[i]][0,1] + segments[a_cluster[j]][0,1]),\
                    0.5*(segments[a_cluster[i]][0,2] + segments[a_cluster[j]][0,2]),\
                    f'Align score: {align_score:.2f}\n Dist: {dist}',fontsize=8)
            plt.savefig(f'/Users/yeonsu/Figures/debug/{i}_{j}.png')
            ax.clear()
    
    N_rods_estimate = int(segment_length_list[a_cluster].sum()//650)
    km = KMeans(n_clusters=N_rods_estimate, random_state=0, n_init="auto", max_iter = 10000).fit(local_ferr_mat)
    # km = KMedoids(n_clusters=N_rods_estimate, random_state=1, method='pam').fit(local_align_mat)
    print(km.labels_)
    # cluster_partitions = partition(a_cluster,N_rods_estimate,enemies)
    km.cluster_centers_
    
    plt.close()
    fig,ax=set_3d_plot()
    for ii in range(km.labels_.max() + 1):
        idx = np.where(km.labels_ == ii)[0]
        joined = np.vstack([segments[ a_cluster[j] ] for j in idx])
        
        clr = np.random.rand(3)
        k = 0
        for each_rod in idx:
            plot_single_rod(segments[a_cluster[each_rod]],'-',ax=ax,color=clr)
            ax.text(segments[a_cluster[each_rod]][0,0],segments[a_cluster[each_rod]][0,1],segments[a_cluster[each_rod]][0,2],str(each_rod),fontsize=8)
    
    local_align_mat[30,27]
        
    
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

