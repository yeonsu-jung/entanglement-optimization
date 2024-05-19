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




def sort_curve(rr):
    centroid = np.mean(rr,axis=0)
    rr_centered = rr - centroid        
    _,_, V = np.linalg.svd(rr_centered, full_matrices=False)
    v1 = V[0,:]
    orientation = v1 * np.sign(np.sum(v1 * (rr_centered[-1, :] - rr_centered[0, :])))
    slist = np.dot((rr - centroid), orientation)
    sorted_indices = np.argsort(slist)
    return centroid + rr_centered[sorted_indices]

def join_segments(seg_i, seg_j):
    # Ensure the segments are numpy arrays
    seg_i = np.array(seg_i)
    seg_j = np.array(seg_j)

    # Get the endpoints of each segment
    p1, p2 = seg_i[0], seg_i[-1]
    q1, q2 = seg_j[0], seg_j[-1]

    # Calculate distances between endpoints
    d11 = np.linalg.norm(p1 - q1)
    d12 = np.linalg.norm(p1 - q2)
    d21 = np.linalg.norm(p2 - q1)
    d22 = np.linalg.norm(p2 - q2)

    # Determine the shortest distance and join segments accordingly
    min_dist = min(d11, d12, d21, d22)
    
    if min_dist == d11:
        return np.vstack([seg_i[::-1], seg_j])
    elif min_dist == d12:
        return np.vstack([seg_i[::-1], seg_j[::-1]])
    elif min_dist == d21:
        return np.vstack([seg_i, seg_j])
    else:  # min_dist == d22
        return np.vstack([seg_i, seg_j[::-1]])
    
def join_multiple_segments(seg_list):
    if not seg_list:
        return np.array([])  # Return an empty array if the input list is empty

    # Initialize the joined segment with the first segment in the list
    joined_segment = seg_list[0]

    # Iterate through the rest of the seg_list and join them one by one
    for i in range(1, len(seg_list)):
        joined_segment = join_segments(joined_segment, seg_list[i])

    return joined_segment

def calculate_curvature(seg):
    # Compute first and second derivatives
    dx_dt = np.gradient(seg[:, 0])
    dy_dt = np.gradient(seg[:, 1])
    d2x_dt2 = np.gradient(dx_dt)
    d2y_dt2 = np.gradient(dy_dt)

    # Compute curvature
    curvature = (dx_dt * d2y_dt2 - dy_dt * d2x_dt2) / (dx_dt**2 + dy_dt**2)**(3/2)
    
    return curvature
                            

def curvature_of_polygonal_curve(nodes):
    tan2 = nodes[2:,:] - nodes[1:-1,:]    
    tan1 = nodes[1:-1,:] - nodes[:-2,:]
    nom = np.linalg.norm(2*np.cross(tan1,tan2,axis=1),axis=1)
    den = np.sum(tan1*tan2,axis=1)
    curvature = (nom/den)
    return curvature

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
    import itertools
    from scipy.spatial import ConvexHull, convex_hull_plot_2d
    from scipy.ndimage import uniform_filter1d
    
    
    
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
    svd_cylinders,centroids,orientations, = prep_svd_cylinder(segments,scale_factor=0.98)
    
    G_mind = nx.Graph()
    G_mind.add_nodes_from(range(len(segments)))
    
    G_mind.add_weighted_edges_from(zip(ijs[:,0],ijs[:,1],dist_score))
    
    
    # Find the minimum weight edge for each node
    min_weight_edges = {}
    for node in G.nodes():
        edges = list(G_mind.edges(node, data='weight'))
        if edges:
            min_weight_edge = min(edges, key=lambda x: x[2])
            if node in min_weight_edges:
                # Ensure we only keep the minimum weight edge among already considered edges
                if min_weight_edges[node][2] > min_weight_edge[2]:
                    min_weight_edges[node] = min_weight_edge
            else:
                min_weight_edges[node] = min_weight_edge
    # Collect edges to remove
    edges_to_remove = set(G_mind.edges()) - {tuple(min_weight_edges[node][:2]) for node in min_weight_edges if node in min_weight_edges}

    # Remove the collected edges
    G_mind.remove_edges_from(edges_to_remove)
    num_edges = G_mind.number_of_edges()
    
    # get neighbor
    mind_edges = []
    for i in range(len(segments)):
        neighbor_list = list(G.neighbors(i))
        weights = []
        for nb in neighbor_list:
            weights.append(G[i][nb]['weight'])
            
        if len(weights) > 0:
            mind_edges.append((i,neighbor_list[np.argmin(weights)]))
            
    G_mind = nx.Graph()
    G_mind.add_nodes_from(range(len(segments)))
    G_mind.add_edges_from(mind_edges)
    
    G_mind_deg = np.array(list(G_mind.degree()))[:,1]
    problems = np.where(G_mind_deg > 2)[0]
    
    plt.subplots(1,1,figsize=(10,10))
    nx.draw(G_mind,with_labels=True)
    
    G_mind.remove_nodes_from(problems)
    plt.subplots(1,1,figsize=(10,10))
    nx.draw(G_mind,with_labels=True)
    
    
    
    
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
    
    i_cluster = 169
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
    
    svd_cylinders,centroids,orientations, = prep_svd_cylinder(segments,scale_factor=0.98)
    
    
    num_a_cluster = len(a_cluster)
    
    points = np.zeros((num_a_cluster*2,3))
    for i in range(len(a_cluster)):
        segment_i = segments[a_cluster[i]]
        points[i] = segment_i[0]
        points[i+num_a_cluster] = segment_i[-1]
    hull = ConvexHull(points)
     
    for i,seg in enumerate(segments):
        segments[i] = np.array(seg,dtype=np.float64)
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
            local_fcurv_mat[i,j] = 1/fit_result['r']
            
            # t,u,d1,d2,d12 = lumelsky_dist_vec(cylinder_i[0:3],cylinder_i[3:6],cylinder_j[0:3],cylinder_j[3:6])
            t,u,d1,d2,d12 = lumelsky_dist_vec(segment_i[0],segment_i[-1],segment_j[0],segment_j[-1])
            # end-facing condition
            vec = d1*t - d2*u - d12
            dist = np.linalg.norm(vec)
            vec /= dist
            align_score = (np.linalg.norm(np.cross(vec,orientation1)) + np.linalg.norm(np.cross(vec,orientation2)))/2
            # align_score = 1/(np.abs(np.dot(vec,orientation1)) + np.abs(np.dot(vec,orientation2)))**2
            local_align_mat[i,j] = align_score
            
            # end-facing condition & alignment condition
            tol = 1e-10
            if (t>(1-tol) or (t<tol)) and ( (u>(1-tol)) or (u<tol) ):
                popt1 = segment_i[0]+t*d1
                popt1_opp = segment_i[0]+(1-t)*d1                
                popt2 = segment_j[0]+u*d2
                popt2_opp = segment_j[0]+(1-u)*d2                
                signed_dist = -dist*np.sign(np.dot(vec,popt2-popt1))
                
                local_dist_mat[i,j] = signed_dist
            else:
                local_dist_mat[i,j] = 1e10


    kkk = 30
    iii,jjj = np.where((local_dist_mat < 1e10))
    
    segment_i = segments[a_cluster[iii[kkk]]]
    segment_j = segments[a_cluster[jjj[kkk]]]
    
    p1 = segment_i[0]
    p2 = segment_i[-1]
    q1 = segment_j[0]
    q2 = segment_j[-1]
    
    t,u,d1,d2,d12 = lumelsky_dist_vec(p1,p2,q1,q2)
    print(np.linalg.norm(d1*t - d2*u - d12))
    print(t,u)
    
    popt1 = p1+t*d1
    popt2 = q1+u*d2
    
    vec = d1*t - d2*u - d12
    plt.close()
    fig,ax=set_3d_plot()
    plot_single_rod(segment_i,'-',ax=ax)
    plot_single_rod(segment_j,'-',ax=ax)
    ax.quiver(popt2[0],popt2[1],popt2[2],vec[0],vec[1],vec[2],color='red')
    
    if t > 0.5:
        edge_vec = p2 - p1
    else:
        edge_vec = p1 - p2
    edge_vec = edge_vec/np.linalg.norm(edge_vec)

    np.dot(vec/np.linalg.norm(vec),edge_vec)
    
    local_align_mat = np.minimum(local_align_mat,local_align_mat.T)
    local_align_mat[local_align_mat == np.inf] = 1e4
    local_align_mat[np.diag_indices(len(a_cluster))] = 0
    
    local_dist_mat = np.minimum(local_dist_mat,local_dist_mat.T)
    local_dist_mat[local_dist_mat == np.inf] = 1e4
    local_dist_mat[np.diag_indices(len(a_cluster))] = 1e4
    
    local_ferr_mat = np.minimum(local_ferr_mat,local_ferr_mat.T)
    local_ferr_mat[local_ferr_mat == np.inf] = 1e4
    local_ferr_mat[np.diag_indices(len(a_cluster))] = 0
    
    local_fcurv_mat = np.minimum(local_fcurv_mat,local_fcurv_mat.T)
    local_fcurv_mat[local_fcurv_mat == np.inf] = 1e4
    local_fcurv_mat[np.diag_indices(len(a_cluster))] = 1e4
    
    local_align_ijs = np.where(local_align_mat < 0.05)
    local_align_ijs = np.where((local_dist_mat < 95) & (local_align_mat < 0.01))
    
    where_high_curvature = np.where(local_fcurv_mat < 0.0001)
    where_high_curvature[0].shape
    fig,ax=set_3d_plot()
    
    # edg-ing by minimal distance
    min_dist_j = np.argmin(local_dist_mat,axis=0)
    min_dist_ij = np.vstack([np.arange(len(a_cluster)),min_dist_j]).T
    
    S_mind = nx.Graph()
    S_mind.add_nodes_from(range(len(a_cluster)))
    S_mind.add_edges_from(min_dist_ij)
    
    plt.subplots(1,1,figsize=(10,10))
    nx.draw(S_mind,with_labels=True)
    
    S_mind_cc = list(nx.connected_components(S_mind))
    S_min_deg = np.array(S_mind.degree())[:,1]
    
    S_mind.remove_nodes_from(np.where(S_min_deg > 2)[0])
    S_mind_cc = list(nx.connected_components(S_mind))
    S_min_deg = np.array(S_mind.degree())[:,1]
    
    fig,ax=set_3d_plot()
    for i,cluster in enumerate(S_mind_cc):
        joined = join_multiple_segments([segments[a_cluster[j]] for j in cluster])
        its_curvature = calculate_curvature(joined)
        total_curv = np.abs(np.sum(its_curvature))
        if total_curv > 0.1:
            print(i)
        plot_single_rod(joined,'.-',ax=ax,markersize=0.2)
        ax.text(joined[0,0],joined[0,1],joined[0,2],f'{i}',fontsize=8)
        plt.savefig(f'/Users/yeonsu/Figures/debug/mindist_cluster_{cluster}.png')
        ax.clear()
        
    fig,ax=set_3d_plot()
    cluster = S_mind_cc[8]
    for i in cluster:
        plot_single_rod(segments[a_cluster[i]],'-',ax=ax)
    joined = join_multiple_segments([segments[a_cluster[j]] for j in cluster])
    plot_single_rod(joined+[10,10,10],'-',ax=ax)

    for i in np.where(S_min_deg > 2)[0]:
        plot_single_rod(segments[a_cluster[i]],'-',ax=ax,color='red')
            
    
    for i,j in zip(where_high_curvature[0],where_high_curvature[1]):
        plot_single_rod(segments[a_cluster[i]],'-',ax=ax)
        plot_single_rod(segments[a_cluster[j]],'-',ax=ax)
        local_ferr_mat[i,j] 
        
        break
    
    end_segments = hull.vertices%num_a_cluster
    
    fig,ax=set_3d_plot()
    for i in range(len(a_cluster)):
        plot_single_rod(segments[a_cluster[i]],'-',ax=ax)
        ax.text(segments[a_cluster[i]][0,0],segments[a_cluster[i]][0,1],segments[a_cluster[i]][0,2],str(i),fontsize=8)
    for eseg in end_segments:
        plot_single_rod(segments[a_cluster[eseg]],'o',ax=ax,markersize=2,color='red')
        # ax.text(segments[a_cluster[eseg]][0,0],segments[a_cluster[eseg]][0,1],segments[a_cluster[eseg]][0,2],str(eseg),fontsize=8)
    
    local_align_ijs = np.where((local_ferr_mat < 2) & (local_align_mat < 0.01))
    local_align_ijs = np.where((local_fcurv_mat < 0.0001) & (local_align_mat < 0.05) )
    
    local_align_ijs = np.where((local_dist_mat < 15) )
    local_align_ijs = np.where()
    
    
    S = nx.Graph()
    S.add_nodes_from(range(len(a_cluster)))
    S.add_edges_from(zip(local_align_ijs[0],local_align_ijs[1]))
    S.add_weighted_edges_from(zip(local_align_ijs[0],local_align_ijs[1],local_dist_mat[local_align_ijs]))

    plt.subplots(1,1,figsize=(10,10))
    nx.draw(S,with_labels=True)

    end_segments = set(hull.vertices%num_a_cluster)
    pairs = itertools.combinations(end_segments, 2)
    num_pairs = len(list(itertools.combinations(end_segments, 2)))
    print(f'Number of pairs: {num_pairs}')
    
    path_fitting_errors = np.full(num_pairs,np.inf)
    path_length = np.full(num_pairs,np.inf)
    path_rad_curv = np.full(num_pairs,np.inf)
    path_curvature = np.full(num_pairs,np.inf)
    paths = []
    for i, pair in enumerate(pairs):
        print(f'Pair {i+1}/{num_pairs}: {pair}')
        try:
            path = nx.shortest_path(S, source=pair[0], target=pair[1], weight='weight')
            print(f'Path: {path}')
            paths.append(path)
            
            joined = join_multiple_segments([segments[a_cluster[j]] for j in path])
            joined = uniform_filter1d(joined, size=20, axis=0, mode='nearest')
            
            fit_result = fit_rod(joined,linearity_threshold=0.1,radius_curvature_threshold=500)
            path_fitting_errors[i] = fit_result['err']
            path_length[i] = np.sum([segment_length_list[ a_cluster[j] ] for j in path])
            path_rad_curv[i] = 1/fit_result['r']
            path_curvature[i] = np.abs(np.sum(calculate_curvature(joined)))
            
        except nx.NetworkXNoPath:
            print(f'No path between {pair[0]} and {pair[1]}')
            paths.append([])
        except nx.NodeNotFound as e:
            print(f'Node not found: {e}')
        except Exception as e:
            print(f'An error occurred: {e}')
    
    
    fig,ax=set_3d_plot()
    idx = paths[np.argmin(path_curvature[(path_length<700)&(path_length>600)])]
    for i in idx:
        plot_single_rod(segments[a_cluster[i]],'-',ax=ax)
        
    # joined = join_multiple_segments([segments[a_cluster[j]] for j in idx])
    # joined = uniform_filter1d(joined, size=100, axis=0, mode='nearest')
    plot_single_rod(joined,'-',ax=ax)
    centroid = np.mean(joined,axis=0)
    ax.text(centroid[0],centroid[1],centroid[2],f'curv{path_curvature[i]:.2f}_err{path_fitting_errors[i]:.2f}_len{path_length[i]}',fontsize=8)  

    fig,ax=set_3d_plot()
    for i in np.where((path_curvature < 0.01) & (path_length<700)&(path_length>600) & (path_fitting_errors<5))[0]:
        idx = paths[i]
        plot_single_rod(joined,'-',ax=ax)
        joined = join_multiple_segments([segments[a_cluster[j]] for j in idx])
        joined = uniform_filter1d(joined, size=100, axis=0, mode='nearest')
        plot_single_rod(joined,'-',ax=ax)
        centroid = np.mean(joined,axis=0)
        ax.text(centroid[0],centroid[1],centroid[2],f'curv{path_curvature[i]:.2f}_err{path_fitting_errors[i]:.2f}_len{path_length[i]}',fontsize=8)  
        
    ax.axis('equal')
    
    
    
    plt.close()    
    fig,ax=set_3d_plot()    
    for i,pth in enumerate(paths):
        idx = paths[i]
        if len(idx) == 0:
            continue
        joined = join_multiple_segments([segments[a_cluster[j]] for j in idx])
        
        _,sval,_ = np.linalg.svd(joined,full_matrices=False)
        # normalize the singular values
        sval /= sval[0]
        
        plot_single_rod(joined,'-',ax=ax)
        window_size = 20
        joined = uniform_filter1d(joined, size=window_size, axis=0, mode='nearest')
        plot_single_rod(joined,'-',ax=ax)
        
        fit_result = fit_rod(joined,linearity_threshold=0.1,radius_curvature_threshold=500)
        err = fit_result['err']
        its_curvature = calculate_curvature(joined)
        total_curv = np.abs(np.sum(its_curvature))
        
        ax.text(joined[0,0],joined[0,1],joined[0,2],f'curv{total_curv:.2f}_err{err:.2f}',fontsize=8)
        plt.savefig(f'/Users/yeonsu/Figures/debug/curvature_{i}.png')
        ax.clear()
    
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
        
    
    
    
        
            
    
    
    
    
    print()

