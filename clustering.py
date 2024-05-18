from scipy.spatial import distance
from scipy.cluster import hierarchy
import numpy as np
from pathlib import Path
import networkx as nx
import pickle
from fitting import fit_rod_error, fit_rod
from data_io import load_xray_data
from visualizations import set_3d_plot, plot_single_rod, plot_centerline_with_container
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN
from example_PhysicalRodRelaxation import prep_svd_cylinder,lumelsky_dist_vec, lumelsky_dist
from numba import jit, prange
from scipy.io import loadmat

from potentials import dist_lin_seg_nonjax


from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids

def get_cluster_properties(connected_component_list,segments,segment_length_list,max_nodes_each_cluster=500):
    cluster_size_list = [len(x) for x in connected_component_list]
    max_cluster_size = np.max(cluster_size_list)
    max_cluster_idx = np.argmax(cluster_size_list)
    max_cluster = connected_component_list[max_cluster_idx]
    num_clusters = len(connected_component_list)
    
    error_list = np.full(num_clusters,np.nan)
    length_list = np.full(num_clusters,np.nan)
    # distance_list = np.full(num_clusters,np.nan)
    
    for iterator,cc in enumerate(connected_component_list):
        if len(cc) > max_nodes_each_cluster:
            continue
        segments_cc = np.vstack([segments[i] for i in cc])
        fit_result = fit_rod(segments_cc,linearity_threshold=0.1,radius_curvature_threshold=100)        
        error_list[iterator] = fit_result['err']
        length_list[iterator] = np.sum( segment_length_list[cc] )
    
    out_dict = {
        'num_clusters': num_clusters,
        'max_cluster_size': max_cluster_size,
        'max_cluster_idx': max_cluster_idx,
        'max_cluster': max_cluster,
        'error_list': error_list,
        'length_list': length_list
    }
    return out_dict

@jit(nopython=True,fastmath=True)
def calculate_alignment_adjacency_numba(svd_cylinders,orientations,threshold=0.1):
    N = svd_cylinders.shape[0]
    
    lst = []    
    for i in range(N):
        point1s = svd_cylinders[i][0:3]
        point1e = svd_cylinders[i][3:6]
        orientation1 = orientations[i]

        for j in range(i+1,N):
            point2s = svd_cylinders[j][0:3]
            point2e = svd_cylinders[j][3:6]
            orientation2 = orientations[j]
            t,u,d1,d2,d12=lumelsky_dist_vec(point1s, point1e, point2s, point2e)
            
            vec = d1 * t - d2 * u - d12
            vec = vec / np.linalg.norm(vec)
            score = (np.linalg.norm(np.cross(vec,orientation1)) + np.linalg.norm(np.cross(vec,orientation2)))/2
            if score < threshold:
                # add to adjacency matrix
                lst.append([i,j])
                if i % 100 == 0:
                    print(i,j)
            
    return lst

@jit(nopython=True,fastmath=True)
def calculate_alignment_dist_matrix_numba(svd_cylinders,orientations):
    N = svd_cylinders.shape[0]
    align_matrix = np.full((N,N),np.inf)
    dist_matrix = np.full((N,N),np.inf)
    
    for i in range(N):
        point1s = svd_cylinders[i][0:3]
        point1e = svd_cylinders[i][3:6]
        orientation1 = orientations[i]

        for j in range(i+1,N):
            point2s = svd_cylinders[j][0:3]
            point2e = svd_cylinders[j][3:6]
            orientation2 = orientations[j]
            t,u,d1,d2,d12=lumelsky_dist_vec(point1s, point1e, point2s, point2e)
            
            vec = d1 * t - d2 * u - d12
            dist_matrix[i,j] = np.linalg.norm(vec)
            vec = vec / np.linalg.norm(vec)
            align_matrix[i,j] = (np.linalg.norm(np.cross(vec,orientation1)) + np.linalg.norm(np.cross(vec,orientation2)))/2
            
    return align_matrix,dist_matrix
            

@jit(nopython=True,fastmath=True)
def calculate_alignment_matrix_numba(svd_cylinders,orientations):
    N = svd_cylinders.shape[0]
    align_matrix = np.full((N,N),np.inf)
    
    for i in range(N):
        point1s = svd_cylinders[i][0:3]
        point1e = svd_cylinders[i][3:6]
        orientation1 = orientations[i]

        for j in range(i+1,N):
            point2s = svd_cylinders[j][0:3]
            point2e = svd_cylinders[j][3:6]
            orientation2 = orientations[j]
            t,u,d1,d2,d12=lumelsky_dist_vec(point1s, point1e, point2s, point2e)
            
            vec = d1 * t - d2 * u - d12
            vec = vec / np.linalg.norm(vec)
            align_matrix[i,j] = (np.linalg.norm(np.cross(vec,orientation1)) + np.linalg.norm(np.cross(vec,orientation2)))/2
            # align_matrix[i,j] = (np.abs(np.dot(vec,orientation1)) + np.abs(np.dot(vec,orientation2)) )/2
            
    return align_matrix

def calculate_alignment_matrix(splitted,svd_cylinders,centroids,orientations):
    N = len(splitted)
    align_matrix = np.full((N,N),np.inf)
    for i in range(N):
        point1s = svd_cylinders[i][0:3]
        point1e = svd_cylinders[i][3:6]
        orientation1 = orientations[i]
        
        # fig,ax=set_3d_plot()
        # plot_centerline_with_container(splitted,svd_cylinders,i,ax)
        for j in range(i+1,N):
            point2s = svd_cylinders[j][0:3]
            point2e = svd_cylinders[j][3:6]
            orientation2 = orientations[j]
            t,u,d1,d2,d12=lumelsky_dist_vec(point1s, point1e, point2s, point2e)
            
            vec = d1 * t - d2 * u - d12
            vec = vec / np.linalg.norm(vec)
            # if (t > 1 or u > 1 or t < 0 or u < 0):
            # align_matrix[i,j] = (np.linalg.norm(np.cross(vec,orientation1)) + np.linalg.norm(np.cross(vec,orientation2)))/2
            align_matrix[i,j] = 2/(np.abs(np.dot(vec,orientation1)) + np.abs(np.dot(vec,orientation2)) )
            
            # fig,ax=set_3d_plot()
            # plot_centerline_with_container(splitted,svd_cylinders,i,ax)
            # plot_centerline_with_container(splitted,svd_cylinders,j,ax)
            # ax.text(point1s[0],point1s[1],point1s[2],'i')
            # ax.text(point2s[0],point2s[1],point2s[2],'j')
            # ax.axis('equal')
            # ax.clear()
            

            
    return align_matrix

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
        
def quick_3d_plot(rr_sorted):
    fig,ax = set_3d_plot()
    plot_single_rod(rr_sorted,'-',ax=ax)
    plt.show()
    
def translating_matlab_to_python():
    i = 55
    glb = group_labels[i]
    group_vertices = np.vstack([centerlines[lb] for lb in glb])
    group_unq,group_count = np.unique(group_vertices,axis=0,return_counts=True)
    
    clustering = DBSCAN(eps=8, min_samples=2).fit(group_unq)    
        
    """ dist_matrix = distance.pdist(group_unq, 'euclidean')
    square_dist_matrix = distance.squareform(dist_matrix)
    linked = hierarchy.linkage(dist_matrix, 'single')
    # Step 3: Plot the dendrogram to help decide the cut-off
    plt.figure(figsize=(10, 7))
    dendrogram = hierarchy.dendrogram(linked)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Sample index')
    plt.ylabel('Distance')
    plt.show()
    clusters = hierarchy.fcluster(linked, 8, criterion='distance')
    print(clusters)
    fig,ax = set_3d_plot()
    for i in range(1,clusters.max()+1):
        idx = np.where(clusters == i)[0]
        plot_single_rod(group_unq[idx],'.',ax=ax,markersize=0.5,color=np.random.rand(3)) """

    splitted = []
    length_list = []
    for i in range(clustering.labels_.max()+1):
        idx = np.where(clustering.labels_ == i)[0]
        rr = group_unq[idx]
        if rr.shape[0] < 6:
            continue
        
        rr_centered = rr - np.mean(rr, axis=0)
        U, S, V = np.linalg.svd(rr_centered, full_matrices=False)
        v1, v2, v3 = V[0,:] , V[1,:], V[2,:]
        orientation = v1 * np.sign(np.sum(v1 * (rr_centered[-1, :] - rr_centered[0, :])))
        slist = np.dot(rr_centered, orientation)
        sorted_indices = np.argsort(slist)
        rr_sorted = rr_centered[sorted_indices] + np.mean(rr, axis=0)
        fit_result = fit_rod(rr_sorted)
        
        length_list.append(fit_result['len'])
        splitted.append(fit_result['rec'])
        
    N = len(splitted)
    fitting_error_matrix = np.full((N,N),np.inf)
    
    for i in range(N):
        r1 = splitted[i]
        for j in range(i+1,N):
            r2 = splitted[j]
            joined = np.vstack([r1,r2])
            joined = joined - np.mean(joined,axis=0)
            res = fit_rod(joined)
            fitting_error_matrix[i,j] = res['err']
            
    # symmetrize using element wise min
    fitting_error_matrix = np.minimum(fitting_error_matrix,fitting_error_matrix.T)
    fitting_error_matrix[np.diag_indices(N)] = np.max(fitting_error_matrix[np.triu_indices_from(fitting_error_matrix,1)])*10
    
    svd_cylinders,centroids,orientations = prep_svd_cylinder(splitted,scale_factor=0.9)
    from visualizations import plot_centerline_with_container
    fig,ax = set_3d_plot()
    for i in range(N):
        plot_centerline_with_container(splitted,svd_cylinders,i,ax)
        
    align_matrix = calculate_alignment_matrix(splitted,svd_cylinders,centroids,orientations)
    align_matrix = np.minimum(align_matrix,align_matrix.T)
    align_matrix[np.diag_indices(N)] = 1
    
    from scipy.cluster.hierarchy import dendrogram, linkage, fclusterdata
    Z = fclusterdata(1/align_matrix, 2, criterion='distance')
    print(Z)
    
    
    # Initializing DBSCAN with metric='precomputed' to use the distance matrix
    dbscan = DBSCAN(eps=0.1, min_samples=2, metric='precomputed')
    # clusters2 = dbscan.fit_predict(fitting_error_matrix)
    clusters2 = dbscan.fit_predict( align_matrix )
    print("Cluster labels:", clusters2)
    
    fig,ax = set_3d_plot()
    sum_errors = 0
    for i in range(0,clusters2.max()+1):
        idx = np.where(clusters2 == i)[0]
        joined = np.vstack([splitted[j] for j in idx])
        sum_errors = np.sum([err for err in fitting_error_matrix[i,idx]])
        print(sum_errors)
        plot_single_rod(joined,'o',ax=ax,markersize=2)        
    plot_single_rod(np.vstack(splitted),'.',ax=ax,markersize=0.5)
    
    adj_mat = align_matrix < 0.03
    edges = np.where(adj_mat)
    
    G = nx.Graph()
    G.add_nodes_from(range(N))
    G.add_edges_from(zip(edges[0],edges[1]))
    conn_comp = list(nx.connected_components(G))

    fig,ax = set_3d_plot()    
    for components in conn_comp:
        joined = np.vstack([splitted[j] for j in components])
        plot_single_rod(joined,'o',ax=ax,markersize=2)

    N_trials = np.min([N-1,20])
    fitting_score_over_trials = np.full(N_trials+1,np.inf)    
    for i in range(1,N_trials+1):
        # km = KMeans(n_clusters=i, random_state=0, n_init="auto").fit(align_matrix)        
        km = KMedoids(n_clusters=i, random_state=0).fit(align_matrix)
        fit_score = 0
        
        cluster_fit_errors = []
        cluster_lengths = []
        for ii in range(km.labels_.max() + 1):
            idx = np.where(km.labels_ == ii)[0]
            joined = np.vstack([splitted[j] for j in idx])
            res = fit_rod(joined)
            cluster_fit_errors.append(res['err']**5)
            cluster_lengths.append(res['len']**2)
        
        fitting_score_over_trials[i] = np.sum(cluster_fit_errors)/np.mean(cluster_lengths)
    
    j_min = np.argmin(fitting_score_over_trials)
    # km = KMeans(n_clusters=i, random_state=0, n_init="auto").fit(align_matrix)
    km = KMedoids(n_clusters=j_min, random_state=0).fit(align_matrix)
    
    
    # fig,ax=set_3d_plot()
    # for ii in range(km.labels_.max()+1):
    #     idx = np.where(km.labels_ == ii)[0]
    #     joined=np.vstack([splitted[j] for j in idx])
    #     plot_single_rod(joined,'o',ax=ax,markersize=2)
    
def got_alignment_matrix():
    pth = Path('/Users/yeonsu/Documents/GitHub/entanglement-optimization/xray_raw_data/alpha200_epsilon00/centerlines.mat')
    centerlines,_ = load_xray_data(pth)
    for i,cl in enumerate(centerlines):
        centerlines[i] = np.unique(cl,axis=0)
    
    pickle_in = open('duplicate_groups.pkl','rb')
    duplicate_groups = pickle.load(pickle_in)
    pickle_in = open('group_labels.pkl','rb')
    group_labels = pickle.load(pickle_in)
    
    unpacked = np.vstack(centerlines)
    labels = np.zeros(unpacked.shape[0],dtype=np.int64)
    start_idx = 0
    for i,cl in enumerate(centerlines):
        end_idx = start_idx + cl.shape[0]
        labels[start_idx:end_idx] = i
        start_idx = end_idx
        
    unq,ind,inv,cnt = np.unique(unpacked,axis=0,return_counts=True,return_inverse=True,return_index=True)
    nonoverlap_labels = np.unique(labels[(cnt == 1)[inv]])
    
    # fig,ax = set_3d_plot()
    # for lb in nonoverlap_labels:
    #     rr = centerlines[lb]        
    #     plot_single_rod(rr,'-',ax=ax)

    broken_pieces = []
    for i in range(len(group_labels)):
        glb = group_labels[i]
        group_vertices = np.vstack([centerlines[lb] for lb in glb])
        group_unq,group_count = np.unique(group_vertices,axis=0,return_counts=True)
        
        clustering = DBSCAN(eps=8, min_samples=2).fit(group_unq)    
        splitted = []
        for j in range(clustering.labels_.max()+1):
            idx = np.where(clustering.labels_ == j)[0]
            rr = group_unq[idx]
            if rr.shape[0] < 10:
                continue            
            
            rr_centered = rr - np.mean(rr, axis=0)
            U, S, V = np.linalg.svd(rr_centered, full_matrices=False)
            v1, v2, v3 = V[0,:] , V[1,:], V[2,:]
            orientation = v1 * np.sign(np.sum(v1 * (rr_centered[-1, :] - rr_centered[0, :])))
            slist = np.dot(rr_centered, orientation)
            sorted_indices = np.argsort(slist)
            rr_sorted = rr_centered[sorted_indices] + np.mean(rr, axis=0)
            splitted.append(rr_sorted)
            
        broken_pieces.extend(splitted)
    len(broken_pieces)
    
    svd_cylinders,centroids,orientations = prep_svd_cylinder(broken_pieces,scale_factor=1.5)

    import time
    start = time.time()
    align_matrix = calculate_alignment_matrix_numba(svd_cylinders,orientations)
    print(f'Elapsed time: {time.time()-start}')    
    
    # clustering
    
    align_matrix = np.minimum(align_matrix,align_matrix.T)
    align_matrix[np.diag_indices(len(broken_pieces))] = 0
    pickle_out = open('align_matrix_AR200.pkl','wb')
    pickle.dump(align_matrix,pickle_out)
    
    edges = np.where(align_matrix > 0.98)
    edges[0].shape
    N = len(broken_pieces)
    G = nx.Graph()
    G.add_nodes_from(range(N))
    G.add_edges_from(zip(edges[0],edges[1]))
    conn_comp = list(nx.connected_components(G))
    print(len(conn_comp))   
    
def developing_clustering():
    pth = Path('/Users/yeonsu/Documents/GitHub/entanglement-optimization/xray_raw_data/alpha200_epsilon00/centerlines.mat')
    centerlines,_ = load_xray_data(pth)
    for i,cl in enumerate(centerlines):
        centerlines[i] = np.unique(cl,axis=0)
    
    pickle_in = open('duplicate_groups.pkl','rb')
    duplicate_groups = pickle.load(pickle_in)
    pickle_in = open('group_labels.pkl','rb')
    group_labels = pickle.load(pickle_in)
    
    unpacked = np.vstack(centerlines)
    labels = np.zeros(unpacked.shape[0],dtype=np.int64)
    start_idx = 0
    for i,cl in enumerate(centerlines):
        end_idx = start_idx + cl.shape[0]
        labels[start_idx:end_idx] = i
        start_idx = end_idx
        
    unq,ind,inv,cnt = np.unique(unpacked,axis=0,return_counts=True,return_inverse=True,return_index=True)
    nonoverlap_labels = np.unique(labels[(cnt == 1)[inv]])
    
    # fig,ax = set_3d_plot()
    # for lb in nonoverlap_labels:
    #     rr = centerlines[lb]        
    #     plot_single_rod(rr,'-',ax=ax)

    broken_pieces = []
    for i in range(len(group_labels)):
        glb = group_labels[i]
        group_vertices = np.vstack([centerlines[lb] for lb in glb])
        group_unq,group_count = np.unique(group_vertices,axis=0,return_counts=True)
        
        clustering = DBSCAN(eps=8, min_samples=2).fit(group_unq)    
        splitted = []
        for j in range(clustering.labels_.max()+1):
            idx = np.where(clustering.labels_ == j)[0]
            rr = group_unq[idx]
            if rr.shape[0] < 10:
                continue            
            
            rr_centered = rr - np.mean(rr, axis=0)
            U, S, V = np.linalg.svd(rr_centered, full_matrices=False)
            v1, v2, v3 = V[0,:] , V[1,:], V[2,:]
            orientation = v1 * np.sign(np.sum(v1 * (rr_centered[-1, :] - rr_centered[0, :])))
            slist = np.dot(rr_centered, orientation)
            sorted_indices = np.argsort(slist)
            rr_sorted = rr_centered[sorted_indices] + np.mean(rr, axis=0)
            splitted.append(rr_sorted)
            
        broken_pieces.extend(splitted)
    len(broken_pieces)
    
    svd_cylinders,centroids,orientations = prep_svd_cylinder(broken_pieces,scale_factor=0.95)
    
    length_list = np.zeros(len(broken_pieces))
    for i in range(len(broken_pieces)):
        p1 = svd_cylinders[i,0:3]
        p2 = svd_cylinders[i,3:6]
        length_list[i] = np.linalg.norm(p2-p1)
    
    pickle_in = open('align_matrix_AR200.pkl','rb')
    align_matrix = pickle.load(pickle_in)
    
    edges = np.where(align_matrix < 0.1)
    edges[0].shape
    N = len(broken_pieces)
    G = nx.Graph()
    G.add_nodes_from(range(N))
    G.add_edges_from(zip(edges[0],edges[1]))
    conn_comp = list(nx.connected_components(G))
    print(len(conn_comp))
    
    bridges = list(nx.bridges(G))    
    G.remove_edges_from(bridges)
    conn_comp = list(nx.connected_components(G))
    print(len(conn_comp))
    
    error_list = []
    for cc in conn_comp:
        joined = np.vstack([broken_pieces[i] for i in cc])
        # for i in cc:
        #     plot_single_rod(broken_pieces[i],'-',ax=ax)
        fit_res = fit_rod(joined)
        error_list.append(fit_res['err'])
        
    error_list = np.array(error_list)
    plt.plot(error_list)
    np.count_nonzero(error_list > 1)
    
    bac_clusters = [x for i,x in enumerate(conn_comp) if error_list[i] > 1]
    
    splitted_again = []
    for cc in bac_clusters:
        joined = np.vstack([broken_pieces[i] for i in cc])
        clustering = DBSCAN(eps=2, min_samples=2).fit(joined)
        
        
        for j in range(clustering.labels_.max()+1):
            idx = np.where(clustering.labels_ == j)[0]
            rr = joined[idx]
            if rr.shape[0] < 10:
                continue            
            
            rr_centered = rr - np.mean(rr, axis=0)
            U, S, V = np.linalg.svd(rr_centered, full_matrices=False)
            v1, v2, v3 = V[0,:] , V[1,:], V[2,:]
            orientation = v1 * np.sign(np.sum(v1 * (rr_centered[-1, :] - rr_centered[0, :])))
            slist = np.dot(rr_centered, orientation)
            sorted_indices = np.argsort(slist)
            rr_sorted = rr_centered[sorted_indices] + np.mean(rr, axis=0)
            splitted_again.append(rr_sorted)
        
    len(splitted_again)
    
    svd_cylinders,centroids,orientations = prep_svd_cylinder(splitted_again,scale_factor=0.95)
        
    import time
    start = time.time()
    align_matrix_again = calculate_alignment_matrix_numba(svd_cylinders,orientations)
    align_matrix_again,distance_matrix_again = calculate_alignment_dist_matrix_numba(svd_cylinders,orientations)
    print(f'Elapsed time: {time.time()-start}')
    
    # clustering
    align_matrix_again = np.minimum(align_matrix_again,align_matrix_again.T)
    align_matrix_again[np.diag_indices(len(splitted_again))] = 1
    
    edges = np.where(align_matrix_again < 0.1)
    edges[0].shape
    N = len(splitted_again)
    G = nx.Graph()
    G.add_nodes_from(range(N))
    G.add_edges_from(zip(edges[0],edges[1]))
    conn_comp = list(nx.connected_components(G))
    print(len(conn_comp))
    
    bridges = list(nx.bridges(G))    
    G.remove_edges_from(bridges)
    conn_comp = list(nx.connected_components(G))
    print(len(conn_comp))
    
    error_list = []
    length_list = []
    number_list = []
    for cc in conn_comp:
        joined = np.vstack([splitted_again[i] for i in cc])
        # for i in cc:
        #     plot_single_rod(broken_pieces[i],'-',ax=ax)
        fit_res = fit_rod(joined)
        error_list.append(fit_res['err'])
        length_list.append(fit_res['len'])
        number_list.append(len(cc))
        
    
        
    error_list = np.array(error_list)
    length_list = np.array(length_list)
    
    plt.plot(error_list)
    np.count_nonzero(error_list > 1)
    np.where(error_list > 1)
    
    ii = 35
    cc = np.array([*conn_comp[ii]])
    print(length_list[ii])
    print(error_list[ii])    
    
    G_conncomp = nx.Graph()
    G_conncomp.add_nodes_from(range(len(cc)))
    edges = np.where(align_matrix_again[np.ix_(cc,cc)] < 0.1)
    G_conncomp.add_edges_from(zip(edges[0],edges[1]))
    
    list(nx.articulation_points(G_conncomp))
    
    nx.draw(G_conncomp,with_labels=True)
    
    for deg in G_conncomp.degree:
        print(deg)
    
    fig,ax=set_3d_plot()
    
    for i in cc:
        plot_single_rod(splitted_again[i],'-',ax=ax)
    ax.axis('equal')
    print()
    
    
    
    def clustering_staircase(centerlines):
        N = len(centerlines)
        fitting_error_matrix = np.full((N,N),np.inf)
        
        for i in range(N):
            r1 = centerlines[i]
            for j in range(i+1,N):
                r2 = centerlines[j]
                joined = np.vstack([r1,r2])
                joined = joined - np.mean(joined,axis=0)
                res = fit_rod(joined)
                fitting_error_matrix[i,j] = res['err']
                
        # symmetrize using element wise min
        fitting_error_matrix = np.minimum(fitting_error_matrix,fitting_error_matrix.T)
        fitting_error_matrix[np.diag_indices(N)] = np.max(fitting_error_matrix[np.triu_indices_from(fitting_error_matrix,1)])*10

def local_group(G_i):
    # G_i is a subgraph of connected components
    
    N_trials = np.min([N-1,20])
    fitting_score_over_trials = np.full(N_trials+1,np.inf)
    
    for i in range(1,N_trials+1):
        # km = KMeans(n_clusters=i, random_state=0, n_init="auto").fit(align_matrix)        
        km = KMedoids(n_clusters=i, random_state=0).fit(align_matrix)
        fit_score = 0
        
        cluster_fit_errors = []
        cluster_lengths = []
        for ii in range(km.labels_.max() + 1):
            idx = np.where(km.labels_ == ii)[0]
            joined = np.vstack([splitted[j] for j in idx])
            res = fit_rod(joined)
            cluster_fit_errors.append(res['err']**5)
            cluster_lengths.append(res['len']**2)
        
        fitting_score_over_trials[i] = np.sum(cluster_fit_errors)/np.mean(cluster_lengths)
    
    j_min = np.argmin(fitting_score_over_trials)
    # km = KMeans(n_clusters=i, random_state=0, n_init="auto").fit(align_matrix)
    km = KMedoids(n_clusters=j_min, random_state=0).fit(align_matrix)
    
def implementation_without_precomputed_adjij():
    rod_data_root_dir = Path('/Users/yeonsu/Data/steel-rods-xray-data')
    segments_file_path = rod_data_root_dir / 'alpha200_epsilon00' / 'segments.mat'
    adjacency_file_path = rod_data_root_dir / 'alpha200_epsilon00' / 'adjacency_scale0p95_threshold0p1_ij_score.pkl'
    
    # "Literal" values - those are parameters
    rod_length = 650
    rod_length_margin = 30
    max_nodes_each_cluster = 500
    stairs = np.linspace(0.03,0.1,5)
    # "Derived" values - those are calculated from the parameters
    
    mat_obj = loadmat(segments_file_path)
    segments = mat_obj['segments']
    segments = [seg[0] for seg in segments]
    with open(adjacency_file_path, 'rb') as f:
        adjij = pickle.load(f)
    print(len(adjij))
    
    max(adjij,key=lambda x: x[2])    
    
    for stair in stairs:
        filtered = [x for x in adjij if x[2] < stair]
        print(len(filtered))
    
    N_segments = len(segments)
    G = nx.Graph()
    G.add_nodes_from(range(N_segments))
    filtered = np.array(adjij)
    

    reconnection_nodes = []
    reconnected_rod_list = []
    error_list = []
    length_list = []
    
    stairs = np.linspace(0.001,0.1,10)[::-1]
    
    for stair in stairs:
        
        print(f"Stair: {stair}")
        
        # filtered = filtered[filtered[:,2] < stair,:]
        filtered = [x for x in adjij if x[2] < stair]
        filtered = np.array(filtered)
        print(f"Previous filtered edge info: ", filtered.shape[0])
                
        ijs = filtered[:,0:2]
        ijs = np.array(ijs,dtype=int)
        
        G.remove_edges_from(list(G.edges))
        G.add_edges_from(ijs)
        G.remove_edges_from(nx.bridges(G))
        conn_comp = list(nx.connected_components(G))
        print(len(conn_comp))
        

        certified_nodes = []
        import time
        start = time.time()
        for i,cc in enumerate(conn_comp):
            idx = np.array([*cc])
            if len(idx) > max_nodes_each_cluster:
                continue
            joined = np.vstack([segments[i] for i in idx])
            fit_result = fit_rod(joined,linearity_threshold=0.1,radius_curvature_threshold=100)
            
            if (fit_result['err'] < 1) & (fit_result['len'] > rod_length-rod_length_margin):
                reconnected_rod_list.append(joined)
                error_list.append(fit_result['err'])
                length_list.append(fit_result['len'])
                certified_nodes.append(idx)
                
        print(f'Elapsed time: {time.time()-start}')
        certified_nodes = np.hstack(certified_nodes)
        reconnection_nodes.append(certified_nodes)
        
        all_is = filtered[:,0]
        all_is = np.array(all_is,dtype=int)
        all_js = filtered[:,1]
        all_js = np.array(all_js,dtype=int)
        
        np.isin(all_is,certified_nodes).any()
        
        num_certified_nodes = np.count_nonzero((np.isin(all_is,certified_nodes)) | (np.isin(all_js,certified_nodes)))
        print(f"Number of reconnected nodes: {num_certified_nodes}")
        # if num_certified_nodes == 0:
        #     break
        
        filtered = filtered[((~np.isin(all_is,certified_nodes)) & (~np.isin(all_js,certified_nodes)))]
        print(f"Current filtered edge info: ", filtered.shape[0])
    
    num_nodes_removed = np.sum([len(x) for x in reconnection_nodes])
    print(f'Filtration result.')
    print(f'Number of reconnected nodes: {len(reconnection_nodes)}')
    print(f'Number of removed nodes: {num_nodes_removed}')

    [len(x) for x in reconnection_nodes]
    
    
    pickle_out = open('reconnection_nodes.pkl','wb')
    pickle.dump(reconnection_nodes,pickle_out)
    
    print
    
    
def staircase(distance_threshold,first_stair,last_stair,num_stairs,max_nodes_each_cluster=500):
    stairs = np.linspace(first_stair,last_stair,num_stairs)        
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
    # t_list = adjij[:,4]
    # u_list = adjij[:,5]
    
    G = nx.Graph()
    G.add_nodes_from(range(N_segments)) # play with edges only
    list_of_cluster_stats = []
    for stair in stairs:
        print(f'Stepping up to {stair}')
        G.add_edges_from(ijs[(align_score < stair) & (dist_score < distance_threshold)])
        connected_component_list = list(nx.connected_components(G))
        connected_component_list = [list(x) for x in connected_component_list]
        
        # sort inner list
        for i in range(len(connected_component_list)):
            connected_component_list[i] = sorted(connected_component_list[i])
            
        props = get_cluster_properties(connected_component_list,segments,segment_length_list,max_nodes_each_cluster=500)
        
        local_dict = {
            'stiar': stair,
            'distance_threshold': distance_threshold,
            'max_cluster_size': props['max_cluster_size'],
            'connected_component_list': connected_component_list,
            'max_cluster_idx': props['max_cluster_idx'],
            'num_clusters': props['num_clusters'],
            'error_list': props['error_list'],
            'length_list': props['length_list'],
            'segment_file': segments_file_path.parent,
            'adjacency_file': adjacency_file_path.name,
        }
        props.update({'connected_component_list': connected_component_list})
        max_cluster_size = props['max_cluster_size']
        max_cluster_idx = props['max_cluster_idx']
        num_clusters = props['num_clusters']
        
        print(f'Maximum size {max_cluster_size} at index {max_cluster_idx} / out of {num_clusters} clusters\n')
        
        # error_list = cluster_stats['error_list']
        # length_list = cluster_stats['length_list']
        list_of_cluster_stats.append(local_dict)
        
        G.remove_edges_from(G.edges)
        
    return list_of_cluster_stats

def adaptive_staircase(segments,svd_cylinders,ijs,align_score,dist_score,segment_length_list,
                       distance_threshold,first_stair,last_stair,num_stairs,max_nodes_each_cluster=500):
    
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
    for stair in stairs:
        print(f'Stepping up to {stair}')
        for distance_threshold in distance_thresholds:
            print(f'Distance threshold: {distance_threshold}')
            
            from partition import partition
            
            G.add_edges_from(ijs[(align_score < stair) & (dist_score < distance_threshold)])
            G.remove_nodes_from(good_cluster_nodes)
            G.number_of_nodes()
            connected_component_list = list(nx.connected_components(G))
            connected_component_list = [list(x) for x in connected_component_list]
            for i in range(len(connected_component_list)):
                connected_component_list[i] = sorted(connected_component_list[i])
                
            cluster_size_list = [len(x) for x in connected_component_list]
            max_cluster_size = np.max(cluster_size_list)
            print(f'Max cluster size: {max_cluster_size}')
        
            if max_cluster_size > max_cluster_size_criterion:
                continue
            
            # use frozenset to update cc list; if cc hasn't been updated, don't compute again.
            props = get_cluster_properties(connected_component_list,segments,segment_length_list,max_nodes_each_cluster)
        
            local_dict = {
                'stair': stair,
                'distance_threshold': distance_threshold,
                'max_cluster_size': props['max_cluster_size'],
                'connected_component_list': connected_component_list,
                'max_cluster_idx': props['max_cluster_idx'],
                'num_clusters': props['num_clusters'],
                'error_list': props['error_list'],
                'length_list': props['length_list']
            }
            props.update({'connected_component_list': connected_component_list})
            
            max_cluster_size = props['max_cluster_size']
            max_cluster_idx = props['max_cluster_idx']
            num_clusters = props['num_clusters']        
            error_list = props['error_list']
            length_list = props['length_list']
            
            a_criterion = (error_list < error_criterion) & (length_list > length_criterion) 
            if np.count_nonzero(a_criterion) == 0:
                continue
        
            new_good_clusters = []
            for i in np.where(a_criterion)[0]:
                good_clusters.append(connected_component_list[i])
                new_good_clusters.append(connected_component_list[i])
                
            # plt.close()
            # fig,ax=set_3d_plot()
            # for i in np.where(a_criterion)[0]:
            #     joined = np.vstack([segments[j] for j in connected_component_list[i]])
            #     plot_single_rod(joined,'-',ax=ax)
            # ax.axis('equal')
            
            new_good_clusters = np.hstack(new_good_clusters)
            good_cluster_nodes = np.hstack(good_clusters)
            print(f'Number of removed nodes: {len(new_good_clusters)}')
        
            list_of_cluster_stats.append(local_dict)
        
    pickle_out = open(f'cache.pkl','wb')
    pickle.dump(good_clusters,pickle_out)
        
    all_nodes = []
    plt.close()
    ll = []
    fig,ax=set_3d_plot()
    for i in good_clusters:
        indices = np.array(i)
        joined = np.vstack([segments[j] for j in indices])
        plot_single_rod(joined,'-',ax=ax)
        all_nodes.append(indices)
        # length of joined
        ll.append( np.sum([segment_length_list[x] for x in indices]) )
        
    ax.axis('equal')
    print(np.hstack(all_nodes).size)
    len(good_clusters) # easy get
    ll = np.array(ll)
    plt.close()
    plt.hist(ll,bins=100)
    np.where(ll>1000.)[0]
    
    ii = 375
    cc = good_clusters[ii]
    fig,ax=set_3d_plot()
    for i in cc:
        plot_single_rod(segments[i],'-',ax=ax)
    ax.axis('equal')
    ll[ii]

    stair = 0.02
    distance_threshold = 100
    G = nx.Graph()
    G.add_nodes_from(range(len(segments))) # play with edges only
    
    max_cluster_size = 1e10
    
    G.add_edges_from(ijs[(align_score < stair) & (dist_score < distance_threshold)])
    G.remove_nodes_from(good_cluster_nodes)
    connected_component_list = list(nx.connected_components(G))
    connected_component_list = [list(x) for x in connected_component_list]
    for i in range(len(connected_component_list)):
        connected_component_list[i] = sorted(connected_component_list[i])
        
    cluster_size_list = [len(x) for x in connected_component_list]
    max_cluster_size = np.max(cluster_size_list)
    print(f'Max cluster size: {max_cluster_size}')
    
    cc0 = connected_component_list[np.argmax(cluster_size_list)]
    fig,ax=set_3d_plot()
    for i in np.random.choice(len(cc0),1000):
        plot_single_rod(segments[cc0[i]],'-',ax=ax)
    ax.axis('equal')        
    
        
    cc = connected_component_list[10]
        
    return list_of_cluster_stats
    
def do_staircase():
    distance_threshold = 30
    num_stairs = 10
    first_stair = 0.001
    last_stair = 0.1
    max_nodes_each_cluster = 500
    
    staircase_stats = staircase(distance_threshold,first_stair,last_stair,num_stairs,max_nodes_each_cluster)
    
    pickle_out = open(f'staircase_stats_{first_stair}_{last_stair}_{num_stairs}_{distance_threshold}.pkl','wb')
    pickle.dump(staircase_stats,pickle_out)
    
def do_adaptive_staircase():
    distance_threshold = 200

    first_stair = 0.05
    last_stair = 0.06
    num_stairs = 1
    
    max_nodes_each_cluster = 500
    
    # load data
    if 1:
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
        # t_list = adjij[:,4]
        # u_list = adjij[:,5]
        
        """
        adjacency_file_path.parent / 'adj_cutoff.pkl'
        pickle_out = open(adjacency_file_path.parent / 'adj_cutoff.pkl','wb')
        adjij_cutoff = adjij[:100000]
        pickle.dump(adjij_cutoff,pickle_out)
        
        import time
        start = time.time()
        pickle_in = open(adjacency_file_path.parent / 'adj_cutoff.pkl','rb')
        adjij_cutoff = pickle.load(pickle_in)
        print(f'Elapsed time: {time.time()-start}')
        adjij = adjij_cutoff
        """
    
    
    staircase_stats = adaptive_staircase(segments,svd_cylinders,ijs,align_score,dist_score,segment_length_list,
        distance_threshold,first_stair,last_stair,num_stairs,max_nodes_each_cluster)
    
    to_add = {'segment_file': segments_file_path.parent,'adjacency_file': adjacency_file_path.name}
    for i in range(num_stairs):
        staircase_stats[i].update(to_add)
    
    pickle_out = open(f'staircase_stats_{first_stair}_{last_stair}_{num_stairs}_{distance_threshold}.pkl','wb')
    pickle.dump(staircase_stats,pickle_out)
    
def do_what():
    
    rod_data_root_dir = Path('/Users/yeonsu/Data/steel-rods-xray-data')
    
    rod_length = 650
    rod_length_told = 30
    
    pickle_in = open('staircase_stats_0.001_0.1_10_30.pkl','rb')
    staircase_stats = pickle.load(pickle_in)
    
    segments_file_path = staircase_stats[0]['segment_file'] / 'segments.mat'
    adjacency_file_path = staircase_stats[0]['segment_file'] / staircase_stats[0]['adjacency_file']

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
    # t_list = adjij[:,4]
    # u_list = adjij[:,5]
    
    G = nx.Graph()
    G.add_nodes_from(range(N_segments)) # play with edges only

    def update_connected_components(prev,curr):
        prev_set = set([frozenset(x) for x in prev])
        curr_set = set([frozenset(x) for x in curr])
        new_set = curr_set - prev_set
        new_list = [list(x) for x in new_set]
        return new_list
    
    i_stair = 8
    a_criterion = (staircase_stats[i_stair]['error_list'] < 1) & (staircase_stats[i_stair]['length_list'] > rod_length-rod_length_told) & (staircase_stats[i_stair]['length_list'] < rod_length+rod_length_told)
    good_connected_components_prev = [staircase_stats[i_stair]['connected_component_list'][i] for i in np.where(a_criterion)[0]]
    a_criterion = (staircase_stats[i_stair+1]['error_list'] < 1) & (staircase_stats[i_stair+1]['length_list'] > rod_length-rod_length_told) & (staircase_stats[i_stair+1]['length_list'] < rod_length+rod_length_told)
    good_connected_components_curr = [staircase_stats[i_stair+1]['connected_component_list'][i] for i in np.where(a_criterion)[0]]
    
    cc_diff = update_connected_components(good_connected_components_prev,good_connected_components_curr)
    fig,ax=set_3d_plot()            
    for cc in cc_diff:
        for j in cc:
            plot_single_rod(segments[j],'-',ax=ax)
            ax.axis('equal')
        ax.clear()
        
    
    
    # print(cluster_stats['num_clusters'])
    
    # a_criterion = (error_list < 1) & (length_list > rod_length-rod_length_told) & (length_list < rod_length+rod_length_told)
    # print(np.count_nonzero(a_criterion))
    
    print()
    
    # staircase_stats = staircase(distance_threshold,first_stair,last_stair,num_stairs,max_nodes_each_cluster)
    # pickle_out = open(f'staircase_stats_{first_stair}_{last_stair}_{num_stairs}_{distance_threshold}.pkl','wb')
    # pickle.dump(staircase_stats,pickle_out)
    
    # i = 1979
    # plt.close()
    # fig,ax=set_3d_plot()
    # plot_single_rod(np.vstack([segments[j] for j in connected_component_list[i]]),'.',ax=ax,markersize=0.2)
    # plot_single_rod(np.vstack([segments[cc[j]] for j in deg0]),'o',ax=ax,markersize=1)
    # ax.axis('equal')
    
    
if __name__ == '__main__':
    do_adaptive_staircase()
    # distance_threshold = 200

    # first_stair = 0.05
    # last_stair = 0.06
    # num_stairs = 1
    
    # max_nodes_each_cluster = 500
    
    # # load data
    # if 1:
    #     rod_data_root_dir = Path('/Users/yeonsu/Data/steel-rods-xray-data')
    #     segments_file_path = rod_data_root_dir / 'alpha200_epsilon00' / 'segments.mat'
    #     adjacency_file_path = rod_data_root_dir / 'alpha200_epsilon00' / 'adjacency_distance_scale0p98_threshold0p3_ij_score.pkl'
        
    #     mat_obj = loadmat(segments_file_path)
    #     segments = mat_obj['segments']
    #     segments = [seg[0] for seg in segments]
    #     N_segments = len(segments)
        
    #     print(f'Staircase clustering: {segments_file_path.parent}')
    #     print(f'Number of segments: {N_segments}')

    #     pickle_in = open(adjacency_file_path,'rb')
    #     adjij = pickle.load(pickle_in)
    #     svd_cylinders,centroids,orientations, = prep_svd_cylinder(segments,scale_factor=0.98)
    #     segment_length_list = np.zeros(len(segments))
    #     for i,seg in enumerate(segments):
    #         segment_length_list[i] = np.sum(np.sqrt(np.sum(np.diff(seg,axis=0)**2,axis=1)))
    
    #     #adjij: i,j,score,dist,t,u
    #     adjij = np.array(adjij)
    #     ijs = adjij[:,0:2].astype(int)
    #     align_score = adjij[:,2]
    #     dist_score = adjij[:,3]
    #     # t_list = adjij[:,4]
    #     # u_list = adjij[:,5]
    
    
    
    print