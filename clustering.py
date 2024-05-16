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
from sklearn.cluster import DBSCAN
from example_PhysicalRodRelaxation import prep_svd_cylinder,lumelsky_dist_vec
from numba import jit

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
        
    from sklearn.cluster import KMeans
    from sklearn_extra.cluster import KMedoids

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

if __name__ == '__main__':
    
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
    
    length_list = np.zeros(len(broken_pieces))
    for i in range(len(broken_pieces)):
        p1 = svd_cylinders[i,0:3]
        p2 = svd_cylinders[i,3:6]
        length_list[i] = np.linalg.norm(p2-p1)

    import time
    start = time.time()
    align_matrix = calculate_alignment_matrix_numba(svd_cylinders,orientations)
    print(f'Elapsed time: {time.time()-start}')
    
    # clustering
    
    align_matrix = np.minimum(align_matrix,align_matrix.T)
    align_matrix[np.diag_indices(len(broken_pieces))] = 1
    pickle_out = open('align_matrix_AR200.pkl','wb')
    pickle.dump(align_matrix,pickle_out)
    
    
    # pickle_in = open('align_matrix_AR200.pkl','rb')
    # align_matrix = pickle.load(pickle_in)
    
    edges = np.where(align_matrix < 0.1)
    edges[0].shape
    N = len(broken_pieces)
    G = nx.Graph()
    G.add_nodes_from(range(N))
    G.add_edges_from(zip(edges[0],edges[1]))
    conn_comp = list(nx.connected_components(G))
    print(len(conn_comp))
    
    list(nx.bridges(G))
    
    align_matrix2 = align_matrix
    
    plt.hist(length_list[length_list<100],bins=100)
    fig,ax=set_3d_plot()
    for i in length_list[length_list<25]:
        j = np.where(length_list == i)[0][0]
        plot_single_rod(broken_pieces[j],'-',ax=ax)
    
    fig,ax=set_3d_plot()
    for cc in conn_comp:
        for i in cc:
            plot_single_rod(broken_pieces[i],'-',ax=ax)
        plt.savefig(f'/Users/yeonsu/Figures/AR200_clustering/{i}.png')
        ax.clear()
    
    
    # fig,ax=set_3d_plot()
    # for i in range(len(broken_pieces)):
    #     plot_single_rod(broken_pieces[i],'-',ax=ax)

    
    print