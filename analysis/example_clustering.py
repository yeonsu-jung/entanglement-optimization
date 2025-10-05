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

    
def inspect_clustering(good_cl,segments):
    fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
    for i_gcl in range(len(good_cl)):
        gcl = good_cl[i_gcl]
        joined = np.vstack([segments[i] for i in gcl])
        joined = sort_curve(joined)
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
        if len(gcl) < 1:
            continue
        if np.isin(gcl,already_cl).any():
            tmp_nodes = gcl[np.isin(gcl,already_cl)]
            
            print(f'Cluster {i_gcl} has already been clustered')
            print(f'Nodes: {tmp_nodes}')
            
            print(f'Corresponding cluster: ', gcl)
            continue
        else:
            no_overlap.append(gcl)
            
    return no_overlap


def inspect_a_cluster(a_cluster,segments,svd_cylinders,subcluster_error_threshold,subcluster_length_threshold,subcluster_length_tolerance=50):
    good_clusters = []
    
    subclusters = explode_local_cluster(a_cluster,svd_cylinders,segments)
    # check quality
    subcluster_error_list = np.zeros(len(subclusters))
    subcluster_length_list = np.zeros(len(subclusters))
    for i_subcluster,subcluster in enumerate(subclusters):
        joined = np.vstack([segments[ a_cluster[iii] ] for iii in subcluster])
        fit_result = fit_rod(joined,linearity_threshold=0.0001,radius_curvature_threshold=100000)
        subcluster_error_list[i_subcluster] = fit_result['err']
        # rec = fit_result['rec']
        subcluster_length_list[i_subcluster] = np.sum(np.sqrt(np.sum(np.diff(fit_result['rec'],axis=0)**2,axis=1)))
    
    certification = (subcluster_error_list < subcluster_error_threshold) & (np.abs(subcluster_length_list - subcluster_length_threshold) < subcluster_length_tolerance)
        
    for i_cert in np.where(certification)[0]:
        good_clusters.append([a_cluster[i] for i in subclusters[i_cert]])
        
    return good_clusters,subclusters


















linearity_threshold = 0.5
radius_curvature_threshold = 500
already_clustered = []
    
rod_data_root_dir = Path('/Users/yeonsu/Data/steel-rods-xray-data')
segments_file_path = rod_data_root_dir / 'alpha200_epsilon00' / 'pruned_segments.pkl'
connectivity_file_path = rod_data_root_dir / 'alpha200_epsilon00' / 'ijscores_total.pkl'

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

# segments = segments[:20000]

# local_segments = []
# for i,segment in enumerate(segments):
#     if np.all(np.sum((segment - np.array([1000,1000,500]))**2,axis=1) < 50000):
#         local_segments.append(segment)
        
# fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
# for i in range(len(local_segments)):
#     ax.plot(local_segments[i][:,0],local_segments[i][:,1],local_segments[i][:,2],linewidth=0.5)
# segments = local_segments

if os.path.exists(connectivity_file_path):
    with open(connectivity_file_path, 'rb') as f:
        ijscore = pickle.load(f)
else:
    # svd_cylinders,centroids,orientations, = prep_svd_cylinder(segments,scale_factor=0.98)
    fp = filamentprocessing.FilamentProcessing(segments,2500,0.15,0.99)
    ij = fp.get_svd_ij()
    scores = fp.get_svd_scores()
    
    ijscore = np.hstack([ij,scores])
        
    # pickle_out = open(connectivity_file_path,'wb')
    with open(connectivity_file_path, 'wb') as f:
        pickle.dump(ijscore,f)


    

N_segments = len(segments)
# segments_length_list,segments_error_list = inspect_segments(segments) # turn off this if you are sure about the segments
svd_cylinders,centroids,orientations, = prep_svd_cylinder(segments,scale_factor=0.99)


def initial_guess(self, alignment_threshold=0.01, distance_threshold=2000, fitting_threshold=1):
        ij = self.ijscore[:, :2]
        scores = self.ijscore[:, 2:]

        dist_score = scores[:, 0]
        align_score = scores[:, 1]
        fit_score = scores[:, 2]
        initial_mask = (align_score < alignment_threshold) & (dist_score < distance_threshold)

        Graph0 = nx.Graph()
        Graph0.add_nodes_from(range(len(self.segments)))
        Graph0.add_weighted_edges_from(zip(ij[initial_mask, 0], ij[initial_mask, 1], dist_score[initial_mask]))
        clusters = find_connected_components(Graph0)
        cluster_size_list = [len(x) for x in clusters]
        max_cluster_size = np.max(cluster_size_list)
        print(f'Number of clusters: {len(clusters)}')
        print(f'Max cluster size: {max_cluster_size}')

        return clusters

distance_threshold = 2000
alignment_threshold = 0.015

fitting_threshold = 1
ij = ijscore[:,:2]
scores = ijscore[:,2:]

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



cc = clusters[np.argmax(cluster_size_list)]

# graph = Graph0.subgraph(cc).copy()
graph = nx.Graph()
for i in cc:
    graph.add_node(i,weight=int(seg_len(segments[i])))
    
local_segments = [segments[i] for i in cc]
fp_local = filamentprocessing.FilamentProcessing(local_segments,2500,0.15,0.99)
ij = fp_local.get_svd_ij()
scores = fp_local.get_svd_scores()
ij = np.array(ij)
scores = np.array(scores)

dist_score = scores[:,0]
align_score = scores[:,1]
fit_score = scores[:,2]
mask = align_score < 0.03
    
graph.add_weighted_edges_from(zip(ij[mask,0],ij[mask,1],dist_score[mask]))
mst = nx.minimum_spanning_tree(graph, weight='weight')
comm = nx.algorithms.community.lukes_partitioning(mst, max_size=10)

fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
for i in comm:    
    joined = np.vstack([segments[int(j)] for j in i])
    joined = sort_curve(joined)
    ax.plot(joined[:,0],joined[:,1],joined[:,2],'.',linewidth=0.5)

fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
for i in range(len(cc)):
    ax.plot(segments[cc[i]][:,0],segments[cc[i]][:,1],segments[cc[i]][:,2],linewidth=0.5)
ax.axis('equal')

np.vstack([segments[x] for x in cc]).shape[0]/650

# segments_in_cluster = []
# for i in range(len(cc)):
#     segments_in_cluster.append(segments[cc[i]])
    
segments_in_cluster = [segments[x] for x in cc]
fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
for seg in segments_in_cluster:
    ax.plot(seg[:,0],seg[:,1],seg[:,2],linewidth=0.5)
    
with open('cluster_cache.pkl','wb') as f:
    pickle.dump(segments_in_cluster,f)

import importlib
import clustering
importlib.reload(clustering)
from clustering import subclustering_by_mst, subclustering_by_mst_length_lowerbound
good_clusters,not_yet_nodes = subclustering_by_mst_length_lowerbound(clusters,segments,svd_cylinders,subcluster_error_threshold,subcluster_length_threshold)
jsll = inspect_clustering(good_clusters,segments)

fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
for i_gcl in range(len(good_clusters)):
    gcl = good_clusters[i_gcl]
    joined = np.vstack([segments[i] for i in gcl])
    joined = sort_curve(joined)
    ax.plot(joined[:,0],joined[:,1],joined[:,2],linewidth=0.5)
    
for i_ny in not_yet_nodes:
    ax.plot(segments[i_ny][:,0],segments[i_ny][:,1],segments[i_ny][:,2],color='k',linewidth=0.5,alpha=0.1)


segments2 = []
for i_gcl in range(len(good_clusters)):
    gcl = good_clusters[i_gcl]
    joined = np.vstack([segments[i] for i in gcl])
    joined = sort_curve(joined)
    segments2.append(joined)
segments2 = segments2 + [segments[i] for i in not_yet_nodes]

fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
for i in range(len(segments2)):
    ax.plot(segments2[i][:,0],segments2[i][:,1],segments2[i][:,2],linewidth=0.5)

print(f'Number of segments: {len(segments)}')
print(f'Number of segments after clustering: {len(segments2)}')

fp2 = filamentprocessing.FilamentProcessing(segments2,2500,0.15,0.99)
ij2 = fp2.get_svd_ij()
scores2 = fp2.get_svd_scores()
ij2 = np.array(ij2)
scores2 = np.array(scores2)

distance_threshold = 2000
alignment_threshold = 0.02
fitting_threshold = 1

dist_score = scores2[:,0]
align_score = scores2[:,1]
fit_score = scores2[:,2]
initial_mask = (align_score < alignment_threshold) & (dist_score < distance_threshold)

N_segments2 = len(segments2)
Graph0 = nx.Graph()
Graph0.add_nodes_from(range(N_segments2))
Graph0.add_weighted_edges_from(zip(ij2[initial_mask,0],ij2[initial_mask,1],dist_score[initial_mask]))    
clusters = find_connected_components(Graph0)
cluster_size_list = [len(x) for x in clusters]
max_cluster_size = np.max(cluster_size_list)
print(f'Number of clusters: {len(clusters)}')
print(f'Max cluster size: {max_cluster_size}')





good_clusters,not_yet_nodes = subclustering_by_mst_length_lowerbound(clusters,segments,svd_cylinders,subcluster_error_threshold,subcluster_length_threshold)
jsll = inspect_clustering(good_clusters,segments)


good_clusters











segments_length_list = np.zeros(len(segments))
for i in range(len(segments)):
    segments_length_list[i] = seg_len(segments[i])
    
from distances import lumelsky_dist_vec

num_cc = len(cc)
dist_mat = np.full((num_cc,num_cc),np.inf)
for i in range(num_cc):
    segment_i = segments[cc[i]]
    cylinder_i = svd_cylinders[cc[i]]
    fr_i = fit_rod(segment_i,0.00001,10000)
    slopes_i = fr_i['slopes']
    
    
    for j in range(i+1,num_cc):
        segment_j = segments[cc[j]]
        cylinder_j = svd_cylinders[cc[j]]
        fr_j = fit_rod(segment_j,0.00001,10000)
        slopes_j = fr_j['slopes']
        
        joined = np.vstack([segment_i,segment_j])
        joined = sort_curve(joined)
        fr_joined = fit_rod(joined,0.00001,10000)
        err_joined = fr_joined['err']        

        t,u,d1,d2,d12 = lumelsky_dist_vec(cylinder_i[0:3],cylinder_i[3:6],cylinder_j[0:3],cylinder_j[3:6])
        popt1 = cylinder_i[0:3] + t*d1
        popt2 = cylinder_j[0:3] + u*d2
        
        if t == 0:
            orientation1 = slopes_i[0]
        elif t == 1:
            orientation1 = slopes_i[-1]
        else: # 0 < t < 1
            orientation1 = slopes_i[0] + t*(slopes_i[-1] - slopes_i[0])
            
        if u == 0:
            orientation2 = slopes_j[0]
        elif u == 1:
            orientation2 = slopes_j[-1]
        else: # 0 < u < 1
            orientation2 = slopes_j[0] + u*(slopes_j[-1] - slopes_j[0])
        
        popt1_opp = cylinder_i[0:3] + (1-t)*d1
        popt2_opp = cylinder_j[0:3] +(1-u)*d2
        
        dvec = popt1 - popt2
        dvec_normalized = dvec/np.linalg.norm(dvec)
        
        orientation1 = orientation1/np.linalg.norm(orientation1)
        orientation1 = orientation1*np.sign( np.dot(popt1-popt1_opp,orientation1) )
        
        orientation2 = orientation2/np.linalg.norm(orientation2)
        orientation2 = orientation2*np.sign( np.dot(popt2-popt2_opp,orientation2) )
        
        # align_score = -np.sign(np.dot(dvec_normalized,orientation1))
        
        sgn = 1        
        the_ultimate_distance = err_joined*sgn        
        # 2/( np.abs(np.dot(dvec_normalized,orientation1)) + np.abs(np.dot(dvec_normalized,orientation2)) )
        dist_mat[i,j] = (np.linalg.norm(np.cross(dvec_normalized,orientation1) ) + np.linalg.norm(np.cross(dvec_normalized,orientation2) ))/2
    
    if (np.linalg.norm(dvec) < 100) and (np.linalg.norm(dvec) > 50):
        
        fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
        rr_i = segments[cc[i]]
        rr_j = segments[cc[j]]
        ax.plot(rr_i[:,0],rr_i[:,1],rr_i[:,2],linewidth=0.5)
        ax.plot(rr_j[:,0],rr_j[:,1],rr_j[:,2],linewidth=0.5)
        ax.quiver(popt1[0],popt1[1],popt1[2],orientation1[0],orientation1[1],orientation1[2],length=30,color='r')
        ax.quiver(popt2[0],popt2[1],popt2[2],orientation2[0],orientation2[1],orientation2[2],length=30,color='b')
        ax.quiver(popt2[0],popt2[1],popt2[2],dvec[0],dvec[1],dvec[2],length=1,color='b')
        break
        
dist_mat[dist_mat < 0] = 1e10
dist_mat = np.minimum(dist_mat,dist_mat.T)
dist_mat[np.diag_indices(num_cc)] = 0

dist_mat_linear = dist_mat[np.triu_indices(num_cc,1)]
ij = np.triu_indices(num_cc,1)

local_graph = nx.Graph()
local_graph.add_nodes_from(range(num_cc))
local_mask = dist_mat_linear < 0.4
# local_graph.add_weighted_edges_from(zip(ij[0][local_mask],ij[1][local_mask],dist_mat_linear[local_mask]))
local_graph.add_edges_from(zip(ij[0][local_mask],ij[1][local_mask]))

fig,ax=plt.subplots(1,1)
nx.draw(local_graph,with_labels=True)

graph_cc = find_connected_components(local_graph)

fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
for lb in graph_cc:
    joined = np.vstack([segments[cc[j]] for j in lb])
    joined = sort_curve(joined)
    ax.plot(joined[:,0],joined[:,1],joined[:,2],'.',linewidth=0.5)

from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids

N_estimate = np.vstack([segments[x] for x in cc]).shape[0]//650

N_trials = np.min([N_estimate-1,20]) + 3
fitting_score_over_trials = np.full(N_trials+1,np.inf)

for i in range(1,N_trials+1):
    # km = KMeans(n_clusters=i, random_state=0, n_init="auto").fit(align_matrix)        
    km = KMedoids(n_clusters=i, random_state=0).fit(dist_mat)
    fit_score = 0
    
    cluster_fit_errors = []
    cluster_lengths = []
    for ii in range(km.labels_.max() + 1):
        idx = np.where(km.labels_ == ii)[0]
        lab_len = np.sum([segments_length_list[cc[j]] for j in idx])
        joined = np.vstack([segments[j] for j in idx])
        res = fit_rod(joined,1e-10,1e10)
        cluster_fit_errors.append(res['err'])
        cluster_lengths.append( lab_len )
    
    fitting_score_over_trials[i] = np.sum(cluster_fit_errors)

j_min = np.argmin(fitting_score_over_trials)
# km = KMeans(n_clusters=i, random_state=0, n_init="auto").fit(align_matrix)
km = KMedoids(n_clusters=j_min, random_state=0).fit(dist_mat)

fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
for lb in km.labels_:
    idx = np.where(km.labels_ == lb)[0]
    joined = np.vstack([segments[cc[j]] for j in idx])
    joined = sort_curve(joined)
    ax.plot(joined[:,0],joined[:,1],joined[:,2],'.',linewidth=0.5)
    

with open('local_segments.txt','w'):
    for i in cc:
        n_rows = segments[i].shape[0]
        for row in range(n_rows):
            print(f'{segments[i][row,0]} {segments[i][row,1]} {segments[i][row,2]}',file=open('local_segments.txt','a'))
            
        print('',file=open('local_segments.txt','a'))
        


fig,ax=plt.subplots(1,1)
nx.draw(local_graph,with_labels=True)

local_good_clusters,subclusters = inspect_a_cluster(cc,segments,svd_cylinders,subcluster_error_threshold,subcluster_length_threshold,subcluster_length_tolerance=50)

fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
for i_gcl in range(len(subclusters)):
    gcl = subclusters[i_gcl]
    joined = np.vstack([segments[cc[i]] for i in gcl])
    joined = sort_curve(joined)
    ax.plot(joined[:,0],joined[:,1],joined[:,2],'-',linewidth=0.5)
ax.axis('equal')

subcluster_error_threshold = 1
subcluster_length_threshold = 600



import importlib
import clustering
importlib.reload(clustering)
from clustering import subclustering_by_mst
good_clusters,not_yet_nodes = subclustering_by_mst(clusters,segments,svd_cylinders,subcluster_error_threshold,subcluster_length_threshold,subcluster_length_tolerance=50)
jsll = inspect_clustering(good_clusters,segments)


already_clustered.extend(good_clusters)

alreday_clustered_nodes = np.concatenate(already_clustered)
print(f'Number of already clustered nodes: {len(alreday_clustered_nodes)}')
print(f'Number of good clusters so far: ', len(already_clustered))














distance_threshold = 200
alignment_threshold = 0.02
fitting_threshold = 2

initial_mask = (align_score < alignment_threshold) & (dist_score < distance_threshold) & (fit_score < fitting_threshold)

Graph0 = nx.Graph()
Graph0.add_nodes_from(range(N_segments))
Graph0.add_weighted_edges_from(zip(ij[initial_mask,0],ij[initial_mask,1],dist_score[initial_mask]))    
Graph0.remove_nodes_from(alreday_clustered_nodes)

clusters = find_connected_components(Graph0)
cluster_size_list = [len(x) for x in clusters]
max_cluster_size = np.max(cluster_size_list)
print(f'Number of clusters: {len(clusters)}')
print(f'Max cluster size: {max_cluster_size}')

cc = clusters[np.argmax(cluster_size_list)]

import importlib
import fitting
import clustering
importlib.reload(clustering)
importlib.reload(fitting)
from clustering import subclustering_by_mst, explode_local_cluster
from fitting import fit_rod


local_good_clusters,subclusters = inspect_a_cluster(cc,segments,svd_cylinders)

fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
for i_gcl in range(len(local_good_clusters)):
    gcl = local_good_clusters[i_gcl]
    joined = np.vstack([segments[cc[i]] for i in gcl])
    joined = sort_curve(joined)
    ax.plot(joined[:,0],joined[:,1],joined[:,2],'-',linewidth=0.5)


rr = segments[25]
fit_result = fit_rod(rr,0.00001,10000)
fit_result['slope']


fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
for i in range(len(cc)):
    ax.plot(segments[cc[i]][:,0],segments[cc[i]][:,1],segments[cc[i]][:,2],linewidth=0.5)
ax.axis('equal')





good_clusters,not_yet_nodes = subclustering_by_mst(clusters,segments,svd_cylinders,subcluster_error_threshold,subcluster_length_threshold,subcluster_length_tolerance=50)
    

jsll = inspect_clustering(good_clusters,segments)
no_overlap = check_overlap(good_clusters,alreday_clustered_nodes)








already_clustered.extend(no_overlap)

alreday_clustered_nodes = np.concatenate(already_clustered)
print(f'Number of already clustered nodes: {len(alreday_clustered_nodes)}')
print(f'Number of good clusters so far: ', len(already_clustered))










distance_threshold = 200
alignment_threshold = 0.02
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


cc = clusters[np.argmax(cluster_size_list)]
local_good_clusters,subclusters = inspect_a_cluster(cc,segments,svd_cylinders)
fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
for i_gcl in range(len(subclusters)):
    gcl = subclusters[i_gcl]
    joined = np.vstack([segments[cc[i]] for i in gcl])
    joined = sort_curve(joined)
    ax.plot(joined[:,0],joined[:,1],joined[:,2],'-',linewidth=0.5)




good_clusters,not_yet_nodes = subclustering_by_mst(clusters,segments,svd_cylinders,subcluster_error_threshold,subcluster_length_threshold,subcluster_length_tolerance=50)
jsll = inspect_clustering(good_clusters,segments)



no_overlap = check_overlap(good_clusters,alreday_clustered_nodes)







already_clustered.extend(no_overlap)
alreday_clustered_nodes = np.concatenate(already_clustered)
print(f'Number of already clustered nodes: {len(alreday_clustered_nodes)}')










###

# Changed ones
subcluster_error_threshold = 1.2


distance_threshold = 200
alignment_threshold = 0.1
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

cc = clusters[np.argmax(cluster_size_list)]
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
print(f'Number of good clusters so far: ', len(already_clustered))







# Changed ones
subcluster_error_threshold = 1.5


distance_threshold = 200
alignment_threshold = 0.12
fitting_threshold = 2
initial_mask = (align_score < alignment_threshold) & (dist_score < distance_threshold) & (fit_score < fitting_threshold)

Graph0 = nx.Graph()
Graph0.add_nodes_from(range(N_segments))
Graph0.add_weighted_edges_from(zip(ij[initial_mask,0],ij[initial_mask,1],dist_score[initial_mask]))    
Graph0.remove_nodes_from(alreday_clustered_nodes)

clusters = find_connected_components(Graph0)
cluster_size_list = [len(x) for x in clusters]
max_cluster_size = np.max(cluster_size_list)
print(f'Number of clusters: {len(clusters)}')
print(f'Max cluster size: {max_cluster_size}')

cc = clusters[np.argmax(cluster_size_list)]
fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
for i in range(len(cc)):
    ax.plot(segments[cc[i]][:,0],segments[cc[i]][:,1],segments[cc[i]][:,2],linewidth=0.5)
ax.axis('equal')
good_clusters,not_yet_nodes = subclustering_by_mst(clusters,segments,subcluster_error_threshold,subcluster_length_threshold,subcluster_length_tolerance=50)
jsll = inspect_clustering(good_clusters,segments)




good_clusters



