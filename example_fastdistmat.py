from clustering import print_connected_components,explode_local_cluster
from fitting import fit_rod
from pathlib import Path
from scipy.io import loadmat
import numpy as np
import fastdistmat
from fitting import prep_svd_cylinder
from matplotlib import pyplot as plt
import networkx as nx

rod_data_root_dir = Path('/Users/yeonsu/Data/steel-rods-xray-data')
segments_file_path = rod_data_root_dir / 'alpha200_epsilon00' / 'segments.mat'

mat_obj = loadmat(segments_file_path)
segments = mat_obj['segments']
segments = [seg[0] for seg in segments]
N_segments = len(segments)

for i,segment in enumerate(segments):
    segments[i] = np.array(segment,dtype=np.float64)

# print(f'Staircase clustering: {segments_file_path.parent}')
print(f'Number of segments: {N_segments}')

svd_cylinders,centroids,orientations, = prep_svd_cylinder(segments,scale_factor=0.98)

# Create an instance of FastDistMat
fdm = fastdistmat.FastDistMat(svd_cylinders)

# Calculate adjacency with a threshold of 0.3
fdm.calculate_alignment_adjacency(0.15)

ij = fdm.get_ij()
scores = fdm.get_scores()



from distances import lumelsky_dist_vec
ij = np.array(ij)
scores = np.array(scores)

i_inspect = 980
p1 = svd_cylinders[ij[i_inspect,0], 0:3]
p2 = svd_cylinders[ij[i_inspect,0], 3:6]
q1 = svd_cylinders[ij[i_inspect,1], 0:3]
q2 = svd_cylinders[ij[i_inspect,1], 3:6]

t,u,d1,d2,d12 = lumelsky_dist_vec(p1,p2,q1,q2)
vec = d1*t - d2*u - d12

popt1 = p1+t*d1
popt2 = q1+u*d2

vec = popt1-popt2
unit_vec = vec/np.linalg.norm(vec)

popt1_opp = p1+ np.clip(1-t,0,1) *d1
popt2_opp = q1+ np.clip(1-u,0,1) *d2

axis1 = popt1_opp-popt1
axis1 = axis1/np.linalg.norm(axis1)
axis2 = popt2_opp-popt2
axis2 = axis2/np.linalg.norm(axis2)

orientations[ij[0,0]]
orientations[ij[0,1]]

np.linalg.norm(np.cross(unit_vec,axis1))
np.linalg.norm(np.cross(unit_vec,axis2))


fig= plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot([p1[0],p2[0]],[p1[1],p2[1]],[p1[2],p2[2]],'r')
ax.plot([q1[0],q2[0]],[q1[1],q2[1]],[q1[2],q2[2]],'b')
ax.plot([popt1[0],popt2[0]],[popt1[1],popt2[1]],[popt1[2],popt2[2]],'g')
ax.axis('equal')

np.cross(vec,axis2)

dist = np.linalg.norm(vec)

align_score = scores[:,0]
dist_score = scores[:,1]
 
distance_threshold = 50
alignment_threshold = 0.05
initial_mask = (align_score < alignment_threshold) & (dist_score < distance_threshold)

Graph0 = nx.Graph()
Graph0.add_nodes_from(range(N_segments))
Graph0.add_weighted_edges_from(zip(ij[initial_mask,0],ij[initial_mask,1],dist_score[initial_mask]))


clusters = print_connected_components(Graph0)
        
cluster_size_list = [len(x) for x in clusters]
max_cluster_size = np.max(cluster_size_list)
print(f'Number of clusters: {len(clusters)}')
print(f'Max cluster size: {max_cluster_size}')

segment_length_list = np.zeros(len(segments))
for i,seg in enumerate(segments):
    segment_length_list[i] = np.sum(np.sqrt(np.sum(np.diff(seg,axis=0)**2,axis=1)))

cluster_length_list = np.zeros(len(clusters))
for i in range(len(clusters)):
    cluster_length_list[i] = segment_length_list[clusters[i]].sum()/650


subcluster_error_threshold = 1
subcluster_length_threshold = 200



good_clusters = []
for i_cluster, a_cluster in enumerate(clusters):
    subclusters = explode_local_cluster(segments,a_cluster)
    # check quality
    subcluster_error_list = np.zeros(len(subclusters))
    subcluster_length_list = np.zeros(len(subclusters))
    for i_subcluster,subcluster in enumerate(subclusters):
        joined = np.vstack([segments[ a_cluster[iii] ] for iii in subcluster])
        fit_result = fit_rod(joined,linearity_threshold=0.0001,radius_curvature_threshold=100000)
        subcluster_error_list[i_subcluster] = fit_result['err']
        # rec = fit_result['rec']
        subcluster_length_list[i_subcluster] = segment_length_list[ a_cluster[subcluster] ].sum()
    
    certification = (subcluster_error_list < subcluster_error_threshold) & (subcluster_length_list > subcluster_length_threshold)
    
    for i_cert in np.where(certification)[0]:
        good_clusters.append([a_cluster[i] for i in subclusters[i_cert]])
    

# fig,ax = set_3d_plot()
# for i_gcl in np.random.choice(len(good_clusters),1000):
#     gcl = good_clusters[i_gcl]
#     joined = np.vstack([segments[i] for i in gcl])
#     plot_single_rod(joined,ax=ax)

good_nodes = np.hstack(good_clusters)
not_yet_nodes = np.setdiff1d(range(N_segments),good_nodes)
for i_ny in not_yet_nodes:
    rr = segments[i_ny]
    fit_result = fit_rod(rr,linearity_threshold=0.0001,radius_curvature_threshold=100000)
    rr_len = np.sum(np.sqrt(np.sum(np.diff(rr,axis=0)**2,axis=1)))
    
    if fit_result['err'] < 1 and rr_len > 200:
        good_clusters.append(np.array([i_ny]))
        
good_nodes = np.hstack(good_clusters)
not_yet_nodes = np.setdiff1d(range(N_segments),good_nodes)

print(f'Number of good clusters: ', len(good_clusters))
print(f'Number of not yet nodes: ', len(not_yet_nodes))



def sort_curve(rr):
    centroid = np.mean(rr,axis=0)
    rr_centered = rr - centroid        
    _,_, V = np.linalg.svd(rr_centered, full_matrices=False)
    v1 = V[0,:]
    orientation = v1 * np.sign(np.sum(v1 * (rr_centered[-1, :] - rr_centered[0, :])))
    slist = np.dot((rr - centroid), orientation)
    sorted_indices = np.argsort(slist)
    return centroid + rr_centered[sorted_indices]

good_segments = []
for i_gcl in range(len(good_clusters)):
    gcl = good_clusters[i_gcl]
    joined = np.vstack([segments[i] for i in gcl])
    good_segments.append(sort_curve(joined))
    
not_yet_segments = []
for i_ny in range(len(not_yet_nodes)):
    rr = segments[not_yet_nodes[i_ny]]
    not_yet_segments.append(rr)
    
len(not_yet_nodes)

segments_next_round = good_segments + not_yet_segments
len(segments_next_round)

segment_length_list_next_round = np.zeros(len(segments_next_round))
for i,seg in enumerate(segments_next_round):
    segment_length_list_next_round[i] = np.sum(np.sqrt(np.sum(np.diff(seg,axis=0)**2,axis=1)))


svd_cylinders_next_round,_,_ = prep_svd_cylinder(segments_next_round,scale_factor=0.98)

fdm_next_round = fastdistmat.FastDistMat(svd_cylinders_next_round)

# Calculate adjacency with a threshold of 0.3
fdm_next_round.calculate_alignment_adjacency(0.15)

ij = fdm_next_round.get_ij()
scores = fdm_next_round.get_scores()

ij = np.array(ij)
scores = np.array(scores)
align_score = np.array(scores[:,0])
dist_score = np.array(scores[:,1])

ij.shape
align_score.shape

distance_threshold = 100
alignment_threshold = 0.05

initial_mask = (align_score < alignment_threshold) & (dist_score < distance_threshold)

Graph0_next_round = nx.Graph()
Graph0_next_round.add_nodes_from(range(len(segments_next_round)))
Graph0_next_round.add_weighted_edges_from(zip(ij[initial_mask,0],ij[initial_mask,1],dist_score[initial_mask]))

clusters_next_round = print_connected_components(Graph0_next_round)
        
cluster_size_list = [len(x) for x in clusters_next_round]
max_cluster_size = np.max(cluster_size_list)
print(f'Number of clusters: {len(clusters_next_round)}')
print(f'Max cluster size: {max_cluster_size}')


cluster_length_list = np.zeros(len(clusters_next_round))
for i in range(len(clusters_next_round)):
    cluster_length_list[i] = segment_length_list_next_round[clusters_next_round[i]].sum()/650


subcluster_error_threshold = 1
subcluster_length_threshold = 200

good_clusters_next_round = []
for i_cluster, a_cluster in enumerate(clusters_next_round):
    subclusters_next_round = explode_local_cluster(segments,a_cluster)
    # check quality
    subcluster_error_list = np.zeros(len(subclusters_next_round))
    subcluster_length_list = np.zeros(len(subclusters_next_round))
    for i_subcluster,subcluster in enumerate(subclusters_next_round):
        joined = np.vstack([segments[ a_cluster[iii] ] for iii in subcluster])
        fit_result = fit_rod(joined,linearity_threshold=0.0001,radius_curvature_threshold=100000)
        subcluster_error_list[i_subcluster] = fit_result['err']
        # rec = fit_result['rec']
        subcluster_length_list[i_subcluster] = segment_length_list_next_round[ a_cluster[subcluster] ].sum()
    
    certification = (subcluster_error_list < subcluster_error_threshold) & (subcluster_length_list > subcluster_length_threshold)
    
    for i_cert in np.where(certification)[0]:
        good_clusters_next_round.append([a_cluster[i] for i in subclusters_next_round[i_cert]])
    
N_segments_next_round = len(segments_next_round)
good_nodes_next_round = np.hstack(good_clusters_next_round)
not_yet_nodes_next_round = np.setdiff1d(range(N_segments_next_round),good_nodes_next_round)

for i_ny in not_yet_nodes_next_round:
    rr = segments[i_ny]
    fit_result = fit_rod(rr,linearity_threshold=0.0001,radius_curvature_threshold=100000)
    rr_len = np.sum(np.sqrt(np.sum(np.diff(rr,axis=0)**2,axis=1)))
    
    if fit_result['err'] < 1 and rr_len > 200:
        good_clusters_next_round.append(np.array([i_ny]))
        
good_nodes_next_round = np.hstack(good_clusters_next_round)
not_yet_nodes_next_round = np.setdiff1d(range(N_segments_next_round),good_nodes_next_round)

print(f'Number of good clusters_next_round: ', len(good_clusters_next_round))
print(f'Number of not yet nodes: ', len(not_yet_nodes_next_round))


fig= plt.figure()
ax = fig.add_subplot(111, projection='3d')
# for i_gcl in range(len(good_clusters_next_round)):
for i_gcl in np.random.choice(len(good_clusters_next_round),1000):
    gcl = good_clusters_next_round[i_gcl]
    joined = np.vstack([segments_next_round[i] for i in gcl])
    
    joined_len = np.sum(np.sqrt(np.sum(np.diff(joined,axis=0)**2,axis=1)))
    
    ax.plot(joined[:,0],joined[:,1],joined[:,2])

fig= plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i_not_yet in np.random.choice(not_yet_nodes_next_round,3000):
    rr = segments_next_round[i_not_yet]
    ax.plot(rr[:,0],rr[:,1],rr[:,2])
    
len(good_clusters)
good_nodes.shape

good_nodes_next_round.shape
not_yet_nodes_next_round.shape

print