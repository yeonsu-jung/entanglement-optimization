import networkx as nx
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

from clustering import find_connected_components, explode_local_cluster, find_edges_to_remove_by_weight
import numpy as np
def seg_len(seg):
    return np.sum(np.sqrt(np.sum(np.diff(seg, axis=0) ** 2, axis=1)))

def sort_curve(rr):
    centroid = np.mean(rr, axis=0)
    rr_centered = rr - centroid
    _, _, V = np.linalg.svd(rr_centered, full_matrices=False)
    v1 = V[0, :]
    orientation = v1 * np.sign(np.sum(v1 * (rr_centered[-1, :] - rr_centered[0, :])))
    slist = np.dot((rr - centroid), orientation)
    sorted_indices = np.argsort(slist)
    return centroid + rr_centered[sorted_indices]
    
with open('cluster_cache.pkl','rb') as f:
    cluster = pickle.load(f)
    
    
print('Number of clusters:', len(cluster))
# fig,ax=plt.subplots(1,1,figsize=(10,10),subplot_kw={'projection':'3d'})
# for i,seg in enumerate(cluster):
#     ax.plot(seg[:,0],seg[:,1],seg[:,2],label=f'Cluster {i}')
# plt.show()


smooth_cluster = []
for seg in cluster:
    fit_result = fit_rod(seg,1e-10,1e10)
    smooth_cluster.append(fit_result['rec'])
    
# fig,ax=plt.subplots(1,1,figsize=(10,10),subplot_kw={'projection':'3d'})
# for i,seg in enumerate(smooth_cluster):
#     ax.plot(seg[:,0],seg[:,1],seg[:,2],label=f'Cluster {i}')
    
def fit_line(seg):
    cen = np.mean(seg,axis=0)
    centered = seg-cen
    u,s,v = np.linalg.svd(centered)
    orientation = v[0]
    slist = np.dot((seg - cen), orientation)
    
    s1, s2 = np.min(slist), np.max(slist)
    r1 = cen + s1 * orientation
    r2 = cen + s2 * orientation
    best_estimation = cen + np.outer(slist, orientation)
    return best_estimation

# fig,ax=plt.subplots(1,1,figsize=(10,10),subplot_kw={'projection':'3d'})
# for seg in smooth_cluster:
#     best_estimation = fit_line(seg)
#     ax.plot(best_estimation[:,0],best_estimation[:,1],best_estimation[:,2],label='Best Estimation')

import filamentprocessing

segments = cluster
fp = filamentprocessing.FilamentProcessing(segments,2500,0.15,0.99)
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
mask = align_score < 0.01



G = nx.Graph()
for i in range(len(segments)):    
    G.add_node(i,weight=int(seg_len(segments[i])))
G.add_weighted_edges_from(zip(ij[mask,0],ij[mask,1],dist_score[mask]))

# svd_cylinders,_,_ = prep_svd_cylinder(segments,scale_factor=1)
fig,ax=plt.subplots(1,1)
nx.draw(G)

edges_to_remove = find_edges_to_remove_by_weight(G)
G.remove_edges_from(edges_to_remove)

subclusters = list(nx.connected_components(G))
fig,ax=plt.subplots(1,1,figsize=(10,10),subplot_kw={'projection':'3d'})
for subcl in subclusters:
    joined = np.concatenate([segments[j] for j in subcl],axis=0)
    ax.plot(joined[:,0],joined[:,1],joined[:,2],'.')
    
np.argmax([len(subcl) for subcl in subclusters])
subcl = subclusters[np.argmax([len(subcl) for subcl in subclusters])]
subgraph = G.subgraph(subcl)

node_len = []
for i in range(len(segments)):
    # node_len.append((i,seg_len(segments[i])))
    node_len.append(int(seg_len(segments[i])))
    
# node_dict = {}
# for i in range(len(segments)):
#     node_dict[i] = int(seg_len(segments[i]))

mst = nx.minimum_spanning_tree(subgraph, weight='weight')
fig,ax=plt.subplots()
nx.draw(mst)
cc = nx.algorithms.community.lukes_partitioning(mst, max_size=650, node_weight="weight")

ccs = nx.algorithms.community.louvain_partitions(G, weight='weight', resolution=1e-1, threshold=1e-7, seed=None)
ccs = list(ccs)
cc = ccs[-1]
print(len(cc))
    
fig,ax=plt.subplots(1,1,figsize=(10,10),subplot_kw={'projection':'3d'})
for i in cc:    
    joined = np.concatenate([segments[j] for j in i],axis=0)
    joined = sort_curve(joined)
    ax.plot(joined[:,0],joined[:,1],joined[:,2],'.')
    # for j in i:
    #     ax.plot(segments[j][:,0],segments[j][:,1],segments[j][:,2])

print("\n===== Compute the Ollivier-Ricci curvature of the given graph G =====")
# compute the Ollivier-Ricci curvature of the given graph G
orc = OllivierRicci(G, alpha=0.5, verbose="INFO")
orc.compute_ricci_curvature()

orc.G[0][2]

orc_OTD = OllivierRicci(G, alpha=0.5, method="OTD", verbose="INFO")
orc_OTD.compute_ricci_flow(iterations=10)
print("\n=====  Compute Ricci community - by Ricci flow =====")
clustering = orc_OTD.ricci_community()

fig,ax=plt.subplots(1,1,figsize=(10,10),subplot_kw={'projection':'3d'})

clustering[1][2]

from sklearn import preprocessing, metrics
def draw_graph(G, clustering_label="club"):
    """
    A helper function to draw a nx graph with community.
    """
    complex_list = nx.get_node_attributes(G, clustering_label)

    le = preprocessing.LabelEncoder()
    node_color = le.fit_transform(list(complex_list.values()))

    nx.draw_spring(G, nodelist=G.nodes(),
                    node_color=node_color,
                    cmap=plt.cm.rainbow,
                    alpha=0.8)

draw_graph(G)



print("\n===== Compute the Forman-Ricci curvature of the given graph G =====")
frc = FormanRicci(G)
frc.compute_ricci_curvature()
print("Karate Club Graph: The Forman-Ricci curvature of edge (0,1) is %f" % frc.G[0][1]["formanCurvature"])

# -----------------------------------
print("\n=====  Compute Ricci flow metric - Optimal Transportation Distance =====")
G = nx.karate_club_graph()
orc_OTD = OllivierRicci(G, alpha=0.5, method="OTD", verbose="INFO")
orc_OTD.compute_ricci_flow(iterations=10)
print("\n=====  Compute Ricci community - by Ricci flow =====")
clustering = orc_OTD.ricci_community()



    
print