# %%
%matplotlib qt
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
from distances import lumelsky_dist_vec
import filamentprocessing

import jax
import jax.numpy as jnp

import pickle

from scipy.special import comb
from scipy.spatial.distance import cdist
from scipy.interpolate import make_interp_spline
from scipy.optimize import minimize

class Filament:
    def __init__(self,nodes):
        self.nodes = nodes # nodes are centerline points
        
        self.certification = None
        
        
        # self.svd_cylinders = None # cylinder representation of the filament
        # self.scores = None
        # self.ij = None
        # self.graph = None
        # self.connected_components = None
        # self.cluster_size_list = None
        # self.good_clusters = None
        # self.subclusters = None
        # self.subcluster_error_list = None
        # self.subcluster_length_list = None
        
        # self.local_segments = None
        
class CollectiveFilaments:
    def __init__(self,filaments):
        self.filaments = filaments # type Filament
        
        # graph representation
        
        # partitions (clusters)
        
        # partition, assess, and merge/reject
        
        
        

def calculate_2d_align_matrix(segments):
    num_segments = len(segments)
    align_matrix = np.zeros((num_segments,num_segments))
    
    for i in range(num_segments):
        for j in range(i+1,num_segments):
            seg_i = segments[i]
            seg_j = segments[j]

            joined = np.vstack([seg_i,seg_j])
            _,_,v = np.linalg.svd(joined)
            v1 = v[0,:]
            v2 = v[1,:]

            # project to the plane
            proj_i = seg_i @ np.column_stack([v1, v2])
            proj_j = seg_j @ np.column_stack([v1, v2])
            fitted_i = fit_line(proj_i)
            fitted_j = fit_line(proj_j)

            p1 = np.array([fitted_i[0,0],fitted_i[0,1],0])
            p2 = np.array([fitted_i[-1,0],fitted_i[-1,1],0])

            q1 = np.array([fitted_j[0,0],fitted_j[0,1],0])
            q2 = np.array([fitted_j[-1,0],fitted_j[-1,1],0])

            t,u,_,_,_ = lumelsky_dist_vec(p1,p2,q1,q2)

            popt1 = p1 + t * (p2 - p1)
            popt2 = q1 + u * (q2 - q1)
            dvec = popt1 - popt2

            t_opp = np.clip(1-t,0,1)
            u_opp = np.clip(1-u,0,1)

            popt1_opp = p1 + t_opp * (p2 - p1)
            popt2_opp = q1 + u_opp * (q2 - q1)

            axis1 = popt1_opp - popt1
            axis2 = popt2_opp - popt2

            dist = np.linalg.norm(dvec)
            
            axis1 = axis1 / np.linalg.norm(axis1)
            axis2 = axis2 / np.linalg.norm(axis2)
            dvec = dvec / np.linalg.norm(dvec)
            align_score = (np.linalg.norm(np.cross(dvec,axis1)) + np.linalg.norm(np.cross(dvec,axis2)) )
            align_matrix[i,j] = align_score
        
        print(f"Segment {i} done")
            

            
    return align_matrix

    # align_matrix = calculate_2d_align_matrix(segments)
    
def fit_line(seg):
    cen = np.mean(seg,axis=0)
    centered = seg-cen
    u,s,v = np.linalg.svd(centered)
    orientation = v[0]
    orientation *= np.sign(np.sum(orientation * (seg[-1, :] - seg[0, :])))
    orientation = orientation / np.linalg.norm(orientation)
    slist = np.dot((seg - cen), orientation)
    
    # s1, s2 = np.min(slist), np.max(slist)
    best_estimation = cen + np.outer(slist, orientation)
    return best_estimation


def check_align_score(i,j,segments,svd_cylinders):
    seg_i = segments[i]
    seg_j = segments[j]

    p1 = svd_cylinders[i,0:3]
    p2 = svd_cylinders[i,3:6]
    q1 = svd_cylinders[j,0:3]
    q2 = svd_cylinders[j,3:6]

    t,u,d1,d2,d12 = lumelsky_dist_vec(p1,p2,q1,q2)

    popt1 = p1 + t * (p2 - p1)
    popt2 = q1 + u * (q2 - q1)
    dvec = popt1 - popt2
    dist = np.linalg.norm(dvec)

    t_opp = np.clip(1-t,0,1)
    u_opp = np.clip(1-u,0,1)

    popt1_opp = p1 + t_opp * (p2 - p1)
    popt2_opp = q1 + u_opp * (q2 - q1)

    axis1 = popt1_opp - popt1
    axis2 = popt2_opp - popt2

    plt.close('all')
    fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
    ax.plot(seg_i[:,0],seg_i[:,1],seg_i[:,2],'-',linewidth=1)
    ax.plot(seg_j[:,0],seg_j[:,1],seg_j[:,2],'-',linewidth=1)
    ax.quiver(popt2[0],popt2[1],popt2[2],dvec[0],dvec[1],dvec[2],color='r',linewidth=0.5)
    ax.quiver(popt1[0],popt1[1],popt1[2],axis1[0],axis1[1],axis1[2],color='g',linewidth=0.5)
    ax.quiver(popt2[0],popt2[1],popt2[2],axis2[0],axis2[1],axis2[2],color='g',linewidth=0.5)
    

    axis1 = axis1 / np.linalg.norm(axis1)
    axis2 = axis2 / np.linalg.norm(axis2)
    normalized_dvec = dvec / np.linalg.norm(dvec)
    align_score = (np.linalg.norm(np.cross(normalized_dvec,axis1)) + np.linalg.norm(np.cross(normalized_dvec,axis2)) )/2
    print(f'Min. distance: {dist}')
    print(f'Align score: {align_score}')
    return ax

def align_score_2d(i,j,segments):
    seg_i = segments[i]
    seg_j = segments[j]

    joined = np.vstack([seg_i,seg_j])
    u,s,v = np.linalg.svd(joined)
    v1 = v[0,:]
    v2 = v[1,:]

    # project to the plane
    proj_i = seg_i @ np.column_stack([v1, v2])
    proj_j = seg_j @ np.column_stack([v1, v2])

    fitted_i = fit_line(proj_i)
    fitted_j = fit_line(proj_j)

    p1 = np.array([fitted_i[0,0],fitted_i[0,1],0])
    p2 = np.array([fitted_i[-1,0],fitted_i[-1,1],0])

    q1 = np.array([fitted_j[0,0],fitted_j[0,1],0])
    q2 = np.array([fitted_j[-1,0],fitted_j[-1,1],0])

    t,u,_,_,_ = lumelsky_dist_vec(p1,p2,q1,q2)

    popt1 = p1 + t * (p2 - p1)
    popt2 = q1 + u * (q2 - q1)
    dvec = popt1 - popt2

    t_opp = np.clip(1-t,0,1)
    u_opp = np.clip(1-u,0,1)

    popt1_opp = p1 + t_opp * (p2 - p1)
    popt2_opp = q1 + u_opp * (q2 - q1)

    axis1 = popt1_opp - popt1
    axis2 = popt2_opp - popt2


    dist = np.linalg.norm(dvec)

    plt.close()
    fig,ax=plt.subplots(1,1,figsize=(10,10))
    ax.plot(proj_i[:,0],proj_i[:,1],'.-')
    ax.plot(proj_j[:,0],proj_j[:,1],'-')
    ax.plot(popt1[0],popt1[1],'b.')
    ax.plot(popt1_opp[0],popt1_opp[1],'bo')

    ax.plot(popt2[0],popt2[1],'r.')
    ax.plot(popt2_opp[0],popt2_opp[1],'ro')
    ax.quiver(popt2[0],popt2[1],dvec[0],dvec[1],color='r')

    ax.quiver(popt1[0],popt1[1],axis1[0],axis1[1],color='g')
    ax.quiver(popt2[0],popt2[1],axis2[0],axis2[1],color='g')

    axis1 = axis1 / np.linalg.norm(axis1)
    axis2 = axis2 / np.linalg.norm(axis2)
    dvec = dvec / np.linalg.norm(dvec)
    align_score = (np.linalg.norm(np.cross(dvec,axis1)) + np.linalg.norm(np.cross(dvec,axis2)) )
    
    print(f'Min. distance: {dist}')
    print(f'Align score: {align_score}')
    return ax



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
        gcl = good_cl[i_gcl]
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

# %%
linearity_threshold = 0.5
radius_curvature_threshold = 500
already_clustered = []
    
rod_data_root_dir = Path('/Users/yeonsu/Data/steel-rods-xray-data')
segments_file_path = rod_data_root_dir / 'alpha200_epsilon00' / 'pruned_segments.pkl'
segments = pickle.load(open(segments_file_path,'rb'))

# %%
# local_segments = []
# for i,segment in enumerate(segments):
#     centroid = np.mean(segment,axis=0)
    
#     if (np.linalg.norm(centroid - np.array([700,700,500])) < 150):
#         local_segments.append(segment)        
# print(f'Number of segments: {len(local_segments)}')
        
# fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
# for i in range(len(local_segments)):
#     ax.plot(local_segments[i][:,0],local_segments[i][:,1],local_segments[i][:,2],'.')
# segments = local_segments

# %%


if os.path.exists('ij.pkl'):
    with open('ij.pkl','rb') as f:
        ij = pickle.load(f)
    with open('scores.pkl','rb') as f:
        scores = pickle.load(f)
    with open('svd_cylinders.pkl','rb') as f:
        svd_cylinders = pickle.load(f)
        
else:

    import time
    start = time.time()
    fp = filamentprocessing.FilamentProcessing(segments,200,1,0.99)
    print(f'Elapsed time: {time.time()-start}')
    ij = fp.get_svd_ij()
    scores = fp.get_svd_scores()
    ij = np.array(ij)
    scores = np.array(scores)

    svd_cylinders = fp.get_svd_cylinders()
    svd_cylinders = np.array(svd_cylinders)
    svd_cylinders.shape

    with open('ij.pkl','wb') as f:
        pickle.dump(ij,f)
    with open('scores.pkl','wb') as f:
        pickle.dump(scores,f)
    with open('svd_cylinders.pkl','wb') as f:
        pickle.dump(svd_cylinders,f)
# %%
fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
for i in range(len(svd_cylinders)):
    p1 = svd_cylinders[i,:3]
    p2 = svd_cylinders[i,3:6]
    ax.plot([p1[0],p2[0]],[p1[1],p2[1]],[p1[2],p2[2]])


# from visualizations import plot_centerline_with_container
# fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
# for i in range(len(svd_cylinders)):
#     plot_centerline_with_container(segments,svd_cylinders,i,ax)
    
    
# %%
dist_score = scores[:,0]
align_score = scores[:,1]

mask = (dist_score < 50) & (align_score < 0.05)

graph = nx.Graph()
graph.add_nodes_from(range(len(segments)))
graph.add_edges_from(ij[mask,:])

connected_components = list(nx.connected_components(graph))
connected_components = [list(x) for x in connected_components]
cluster_size_list = [len(x) for x in connected_components]
print(f'Number of segments: {len(segments)}')
print(f'Number of connected components: {len(connected_components)}')
print(f'Max. cluster size {np.max(cluster_size_list)} at {np.argmax(cluster_size_list)}')
# %%


# %%

# plot max cluster
i_max = np.argmax(cluster_size_list)
cc_max = connected_components[i_max]


plt.close('all')
fig,ax=plt.subplots(1,1,figsize=(10,10),subplot_kw={'projection':'3d'})
for i_ in cc_max:
    deg = graph.degree[i_]
    ax.plot(segments[i_][:,0],segments[i_][:,1],segments[i_][:,2],linewidth=deg)


for i_ in cc_max:
    cen_i = np.mean(segments[i_],axis=0)
    
    for j_ in cc_max:
        if j_ <= i_:
            continue
        
        cen_j = np.mean(segments[j_],axis=0)
        if graph.has_edge(i_,j_):
            # curve connecting cen_i and cen_j
            
            random_offset = np.random.rand(3) * 10
            # cen_i to cen_i + offset
            ax.plot([cen_i[0],cen_i[0]+random_offset[0]],[cen_i[1],cen_i[1]+random_offset[1]],[cen_i[2],cen_i[2]+random_offset[2]],'k-',linewidth=0.5)
            # cen_j to cen_j + offset
            ax.plot([cen_j[0],cen_j[0]+random_offset[0]],[cen_j[1],cen_j[1]+random_offset[1]],[cen_j[2],cen_j[2]+random_offset[2]],'k-',linewidth=0.5)
            # cen_i + offset to cen_j + offset
            ax.plot([cen_i[0]+random_offset[0],cen_j[0]+random_offset[0]],[cen_i[1]+random_offset[1],cen_j[1]+random_offset[1]],[cen_i[2]+random_offset[2],cen_j[2]+random_offset[2]],'k-',linewidth=0.5)
            
            
# %%
# inspect a cluster
i_ = 25

# length
joined = np.vstack([segments[i] for i in connected_components[i_]])
joined = sort_curve(joined)

print(connected_components[i_])
print(seg_len(joined))

length_list = []
error_list = []
for i_ in range(len(connected_components)):
    joined = np.vstack([segments[i] for i in connected_components[i_]])
    joined = sort_curve(joined)
    length_list.append(seg_len(joined))
    
    fit_result = fit_rod(joined,0.00001,10000)
    error_list.append(fit_result['err'])
# %%
plt.close('all')
plt.hist(length_list,bins=100)
# %%
np.count_nonzero(np.array(length_list) > 600)

good_clusters_labels = np.where(np.array(length_list) > 600)[0]
# %%
error_list = np.array(error_list)
local_where = np.argmax(error_list[np.array(length_list) > 600])
i_weird = good_clusters_labels[local_where]


# %%

cc = connected_components[i_weird]
joined = np.vstack([segments[i] for i in cc])
joined = sort_curve(joined)
fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
ax.plot(joined[:,0],joined[:,1],joined[:,2],linewidth=1)
ax.axis('equal')

seg_len(joined)

# %%
legit_clusters = []
for i_ in range(len(connected_components)):
    joined = np.vstack([segments[i] for i in connected_components[i_]])
    joined = sort_curve(joined)
    if seg_len(joined) > 600:
        legit_clusters.append(connected_components[i_])
        
# %%
plt.close('all')
fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
for i_ in legit_clusters:
    joined = np.vstack([segments[i] for i in i_])
    joined = sort_curve(joined)
    ax.plot(joined[:,0],joined[:,1],joined[:,2],linewidth=0.5)
            
            
# %% second round
to_be_legit_clusters = []
for i_ in range(len(connected_components)):
    joined = np.vstack([segments[i] for i in connected_components[i_]])
    joined = sort_curve(joined)
    if seg_len(joined) < 600:
        to_be_legit_clusters.append(connected_components[i_])
# %%
length_list = []
error_list = []
for i_ in range(len(to_be_legit_clusters)):
    joined = np.vstack([segments[i] for i in to_be_legit_clusters[i_]])
    joined = sort_curve(joined)
    
    length_list.append(seg_len(joined))
    error_list.append(fit_rod(joined,0.00001,10000)['err'])
    
# %%
plt.close('all')
fig,ax=plt.subplots(1,1)
log_bins = np.logspace(0,3,100)
ax.hist(length_list,bins=log_bins)
ax.set_xscale('log')
# %%
filtered = []
fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
for i_ in range(len(to_be_legit_clusters)):
    joined = np.vstack([segments[i] for i in to_be_legit_clusters[i_]])
    joined = sort_curve(joined)    
    if np.all(np.linalg.norm(joined[:,:2] - [1000,1000],axis=1) > 700) and (seg_len(joined) > 100):
        ax.plot(joined[:,0],joined[:,1],joined[:,2],linewidth=1)
        filtered.append(joined)
        
ax.axis('equal')

# %%


# plt.hist(error_list,bins=100)
# %%
filtered = []
for i_ in range(len(to_be_legit_clusters)):
    joined = np.vstack([segments[i] for i in to_be_legit_clusters[i_]])
    joined = sort_curve(joined)
    if (seg_len(joined) > 100):
        filtered.append(joined)
        
# %%
len(filtered)
# %%
# second round
import time
start = time.time()
fp2 = filamentprocessing.FilamentProcessing(filtered,200,1,0.99)
print(f'Elapsed time: {time.time()-start}')

ij = fp2.get_svd_ij()
scores = fp2.get_svd_scores()
ij = np.array(ij)
scores = np.array(scores)

svd_cylinders = fp2.get_svd_cylinders()
svd_cylinders = np.array(svd_cylinders)
svd_cylinders.shape
# %%



dist_score = scores[:,0]
align_score = scores[:,1]

mask = (dist_score < 30) & (align_score < 0.1)

graph = nx.Graph()
graph.add_nodes_from(range(len(filtered)))
graph.add_edges_from(ij[mask,:])

connected_components = list(nx.connected_components(graph))
connected_components = [list(x) for x in connected_components]
cluster_size_list = [len(x) for x in connected_components]
print(f'Number of segments: {len(filtered)}')
print(f'Number of connected components: {len(connected_components)}')
print(f'Max. cluster size {np.max(cluster_size_list)} at {np.argmax(cluster_size_list)}')

# %%

length_list = []
error_list = []
for i_ in range(len(connected_components)):
    joined = np.vstack([filtered[i] for i in connected_components[i_]])
    joined = sort_curve(joined)
    length_list.append(seg_len(joined))
    
    fit_result = fit_rod(joined,0.00001,10000)
    error_list.append(fit_result['err'])
    
error_list = np.array(error_list)
log_bins = np.logspace(0,3,100)
# %%
plt.close('all')
fig,ax=plt.subplots(1,1)
ax.hist(error_list,bins=log_bins)
ax.set_xscale('log')
# %%
plt.close('all')
fig,ax=plt.subplots(1,1)
ax.hist(length_list,bins=log_bins)
ax.set_xscale('log')

# %%

i_max = np.argmax(cluster_size_list)
# i_max = np.argmax(error_list)
error_list[i_max]

joined = np.vstack([filtered[i] for i in connected_components[i_max]])
joined = sort_curve(joined)

fit_result = fit_rod(joined,0.00001,10000)
print(fit_result['err'])

cc_max = connected_components[i_max]

# %%
plt.close('all')
fig,ax=plt.subplots(1,1,figsize=(10,10),subplot_kw={'projection':'3d'})
for i_ in cc_max:
    ax.plot(filtered[i_][:,0],filtered[i_][:,1],filtered[i_][:,2])
    
# %%
# unclustered
unclustered = []
for i_ in range(len(connected_components)):
    cc = connected_components[i_]
    if len(cc) < 2:
        unclustered.append(cc)
        
# %%
plt.close('all')
fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
for i_ in unclustered:
    joined = np.vstack([filtered[i] for i in i_])
    joined = sort_curve(joined)
    
    if np.all(np.linalg.norm(joined[0,:2] - [1000,1000]) > 900):
    # if np.any(np.linalg.norm(joined - [500,1000,500],axis=1) < 100):
        ax.plot(joined[:,0],joined[:,1],joined[:,2],linewidth=1)
    
# %%



@jax.jit
def calculate_distances(cyl_i, cyl_j):
    d1 = jnp.linalg.norm(cyl_i[:3] - cyl_j[:3])
    d2 = jnp.linalg.norm(cyl_i[3:6] - cyl_j[3:6])
    d3 = jnp.linalg.norm(cyl_i[:3] - cyl_j[3:6])
    d4 = jnp.linalg.norm(cyl_i[3:6] - cyl_j[:3])
    return jnp.min(jnp.array([d1, d2, d3, d4]))

@jax.jit
def calculate_e2e_dist_mat(svd_cylinders):
    n = len(svd_cylinders)
    indices = jnp.tril_indices(n, -1)
    distances = jax.vmap(lambda i, j: calculate_distances(svd_cylinders[i], svd_cylinders[j]))(indices[0], indices[1])
    e2e_distance = jnp.zeros((n, n))
    e2e_distance = e2e_distance.at[indices].set(distances)
    return e2e_distance
# %%
@jax.jit
def calculate_alignment(cyl_i, cyl_j):
    # d1 = jnp.linalg.norm(cyl_i[:3] - cyl_j[:3])
    # d2 = jnp.linalg.norm(cyl_i[3:6] - cyl_j[3:6])
    # d3 = jnp.linalg.norm(cyl_i[:3] - cyl_j[3:6])
    # d4 = jnp.linalg.norm(cyl_i[3:6] - cyl_j[:3])
    
    # # dvec is the vector connecting the two closest points
    # i_min = jnp.argmin(jnp.array([d1, d2, d3, d4]))
    
    # dvec = jax.lax.switch(i_min, [
    #     lambda: cyl_i[:3] - cyl_j[:3],
    #     lambda: cyl_i[3:6] - cyl_j[3:6],
    #     lambda: cyl_i[:3] - cyl_j[3:6],
    #     lambda: cyl_i[3:6] - cyl_j[:3]
    # ])
    cen_i = (cyl_i[:3] + cyl_i[3:6]) / 2
    cen_j = (cyl_j[:3] + cyl_j[3:6]) / 2
    dvec = cen_i - cen_j
    dvec = dvec / jnp.linalg.norm(dvec)
    
    ori_i = cyl_i[3:6] - cyl_i[:3]
    ori_j = cyl_j[3:6] - cyl_j[:3]
    
    ori_i = ori_i / jnp.linalg.norm(ori_i)
    ori_j = ori_j / jnp.linalg.norm(ori_j)    
    
    alignment = (jnp.linalg.norm(jnp.cross(dvec, ori_i)) + jnp.linalg.norm(jnp.cross(dvec, ori_j))) / 2
    
    return alignment

@jax.jit
def calculate_alignment_dist_mat(svd_cylinders):
    n = len(svd_cylinders)
    indices = jnp.tril_indices(n, -1)
    alignments = jax.vmap(lambda i, j: calculate_alignment(svd_cylinders[i], svd_cylinders[j]))(indices[0], indices[1])
    alignment_matrix = jnp.zeros((n, n))
    alignment_matrix = alignment_matrix.at[indices].set(alignments)
    return alignment_matrix


# %%
e2e_alignment = calculate_alignment_dist_mat(svd_cylinders)
e2e_distance = calculate_e2e_dist_mat(svd_cylinders)
# %%
e2e_distance = np.array(e2e_distance)
e2e_alignment = np.array(e2e_alignment)

e2e_distance = e2e_distance + e2e_distance.T
e2e_alignment = e2e_alignment + e2e_alignment.T

e2e_distance[np.diag_indices(len(svd_cylinders))] = np.inf
e2e_alignment[np.diag_indices(len(svd_cylinders))] = np.inf
# %%
i_ = 421
e2e_neighbors = np.where(e2e_distance[i_] < 10)[0]
fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
ax.plot(filtered[i_][:,0],filtered[i_][:,1],filtered[i_][:,2],linewidth=1)
for nb in e2e_neighbors:
    ax.plot(filtered[nb][:,0],filtered[nb][:,1],filtered[nb][:,2],linewidth=1)

ax.axis('equal')

# %%
mask = (e2e_distance < 15) & (e2e_alignment < 0.03)
# fill inf to diag
np.count_nonzero(mask)

# %%
e2e_graph = nx.Graph()
e2e_graph.add_nodes_from(range(len(filtered)))
e2e_graph.add_edges_from(np.array(np.where(mask)).T)
e2e_clusters = list(nx.connected_components(e2e_graph))
length_list = []
error_list = []
for i_ in e2e_clusters:
    joined = np.vstack([filtered[i] for i in i_])
    joined = sort_curve(joined)    
    length_list.append(seg_len(joined))
    fit_result = fit_rod(joined,0.00001,10000)
    error_list.append(fit_result['err'])
    
np.count_nonzero(np.array(length_list) > 600)
# %%
fig,ax=plt.subplots(1,1)
ax.hist(length_list,bins=100)

# %%
log_bins = np.logspace(2,np.log(800)/np.log(10),100)
plt.close('all')
fig,ax=plt.subplots(1,1)
ax.hist(length_list,bins=log_bins)
ax.set_xscale('log')
# %%

# %%
i_=np.argmax(error_list)
i_=np.argsort(length_list)[-3]
rr = np.vstack([filtered[i] for i in e2e_clusters[i_]])
rr = sort_curve(rr)
fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
ax.plot(rr[:,0],rr[:,1],rr[:,2],linewidth=1)
ax.axis('equal')
print(length_list[i_])

# %%
log_bins = np.logspace(0,2,100)
plt.close('all')
fig,ax=plt.subplots(1,1)
ax.hist(error_list,bins=log_bins)
ax.set_xscale('log')
# %%
np.count_nonzero(np.array(length_list) > 600)

# %%
plt.close('all')
fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
for i_,cc in enumerate(e2e_clusters):
    if length_list[i_] < 600:
        continue
    
    rr = np.vstack([filtered[i] for i in cc])
    rr = sort_curve(rr)
    ax.plot(rr[:,0],rr[:,1],rr[:,2],linewidth=1)
    
# %%
fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
for i_,cc in enumerate(e2e_clusters):
    if len(cc) > 1:
        continue
    
    if length_list[i_] > 300:
        continue
    
    rr = np.vstack([filtered[i] for i in cc])
    ax.plot(rr[:,0],rr[:,1],rr[:,2],linewidth=1)
    
# %%
legit_clusters2 = []
to_be_legit_clusters2 = []


clustered_cc = []
for i_,cc in enumerate(e2e_clusters):
    if length_list[i_] < 600:
        continue
    
    if length_list[i_] > 800:
        continue
    
    if error_list[i_] > 10:
        continue
    
    legit_clusters2.append(cc)
    # append all cc to clustered_cc
    clustered_cc.extend(cc)
    
# %%
error_list = []
length_list = []

fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
for i_,cc in enumerate(legit_clusters2):
    joined = np.vstack([filtered[i] for i in cc])
    joined = sort_curve(joined)
    ax.plot(joined[:,0],joined[:,1],joined[:,2],linewidth=1)
    
    length_list.append(seg_len(joined))
    fit_result = fit_rod(joined,0.00001,10000)
    error_list.append(fit_result['err'])
# %%
plt.close('all')
fig,ax=plt.subplots(1,1)
# ax.hist(length_list)
ax.hist(error_list,bins=100)
# %%
i_max = np.argmax(error_list)
# i_max = np.argmax(length_list)

cc = legit_clusters2[i_max]
joined = np.vstack([filtered[i] for i in cc])
joined = sort_curve(joined)
plt.close('all')
fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
ax.plot(joined[:,0],joined[:,1],joined[:,2],linewidth=1)
ax.axis('equal')

    
# %%
np.unique(clustered_cc).shape
len(clustered_cc)
# %%
unclustered = np.setdiff1d(np.arange(len(filtered)),clustered_cc)
# %%
fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})

for i_ in np.random.choice(len(unclustered),100):#range(len(unclustered)):    
    rr = filtered[unclustered[i_]]    
    ax.plot(rr[:,0],rr[:,1],rr[:,2],linewidth=1)
    
# %%
len(unclustered)

# %%
# round 3
filtered2 = []
for i_ in unclustered:
    rr = filtered[i_]
    filtered2.append(rr)
    
# %%
error_list = []
length_list = []
for i_ in range(len(filtered2)):
    joined = filtered2[i_]
    joined = sort_curve(joined)
    length_list.append(seg_len(joined))
    fit_result = fit_rod(joined,0.00001,10000)
    error_list.append(fit_result['err'])
# %%
np.argmax(error_list)
    
# %%
fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
for i_ in np.random.choice(len(unclustered),300):
    rr = filtered2[i_]
    ax.plot(rr[:,0],rr[:,1],rr[:,2],linewidth=1)
    
# %%
len(filtered2)

# %%
import time
start = time.time()
fp2 = filamentprocessing.FilamentProcessing(filtered2,200,1,0.99)
print(f'Elapsed time: {time.time()-start}')

ij = fp2.get_svd_ij()
scores = fp2.get_svd_scores()
ij = np.array(ij)
scores = np.array(scores)

svd_cylinders = fp2.get_svd_cylinders()
svd_cylinders = np.array(svd_cylinders)
svd_cylinders.shape
# %%
e2e_alignment = calculate_alignment_dist_mat(svd_cylinders)
e2e_distance = calculate_e2e_dist_mat(svd_cylinders)
e2e_distance = np.array(e2e_distance)
e2e_alignment = np.array(e2e_alignment)
e2e_distance = e2e_distance + e2e_distance.T
e2e_alignment = e2e_alignment + e2e_alignment.T
e2e_distance[np.diag_indices(len(svd_cylinders))] = np.inf
e2e_alignment[np.diag_indices(len(svd_cylinders))] = np.inf
# %%
i_ = 30
e2e_neighbors = np.where(e2e_distance[i_] < 10)[0]
plt.close('all')
fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
ax.plot(filtered2[i_][:,0],filtered2[i_][:,1],filtered2[i_][:,2],linewidth=1)
for nb in e2e_neighbors:
    ax.plot(filtered2[nb][:,0],filtered2[nb][:,1],filtered2[nb][:,2],linewidth=1)
ax.axis('equal')

# %%
e2e_alignment[i_,851]
    
# %%



# Example Nx3 numpy array
# N = 100  # Example number of points
# t = np.linspace(0, 1, N)
# x = np.sin(2 * np.pi * t)  # Example x-coordinates
# y = np.cos(2 * np.pi * t)  # Example y-coordinates
# z = t                      # Example z-coordinates

# points = np.vstack((x, y, z)).T
points = filtered2[i_]
t = np.linspace(0, 1, len(points))

# Perform cubic spline interpolation (Bezier splines are typically piecewise cubic)
spl = make_interp_spline(t, points, k=3)

# Generate new points on the smooth curve
unew = np.linspace(0, 1, 1000)  # 1000 points for smoothness
smooth_points = spl(unew)

# Plot the original and smooth curves
plt.close('all')
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(points[:, 0], points[:, 1], points[:, 2], 'r.', label='Original Curve')
ax.plot(smooth_points[:, 0], smooth_points[:, 1], smooth_points[:, 2], 'b-', label='Smooth Curve')
ax.legend()
plt.show()


# %%
# Function to compute Bezier curve points
def bezier_curve(control_points, n_points=1000):
    n = len(control_points) - 1
    t = np.linspace(0, 1, n_points)
    curve = np.zeros((n_points, 3))
    for i in range(n + 1):
        binomial_coeff = comb(n, i)
        curve += np.outer(binomial_coeff * (t ** i) * ((1 - t) ** (n - i)), control_points[i])
    return curve

# Error function to minimize
def error_function(control_points_flat, points, n_control_points):
    control_points = control_points_flat.reshape((n_control_points, 3))
    bezier_points = bezier_curve(control_points, len(points))
    error = np.sum(np.linalg.norm(points - bezier_points, axis=1)**2)
    return error

# Example Nx3 numpy array (use your own data here)
i_ = 3103
points = filtered2[i_]
t = np.linspace(0, 1, len(points))
N = len(points)

# Function to get initial guess for control points
def initial_control_points(points, n_control_points):
    indices = np.linspace(0, len(points) - 1, n_control_points).astype(int)
    return points[indices]

# Parameterize number of anchor points
n_control_points = 10  # Change this to any number of control points you want

# Initial guess for control points
initial_guess = initial_control_points(points, n_control_points)
initial_guess_flat = initial_guess.flatten()

# Perform optimization
result = minimize(error_function, initial_guess_flat, args=(points, n_control_points), method='L-BFGS-B')
optimized_control_points = result.x.reshape((n_control_points, 3))

# Generate the optimized Bezier curve
smooth_points = bezier_curve(optimized_control_points, len(points))



plt.close('all')
# Plot the original points and the optimized Bezier curve
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(points[:, 0], points[:, 1], points[:, 2], 'r.', label='Original Curve')
ax.plot(smooth_points[:, 0], smooth_points[:, 1], smooth_points[:, 2], 'b-', label='Optimized Bezier Curve')
ax.scatter(optimized_control_points[:, 0], optimized_control_points[:, 1], optimized_control_points[:, 2], color='g', label='Optimized Control Points')
ax.legend()
ax.axis('equal')
plt.show()
# %%
def optimal_bezier_curve(points, n_control_points=4):
    def bezier_curve(control_points, n_points=1000):
        n = len(control_points) - 1
        t = np.linspace(0, 1, n_points)
        curve = np.zeros((n_points, 3))
        for i in range(n + 1):
            binomial_coeff = comb(n, i)
            curve += np.outer(binomial_coeff * (t ** i) * ((1 - t) ** (n - i)), control_points[i])
        return curve

    def error_function(control_points_flat, points, n_control_points):
        control_points = control_points_flat.reshape((n_control_points, 3))
        bezier_points = bezier_curve(control_points, len(points))
        error = np.sum(np.linalg.norm(points - bezier_points, axis=1)**2)
        return error

    def initial_control_points(points, n_control_points):
        indices = np.linspace(0, len(points) - 1, n_control_points).astype(int)
        return points[indices]

    initial_guess = initial_control_points(points, n_control_points)
    initial_guess_flat = initial_guess.flatten()

    result = minimize(error_function, initial_guess_flat, args=(points, n_control_points), method='L-BFGS-B')
    optimized_control_points = result.x.reshape((n_control_points, 3))

    smooth_points = bezier_curve(optimized_control_points, len(points))
    return smooth_points,optimized_control_points
# %%
obc1,ocp1 = optimal_bezier_curve(filtered2[30],n_control_points=10)
obc2,ocp2 = optimal_bezier_curve(filtered2[851],n_control_points=10)

plt.close('all')
fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
ax.plot(filtered2[30][:,0],filtered2[30][:,1],filtered2[30][:,2],linewidth=1)
ax.plot(obc1[:,0],obc1[:,1],obc1[:,2],linewidth=1)
ax.plot(filtered2[851][:,0],filtered2[851][:,1],filtered2[851][:,2],linewidth=1)
ax.plot(obc2[:,0],obc2[:,1],obc2[:,2],linewidth=1)
# %%
# test colliniearity and continuity


# Function to compute Bezier curve points
def bezier_curve(control_points, n_points=1000):
    n = len(control_points) - 1
    t = np.linspace(0, 1, n_points)
    curve = np.zeros((n_points, 3))
    for i in range(n + 1):
        binomial_coeff = comb(n, i)
        curve += np.outer(binomial_coeff * (t ** i) * ((1 - t) ** (n - i)), control_points[i])
    return curve

# Function to compute the derivative of the Bezier curve
def bezier_curve_derivative(control_points, n_points=1000):
    n = len(control_points) - 1
    t = np.linspace(0, 1, n_points)
    derivative = np.zeros((n_points, 3))
    for i in range(n):
        binomial_coeff = comb(n-1, i)
        derivative += np.outer(binomial_coeff * (t ** i) * ((1 - t) ** (n-1-i)) * n, (control_points[i+1] - control_points[i]))
    return derivative

# Function to check if two Bezier curves form a continuous and smooth curve
def check_bezier_continuity(control_points1, control_points2, tolerance=1e-3):
    # Sample points on both Bezier curves
    curve1 = bezier_curve(control_points1)
    curve2 = bezier_curve(control_points2)
    
    # Compute pairwise distances between the points on both curves
    distances = cdist(curve1, curve2)
    
    # Find the indices of the closest points
    min_idx1, min_idx2 = np.unravel_index(np.argmin(distances), distances.shape)
    
    # Check if the closest points are within the tolerance
    if distances[min_idx1, min_idx2] > tolerance:
        return False
    
    # Compute the derivatives (tangents) at the closest points
    derivative1 = bezier_curve_derivative(control_points1)
    derivative2 = bezier_curve_derivative(control_points2)
    
    tangent1 = derivative1[min_idx1]
    tangent2 = derivative2[min_idx2]
    
    # Normalize tangents
    tangent1 /= np.linalg.norm(tangent1)
    tangent2 /= np.linalg.norm(tangent2)
    
    # Check if the tangents are nearly parallel (dot product close to 1 or -1)
    dot_product = np.dot(tangent1, tangent2)
    if np.abs(dot_product) > 1 - tolerance:
        return True
    return False

# %%




# Example control points for two Bezier curves
control_points1 = np.array([
    [0, 0, 0],
    [1, 2, 1],
    [2, 3, 2],
    [3, 4, 3]
])

control_points2 = np.array([
    [3, 4, 3.0001],  # Slightly offset point for testing
    [4, 5, 4],
    [5, 6, 5],
    [6, 7, 6]
])

# Check if the two Bezier curves form a continuous and smooth curve
are_continuous = check_bezier_continuity(control_points1, control_points2)
print(f"Do the two Bezier curves form a continuous and smooth curve? {are_continuous}")

bezier_curve1 = bezier_curve(control_points1)
bezier_curve2 = bezier_curve(control_points2)

fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
ax.plot(bezier_curve1[:,0],bezier_curve1[:,1],bezier_curve1[:,2],linewidth=1)
ax.plot(bezier_curve2[:,0],bezier_curve2[:,1],bezier_curve2[:,2],linewidth=1)






# %%
smooth_filtered2 = []
for i_ in range(len(filtered2)):
    points = filtered2[i_]
    t = np.linspace(0, 1, len(points))
    N = len(points)
    initial_guess = np.array([
        points[0],
        points[N//3],
        points[2*N//3],
        points[-1]
    ])
    initial_guess_flat = initial_guess.flatten()
    result = minimize(error_function, initial_guess_flat, args=(points), method='L-BFGS-B')
    optimized_control_points = result.x.reshape((4, 3))
    smooth_points = bezier_curve(optimized_control_points, len(points))    
    smooth_filtered2.append(smooth_points)


# %%
fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
for i_ in np.random.choice(len(filtered2),100):
    ax.plot(smooth_filtered2[i_][:,0],smooth_filtered2[i_][:,1],smooth_filtered2[i_][:,2],linewidth=1)
    
# %%





        
    
# %%
for i_ in cc_max:
    cen_i = np.mean(filtered[i_],axis=0)
    
    for j_ in cc_max:
        if j_ <= i_:
            continue
        
        cen_j = np.mean(filtered[j_],axis=0)
        if graph.has_edge(i_,j_):
            # curve connecting cen_i and cen_j
            
            random_offset = np.random.rand(3) * 10
            # cen_i to cen_i + offset
            ax.plot([cen_i[0],cen_i[0]+random_offset[0]],[cen_i[1],cen_i[1]+random_offset[1]],[cen_i[2],cen_i[2]+random_offset[2]],'k-',linewidth=0.5)
            # cen_j to cen_j + offset
            ax.plot([cen_j[0],cen_j[0]+random_offset[0]],[cen_j[1],cen_j[1]+random_offset[1]],[cen_j[2],cen_j[2]+random_offset[2]],'k-',linewidth=0.5)
            # cen_i + offset to cen_j + offset
            ax.plot([cen_i[0]+random_offset[0],cen_j[0]+random_offset[0]],[cen_i[1]+random_offset[1],cen_j[1]+random_offset[1]],[cen_i[2]+random_offset[2],cen_j[2]+random_offset[2]],'k-',linewidth=0.5)
            
            
# %%
