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
# %%
@jax.jit
def calculate_euclidean_distances(point1, point2):    
    return jnp.linalg.norm(point1 - point2)

@jax.jit
def calculate_dist_mat(points):
    n = points.shape[0]
    indices = jnp.tril_indices(n, -1)
    distances = jax.vmap(lambda i, j: calculate_euclidean_distances(points[i], points[j]))(indices[0], indices[1])
    dist_mat = jnp.zeros((n, n))
    dist_mat = dist_mat.at[indices].set(distances)
    return dist_mat

def pairwise_alignment_for_ends(p_i, p_j, tan_i, tan_j):
    dvec = p_j - p_i
    dvec = dvec / jnp.linalg.norm(dvec)
    # i to j
    
    alignment = (jnp.linalg.norm(jnp.cross(dvec, tan_i)) + jnp.linalg.norm(jnp.cross(dvec, tan_j))) / 2
    
    return alignment
# %%
def quick_tangent(rr):
    cen = np.mean(rr,axis=0)
    rr_centered = rr - cen
    _,_,V = np.linalg.svd(rr_centered,full_matrices=False)
    v1 = V[0,:]
    
    return v1
    
    
# %%
import jax.numpy as jnp
from jax import vmap, jit, lax

def prune_mst(mst):
    import heapq

    # Track the degree of each node
    node_degrees = {node: 0 for node in mst.nodes}
    pruned_edges = []
    mandatory_edges = []
    added_edges = set()

    # Enforce the mandatory connections for even i
    for i in range(0, max(mst.nodes), 2):
        if i in mst.nodes and i+1 in mst.nodes:
            if mst.has_edge(i, i+1):
                mandatory_edges.append((i, i+1, mst[i][i+1]['weight']))
                node_degrees[i] += 1
                node_degrees[i+1] += 1
                added_edges.add((i, i+1))

    # Priority queue for edges sorted by weight
    edge_queue = []
    for u, v, data in mst.edges(data=True):
        if (u, v) not in added_edges and (v, u) not in added_edges:
            heapq.heappush(edge_queue, (data['weight'], u, v))

    # Union-Find data structure for tracking connected components
    parent = {node: node for node in mst.nodes}

    def find(node):
        if parent[node] != node:
            parent[node] = find(parent[node])
        return parent[node]

    def union(node1, node2):
        root1 = find(node1)
        root2 = find(node2)
        if root1 != root2:
            parent[root2] = root1

    # Add mandatory edges to pruned graph
    for u, v, weight in mandatory_edges:
        pruned_edges.append((u, v, {'weight': weight}))
        union(u, v)

    # Add remaining edges while respecting degree constraints and maintaining connectivity
    while edge_queue:
        weight, u, v = heapq.heappop(edge_queue)
        if node_degrees[u] < 2 and node_degrees[v] < 2 and find(u) != find(v):
            pruned_edges.append((u, v, {'weight': weight}))
            node_degrees[u] += 1
            node_degrees[v] += 1
            union(u, v)

    # Create a new graph from the pruned edges
    pruned_graph = nx.Graph()
    pruned_graph.add_nodes_from(mst.nodes)
    pruned_graph.add_edges_from((u, v, data) for u, v, data in pruned_edges)
    
    return pruned_graph




def plot_segments_with_connectivity(segments, graph):
        
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
                
                
            

def pairwise_alignment(p_i, p_j, tan_i, tan_j):
    tan_i1 = tan_i[:3]
    tan_i2 = tan_i[3:]
    tan_j1 = tan_j[:3]
    tan_j2 = tan_j[3:]
    
    p_i1 = p_i[:3]
    p_i2 = p_i[3:]
    p_j1 = p_j[:3]
    p_j2 = p_j[3:]
    
    d1 = jnp.linalg.norm(p_i1 - p_j1)
    d2 = jnp.linalg.norm(p_i2 - p_j2)
    d3 = jnp.linalg.norm(p_i1 - p_j2)
    d4 = jnp.linalg.norm(p_i2 - p_j1)
    
    dvec = jnp.array([d1, d2, d3, d4])    
    i_min = jnp.argmin(dvec)
    
    dvec = jnp.where(i_min == 0, p_i1 - p_j1,
                     jnp.where(i_min == 1, p_i2 - p_j2,
                               jnp.where(i_min == 2, p_i1 - p_j2,
                                         p_i2 - p_j1)))
    tan_i = jnp.where(i_min == 0, tan_i1,
                      jnp.where(i_min == 1, tan_i2,
                                jnp.where(i_min == 2, tan_i1,
                                          tan_i2)))
    tan_j = jnp.where(i_min == 0, tan_j1,
                      jnp.where(i_min == 1, tan_j2,
                                jnp.where(i_min == 2, tan_j2,
                                          tan_j1)))
    
    dvec = dvec / jnp.linalg.norm(dvec)
    alignment = (jnp.linalg.norm(jnp.cross(dvec, tan_i)) + jnp.linalg.norm(jnp.cross(dvec, tan_j))) / 2
    return alignment



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

def compute_tangent_alignment(end_points, end_tangents):
    def pairwise_alignment(i_, j_):
        tan_i1 = end_tangents[i_, :3]
        tan_i2 = end_tangents[i_, 3:]
        
        p_i1 = end_points[i_][:3]
        p_i2 = end_points[i_][3:]
        
        tan_j1 = end_tangents[j_, :3]
        tan_j2 = end_tangents[j_, 3:]
        
        p_j1 = end_points[j_][:3]
        p_j2 = end_points[j_][3:]
        
        d1 = jnp.linalg.norm(p_i1 - p_j1)
        d2 = jnp.linalg.norm(p_i2 - p_j2)
        d3 = jnp.linalg.norm(p_i1 - p_j2)
        d4 = jnp.linalg.norm(p_i2 - p_j1)
        
        dvec = jnp.array([d1, d2, d3, d4])
        
        i_min = jnp.argmin(dvec)
        
        dvec, tan_i, tan_j = jnp.where(i_min == 0, (p_i1 - p_j1, tan_i1, tan_j1), 
                                       jnp.where(i_min == 1, (p_i2 - p_j2, tan_i2, tan_j2), 
                                                 jnp.where(i_min == 2, (p_i1 - p_j2, tan_i1, tan_j2), 
                                                           (p_i2 - p_j1, tan_i2, tan_j1))))
        
        dvec = dvec / jnp.linalg.norm(dvec)
        alignment = (jnp.linalg.norm(jnp.cross(dvec, tan_i)) + jnp.linalg.norm(jnp.cross(dvec, tan_j))) / 2
        return alignment
    
    indices = jnp.arange(len(end_points))
    compute_for_i = vmap(lambda i_: vmap(lambda j_: pairwise_alignment(i_, j_))(
        lax.dynamic_slice(indices, (i_ + 1,), (len(indices) - i_ - 1,))
    ))(indices[:-1])
    
    end_tangent_alignment = jnp.zeros((len(end_points), len(end_points)))
    end_tangent_alignment = lax.dynamic_update_slice(end_tangent_alignment, compute_for_i, (0, 1))
    
    # Make the matrix symmetric
    end_tangent_alignment = end_tangent_alignment + end_tangent_alignment.T
    
    return end_tangent_alignment



# %%
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
len(segments)

# from scipy.io import loadmat
# dataobj = loadmat('/Users/yeonsu/Dropbox (Harvard University)/Data/steel-rods-xray-data/alpha200_epsilon00/segments.mat')
# segments = dataobj['segments']
# segments = [np.array(seg[0]) for seg in segments]


    
    
    

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
    fp = filamentprocessing.FilamentProcessing(segments,50,1,0.99)
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
# fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
# for i in range(len(svd_cylinders)):
#     p1 = svd_cylinders[i,:3]
#     p2 = svd_cylinders[i,3:6]
#     ax.plot([p1[0],p2[0]],[p1[1],p2[1]],[p1[2],p2[2]])


# from visualizations import plot_centerline_with_container
# fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
# for i in range(len(svd_cylinders)):
#     plot_centerline_with_container(segments,svd_cylinders,i,ax)
# %%

# %%
# plt.close('all')
# fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
# for i in range(len(local_segments)):
#     ax.plot(local_segments[i][:,0],local_segments[i][:,1],local_segments[i][:,2],'-')
        
# %%



# %%
import filamentprocessing
fp = filamentprocessing.FilamentProcessing(segments,50,1,0.99)
# %%
fp.calculate_end_to_end_properties(10)
endpoints = fp.get_end_points()
endtangents = fp.get_end_tangents()
    
    
# %%
fp.calculate_end_to_end_scores(20)
end_ab = fp.get_end_ab()
end_scores = fp.get_end_scores()

# %%
end_ab = np.array(end_ab)
end_scores = np.array(end_scores)
dist_score = end_scores[:,0]
align_score = end_scores[:,1]
mask = (dist_score < 10) & (align_score < 0.15)
weighted_edges = [(i, j, align_score[k]) for k, (i, j) in enumerate(end_ab) if mask[k]] 

e2e_graph = nx.Graph()
e2e_graph.add_nodes_from(range(len(segments)*2))
e2e_graph.add_weighted_edges_from(weighted_edges)
# %%
mst = nx.minimum_spanning_tree(e2e_graph)
pruned_graph = prune_mst(mst)
conn_comp = list(nx.connected_components(pruned_graph))

cluster_size_list = [len(x) for x in conn_comp]
print(f'Number of end points: {len(segments)*2}')
print(f'Number of connected components: {len(conn_comp)}')
print(f'Max. cluster size {np.max(cluster_size_list)} at {np.argmax(cluster_size_list)}')

# %%
# i_max = np.argmax(cluster_size_list)
i_max = np.argsort(cluster_size_list)[-512]
cc_max = conn_comp[i_max]
plt.close('all')
fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
for i in cc_max:
    if (i % 2 == 1):
        continue    
    rr = segments[i//2]
    ax.plot(rr[:,0],rr[:,1],rr[:,2],'-')
ax.axis('equal')
# %%

round2 = []
for cc in conn_comp:
    cc = list(cc)
    subgraph = pruned_graph.subgraph(cc)
    eps = [node for node in subgraph.nodes if subgraph.degree[node] == 1]

    if len(eps) != 2:
        continue
        # raise ValueError("The graph does not have exactly two endpoints.")
    

    # Find the shortest path between the two endpoints
    path = nx.shortest_path(subgraph, source=eps[0], target=eps[1])    
    straight_curve = []
    for i_ in path[::2]:
        if i_ % 2 == 0:
            straight_curve.append(segments[i_//2])
        elif i_ % 2 == 1: 
            straight_curve.append(segments[i_//2][::-1])            
    straight_curve = np.vstack(straight_curve)
    round2.append(straight_curve)
    
# %%
plt.close('all')
length_list = []
for rr in round2:
    length_list.append(seg_len(rr))
    
# %%
log_bins = np.logspace(np.log10(1),np.log10(1000),100)
plt.close('all')
fig,ax=plt.subplots(1,1)
ax.hist(length_list,bins=log_bins)
ax.set_xscale('log')
# %%
# plt.close('all')
# fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
# for rr in round2:
#     if seg_len(rr) > 10:
#         continue
#     ax.plot(rr[:,0],rr[:,1],rr[:,2],'-')
    
# %%
np.count_nonzero(np.array(length_list) < 10)

# %%
round2 = [rr for rr in round2 if seg_len(rr) > 10]
# %%
len(segments)
len(round2)

# %%

fp2 = filamentprocessing.FilamentProcessing(round2,50,1,0.99)
fp2.calculate_end_to_end_properties(30)
endpoints = fp2.get_end_points()
endtangents = fp2.get_end_tangents()

# %%
scale = 10
# plt.close('all')
# fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
# for i in range(len(round2)):
#     rr = round2[i]
#     if np.any(np.linalg.norm(rr - np.array([700,700,500]),axis=1) < 200):
#         ax.plot(rr[:,0],rr[:,1],rr[:,2],'-')
    
    
    
# %%
dist_threshold = 30
fp2.calculate_end_to_end_scores(dist_threshold)
end_ab = fp2.get_end_ab()
end_scores = fp2.get_end_scores()

end_ab = np.array(end_ab)
end_scores = np.array(end_scores)
dist_score = end_scores[:,0]
align_score = end_scores[:,1]
mask = (dist_score < dist_threshold) & (align_score < 0.15)
weighted_edges = [(i, j, align_score[k]) for k, (i, j) in enumerate(end_ab) if mask[k]] 

e2e_graph = nx.Graph()
e2e_graph.add_nodes_from(range(len(round2)*2))
e2e_graph.add_weighted_edges_from(weighted_edges)
mst = nx.minimum_spanning_tree(e2e_graph)
pruned_graph = prune_mst(mst)
conn_comp = list(nx.connected_components(pruned_graph))

cluster_size_list = [len(x) for x in conn_comp]
print(f'Number of end points: {len(round2)*2}')
print(f'Number of connected components: {len(conn_comp)}')
print(f'Max. cluster size {np.max(cluster_size_list)} at {np.argmax(cluster_size_list)}')

# %%
# i_max = np.argmax(cluster_size_list)
i_max = np.argsort(cluster_size_list)[-1]
cc_max = conn_comp[i_max]
plt.close('all')
fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
for i in cc_max:
    if (i % 2 == 1):
        continue    
    rr = round2[i//2]
    ax.plot(rr[:,0],rr[:,1],rr[:,2],'-')
ax.axis('equal')

# %%

round3 = []
for cc in conn_comp:
    cc = list(cc)
    subgraph = pruned_graph.subgraph(cc)
    eps = [node for node in subgraph.nodes if subgraph.degree[node] == 1]

    if len(eps) != 2:
        continue
        # raise ValueError("The graph does not have exactly two endpoints.")
    

    # Find the shortest path between the two endpoints
    path = nx.shortest_path(subgraph, source=eps[0], target=eps[1])    
    straight_curve = []
    for i_ in path[::2]:
        if i_ % 2 == 0:
            straight_curve.append(round2[i_//2])
        elif i_ % 2 == 1: 
            straight_curve.append(round2[i_//2][::-1])            
    straight_curve = np.vstack(straight_curve)
    round3.append(straight_curve)
# %%
length_list = [seg_len(rr) for rr in round3]

log_bins = np.logspace(np.log10(1),np.log10(2000),100)
plt.close('all')
fig,ax=plt.subplots(1,1)
ax.hist(length_list,bins=log_bins)
ax.set_xscale('log')
# %%
np.count_nonzero(np.array(length_list) > 30)

plt.close('all')
fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
for i in range(len(round3)):
    if length_list[i] < 30:
        ax.plot(round3[i][:,0],round3[i][:,1],round3[i][:,2],'-')
        

# %%
plt.close('all')
fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
for i in np.random.choice(len(round3),1000):
    rr = round3[i]
    if seg_len(rr) < 30:
        continue
    
    
    ax.plot(rr[:,0],rr[:,1],rr[:,2],'-')
    

# %%
class Segments:
    def __init__(self,segments):
        import filamentprocessing
        import networkx as nx
        
        self.segments = segments
        self.fp = filamentprocessing.FilamentProcessing(segments,50,1,0.99)
        
    def calculate_end_to_end_properties(self,dist_threshold):
        self.fp.calculate_end_to_end_properties(dist_threshold)
        
    def get_end_points(self):
        return self.fp.get_end_points()
    
    def get_end_tangents(self):
        return self.fp.get_end_tangents()
    
    def calculate_end_to_end_scores(self,dist_threshold):
        self.fp.calculate_end_to_end_scores(dist_threshold)
        
    def get_end_ab(self):
        return self.fp.get_end_ab()
    
    def get_end_scores(self):
        return self.fp.get_end_scores()
    
    def get_svd_ij(self):
        return self.fp.get_svd_ij()
    
    def get_svd_scores(self):
        return self.fp.get_svd_scores()
    
    def get_svd_cylinders(self):
        return self.fp.get_svd_cylinders()
    
    def clustering(self,number_of_endpoint_averaging=10,dist_threshold=30,align_threshold=0.15):
        self.fp.calculate_end_to_end_properties(number_of_endpoint_averaging)
        self.endpoints = self.get_end_points()
        self.endtangents = self.get_end_tangents()
        
        self.fp.calculate_end_to_end_scores(dist_threshold)
        self.end_ab = self.get_end_ab()
        self.end_scores = self.get_end_scores()
        
        self.end_ab = np.array(self.end_ab)
        self.end_scores = np.array(self.end_scores)
        
        dist_score = self.end_scores[:,0]
        align_score = self.end_scores[:,1]
        
        mask = (dist_score < dist_threshold) & (align_score < align_threshold)
        
        weighted_edges = [(i, j, align_score[k]) for k, (i, j) in enumerate(self.end_ab) if mask[k]]
        
        e2e_graph = nx.Graph()
        e2e_graph.add_nodes_from(range(len(self.segments)*2))
        e2e_graph.add_weighted_edges_from(weighted_edges)
        
        mst = nx.minimum_spanning_tree(e2e_graph)
        pruned_graph = prune_mst(mst)
        conn_comp = list(nx.connected_components(pruned_graph))
        
        cluster_size_list = [len(x) for x in conn_comp]
        print(f'Number of end points: {len(self.segments)*2}')        
        print(f'Number of connected components: {len(conn_comp)}')
        print(f'Max. cluster size {np.max(cluster_size_list)} at {np.argmax(cluster_size_list)}')
        
        next_round = []
        
        for cc in conn_comp:
            cc = list(cc)
            subgraph = pruned_graph.subgraph(cc)
            eps = [node for node in subgraph.nodes if subgraph.degree[node] == 1]

            if len(eps) != 2:
                continue
                # raise ValueError("The graph does not have exactly two endpoints.")


            # Find the shortest path between the two endpoints
            path = nx.shortest_path(subgraph, source=eps[0], target=eps[1])    
            straight_curve = []
            for i_ in path[::2]:
                if i_ % 2 == 0:
                    straight_curve.append(self.segments[i_//2])
                elif i_ % 2 == 1: 
                    straight_curve.append(self.segments[i_//2][::-1])            
            straight_curve = np.vstack(straight_curve)
            next_round.append(straight_curve)
            
        self.graph = pruned_graph
            
        return next_round        
        
        
    def inspect_clustering(self):
        plt.close('all')
        fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
        for i in range(len(self.segments)):
            ax.plot(self.segments[i][:,0],self.segments[i][:,1],self.segments[i][:,2],'-')
        
        fig,ax=plt.subplots(1,1)
        length_list = []
        for rr in self.segments:
            length_list.append(seg_len(rr))
        log_bins = np.logspace(np.log10(1),np.log10(2000),100)
        ax.hist(length_list,bins=log_bins)
        ax.set_xscale('log')
        
        return length_list
    
# %%

    
# %%
seg3 = Segments(round3)
round4 = seg3.clustering(number_of_endpoint_averaging=30,dist_threshold=30,align_threshold=0.15)
# length_list = seg3.inspect_clustering()
# %%






# %%


# %%
fp = filamentprocessing.FilamentProcessing(round4,50,1,0.99)
# %%
fp.calculate_svd_scores(50,0.05)
# %%
ij = fp.get_svd_ij()
scores = fp.get_svd_scores()
# %%
ij = np.array(ij)
scores = np.array(scores)
dist_score = scores[:,0]
align_score = scores[:,1]
# %%
dist_score = scores[:,0]
align_score = scores[:,1]

mask = (dist_score < 50) & (align_score < 0.05)

graph = nx.Graph()
graph.add_nodes_from(range(len(round4)))
graph.add_edges_from(ij[mask,:])

connected_components = list(nx.connected_components(graph))
connected_components = [list(x) for x in connected_components]
cluster_size_list = [len(x) for x in connected_components]
print(f'Number of segments: {len(round4)}')
print(f'Number of connected components: {len(connected_components)}')
print(f'Max. cluster size {np.max(cluster_size_list)} at {np.argmax(cluster_size_list)}')

# %%
i_max = np.argmax(cluster_size_list)
i_max = np.argsort(cluster_size_list)[-510]
cc_max = connected_components[i_max]
plt.close('all')
fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
for i_ in cc_max:
    ax.plot(round4[i_][:,0],round4[i_][:,1],round4[i_][:,2],linewidth=1)
    ax.axis('equal')

# %%

# %%
# inspect a cluster
i_ = 25
joined = np.vstack([segments[i] for i in connected_components[i_]])
joined = sort_curve(joined)
print(connected_components[i_])
print(seg_len(joined))
# %%
length_list = []
error_list = []
for i_ in range(len(connected_components)):
    joined = np.vstack([segments[i] for i in connected_components[i_]])
    joined = sort_curve(joined)
    length_list.append(seg_len(joined))
    
    fit_result = fit_rod(joined,0.00001,10000)
    error_list.append(fit_result['err'])
# %%
log_bins = np.logspace(1,3,100)
plt.close('all')
fig,ax=plt.subplots(1,1)
ax.hist(length_list,bins=log_bins)
ax.set_xscale('log')
# %%
np.count_nonzero(np.array(length_list) > 600)

good_clusters_labels = np.where(np.array(length_list) > 600)[0]
# %%
error_list = np.array(error_list)
local_where = np.argmax(error_list[np.array(length_list) > 600])
i_weird = good_clusters_labels[local_where]
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
# %%
plt.close('all')
fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
for i_ in range(len(to_be_legit_clusters)):
    joined = np.vstack([segments[i] for i in to_be_legit_clusters[i_]])
    joined = sort_curve(joined)    
    if np.all(np.linalg.norm(joined[:,:2] - [1000,1000],axis=1) > 700) and (seg_len(joined) < 20):
    # if np.all(np.linalg.norm(joined - [500,1000,500],axis=1) < 250):
        ax.plot(joined[:,0],joined[:,1],joined[:,2],linewidth=1)
        
        
ax.axis('equal')

# %%
np.count_nonzero(np.array(length_list) < 20)

# %%
filtered = []
for i_ in range(len(to_be_legit_clusters)):
    joined = np.vstack([segments[i] for i in to_be_legit_clusters[i_]])
    joined = sort_curve(joined)
    filtered.append(joined)
    # if (seg_len(joined) > 10):
    #     filtered.append(joined)

len(filtered)
# %%




# %%

mask = (end_dist_mat < 10) & (end_alig_mat < 1)
e2e_graph = nx.Graph()
e2e_graph.add_nodes_from(range(len(long_segments)*2))

edges_indices = np.array(np.where(mask)).T
weights = end_alig_mat[mask]
weighted_edges = [(i, j, w) for (i, j), w in zip(edges_indices, weights)]
e2e_graph.add_weighted_edges_from(weighted_edges)

# e2e_graph.add_edges_from(np.array(np.where(mask)).T)

e2e_clusters = list(nx.connected_components(e2e_graph))

cluster_size_list = [len(x) for x in e2e_clusters]
print(f'Number of end points: {len(long_segments)*2}')
print(f'Number of connected components: {len(e2e_clusters)}')
print(f'Max. cluster size {np.max(cluster_size_list)} at {np.argmax(cluster_size_list)}')

# %%
mst = nx.minimum_spanning_tree(e2e_graph)
pruned_graph = prune_mst(mst)
conn_comp = list(nx.connected_components(pruned_graph))

cluster_size_list = [len(x) for x in conn_comp]
print(f'Number of end points: {len(long_segments)*2}')
print(f'Number of connected components: {len(conn_comp)}')
print(f'Max. cluster size {np.max(cluster_size_list)} at {np.argmax(cluster_size_list)}')

# %%
cluster_size_list = [len(x) for x in conn_comp]
i_max = np.argmax(cluster_size_list)
# i_max = np.argsort(cluster_size_list)[0]
cc_max = conn_comp[i_max]

plt.close('all')
fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
scale=10
for i_ in cc_max:
    # plot each end    
    if i_ % 2 == 0:
        clr = np.random.rand(3)
        ax.plot(long_segments[i_//2][:,0],long_segments[i_//2][:,1],long_segments[i_//2][:,2],linewidth=1,color=clr)        
    ax.plot(end_points2[i_][0],end_points2[i_][1],end_points2[i_][2],'o',markersize=2,color=clr)
ax.axis('equal')
# %%

round2 = []
for cc in conn_comp:
    cc = list(cc)
    subgraph = pruned_graph.subgraph(cc)
    eps = [node for node in subgraph.nodes if subgraph.degree[node] == 1]

    if len(eps) != 2:
        raise ValueError("The graph does not have exactly two endpoints.")

    # Find the shortest path between the two endpoints
    path = nx.shortest_path(subgraph, source=eps[0], target=eps[1])    
    straight_curve = []
    for i_ in path[::2]:
        if i_ % 2 == 0:
            straight_curve.append(long_segments[i_//2])
        elif i_ % 2 == 1: 
            straight_curve.append(long_segments[i_//2][::-1])            
    straight_curve = np.vstack(straight_curve)
    round2.append(straight_curve)

# %%
plt.close('all')
fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
for i_ in np.random.choice(range(len(round2)),1000):
    if seg_len(round2[i_]) > 100:
        continue
    rr = round2[i_]
    ax.plot(rr[:,0],rr[:,1],rr[:,2],linewidth=1)
    


# %%
length_list = []
error_list = []

for i_ in range(len(round2)):
    joined = round2[i_]
    length_list.append(seg_len(joined))
    error_list.append(fit_rod(joined,0.00001,10000)['err'])
    
# %%
plt.close('all')
fig,ax=plt.subplots(1,1)
log_bins = np.logspace(0,3,100)
ax.hist(length_list,bins=log_bins)
ax.set_xscale('log')
# %%
np.count_nonzero(np.array(length_list) > 600)


# %%







# %%

# %%
mask = e2e_dist_flat < 5
np.count_nonzero(mask)

graph = nx.Graph()
graph.add_nodes_from(range(len(filtered)))
graph.add_edges_from(ij[:,mask].T)

connected_components = list(nx.connected_components(graph))
connected_components = [list(x) for x in connected_components]
cluster_size_list = [len(x) for x in connected_components]
print(f'Number of segments: {len(filtered)}')
print(f'Number of connected components: {len(connected_components)}')
print(f'Max. cluster size {np.max(cluster_size_list)} at {np.argmax(cluster_size_list)}')
# %%
isolated_debris = []
plt.close('all')
fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
for isolated in np.where(np.array(cluster_size_list) < 2)[0]:
    joined = np.vstack([filtered[i] for i in connected_components[isolated]])
    joined = sort_curve(joined)
    if seg_len(joined) < 50:
        ax.plot(joined[:,0],joined[:,1],joined[:,2],linewidth=1)
        isolated_debris.append(joined)
        
        
# %%
plt.close('all')
fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
ax.plot(joined[:,0],joined[:,1],joined[:,2],linewidth=1)

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

np.count_nonzero(np.array(length_list) > 600)

# %%







# %%


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
    rr = rr[5:-5,:]
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

    

end_tangents = np.zeros((len(svd_cylinders),6))
for i_ in range(len(filtered2)):
    rr = filtered2[i_]
    
    # first 20 points
    rr_first = rr[:20]
    # last 20 points
    rr_last = rr[-20:]
    
    first_to_last = rr_last[0] - rr_first[-1]
    
    
    tan1 = quick_tangent(rr_first)
    tan1 = -tan1 * np.sign(np.dot(tan1,first_to_last))
    tan2 = quick_tangent(rr_last)
    tan2 = tan2 * np.sign(np.dot(tan2,first_to_last))
    
    
    # tan1 *= 100
    # tan2 *= 100
    # ax.plot(rr[:,0],rr[:,1],rr[:,2],linewidth=1)
    # ax.quiver(rr[0,0],rr[0,1],rr[0,2],tan1[0],tan1[1],tan1[2],color='r')
    # ax.quiver(rr[-1,0],rr[-1,1],rr[-1,2],tan2[0],tan2[1],tan2[2],color='r')
    # ax.axis('equal')
    
    end_tangents[i_,:3] = tan1
    end_tangents[i_,3:] = tan2
    
# %%
end_points = np.zeros((len(filtered2),6))
for i_ in range(len(filtered2)):
    end_points[i_,:3] = filtered2[i_][10]
    end_points[i_,3:] = filtered2[i_][-10]
    
# %%


from jax import vmap

end_tangent_alignment = np.zeros((len(filtered2), len(filtered2)))

        
# %%
# Vectorizing the pairwise alignment computation
vectorized_pairwise_alignment = vmap(vmap(pairwise_alignment, in_axes=(None, 0, None, 0)), in_axes=(0, None, 0, None))
end_tangent_alignment = vectorized_pairwise_alignment(end_points, end_points, end_tangents, end_tangents)

print(end_tangent_alignment)
#
# %%
end_tangent_alignment = np.array(end_tangent_alignment)
end_tangent_alignment[np.diag_indices(len(filtered2))] = 1

# %%
e2e_distance = calculate_e2e_dist_mat(end_points)
e2e_distance = np.array(e2e_distance)
e2e_distance = e2e_distance + e2e_distance.T
e2e_distance[np.diag_indices(len(filtered2))] = np.inf
# %%
p_i1 = end_points[3881,:3]
p_i2 = end_points[3881,3:]
p_j1 = end_points[3845,:3]
p_j2 = end_points[3845,3:]

d1 = np.linalg.norm(p_i1 - p_j1)
d2 = np.linalg.norm(p_i2 - p_j2)
d3 = np.linalg.norm(p_i1 - p_j2)
d4 = np.linalg.norm(p_i2 - p_j1)

dmin = np.min([d1,d2,d3,d4])
# %%        
mask = (e2e_distance < 50) & (end_tangent_alignment < 0.05)

e2e_graph = nx.Graph()
e2e_graph.add_nodes_from(range(len(filtered2)))
e2e_graph.add_edges_from(np.array(np.where(mask)).T)
e2e_clusters = list(nx.connected_components(e2e_graph))

e2e_clusters = list(nx.connected_components(e2e_graph))
e2e_clusters = [list(x) for x in e2e_clusters]
cluster_size_list = [len(x) for x in e2e_clusters]
print(f'Number of segments: {len(filtered2)}')
print(f'Number of connected components: {len(e2e_clusters)}')
print(f'Max. cluster size {np.max(cluster_size_list)} at {np.argmax(cluster_size_list)}')

error_list = []
length_list = []

for i_ in range(len(e2e_clusters)):
    joined = np.vstack([filtered2[i] for i in e2e_clusters[i_]])
    joined = sort_curve(joined)
    
    
    length_list.append(seg_len(joined))
    fit_result = fit_rod(joined,0.00001,10000)
    error_list.append(fit_result['err'])
    
np.count_nonzero(np.array(length_list) > 600)

# %%
# i_ = np.argsort(length_list)[-100]
# i_ = np.argmax(error_list)
i_ = np.argsort(length_list)[-32]
cc_max = e2e_clusters[i_]
plt.close('all')
fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
for i_ in cc_max:
    rr = filtered2[i_]
    rr = sort_curve(rr)    
    ax.plot(rr[:,0],rr[:,1],rr[:,2],linewidth=1)
    ax.axis('equal')
    
# %%

    
# %%
plt.close('all')
fig,ax=plt.subplots(1,1)
ax.hist(error_list,bins=100)
# %%
plt.close('all')
fig,ax=plt.subplots(1,1)
ax.hist(length_list,bins=100)

# %%
N1 = len(legit_clusters)
N2 = len(legit_clusters2)
N3 = np.count_nonzero(np.array(length_list) > 600)

print(f'Number of clusters in round 1: {N1}')
print(f'Number of clusters in round 2: {N2}')
print(f'Number of clusters in round 3: {N3}')

print(f'Total {N1 + N2 + N3} clusters')
# %%
N4 = np.count_nonzero(np.array(length_list) < 600)




# %%
plt.close('all')
fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
for i_ in range(len(e2e_clusters)):
    if length_list[i_] > 200:
        continue    
    
    joined = np.vstack([filtered2[i] for i in e2e_clusters[i_]])
    
    if np.any(np.linalg.norm(joined - [500,1000,500],axis=1) > 400):
        continue    
    
    ax.plot(joined[:,0],joined[:,1],joined[:,2],linewidth=1)
    ax.text(joined[0,0],joined[0,1],joined[0,2],f'{i_}',fontsize=6)

# %%
i_ = np.where(np.array(length_list) < 400)[0][300]
i_ = 3845

# %%
i_ = 3881
cc = e2e_clusters[i_]
joined = np.vstack([filtered2[i] for i in cc])
joined = sort_curve(joined)
plt.close('all')
fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
ax.plot(joined[:,0],joined[:,1],joined[:,2],linewidth=1)
ax.axis('equal')

neighbors = np.where(e2e_distance[i_] < 50)[0]
for nb in neighbors:
    rr = filtered2[nb]
    ax.plot(rr[:,0],rr[:,1],rr[:,2],linewidth=1)
    
# %%
e2e_distance[5149,5197]
end_tangent_alignment[5149,5197]

# %%

# %%
# %%
scale = 100
i_ = 30
e2e_neighbors = np.where(e2e_distance[i_] < 30)[0]
plt.close('all')
fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
ax.plot(filtered2[i_][:,0],filtered2[i_][:,1],filtered2[i_][:,2],linewidth=1)
ax.quiver(filtered2[i_][0,0],filtered2[i_][0,1],filtered2[i_][0,2],scale*end_tangents[i_,0],scale*end_tangents[i_,1],scale*end_tangents[i_,2],color='r')
ax.quiver(filtered2[i_][-1,0],filtered2[i_][-1,1],filtered2[i_][-1,2],scale*end_tangents[i_,3],scale*end_tangents[i_,4],scale*end_tangents[i_,5],color='r')

for nb in e2e_neighbors:
    ax.plot(filtered2[nb][:,0],filtered2[nb][:,1],filtered2[nb][:,2],linewidth=1)
    ax.quiver(filtered2[nb][0,0],filtered2[nb][0,1],filtered2[nb][0,2],scale*end_tangents[nb,0],scale*end_tangents[nb,1],scale*end_tangents[nb,2],color='r')
    ax.quiver(filtered2[nb][-1,0],filtered2[nb][-1,1],filtered2[nb][-1,2],scale*end_tangents[nb,3],scale*end_tangents[nb,4],scale*end_tangents[nb,5],color='r')
ax.axis('equal')

# %%
plt.close('all')
i_ = 5149
rr_i = filtered2[i_]
p_i1 = filtered2[i_][0]
p_i2 = filtered2[i_][-1]
tan_i1 = end_tangents[i_,:3]
tan_i2 = end_tangents[i_,3:]

j_ = 5197
rr_j = filtered2[j_]
p_j1 = filtered2[j_][0]
p_j2 = filtered2[j_][-1]
tan_j1 = end_tangents[j_,:3]
tan_j2 = end_tangents[j_,3:]

d1 = np.linalg.norm(p_i1 - p_j1)
d2 = np.linalg.norm(p_i2 - p_j2)
d3 = np.linalg.norm(p_i1 - p_j2)
d4 = np.linalg.norm(p_i2 - p_j1)

ds = np.array([d1,d2,d3,d4])
i_min = np.argmin(ds)

if i_min == 0:
    dvec = p_i1 - p_j1
    tan_i = tan_i1
    tan_j = tan_j1
    p_i_opt = p_i1
    p_j_opt = p_j1
    
elif i_min == 1:
    dvec = p_i2 - p_j2
    tan_i = tan_i2
    tan_j = tan_j2
    p_i_opt = p_i2
    p_j_opt = p_j2
    
elif i_min == 2:
    dvec = p_i1 - p_j2
    tan_i = tan_i1
    tan_j = tan_j2
    p_i_opt = p_i1
    p_j_opt = p_j2
    
elif i_min == 3:
    dvec = p_i2 - p_j1
    tan_i = tan_i2
    tan_j = tan_j1
    p_i_opt = p_i2
    p_j_opt = p_j1

dvec = dvec / np.linalg.norm(dvec)

alignment = (np.linalg.norm(np.cross(dvec,tan_i)) + np.linalg.norm(np.cross(dvec,tan_j))) / 2
scale = 10
plt.close('all')
fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
ax.plot(rr_i[:,0],rr_i[:,1],rr_i[:,2],linewidth=1)
ax.plot(rr_j[:,0],rr_j[:,1],rr_j[:,2],linewidth=1)
ax.plot([p_i_opt[0],p_j_opt[0]],[p_i_opt[1],p_j_opt[1]],[p_i_opt[2],p_j_opt[2]],'k--')
ax.quiver(p_i1[0],p_i1[1],p_i1[2],scale*tan_i[0],scale*tan_i[1],scale*tan_i[2],color='b')
ax.quiver(p_j2[0],p_j2[1],p_j2[2],scale*tan_j[0],scale*tan_j[1],scale*tan_j[2],color='k')
ax.axis('equal')

# %%
np.linalg.norm(np.cross(tan_i,tan_j))
dvec





# %%
e2e_alignment[i_,851]
end_tangent_alignment[i_,851]
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
