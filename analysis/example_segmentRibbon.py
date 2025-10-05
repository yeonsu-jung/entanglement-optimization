# %%
%matplotlib qt
from matplotlib import pyplot as plt
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
def curvature_of_polygonal_curve(nodes):
    tan2 = nodes[2:,:] - nodes[1:-1,:]    
    tan1 = nodes[1:-1,:] - nodes[:-2,:]
    
    nom = np.linalg.norm(2*np.cross(tan1,tan2,axis=1),axis=1)
    den = np.sum(tan1*tan2,axis=1)
    # curvature = np.sum(nom/den)
    return nom/den

def break_curved_rods(seg,curvature_threshold):
    curvature = curvature_of_polygonal_curve(seg)
    break_points = np.where(np.abs(curvature)>curvature_threshold)[0]
    if len(break_points)==0:
        return [seg]
    else:
        segs = []
        start_idx = 0
        for bp in break_points:
            segs.append(seg[start_idx:bp+1])
            start_idx = bp
        segs.append(seg[start_idx:])
        return segs
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

def prune_mst2(mst):
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

    
def prune_mst(mst):
    import heapq
    # Track the degree of each node
    node_degrees = {node: 0 for node in mst.nodes}
    pruned_edges = []
    mandatory_edges = []
    added_edges = set()
    
    i = 0
    mst.has_edge(i, i+1)
    
    for i in range(0,max(mst.nodes),2):
        if mst.has_edge(i,i+1):
            continue
        else:
            print(f'Edge {i} - {i+1} does not exist')
    
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
            
    # Ensure each connected component is a path
    def is_path(graph):
        for component in nx.connected_components(graph):
            if sum(1 for node in component if graph.degree[node] == 1) != 2:
                return False
        return True

    pruned_graph = nx.Graph()
    pruned_graph.add_nodes_from(mst.nodes)
    pruned_graph.add_edges_from((u, v, data) for u, v, data in pruned_edges)
    
    # if not is_path(pruned_graph):
        # raise ValueError("The pruned graph is not a path graph in all connected components")
    
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
    
# rod_data_root_dir = Path('/Users/yeonsu/Data/steel-rods-xray-data')
# segments_file_path = rod_data_root_dir / 'alpha200_epsilon00' / 'pruned_segments.pkl'
# segments = pickle.load(open(segments_file_path,'rb'))

from scipy.io import loadmat
dataobj = loadmat('/Users/yeonsu/Dropbox (Harvard University)/Data/ribbon/segments.mat')


segments = dataobj['segments']
segments = [np.array(seg[0]) for seg in segments]
# %%
len(segments)


# from scipy.io import loadmat
# dataobj = loadmat('/Users/yeonsu/Dropbox (Harvard University)/Data/steel-rods-xray-data/alpha200_epsilon00/segments.mat')
# segments = dataobj['segments']
# segments = [np.array(seg[0]) for seg in segments]


    
    
# %%

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
import filamentprocessing


# %%
class Segments:
    def __init__(self,segments):
        import filamentprocessing
        import networkx as nx
        
        self.segments = segments
        self.fp = filamentprocessing.FilamentProcessing(segments,50,1,0.99)
        
    def update_segments(self,segments):
        self.segments = segments
        self.fp.update_filaments(segments)
        
    def calculate_end_to_end_properties(self,dist_threshold):
        self.fp.calculate_end_to_end_properties(dist_threshold)
        
    def get_end_points(self):
        return self.fp.get_end_points()
    
    def get_corrected_end_points(self):
        return self.fp.get_corrected_end_points()
    
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
    
    def end_to_end_clustering(self,number_of_endpoint_averaging=10,dist_threshold=30,align_threshold=0.15):
        self.fp.calculate_end_to_end_properties(number_of_endpoint_averaging)
        self.endpoints = self.get_end_points()
        self.corrected_end_points = self.get_corrected_end_points()
        self.endtangents = self.get_end_tangents()
        
        self.fp.calculate_end_to_end_scores(dist_threshold)
        self.end_ab = self.get_end_ab()
        self.end_scores = self.get_end_scores()
        
        self.end_ab = np.array(self.end_ab)
        self.end_scores = np.array(self.end_scores)
        
        dist_score = self.end_scores[:,0]
        align_score = self.end_scores[:,1]
        
        # sanity check
        # ij = np.array(self.end_ab)        
        # even_is = np.where(np.mod(ij[:,0],2) == 0)[0]
        # conjugates = np.where( ij[:,1] == ij[:,0] + 1 )[0]
        # both_cond = np.intersect1d(even_is,conjugates)                
        # pathologies = np.where(dist_score[both_cond] != -1)[0]        
        # ij[both_cond[pathologies],:]
        
            
            
        
        
        
        for k, (i, j) in enumerate(self.end_ab):
            
            i_conj = i + 1 if i % 2 == 0 else i - 1
            j_conj = j + 1 if j % 2 == 0 else j - 1
            
            # if i and j are conjugate, skip
            if i_conj == j or j_conj == i:                
                continue
            
            ep_i = self.corrected_end_points[i]
            ep_j = self.corrected_end_points[j]
            
            
            
            inward_i = self.corrected_end_points[i_conj] - ep_i
            inward_j = self.corrected_end_points[j_conj] - ep_j
            
            inward_i = inward_i / np.linalg.norm(inward_i)
            inward_j = inward_j / np.linalg.norm(inward_j)
            
            dvec = ep_j - ep_i
            
            if np.dot(dvec, inward_i) < 0.5:
                dist_score[k] = np.inf
                align_score[k] = np.inf
        
        mask = (dist_score < dist_threshold) & (align_score < align_threshold)        
        edges_with_alignment_weights = [(i, j, align_score[k]) for k, (i, j) in enumerate(self.end_ab)]
        edges_with_distance_weights = [(i, j, dist_score[k]) for k, (i, j) in enumerate(self.end_ab)]
        
        self.alignment_graph = nx.Graph()
        self.alignment_graph.add_nodes_from(range(len(self.segments)*2))
        self.alignment_graph.add_weighted_edges_from(edges_with_alignment_weights)
        
        self.distance_graph = nx.Graph()
        self.distance_graph.add_nodes_from(range(len(self.segments)*2))
        self.distance_graph.add_weighted_edges_from(edges_with_distance_weights)
        
        
        for i_ in range(0,len(self.segments)*2,2):
            i_conj = i_ + 1
            
            # i_ th node's weight for i_conj
            if self.alignment_graph[i_][i_conj]['weight'] != -1:
                print(f'Node {i_} does not have negative weight for its conjugate {i_conj}')
                
            if self.distance_graph[i_][i_conj]['weight'] != -1:
                print(f'Node {i_} does not have negative weight for its conjugate {i_conj}')
            
        
        
        
        filtered_edges = [(i, j, align_score[k]) for k, (i, j) in enumerate(self.end_ab) if mask[k]]
        filtered_graph = nx.Graph()
        filtered_graph.add_nodes_from(range(len(self.segments)*2))
        filtered_graph.add_weighted_edges_from(filtered_edges)
        
        # sanity check: i and i conjugate should be in filtered edges
        for i_ in range(0,len(self.segments)*2,2):
            i_conj = i_ + 1
            
            # i_ th node's weight for i_conj
            if filtered_graph[i_][i_conj]['weight'] != -1:
                print(f'Node {i_} does not have negative weight for its conjugate {i_conj}')
                
            if filtered_graph[i_][i_conj]['weight'] != -1:
                print(f'Node {i_} does not have negative weight for its conjugate {i_conj}')
        
        
        # mst = nx.minimum_spanning_tree(filtered_graph)
        
        
        self.pruned_graph = prune_mst(filtered_graph)
        self.end_to_end_cluster = list(nx.connected_components(self.pruned_graph))
        self.cluster_size_list = [len(x) for x in self.end_to_end_cluster]
        
        print(f'Number of end points: {len(self.segments)*2}')
        print(f'Number of connected components: {len(self.end_to_end_cluster)}')
        print(f'Max. cluster size {np.max(self.cluster_size_list)} at {np.argmax(self.cluster_size_list)}')
        
        next_round = []
        self.length_list = []
        for i_,cc in enumerate(self.end_to_end_cluster):
            cc = list(cc)
            subgraph = self.pruned_graph.subgraph(cc)
            eps = [node for node in subgraph.nodes if subgraph.degree[node] == 1]

            if len(eps) != 2:
                print(f'Cluster {i_} does not have exactly two endpoints.')
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
            straight_curve = sort_curve(straight_curve)
            next_round.append(straight_curve)            
            self.length_list.append(seg_len(straight_curve))
            
        # sort by length
        next_round = [x for _, x in sorted(zip(self.length_list, next_round), key=lambda pair: -pair[0])]
        
        self.next_round = next_round
            
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

    def plot_large_clusters(self,num_to_show):
        cluster_size_list = [len(x) for x in self.end_to_end_cluster]
        i_max_list = np.argsort(cluster_size_list)[-num_to_show:]
        
        plt.close('all')
        fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
        for i_max in i_max_list:            
            cc_max = self.end_to_end_cluster[i_max]
            
            joined = []
            for i_ in cc_max:
                if i_ % 2 == 1:
                    continue
                ax.plot(self.segments[i_//2][:,0],self.segments[i_//2][:,1],self.segments[i_//2][:,2],'.',alpha=0.2)
                joined.append(self.segments[i_//2])
                
            if len(joined) == 0:
                continue
            
            joined = np.vstack(joined)
            joined = sort_curve(joined)
            ax.plot(joined[:,0],joined[:,1],joined[:,2],linewidth=1,color='k',alpha=0.5)
            
        ax.axis('equal')
        return ax
            
    def check_nearby_segments(self,i_segment,search_radius):
        ep1 = self.endpoints[i_segment*2]
        ep2 = self.endpoints[i_segment*2+1]
        
        dist1 = np.linalg.norm(self.endpoints - ep1,axis=1)
        dist2 = np.linalg.norm(self.endpoints - ep2,axis=1)
        
        mask = (dist1 < search_radius) | (dist2 < search_radius)
        return np.where(mask)[0]
    
    def merge_by_fitting(self,distance_threshold=30,fitting_error_threshold=1.5):
        ab_ = self.fp.get_end_ab()
        scores_ = self.fp.get_end_scores()
        
        ab_ = np.array(ab_)
        scores_ = np.array(scores_)
        
        dist_scores_ = scores_[:,0]
        # align_scores_ = scores_[:,1] # useless here       
        
        test = 0
        if test:
            i_min_list = np.argsort(dist_scores_[dist_scores_>0])
            i_min = i_min_list[321]
            i_min_global = np.where(dist_scores_>0)[0][i_min]
            print(scores_[i_min_global,:])                
        
            _ij = ab_[i_min_global,:]
            rr0 = round5[_ij[0]//2]
            rr = round5[_ij[1]//2]
            plt.close('all')
            fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
            ax.plot(rr0[:,0],rr0[:,1],rr0[:,2],'-')
            ax.plot(rr[:,0],rr[:,1],rr[:,2],'-')
            ax.axis('equal')
            joined = np.vstack([rr0,rr])
            fr = fit_rod(joined,linearity_threshold=0.0001,radius_curvature_threshold=100000)
            print(f'Error: {fr["err"]}') # good, merge it.

            ep1 = self.endpoints[_ij[0]]
            ep2 = self.endpoints[_ij[1]]
            dvec = ep1 - ep2
            dvec /= np.linalg.norm(dvec)

            # dist = np.linalg.norm(ep1 - ep2)
            tan1 = self.endtangents[_ij[0]]
            tan2 = self.endtangents[_ij[1]]

        # align_score = (np.linalg.norm(np.cross(dvec,tan1)) + np.linalg.norm(np.cross(dvec,tan2))) / 2

        visualize = 0
        if visualize:
            plt.close('all')
            fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
            ax.plot(rr0[:,0],rr0[:,1],rr0[:,2],'-')
            ax.plot(rr[:,0],rr[:,1],rr[:,2],'-')
            ax.plot(ep1[0],ep1[1],ep1[2],'ro')
            ax.plot(ep2[0],ep2[1],ep2[2],'bo')
            scale = 100
            ax.quiver(ep2[0],ep2[1],ep2[2],scale*dvec[0],scale*dvec[1],scale*dvec[2],color='r')
            ax.quiver(ep1[0],ep1[1],ep1[2],scale*tan1[0],scale*tan1[1],scale*tan1[2],color='b')
            ax.quiver(ep2[0],ep2[1],ep2[2],scale*tan2[0],scale*tan2[1],scale*tan2[2],color='g')
            ax.axis('equal')

        merged = []
        length_list = []

        # union find?
        parent = {i:i for i in range(len(round5))}
        
        def find(i):
            if parent[i] != i:
                parent[i] = find(parent[i])
            return parent[i]
        
        def union(i,j):
            parent[find(i)] = find(j)
            
        # def union(node1, node2):
        #     root1 = find(node1)
        #     root2 = find(node2)
        #     if root1 != root2:
        #         parent[root2] = root1
            

        for i_ in np.where((dist_scores_ < distance_threshold) & (dist_scores_ > 0))[0]:    
            _ij = ab_[i_,:]
            rr0 = round5[_ij[0]//2]
            rr = round5[_ij[1]//2]
            
            joined = np.vstack([rr0,rr])
            joined = sort_curve(joined)
            fr = fit_rod(joined,linearity_threshold=0.0001,radius_curvature_threshold=100000)
            if fr['err'] < fitting_error_threshold:
                print(f'Error: {fr["err"]}')
                
                # connect
                union(_ij[0]//2,_ij[1]//2)
                
                
                # merged.append(joined)
                # length_list.append(seg_len(joined))
                
            
                
                
            
                
                
# %%
# global_centroid = np.mean(np.vstack(segments),axis=0)

# local_segments = []
# for i,segment in enumerate(segments):   
    
#     if np.any(np.linalg.norm(segment - global_centroid,axis=1) < 500):
#         local_segments.append(segment)
        
# %%
seg = Segments(segments)
segments = seg.end_to_end_clustering(number_of_endpoint_averaging=30,dist_threshold=10,align_threshold=0.1)
# %%

seg = Segments(segments)
segments = seg.end_to_end_clustering(number_of_endpoint_averaging=50,dist_threshold=100,align_threshold=0.05)


# %%
i_max_list = np.argsort(seg.length_list)[-100:]

plt.close('all')
fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
for i_max in i_max_list:
    rr = seg.next_round[i_max]    
    ax.plot(rr[:,0],rr[:,1],rr[:,2],'-')    
    ax.axis('equal')
    
# %%
i_max_list = np.argsort(seg.cluster_size_list)[-540:]
cc_max = seg.end_to_end_cluster[i_max_list[0]]

plt.close('all')
fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
for i_ in cc_max:
    if i_ % 2 == 0:
        ax.plot(seg.segments[i_//2][:,0],seg.segments[i_//2][:,1],seg.segments[i_//2][:,2],'-')

ax.axis('equal')
# %%


# %%
log_bins = np.logspace(np.log10(1),np.log10(1000),100)
plt.close('all')
fig,ax=plt.subplots(1,1)
ax.hist(seg.length_list,bins=log_bins)
ax.set_xscale('log')
    
# %%
seg.plot_large_clusters(20)
# %%
plt.close('all')
fig,ax=plt.subplots(1,1)
log_bins = np.logspace(np.log10(1),np.log10(1000),100)
ax.hist(seg.length_list,bins=log_bins)
ax.set_xscale('log')
# %%
np.count_nonzero(np.array(seg.length_list) < 30)

plt.close('all')
fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
for i in np.where(np.array(seg.length_list) < 30)[0]:
    ax.plot(seg.next_round[i][:,0],seg.next_round[i][:,1],seg.next_round[i][:,2],'-')
    

# %%
round2 = seg.next_round
round2 = [rr for rr in round2 if seg_len(rr) > 30]
# %%
seg = Segments(round2)
round3 = seg.end_to_end_clustering(number_of_endpoint_averaging=100,dist_threshold=200,align_threshold=0.05)

# %%
plt.close('all')
fig,ax=plt.subplots(1,1)
log_bins = np.logspace(np.log10(10),np.log10(1000),100)
ax.hist(seg.length_list,bins=log_bins)
ax.set_xscale('log')

# %%
seg.plot_large_clusters(1000)

# %%
seg_3 = Segments(round3)
round4 = seg_3.end_to_end_clustering(number_of_endpoint_averaging=250,dist_threshold=600,align_threshold=0.025)
# %%
plt.close('all')
fig,ax=plt.subplots(1,1)
# log_bins = np.logspace(np.log10(10),np.log10(1000),100)
# ax.hist(seg.length_list,bins=log_bins)
ax.hist(seg_3.length_list,bins=np.linspace(10,1000,25))
# ax.set_xscale('log')

# %%
seg_3.plot_large_clusters(35)
# %%
i_test = np.where(np.array(seg_3.length_list) < 100)[0][0]
rr = round4[i_test]
plt.close('all')
fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
ax.plot(rr[:,0],rr[:,1],rr[:,2],'-')
ax.axis('equal')

# %%
seg_4 = Segments(round4)
round5 = seg_4.end_to_end_clustering(number_of_endpoint_averaging=250,dist_threshold=600,align_threshold=0.025)

#%%
seg_4.plot_large_clusters(35)
#%%
plt.close('all')
fig,ax=plt.subplots(1,1)
# log_bins = np.logspace(np.log10(10),np.log10(1000),100)
# ax.hist(seg.length_list,bins=log_bins)
ax.hist(seg_4.length_list,bins=np.linspace(10,1000,25))
#%%
# loop over each rod
# try to find nearby rods
# merge and assess
# if not, keep it as it is
# if yes, merge and assess

seg_5 = Segments(round5)
round6 = seg_5.end_to_end_clustering(number_of_endpoint_averaging=250,dist_threshold=600,align_threshold=0.025)
# %%



        
# %%
plt.close('all')
fig,ax=plt.subplots(1,1)
log_bins = np.logspace(np.log10(10),np.log10(1000),100) 
ax.hist(seg_5.length_list,bins=log_bins)

# %%
plt.close('all')
fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
for rr in merged:
    ax.plot(rr[:,0],rr[:,1],rr[:,2],'-')
    
        
# %%
   
ab_ = seg_5.fp.get_end_ab()
scores_ = seg_5.fp.get_end_scores()

ab_ = np.array(ab_)
scores_ = np.array(scores_)

dist_scores_ = scores_[:,0]
# %%        
distance_threshold = 30
fitting_error_threshold = 1.5

parent = {i:i for i in range(len(round5))}
connections = []
for i_ in np.where((dist_scores_ < distance_threshold) & (dist_scores_ > 0))[0]:    
    _ij = ab_[i_,:]
    rr0 = round5[_ij[0]//2]
    rr = round5[_ij[1]//2]
    
    joined = np.vstack([rr0,rr])
    joined = sort_curve(joined)
    
    
    fr = fit_rod(joined,linearity_threshold=0.0001,radius_curvature_threshold=100000)
    if fr['err'] < fitting_error_threshold:
        connections.append( (_ij[0]//2,_ij[1]//2) )
        print(f'Error: {fr["err"]}')
        
        # connect
        # union(_ij[0]//2,_ij[1]//2)
        
            
            
# %%
merge_graph = nx.Graph()
merge_graph.add_nodes_from(range(len(round5)))
merge_graph.add_edges_from(connections)

# %%
ccs = list(nx.connected_components(merge_graph))

ccs = [list(cc) for cc in ccs]
len(ccs)
# %%
plt.close('all')
fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
for cc in ccs:
    joined = []
    for i_ in cc:
        joined.append(round5[i_])
    joined = np.vstack(joined)
    joined = sort_curve(joined)
    ax.plot(joined[:,0],joined[:,1],joined[:,2],'-')
    

    

# %%
# parallel transport


