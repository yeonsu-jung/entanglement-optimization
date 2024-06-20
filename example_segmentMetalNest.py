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
def seg_len(seg):
    return np.sum(np.sqrt(np.sum(np.diff(seg,axis=0)**2,axis=1)))
    
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
            # straight_curve = sort_curve(straight_curve)
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


from scipy.io import loadmat
dataobj = loadmat('/Users/yeonsu/Dropbox (Harvard University)/Data/prunedMetalNest/segments.mat')
segments = dataobj['segments']
segments = [np.array(seg[0]) for seg in segments]
# %%
len(segments)
                
# %%
seg = Segments(segments)
next_round = seg.end_to_end_clustering(number_of_endpoint_averaging=20,dist_threshold=10,align_threshold=1.)
# %%
straightened_list = []
for cc in seg.end_to_end_cluster:
    if len(cc) < 2:
        print(f'Cluster {cc} has less than 2 segments.')
    elif len(cc) == 2:
        cc = list(cc)
        ix = cc[0]
        jx = cc[1]
        assert( ix // 2 == jx // 2 )
        
        # straightened_list.append(segments[ix//2])
        
    elif len(cc) > 2:
        print(cc)
        subgraph = seg.pruned_graph.subgraph(cc)
        subgraph.nodes()
        eps = [node for node in subgraph.nodes if subgraph.degree[node] == 1]
        path = nx.shortest_path(subgraph, source=eps[0], target=eps[1])
        
        straightened = []
        for i_ in path[::2]:
            if i_ % 2 == 0:
                straightened.append(segments[i_//2])
            elif i_ % 2 == 1:
                straightened.append(segments[i_//2][::-1])                
            
            
        straightened = np.vstack(straightened)
        straightened_list.append(straightened)
        
                
# %%
fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
for straightened in straightened_list[:30]:
    ax.plot(straightened[:,0],straightened[:,1],straightened[:,2],'-')

    
# %%
        
    


                
# %%
