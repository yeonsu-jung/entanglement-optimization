# %%
# %matplotlib qt
from matplotlib import pyplot as plt
from fitting import prep_svd_cylinder, fit_rod
from pathlib import Path
import pickle
import numpy as np
import networkx as nx
from clustering import find_connected_components, explode_local_cluster
import os
from distances import lumelsky_dist_vec
import filamentprocessing

from scipy.io import loadmat

import jax
import jax.numpy as jnp

import pickle

from scipy.special import comb
from scipy.spatial.distance import cdist
from scipy.interpolate import make_interp_spline
from scipy.optimize import minimize
# %%
# segment: continguously connected points; d < sqrt(3) 
# vertex: each point in the segment
# edge: two connected vertices
# thus a segment from skeletonization is actually too dense.

class Segment: # needed?
    def __init__(self,segment):
        # segment is a Nx3 numpy array
        assert segment.shape[1] == 3
        self.segment = segment
    # length, curvature, etc...
    
    @staticmethod
    def sort_curve(segment):
        centroid = np.mean(segment,axis=0)
        segment_centered = segment - centroid        
        _,_, V = np.linalg.svd(segment_centered, full_matrices=False)
        v1 = V[0,:]
        orientation = v1 * np.sign(np.sum(v1 * (segment_centered[-1, :] - segment_centered[0, :])))
        slist = np.dot((segment - centroid), orientation)
        sorted_indices = np.argsort(slist)
        return centroid + segment_centered[sorted_indices]

    @staticmethod
    def edge_lengths(segment):
        return (np.sqrt(np.sum(np.diff(segment,axis=0)**2,axis=1)))

    @staticmethod
    def seg_len(segment):
        return np.sum(np.sqrt(np.sum(np.diff(segment,axis=0)**2,axis=1)))

    @staticmethod
    def curvature_of_polygonal_curve(segment):
        tan2 = segment[2:,:] - segment[1:-1,:]    
        tan1 = segment[1:-1,:] - segment[:-2,:]
        
        nom = np.linalg.norm(2*np.cross(tan1,tan2,axis=1),axis=1)
        den = np.sum(tan1*tan2,axis=1)
        # curvature = np.sum(nom/den)
        return nom/den

    @staticmethod
    def break_curved_rods(segment,curvature_threshold):
        curvature = Segment.curvature_of_polygonal_curve(segment)
        break_points = np.where(np.abs(curvature)>curvature_threshold)[0]
        if len(break_points)==0:
            return [segment]
        else:
            segments = []
            start_idx = 0
            for bp in break_points:
                segments.append(segment[start_idx:bp+1])
                start_idx = bp
            segments.append(segment[start_idx:])
            return segments
        
    def plot(segment,ax=None):
        if ax is None:
            fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
            ax.plot(segment[:,0],segment[:,1],segment[:,2])
        else:
            ax.plot(segment[:,0],segment[:,1],segment[:,2])
        ax.axis('equal')
        return ax
        
# %%
a_segment = np.random.rand(10,3)
Segment.sort_curve(a_segment)
Segment.curvature_of_polygonal_curve(a_segment)
Segment.break_curved_rods(a_segment,0.1)

# %%


# %%

# %%



# %%


# %%
def prune_edges(nodes, ij, weight):
    import heapq
    
    node_degrees = {node: 0 for node in nodes}
    pruned_edges = []
    mandatory_edges = []
    added_edges = set()
    
    # Enforce the mandatory connections for even i
    for i in range(0, max(nodes), 2):
        if i in nodes and i+1 in nodes:
            mandatory_edges.append((i, i+1, -1))
            node_degrees[i] += 1
            node_degrees[i+1] += 1
            added_edges.add((i, i+1))

    # Priority queue for edges sorted by weight
    edge_queue = []
    for k,(u,v) in enumerate(ij):
        if (u, v) not in added_edges and (v, u) not in added_edges:
            heapq.heappush(edge_queue, (weight[k], u, v))

    # Union-Find data structure for tracking connected components
    parent = {node: node for node in nodes}

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
            
    return pruned_edges
            

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



# %%

        


class Segments:
    def __init__(self,segments):
        import filamentprocessing
        import networkx as nx        
        self.segments = segments
        
    def prune_segments(self,error_threshold,curvature_threshold):
        
        def break_segments(segs):
            new_segments = []
            for seg in segs:
                edge_len = Segment.edge_lengths(seg)
                grph = nx.Graph()
                grph.add_nodes_from(range(len(seg)))

                for i in range(len(seg)-1):
                    if edge_len[i] <= np.sqrt(3):
                        grph.add_edge(i,i+1)
                    
                clusters = list(nx.connected_components(grph))
                for i,cluster in enumerate(clusters):
                    if len(cluster) == 1:                
                        continue
                    rr = np.array(seg,dtype=np.float64)            
                    new_segments.append(rr[list(cluster)])
                
            return new_segments

        _segments = segm.initial_prune_segments()
        _segments = break_segments(_segments)
        for i,seg in enumerate(_segments):
            _segments[i] = Segment.sort_curve(seg)

        self.segments = _segments
        self.segments = self.break_curved_segments(1,10)
        self.inspect_segments(visualize=True)

        
    def initialize_filament_processing(self,dist_threshold=50,align_threshold=1,svd_cutoff=1.):
        self.fp = filamentprocessing.FilamentProcessing(self.segments,dist_threshold,align_threshold,svd_cutoff)
        
        
    def update_segments(self,segments):
        self.segments = segments
        self.fp.update_filaments(segments)
        
    def calculate_end_to_end_properties(self,dist_threshold,shrinkage=1):
        self.fp.calculate_end_to_end_properties(dist_threshold,shrinkage)
        
    def get_end_points(self):
        return self.fp.get_end_points()
    
    def get_corrected_end_points(self):
        return self.fp.get_corrected_end_points()
    
    def get_end_tangents(self):
        return self.fp.get_end_tangents()
    
    def calculate_end_to_end_scores(self,dist_threshold,same_sideness=0.5):
        self.fp.calculate_end_to_end_scores(dist_threshold,same_sideness)
        
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
    
    def end_to_end_clustering(self,number_of_endpoint_averaging=10,dist_threshold=30,same_sideness=0.5,align_threshold=0.15,shrinkage=1):
        import time
        
        
        self.fp.calculate_end_to_end_properties(number_of_endpoint_averaging,shrinkage)
        self.endpoints = self.get_end_points()
        self.corrected_end_points = self.get_corrected_end_points()
        self.endtangents = self.get_end_tangents()
        
        start = time.time()
        self.fp.calculate_end_to_end_scores(dist_threshold,same_sideness)        
        print(f'Elapsed time for end-to-end distance calculation: {time.time() - start:.2f} s')
        
        start = time.time()
        self.end_ab = self.get_end_ab()
        self.end_scores = self.get_end_scores()
        
        self.end_ab = np.array(self.end_ab)
        self.end_scores = np.array(self.end_scores)
        print(f'Elapsed time for moving data: {time.time() - start:.2f} s')
        
        dist_score = self.end_scores[:,0]
        align_score = self.end_scores[:,1]
        
        start = time.time()
        mask = (dist_score < dist_threshold) & (align_score < align_threshold)        
        edges_with_alignment_weights = [(i, j, align_score[k]) for k, (i, j) in enumerate(self.end_ab)]
        edges_with_distance_weights = [(i, j, dist_score[k]) for k, (i, j) in enumerate(self.end_ab)]
        print(f'Elapsed time for creating edges: {time.time() - start:.2f} s')        
               
        filtered_edges = [(i, j, align_score[k]) for k, (i, j) in enumerate(self.end_ab) if mask[k]]
        self.filtered_graph = nx.Graph()
        self.filtered_graph.add_nodes_from(range(len(self.segments)*2))
        self.filtered_graph.add_weighted_edges_from(filtered_edges)
                
        self.pruned_graph = prune_mst(self.filtered_graph)
        
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
            straight_curve = Segment.sort_curve(straight_curve)
            next_round.append(straight_curve)            
            self.length_list.append(Segment.seg_len(straight_curve))
            
        # sort by length
        next_round = [x for _, x in sorted(zip(self.length_list, next_round), key=lambda pair: -pair[0])]
        
        self.next_round = next_round
            
        return next_round
    
    def end_to_end_clustering_cpp(self,number_of_endpoint_averaging=10,dist_threshold=30,same_sideness=0.5,align_threshold=0.15,shrinkage=1):
        import time
        
        
        self.fp.calculate_end_to_end_properties(number_of_endpoint_averaging,shrinkage)
        self.endpoints = self.get_end_points()
        self.corrected_end_points = self.get_corrected_end_points()
        self.endtangents = self.get_end_tangents()
        
        start = time.time()
        self.fp.calculate_end_to_end_scores(dist_threshold,same_sideness)        
        print(f'Elapsed time for end-to-end distance calculation: {time.time() - start:.2f} s')
        
        # start = time.time()
        # self.end_ab = self.get_end_ab()
        # self.end_scores = self.get_end_scores()
        
        # self.end_ab = np.array(self.end_ab)
        # self.end_scores = np.array(self.end_scores)
        # print(f'Elapsed time for moving data: {time.time() - start:.2f} s')
        
        # dist_score = self.end_scores[:,0]
        # align_score = self.end_scores[:,1]
        
        # start = time.time()
        # mask = (dist_score < dist_threshold) & (align_score < align_threshold)        
        # edges_with_alignment_weights = [(i, j, align_score[k]) for k, (i, j) in enumerate(self.end_ab)]
        # edges_with_distance_weights = [(i, j, dist_score[k]) for k, (i, j) in enumerate(self.end_ab)]
        # print(f'Elapsed time for creating edges: {time.time() - start:.2f} s')
               
        # filtered_edges = [(i, j, align_score[k]) for k, (i, j) in enumerate(self.end_ab) if mask[k]]
        # self.filtered_graph = nx.Graph()
        # self.filtered_graph.add_nodes_from(range(len(self.segments)*2))
        # self.filtered_graph.add_weighted_edges_from(filtered_edges)
                
        # self.pruned_graph = prune_mst(self.filtered_graph)
        _pruned_edges = self.fp.prune_edges(dist_threshold,align_threshold)
        pruned_edges = [(tmp.u,tmp.v,tmp.weight) for tmp in _pruned_edges]
        
        self.pruned_graph = nx.Graph()
        self.pruned_graph.add_nodes_from(range(len(self.segments)*2))
        self.pruned_graph.add_weighted_edges_from(pruned_edges)
        
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
            straight_curve = Segment.sort_curve(straight_curve)
            next_round.append(straight_curve)            
            self.length_list.append(Segment.seg_len(straight_curve))
            
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
            length_list.append(Segment.seg_len(rr))
        log_bins = np.logspace(np.log10(1),np.log10(2000),100)
        ax.hist(length_list,bins=log_bins)
        ax.set_xscale('log')
        
        return length_list
    def plot_length_histogram(self):
        bins = np.linspace(10,1000,100)
        fig,ax=plt.subplots(1,1)
        plt.hist(self.length_list,bins=bins,density=True)
        

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
            joined = Segment.sort_curve(joined)
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
    
    def check_short_segments(self,length_threshold):
        short_segment_labels = np.where(np.array(self.length_list) < length_threshold)[0]
        nnz = np.count_nonzero(short_segment_labels)
        print(f'Number of short segments: {nnz}')
        trimmed = []
        fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
        for i in range(len(self.end_to_end_cluster)):
            cc_max = self.end_to_end_cluster[i]
            joined = []
            for i_ in cc_max:
                if i_ % 2 == 1:
                    continue
                # ax.plot(self.segments[i_//2][:,0],self.segments[i_//2][:,1],self.segments[i_//2][:,2],'.',alpha=0.2)
                joined.append(self.segments[i_//2])

            joined = np.vstack(joined)
            joined = Segment.sort_curve(joined)
            
            if (self.length_list[i] > length_threshold):
                trimmed.append(joined)
            else:            
                ax.plot(joined[:,0],joined[:,1],joined[:,2],linewidth=1,alpha=0.5)
            
        return trimmed
    
    def check_long_segments(self,length_threshold):
        long_segment_labels = np.where(np.array(self.length_list) > length_threshold)[0]
        nnz = np.count_nonzero(long_segment_labels)
        print(f'Number of long segments: {nnz}')
        trimmed = []
        fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
        for i in range(len(self.end_to_end_cluster)):
            cc_max = self.end_to_end_cluster[i]
            joined = []
            for i_ in cc_max:
                if i_ % 2 == 1:
                    continue
                # ax.plot(self.segments[i_//2][:,0],self.segments[i_//2][:,1],self.segments[i_//2][:,2],'.',alpha=0.2)
                joined.append(self.segments[i_//2])

            joined = np.vstack(joined)
            joined = Segment.sort_curve(joined)
            
            if (self.length_list[i] < length_threshold):
                trimmed.append(joined)
            else:            
                ax.plot(joined[:,0],joined[:,1],joined[:,2],linewidth=1,alpha=0.5)
            
        return trimmed
    
    
    
    def break_segments(self):
        new_segments = []
        for seg in self.segments:
            edge_len = Segment.edge_lengths(seg)
            grph = nx.Graph()
            grph.add_nodes_from(range(len(seg)))

            for i in range(len(seg)-1):
                if edge_len[i] <= np.sqrt(3):
                    grph.add_edge(i,i+1)
                
            clusters = list(nx.connected_components(grph))
            for i,cluster in enumerate(clusters):
                if len(cluster) == 1:                
                    continue
                rr = np.array(seg,dtype=np.float64)            
                new_segments.append(rr[list(cluster)])
            
        return new_segments

    
    def initial_prune_segments(self):
        # simple sort; "robust way?" this only works for relatively straight one.
        pruned = []
        for i,seg in enumerate(self.segments):
            seg = np.unique(seg,axis=0)
            pruned.append(Segment.sort_curve(seg))
            
        return pruned
    
    
    def inspect_segments(self,visualize=False):
        N_segments = len(self.segments)
        segments_length_list = np.zeros(N_segments)
        for i,seg in enumerate(self.segments):
            segments_length_list[i] = np.sum(np.sqrt(np.sum(np.diff(seg,axis=0)**2,axis=1)))   
            
        if visualize:
            fig,ax=plt.subplots(1,1)
            ax.hist(segments_length_list,bins=100)
            ax.set_xlim([0,1000])
        
        from fitting import fit_rod

        segments_error_list = np.zeros(N_segments)
        for i,seg in enumerate(self.segments):
            rr = np.array(seg,dtype=np.float64)
            fit_result = fit_rod(rr,0.00001,10000)
            segments_error_list[i] = fit_result['err']
        
        if visualize:    
            fig,ax=plt.subplots(1,1)
            ax.hist(segments_error_list,bins=100)
            
        print(f'Maximum segment length: {np.max(segments_length_list)} at index {np.argmax(segments_length_list)}')
        print(f'Maximum segment error: {np.max(segments_error_list)} at index {np.argmax(segments_error_list)}')
        
        return segments_length_list,segments_error_list
    
    def break_curved_segments(self,error_threshold,curvature_threshold):
        _,segments_error_list = self.inspect_segments(visualize=False)
        broken_segments_list = []
        for i in np.where(segments_error_list>error_threshold)[0]:
            rr = self.segments[i]
            broken_pieces = Segment.break_curved_rods(rr,curvature_threshold)    
            for bp in broken_pieces:                
                broken_segments_list.append(bp)
                
        new_segments = [seg for i,seg in enumerate(self.segments) if segments_error_list[i]<=error_threshold]
        new_segments = new_segments + broken_segments_list
                        
        return new_segments
    
    def plot(self,ax=None):
        if ax is None:
            fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
            for seg in self.segments:
                ax.plot(seg[:,0],seg[:,1],seg[:,2])
        else:
            for seg in self.segments:
                ax.plot(seg[:,0],seg[:,1],seg[:,2])
    
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
            joined = Segment.sort_curve(joined)
            fr = fit_rod(joined,linearity_threshold=0.0001,radius_curvature_threshold=100000)
            if fr['err'] < fitting_error_threshold:
                print(f'Error: {fr["err"]}')
                
                # connect
                union(_ij[0]//2,_ij[1]//2)
                
                
# %%
# rod_data_root_dir = Path('/Users/yeonsu/Data/steel-rods-xray-data')
# segments_file_path = rod_data_root_dir / 'alpha200_epsilon00' / 'segments.mat'
# mat_obj = loadmat(segments_file_path)
# segments = mat_obj['segments']
# segments = [seg[0] for seg in segments]    

# print(f'Number of segments (original): {len(segments)}')
# for seg in segments:
#     # nx3 array
#     assert seg.shape[1] == 3

# # %%
# segm = Segments(segments)
# segm.prune_segments(1,10)

# %%
if __name__ == '__main__':
    rod_data_root_dir = Path('/Users/yeonsu/Data/steel-rods-xray-data')
    segments_file_path = rod_data_root_dir / 'alpha200_epsilon00' / 'pruned_segments.pkl'

    with open(segments_file_path, 'rb') as f:
        pruned_segments = pickle.load(f)

    # # %%
    # segments = pruned_segments

    # global_centroid = np.mean(np.vstack(segments),axis=0)
    # local_segments = []
    # for i,segment in enumerate(segments):    
    #     if np.any(np.linalg.norm(segment - global_centroid,axis=1) < 500):
    #         local_segments.append(segment)
            
    # segments = local_segments
    # print(f'Number of segments (sampled): {len(segments)}')

    # pruned_segments = segments
    # %%
    segm = Segments(pruned_segments)
    segm.initialize_filament_processing()
    next_round = segm.end_to_end_clustering(number_of_endpoint_averaging=30,dist_threshold=10,align_threshold=0.1)
    # plt.close('all')
    # segm.plot_length_histogram()
    # %%
    # segm.end_to_end_cluster
    degrees = [segm.pruned_graph.degree[node] for node in segm.pruned_graph.nodes]
    np.max(degrees)
    
    # %%
    next_round = segm.end_to_end_clustering_cpp(number_of_endpoint_averaging=30,dist_threshold=10,align_threshold=0.1)
    
    # %%
    import time
    start = time.time()
    tmp0 = prune_mst(segm.filtered_graph)
    print(f'Elapsed time: {time.time() - start:.2f} s')
    # %%
    segm.filtered_graph.number_of_edges()
    # %%
    start = time.time()
    tmp = segm.fp.prune_edges(30,0.1)
    print(f'Elapsed time here: {time.time() - start:.2f} s')
    # %%
    edges = []
    for entry in tmp:
        edges.append((entry.u,entry.v,entry.weight))
    # %%
    pruned_graph = nx.Graph()
    pruned_graph.add_nodes_from(range(len(pruned_segments)*2))
    pruned_graph.add_weighted_edges_from(edges)
    # %%
    ccs = list(nx.connected_components(pruned_graph))
    len(ccs)
    # %%
    tmp0.number_of_edges()