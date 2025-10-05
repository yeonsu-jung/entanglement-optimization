# %%
import os
import pickle
import numpy as np
import networkx as nx
from pathlib import Path
from matplotlib import pyplot as plt
import logging
from clustering import find_connected_components, subclustering_by_mst_length_lowerbound
from fitting import prep_svd_cylinder, fit_rod
import filamentprocessing

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_cluster_map(filepath):
    clusters_from_file = []
    with open(filepath, 'r') as f:
        clusters_at_iteration = []
        for ln in f:
            if ln.startswith('='):
                clusters_from_file.append(clusters_at_iteration)
                clusters_at_iteration = []
            else:
                cluster = [int(x) for x in ln.split()]
                clusters_at_iteration.append(cluster)
    return clusters_from_file

def rowwise_norm(x):
        return np.sqrt(np.sum(x**2, axis=1))
    
class SegmentLoader:
    def __init__(self, segments_file_path, connectivity_file_path):
        self.segments_file_path = segments_file_path
        self.connectivity_file_path = connectivity_file_path
    
    def load_segments(self):
        if os.path.exists(self.segments_file_path):
            with open(self.segments_file_path, 'rb') as f:
                segments0 = pickle.load(f)
            # segments = []
            # for seg in segments0:
            #     if np.any(rowwise_norm(seg- np.array([1000,1000,500])) < 250):
            #         segments.append(seg)
            segments = segments0
            print(f'Number of segments: {len(segments)}')            
            # fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
            # for seg in segments[:]:
            #     ax.plot(seg[:,0],seg[:,1],seg[:,2],linewidth=0.5)
                    
            return segments
        else:
            raise FileNotFoundError(f'Segments file not found: {self.segments_file_path}')
    
    def load_ijscore(self, segments):
        if os.path.exists(self.connectivity_file_path):
            with open(self.connectivity_file_path, 'rb') as f:
                return pickle.load(f)
        else:
            fp = filamentprocessing.FilamentProcessing(segments, 2500, 0.15, 0.99)
            ij = fp.get_svd_ij()
            scores = fp.get_svd_scores()
            ijscore = np.hstack([ij, scores])
            with open(self.connectivity_file_path, 'wb') as f:
                pickle.dump(ijscore, f)
            return ijscore


class Partition:
    """
    Partition of a set of elements - to encoding the clustering of segments
    Initial segments in RodProcessing will not be changed, but the partition will be updated    
    """
    def __init__(self, elements):
        self.elements = elements
        self.partition = [[i] for i in range(len(elements))]

class RodProcessing:
    def __init__(self, data_dir, thresholds):
        self.data_dir = Path(data_dir)
        self.segments_file_path = self.data_dir / 'pruned_segments.pkl'
        self.connectivity_file_path = self.data_dir / 'ijscores.pkl'
        
        self.segment_loader = SegmentLoader(self.segments_file_path, self.connectivity_file_path)
        self.initial_segments = self.segment_loader.load_segments() # can we freeze?
        self.segments = self.initial_segments
        self.ijscore = self.segment_loader.load_ijscore(self.initial_segments)
               
        self.clustered = []
        self.unclustered = [[i] for i in range(len(self.segments))]
        self.current_clusters = self.unclustered
        
        self.fixed_nodes = set()
        self.trash_nodes = set()
        
        self.thresholds = thresholds
        self.clustering_alignment_threshold = thresholds['clustering_alignment_threshold']
        self.clustering_distance_threshold = thresholds['clustering_distance_threshold']
        self.subcluster_error_threshold = thresholds['subcluster_error_threshold']
        self.subcluster_length_threshold = thresholds['subcluster_length_threshold']
        
        self.svd_cylinders, self.centroids, self.orientations = prep_svd_cylinder(self.segments, scale_factor=0.99)

    @staticmethod
    def sort_curve(rr):
        centroid = np.mean(rr, axis=0)
        rr_centered = rr - centroid
        _, _, V = np.linalg.svd(rr_centered, full_matrices=False)
        v1 = V[0, :]
        orientation = v1 * np.sign(np.sum(v1 * (rr_centered[-1, :] - rr_centered[0, :])))
        slist = np.dot((rr - centroid), orientation)
        sorted_indices = np.argsort(slist)
        return centroid + rr_centered[sorted_indices]

    @staticmethod
    def seg_len(seg):
        return np.sum(np.sqrt(np.sum(np.diff(seg, axis=0) ** 2, axis=1)))

    def inspect_clustering(self, good_cl):
        joined_segment_length_list = np.zeros(len(good_cl))
        for i, gcl in enumerate(good_cl):
            joined = np.vstack([self.initial_segments[i] for i in gcl])
            joined = self.sort_curve(joined)
            joined_segment_length_list[i] = self.seg_len(joined)
        return joined_segment_length_list

    def initial_guess(self):
        ij = self.ijscore[:, :2]
        scores = self.ijscore[:, 2:]
        dist_score = scores[:, 0]
        align_score = scores[:, 1]
        initial_mask = (align_score < self.clustering_alignment_threshold) & (dist_score < self.clustering_distance_threshold)
        
        Graph0 = nx.Graph()
        Graph0.add_nodes_from(range(len(self.segments)))
        Graph0.add_weighted_edges_from(zip(ij[initial_mask, 0], ij[initial_mask, 1], dist_score[initial_mask]))
        clusters = find_connected_components(Graph0)
        return clusters
    
    def thresholding(self):
        ij = self.ijscore[:, :2]
        scores = self.ijscore[:, 2:]
        dist_score = scores[:, 0]
        align_score = scores[:, 1]
        initial_mask = (align_score < self.clustering_alignment_threshold) & (dist_score < self.clustering_distance_threshold)
        
        Graph0 = nx.Graph()
        Graph0.add_nodes_from(range(len(self.segments)))
        Graph0.add_edges_from(zip(ij[initial_mask, 0], ij[initial_mask, 1]))
        clusters = find_connected_components(Graph0)
        return clusters

    def process_clusters(self, clusters):
        good_clusters, local_trash_nodes = subclustering_by_mst_length_lowerbound(
            clusters, self.segments, self.svd_cylinders, self.subcluster_error_threshold, self.subcluster_length_threshold
        )
        
        good_nodes = []
        if good_clusters:
            good_nodes = np.hstack(good_clusters)
        not_yet_nodes = np.setdiff1d(range(len(self.segments)), good_nodes)
    
        for i_ny in not_yet_nodes:
            rr = self.segments[i_ny]
            fit_result = fit_rod(rr, linearity_threshold=1e-10, radius_curvature_threshold=1e10)
            rr_len = self.seg_len(rr)
            
            if fit_result['err'] < self.subcluster_error_threshold and rr_len > self.subcluster_length_threshold and rr_len < 750:
                good_clusters.append(np.array([i_ny]))
                local_trash_nodes.discard(i_ny)
            
        good_nodes = []
        if good_clusters:
            good_nodes = np.hstack(good_clusters)
        
        return good_clusters, local_trash_nodes
    
    def update_clusters(self, good_clusters, local_trash_nodes):
        global_good_clusters = []
        for local_gcl in good_clusters:
            gcl = np.hstack([np.hstack(self.current_clusters[i]) for i in local_gcl])
            global_good_clusters.append(gcl)
            
        self.trash_nodes.update(local_trash_nodes)
        
        self.clustered = global_good_clusters
        all_clustered_nodes = set(node for cluster in self.clustered for node in cluster)
        self.unclustered = list(set(range(len(self.initial_segments))) - all_clustered_nodes - self.trash_nodes - self.fixed_nodes)
        self.current_clusters = self.clustered + [[i] for i in self.unclustered]
        
        for i_ny in self.unclustered:
            rr = self.initial_segments[i_ny]
            fr = fit_rod(rr, 1e-10, 1e10)
            leng = self.seg_len(rr)
            if fr['err'] < self.subcluster_error_threshold and leng > self.subcluster_length_threshold and leng < 750:
                self.clustered.append(np.array([i_ny]))
            if leng < 5:
                self.trash_nodes.add(i_ny)
        
        for gcl in self.clustered:
            if np.isin(gcl, self.trash_nodes).any() or np.isin(gcl, self.fixed_nodes).any():
                logging.error('Error: Invalid nodes are in clustered segments')
        
        self.unclustered = list(set(range(len(self.initial_segments))) - all_clustered_nodes - self.trash_nodes - self.fixed_nodes)
        self.current_clusters = self.clustered + [[i] for i in self.unclustered]
        
        for cl in self.current_clusters:
            if np.isin(cl, self.trash_nodes).any() or np.isin(cl, self.fixed_nodes).any():
                logging.error('Error: Invalid nodes are in clustered segments')
        
        new_segments = []
        for cl in self.current_clusters:
            joined = np.vstack([self.initial_segments[i] for i in cl])
            joined = self.sort_curve(joined)
            new_segments.append(joined)
    
        self.segments = new_segments
        
        self.svd_cylinders, self.centroids, self.orientations = prep_svd_cylinder(self.segments, scale_factor=0.99)
        fp = filamentprocessing.FilamentProcessing(new_segments, 2500, 0.15, 0.99)
        ij = fp.get_svd_ij()
        scores = fp.get_svd_scores()
        self.ijscore = np.hstack([ij, scores])
        
    def log(self, iteration):
        
        fig, ax = plt.subplots(1, 1, subplot_kw={'projection': '3d'})
        for i_ny in self.unclustered:
            ax.plot(self.initial_segments[i_ny][:, 0], self.initial_segments[i_ny][:, 1], self.initial_segments[i_ny][:, 2], linewidth=0.5)
        plt.savefig(self.data_dir / 'figures' / f'unclustered_alpha200_epsilon00_{iteration}.png')
        plt.close()
        
        length_list = []
        for gcl in self.current_clusters:
            joined = np.vstack([self.initial_segments[i] for i in gcl])
            joined = self.sort_curve(joined)
            length_list.append(self.seg_len(joined))        
        length_list = np.array(length_list)
        
        fig, ax = plt.subplots(1, 1, subplot_kw={'projection': '3d'})
        for i in np.where(length_list < 15)[0]:
            joined = np.vstack([self.initial_segments[i] for i in self.current_clusters[i]])
            joined = self.sort_curve(joined)
            ax.plot(joined[:, 0], joined[:, 1], joined[:, 2], linewidth=0.5)
            ax.set_title(f'Number of short rods: {np.count_nonzero(length_list < 15)}')
        plt.savefig(self.data_dir / 'figures' / f'short_alpha200_epsilon00_{iteration}.png')
        plt.close()            
        
        fig, ax = plt.subplots(1, 1)
        ax.hist(length_list, bins=100)
        ax.set_title(f'Number of long enough rods: {np.count_nonzero(length_list > 550)}')
        plt.savefig(self.data_dir / 'figures' / f'histogram_alpha200_epsilon00_{iteration}.png')
        plt.close()
        
        # with open(self.`data_dir / 'rod_processing.pkl', 'wb') as f:
        #     pickle.dump(rod_processing, f)
            
    def update_thresholds(self, thresholds):
        
        self.clustering_alignment_threshold = min(thresholds['clustering_alignment_threshold'],0.1)
        self.clustering_distance_threshold = min(thresholds['clustering_distance_threshold'], 2000)
        self.subcluster_error_threshold = min(thresholds['subcluster_error_threshold'], 2.5)
        self.subcluster_length_threshold = max(thresholds['subcluster_length_threshold'], 25)
                       
        
    def cluster_matp(self):
        return 1

    def run(self):
        clusters = self.initial_guess()
        good_clusters, local_trash_nodes = self.process_clusters(clusters)
        return good_clusters, local_trash_nodes


def first_run():
    data_dir = Path('/Users/yeonsu/Data/steel-rods-xray-data/alpha200_epsilon00')
    
    thresholds = {
        'subcluster_error_threshold': 1.0,
        'subcluster_length_threshold': 50,
        'clustering_alignment_threshold': 0.01,
        'clustering_distance_threshold': 50
    }
    
    rod_processing = RodProcessing(data_dir, thresholds)
    
    iteration = 0
    max_iterations = 1000 
    while iteration < max_iterations:
        iteration += 1
        logging.info(f"Iteration: {iteration}")
        good_clusters, local_trash_nodes = rod_processing.run()        
        rod_processing.update_clusters(good_clusters, local_trash_nodes)
        
        # update thresholds
        thresholds['subcluster_error_threshold'] = thresholds['subcluster_error_threshold'] + 0.01
        thresholds['subcluster_length_threshold'] = thresholds['subcluster_length_threshold'] - 5
        thresholds['clustering_alignment_threshold'] = thresholds['clustering_alignment_threshold'] + 0.01
        # thresholds['clustering_distance_threshold'] = thresholds['clustering_distance_threshold'] * 0.99        
        rod_processing.update_thresholds(thresholds)        
        
        logging.info(f"Number of good clusters: {len(rod_processing.clustered)} / Number of unclustered segments: {len(rod_processing.unclustered)}")        
        
        if iteration % 1 == 0:
            rod_processing.log(iteration)
    
    logging.info("Processing completed.")

def load_and_analyze(sample_id):
    data_dir = Path('/Users/yeonsu/Data/steel-rods-xray-data/') / sample_id
    with open(data_dir / 'rod_processing.pkl', 'rb') as f:
        rod_processing = pickle.load(f)
        
    print(f'Number of current clusters: {len(rod_processing.current_clusters)}')
    
    initial_segments = rod_processing.initial_segments
    clusters = rod_processing.current_clusters
    
    def surjective_map(clusters, initial_segments):
        current_segments = []
        for i, cl in enumerate(clusters):
            joined = np.vstack([initial_segments[i] for i in cl])
            joined = RodProcessing.sort_curve(joined)
            current_segments.append(joined)
        return current_segments
    
    current_segments = surjective_map(clusters, initial_segments)
    
    fig, ax = plt.subplots(1, 1, figsize=(10,10), subplot_kw={'projection': '3d'})
    for cc in current_segments[-300:-100]:
        ax.plot(cc[:, 0], cc[:, 1], cc[:, 2], linewidth=0.5)
    plt.show()    
    
        
        
    return
    
# if __name__ == "__main__":
    # sample_id = 'alpha200_epsilon00'
    # load_and_analyze(sample_id)
    # first_run()
# %%
sample_id = 'alpha200_epsilon00'

data_dir = Path('/Users/yeonsu/Data/steel-rods-xray-data/alpha200_epsilon00')
thresholds = {
    'subcluster_error_threshold': 1.0,
    'subcluster_length_threshold': 2000,
    'clustering_alignment_threshold': 0.01,
    'clustering_distance_threshold': 50
}

rod_processing = RodProcessing(data_dir, thresholds)

# %%
thresholds['clustering_alignment_threshold'] = 0.05
rod_processing.update_thresholds(thresholds)
clusters = rod_processing.thresholding()
len(clusters)
# %%

class RodEndConditions:
    def __init__(self, segments):
        self.segments = segments
        self.end_points = np.vstack([seg[[0, -1]] for seg in segments])
        self.end_tangents = np.vstack([seg[[1, -1]] - seg[[0, -2]] for seg in segments])
        
        
    
    

