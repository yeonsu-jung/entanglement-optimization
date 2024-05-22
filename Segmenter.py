import os
import pickle
import numpy as np
import networkx as nx
from pathlib import Path
from matplotlib import pyplot as plt
from clustering import find_connected_components, explode_local_cluster
from fitting import prep_svd_cylinder, fit_rod
import filamentprocessing


class RodProcessing:
    def __init__(self, data_dir, subcluster_error_threshold, subcluster_length_threshold):
        self.data_dir = Path(data_dir)
        self.segments_file_path = self.data_dir / 'pruned_segments.pkl'
        self.connectivity_file_path = self.data_dir / 'ijscores.pkl'
        self.segments = self.load_segments()
        
        
        
        
        self.svd_cylinders, self.centroids, self.orientations = None, None, None
        self.ijscore = self.load_ijscore()
        self.subcluster_error_threshold = subcluster_error_threshold
        self.subcluster_length_threshold = subcluster_length_threshold
        self.svd_cylinders, self.centroids, self.orientations = prep_svd_cylinder(self.segments, scale_factor=0.99)
        
    
    def load_segments(self):
        if os.path.exists(self.segments_file_path):
            with open(self.segments_file_path, 'rb') as f:
                segments =  pickle.load(f)
                local_segments = []
                for i,segment in enumerate(segments):
                    if np.all(np.sum((segment - np.array([1000,1000,500]))**2,axis=1) < 50000):
                        local_segments.append(segment)
                        
                return local_segments
        else:
            raise FileNotFoundError(f'Segments file not found: {self.segments_file_path}')

    def load_ijscore(self):
        if os.path.exists(self.connectivity_file_path):
            with open(self.connectivity_file_path, 'rb') as f:
                return pickle.load(f)
        else:
            fp = filamentprocessing.FilamentProcessing(self.segments, 2500, 0.15, 0.99)
            ij = fp.get_svd_ij()
            scores = fp.get_svd_scores()
            ijscore = np.hstack([ij, scores])
            with open(self.connectivity_file_path, 'wb') as f:
                pickle.dump(ijscore, f)
            return ijscore

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
        fig, ax = plt.subplots(1, 1, subplot_kw={'projection': '3d'})
        for gcl in good_cl:
            joined = np.vstack([self.segments[i] for i in gcl])
            joined = self.sort_curve(joined)
            ax.plot(joined[:, 0], joined[:, 1], joined[:, 2], linewidth=0.5)

        joined_segment_length_list = np.zeros(len(good_cl))
        for i, gcl in enumerate(good_cl):
            joined = np.vstack([self.segments[i] for i in gcl])
            joined = self.sort_curve(joined)
            joined_segment_length_list[i] = self.seg_len(joined)

        fig, ax = plt.subplots(1, 1)
        ax.hist(joined_segment_length_list, bins=100)
        return joined_segment_length_list

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

    def process_clusters(self, clusters):
        from clustering import subclustering_by_mst_length_lowerbound

        good_clusters, not_yet_nodes = subclustering_by_mst_length_lowerbound(
            clusters, self.segments, self.svd_cylinders, self.subcluster_error_threshold, self.subcluster_length_threshold
        )
        joined_segment_length_list = self.inspect_clustering(good_clusters)

        fig, ax = plt.subplots(1, 1, subplot_kw={'projection': '3d'})
        for gcl in good_clusters:
            joined = np.vstack([self.segments[i] for i in gcl])
            joined = self.sort_curve(joined)
            ax.plot(joined[:, 0], joined[:, 1], joined[:, 2], linewidth=0.5)

        for i_ny in not_yet_nodes:
            ax.plot(self.segments[i_ny][:, 0], self.segments[i_ny][:, 1], self.segments[i_ny][:, 2], color='k', linewidth=0.5, alpha=0.1)

        return good_clusters, not_yet_nodes

    def run(self):
        clusters = self.initial_guess()
        good_clusters, not_yet_nodes = self.process_clusters(clusters)
        return good_clusters, not_yet_nodes


if __name__ == "__main__":
    data_dir = Path('/Users/yeonsu/Data/steel-rods-xray-data') / 'alpha200_epsilon00'
    
    subcluster_error_threshold = 0.5
    subcluster_length_threshold = 50

    rod_processing = RodProcessing(data_dir, subcluster_error_threshold, subcluster_length_threshold)
    while True:
        good_clusters, not_yet_nodes = rod_processing.run()
        # update clusters
        
        # change the thresholds
        
        # repeat until no more clusters are found
