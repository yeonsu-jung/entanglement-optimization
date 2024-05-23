import os
import pickle
import numpy as np
import networkx as nx
from pathlib import Path
from matplotlib import pyplot as plt
from clustering import find_connected_components, subclustering_by_mst_length_lowerbound
from fitting import prep_svd_cylinder, fit_rod
import filamentprocessing
import datetime



def load_cluster_map(filepath):
    clusters_from_file = []    
    with open(filepath, 'r') as f:
        lines = f.readlines()
        clusters_at_iteration = []
        for ln in lines:
            if ln.startswith('='):
                clusters_from_file.append(clusters_at_iteration)
                clusters_at_iteration = []
            else:
                cluster = [int(x) for x in ln.split()]
                clusters_at_iteration.append(cluster)
            
        
    return clusters_from_file

class RodProcessing:
    def __init__(self, data_dir, thresholds, previous_file=None):
        self.data_dir = Path(data_dir)
        self.segments_file_path = self.data_dir / 'pruned_segments.pkl'
        self.connectivity_file_path = self.data_dir / f'ijscores.pkl'
        
        self.initial_segments = self.load_segments()
        self.segments = self.initial_segments
        self.ijscore = self.load_ijscore()
        
        self.fixed_nodes = set()
        self.trash_nodes = set()
        
        self.thresholds = thresholds  
        self.clustering_alignment_threshold = thresholds['clustering_alignment_threshold']
        self.clustering_distance_threshold = thresholds['clustering_distance_threshold']
        self.subcluster_error_threshold = thresholds['subcluster_error_threshold']
        self.subcluster_length_threshold = thresholds['subcluster_length_threshold']
        
        self.svd_cylinders, self.centroids, self.orientations = prep_svd_cylinder(self.segments, scale_factor=0.99)
        
        if previous_file is None:
            # load the previous
            cluster_output_name = f'clustered_segments_N{len(self.initial_segments)}.txt'
            self.cluster_output_path = self.data_dir / cluster_output_name
            
            self.clustered = []
            self.unclustered = list(range(len(self.initial_segments)))  # Initially, all segments are unclustered
            self.current_clusters = self.clustered + [[i] for i in self.unclustered]
            
        elif os.path.exists(previous_file):
            # read       
            self.cluster_output_path = previous_file
            all_cluster_maps = load_cluster_map(previous_file)
            self.clustered = all_cluster_maps[-1]
            self.unclustered = list(set(range(len(self.initial_segments))))            
            self.current_clusters = self.clustered + [[i] for i in self.unclustered]
    
    def load_segments(self):
        if os.path.exists(self.segments_file_path):
            with open(self.segments_file_path, 'rb') as f:
                segments = pickle.load(f)
                local_segments = [segment for segment in segments if np.all(np.sum((segment - np.array([1000, 1000, 500]))**2, axis=1) < 500**2)]
                segments = local_segments
                return segments
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
        # rr_len = np.sum(np.sqrt(np.sum(np.diff(rr,axis=0)**2,axis=1)))
        return np.sum(np.sqrt(np.sum(np.diff(seg, axis=0) ** 2, axis=1)))

    def inspect_clustering(self, good_cl):
        # fig, ax = plt.subplots(1, 1, subplot_kw={'projection': '3d'})
        for gcl in good_cl:
            joined = np.vstack([self.initial_segments[i] for i in gcl])
            joined = self.sort_curve(joined)
            # ax.plot(joined[:, 0], joined[:, 1], joined[:, 2], linewidth=0.5)

        joined_segment_length_list = np.zeros(len(good_cl))
        for i, gcl in enumerate(good_cl):
            joined = np.vstack([self.initial_segments[i] for i in gcl])
            joined = self.sort_curve(joined)
            joined_segment_length_list[i] = self.seg_len(joined)

        # fig, ax = plt.subplots(1, 1)
        # ax.hist(joined_segment_length_list, bins=100)
        return joined_segment_length_list

    def initial_guess(self):
        ij = self.ijscore[:, :2]
        scores = self.ijscore[:, 2:]

        dist_score = scores[:, 0]
        align_score = scores[:, 1]
        fit_score = scores[:, 2]
        initial_mask = (align_score < self.clustering_alignment_threshold) & (dist_score < self.clustering_distance_threshold)
        
        """
        length_list = []
        length_list2 = []
        for gcl in self.current_clusters:
            joined=np.vstack([self.initial_segments[i] for i in gcl])
            joined=self.sort_curve(joined)
            length_list.append(self.seg_len(joined))
            length_list2.append(np.sum([self.seg_len(self.initial_segments[i]) for i in gcl]))
        length_list = np.array(length_list)
        length_list2 = np.array(length_list2)
        
        np.where(length_list2 > 550)[0]
        
        joined = np.vstack([self.initial_segments[i] for i in self.current_clusters[31]])
        joined = self.sort_curve(joined)
        
        fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
        ax.plot(joined[:,0],joined[:,1],joined[:,2],linewidth=0.5)        
        ax.axis('equal')
        
        fig,ax=plt.subplots(1,1)
        ax.hist(length_list2[length_list2>500], bins=100)
        
        np.count_nonzero(length_list2 > 550)
            
        fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
        for i in np.where(length_list2 < 200)[0][:100]:
            joined=np.vstack([self.initial_segments[i] for i in self.current_clusters[i]])
            joined=self.sort_curve(joined)
            ax.plot(joined[:,0],joined[:,1],joined[:,2],linewidth=0.5)
            # ax.set_title(f'Number of short rods: {np.count_nonzero(length_list < 15)}')
        plt.close()
        
        cc = self.current_clusters[np.argmax(length_list2)]
        fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
        leng = 0
        for i in cc:
            ax.plot(self.initial_segments[i][:,0],self.initial_segments[i][:,1],self.initial_segments[i][:,2],linewidth=0.5)
            leng += self.seg_len(self.initial_segments[i])
            print(self.seg_len(self.initial_segments[i]))
        print(leng)
        
        np.unique(cc)
        """    

        Graph0 = nx.Graph()
        Graph0.add_nodes_from(range(len(self.segments)))
        Graph0.add_weighted_edges_from(zip(ij[initial_mask, 0], ij[initial_mask, 1], dist_score[initial_mask]))
        clusters = find_connected_components(Graph0)
        cluster_size_list = [len(x) for x in clusters]
        max_cluster_size = np.max(cluster_size_list)
        
        return clusters

    def process_clusters(self, clusters):
        good_clusters,local_trash_nodes = subclustering_by_mst_length_lowerbound(
            clusters, self.segments, self.svd_cylinders, self.subcluster_error_threshold, self.subcluster_length_threshold
        )
        
        good_nodes = []
        if len(good_clusters) > 0:
            good_nodes = np.hstack(good_clusters)
        not_yet_nodes = np.setdiff1d(range( len(self.segments) ),good_nodes)
        # print(good_nodes.shape)
    
        for i_ny in not_yet_nodes:
            rr = self.segments[i_ny]
            fit_result = fit_rod(rr,linearity_threshold=1e-10,radius_curvature_threshold=1e10)
            # rr_len = np.sum(np.sqrt(np.sum(np.diff(rr,axis=0)**2,axis=1)))
            rr_len = self.seg_len(rr)
            
            if fit_result['err'] < self.subcluster_error_threshold and (rr_len > self.subcluster_length_threshold) and (rr_len < 750):
                good_clusters.append(np.array([i_ny]))                
                local_trash_nodes.discard(i_ny)
            
        good_nodes = []
        if len(good_clusters) > 0:
            good_nodes = np.hstack(good_clusters)
        
        # not_yet_nodes = np.setdiff1d(range( len(self.segments) ),good_nodes)
        
        return good_clusters, local_trash_nodes
    
    def update_clusters(self, good_clusters,local_trash_nodes):
        global_good_clusters = []
        for local_gcl in good_clusters:
            gcl = np.hstack([ np.hstack(self.current_clusters[i])  for i in local_gcl])
            global_good_clusters.append(gcl)
            
        for local_tn in local_trash_nodes:
            self.trash_nodes.add(local_tn)
        
        self.clustered = global_good_clusters
        all_clustered_nodes = set(node for cluster in self.clustered for node in cluster) # in global
        self.current_clusters = self.clustered + [[i] for i in self.unclustered]
        
        for i_ny in self.unclustered:
            rr = self.initial_segments[i_ny]
            fr = fit_rod(rr,1e-10,1e10)
            leng = self.seg_len(rr)
            if ((fr['err'] < self.subcluster_error_threshold) and (leng > self.subcluster_length_threshold) & (leng < 750)):
                self.clustered.append(np.array([i_ny])) # super ugly...
                
            if leng < 5:
                self.trash_nodes.add(i_ny)
                
        # check sanity
        for gcl in self.clustered:
            if np.isin(gcl, self.trash_nodes).any():
                print('Error: Trash nodes are in clustered segments')
            elif np.isin(gcl, self.fixed_nodes).any():
                print('Error: Excluded nodes are in clustered segments')
        
        self.unclustered = list(set(range(len(self.initial_segments))) - all_clustered_nodes - self.trash_nodes - self.fixed_nodes)
        self.current_clusters = self.clustered + [[i] for i in self.unclustered]
        
        # check sanity
        for cl in self.current_clusters:
            if np.isin(cl, self.trash_nodes).any():
                print('Error: Trash nodes are in clustered segments')
            elif np.isin(cl, self.fixed_nodes).any():
                print('Error: Excluded nodes are in clustered segments')
        
        
        new_segments = []
        for cl in self.current_clusters:
            joined = np.vstack([self.initial_segments[i] for i in cl])
            joined = self.sort_curve(joined)
            new_segments.append(joined)
    
        # Update self.segments with new_segments
        self.segments = new_segments
        
        # Recompute svd_cylinders and ijscore
        self.svd_cylinders, self.centroids, self.orientations = prep_svd_cylinder(self.segments, scale_factor=0.99)
        fp = filamentprocessing.FilamentProcessing(new_segments, 2500, 0.15, 0.99)
        ij = fp.get_svd_ij()
        scores = fp.get_svd_scores()
        self.ijscore = np.hstack([ij, scores])
        
    def cluster_matp(self):
        # whenever update segments, we have a surjective map from initial_segments to segments
        # what is it?
        # it is a list of indices
        return 1
        

    def run(self):
        clusters = self.initial_guess()
        good_clusters,local_trash_nodes = self.process_clusters(clusters)
        return good_clusters,local_trash_nodes


if __name__ == "__main__":
    data_dir = Path('/Users/yeonsu/Data/steel-rods-xray-data') / 'alpha200_epsilon00'
    
    thresholds = {'subcluster_error_threshold': 2.5,
                    'subcluster_length_threshold': 10,
                    'clustering_alignment_threshold': 0.01,
                    'clustering_distance_threshold': 2000}

    # rod_processing = RodProcessing(data_dir,thresholds, previous_file='/Users/yeonsu/Data/steel-rods-xray-data/alpha200_epsilon00/clustered_segments_N8916.txt')
    # rod_processing = RodProcessing(data_dir,thresholds,'/Users/yeonsu/Data/steel-rods-xray-data/alpha200_epsilon00/clustered_segments_N82792.txt')
    
    # rod_processing = RodProcessing(data_dir,thresholds)
    
    with open(data_dir / 'rod_processing.pkl', 'rb') as f:
        rod_processing = pickle.load(f)
    
    # distance_threshold = 2000
    # alignment_threshold = 0.1
    # rod_processing.clustering_alignment_threshold = 0.015

    # clusters = rod_processing.initial_guess()
    # cluster_size_list = [len(x) for x in clusters]
    # print(f'Initial number of clusters: {len(clusters)}')
    # print(f'Max cluster size: {np.max(cluster_size_list)}')

    # cc = clusters[np.argmax(cluster_size_list)]
    # fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
    # for i in cc:
    #     ax.plot(rod_processing.segments[i][:,0],rod_processing.segments[i][:,1],rod_processing.segments[i][:,2],linewidth=0.5)
        
    # import importlib
    # import fitting
    # importlib.reload(fitting)
    # from fitting import fit_rod
    
    # fit_result = fit_rod(rod_processing.segments[cc[3]],1e-10,1e10)
    
    # _,s,_ = np.linalg.svd(rod_processing.segments[cc[0]])
    
    # pl_list = np.zeros(len(clusters))
    # for i,cc in enumerate(clusters):
    #     joined = np.vstack([rod_processing.segments[i] for i in cc])
    #     joined = rod_processing.sort_curve(joined)
    #     fr = fit_rod(joined,1e-10,1e10)
    #     pl = fr['planarity']
    #     pl_list[i] = pl
        
    # fig,ax=plt.subplots(1,1)
    # ax.hist(pl_list,bins=100)
        
    # np.argsort(pl_list)[::-1]
    # np.max(pl_list)
    
    # import clustering
    # importlib.reload(clustering)
    # from clustering import subclustering_by_mst_length_lowerbound
    
    # cc = clusters[np.argmax(pl_list)] 
    # cc = clusters[8111]
    # fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
    # for i in cc:
    #     ax.plot(rod_processing.segments[i][:,0],rod_processing.segments[i][:,1],rod_processing.segments[i][:,2],'-',linewidth=0.5)
    # ax.axis('equal')
        
    # rod_processing.subcluster_error_threshold = 2.5
    # good_clusters,local_trash_nodes = subclustering_by_mst_length_lowerbound(
    #         clusters, rod_processing.segments, rod_processing.svd_cylinders, rod_processing.subcluster_error_threshold, rod_processing.subcluster_length_threshold
    #     )
    
    # good_clusters

    iteration = 0
    while True:
        iteration += 1
        print(f"Iteration: {iteration}")
        good_clusters,local_trash_nodes = rod_processing.run()        
        rod_processing.update_clusters(good_clusters,local_trash_nodes)
        
        # Adjust thresholds if needed
        rod_processing.clustering_alignment_threshold += 0.005
        rod_processing.subcluster_error_threshold += 0.005
        # rod_processing.subcluster_length_threshold *= 1.005
        
        rod_processing.clustering_alignment_threshold = min(rod_processing.clustering_alignment_threshold, 0.5)
        rod_processing.subcluster_error_threshold = min(rod_processing.subcluster_error_threshold, 2.0)
        rod_processing.subcluster_length_threshold = min(rod_processing.subcluster_length_threshold, 600)
        
        print(f"Number of good clusters: {len(rod_processing.clustered)} / Number of unclustered segments: {len(rod_processing.unclustered)}")
            
        # '/Users/yeonsu/Data/steel-rods-xray-data/alpha200_epsilon00/clustered_segments.txt'
        with open(rod_processing.cluster_output_path, 'a') as f:
            for item in rod_processing.clustered:
                for numbers in item:
                    f.write("%s " % numbers)
                f.write("\n")
            f.write("====================================\n")

        # If the stopping condition is reached, break the loop
        if iteration % 1 == 0:
            fig,ax=plt.subplots(1,1,    subplot_kw={'projection':'3d'})
            for i_ny in rod_processing.unclustered:
                ax.plot(rod_processing.initial_segments[i_ny][:,0],rod_processing.initial_segments[i_ny][:,1],rod_processing.initial_segments[i_ny][:,2],linewidth=0.5)
            plt.savefig(f'/Users/yeonsu/Figures/Segmenter_testing/unclustered_alpha200_epsilon00_{iteration}.png')
            plt.close()
            
            
            length_list = []
            for gcl in rod_processing.current_clusters:
                joined=np.vstack([rod_processing.initial_segments[i] for i in gcl])
                joined=rod_processing.sort_curve(joined)
                length_list.append(rod_processing.seg_len(joined))
            
            length_list = np.array(length_list)
            
            fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
            for i in np.where(length_list < 15)[0]:
                joined=np.vstack([rod_processing.initial_segments[i] for i in rod_processing.current_clusters[i]])
                joined=rod_processing.sort_curve(joined)
                ax.plot(joined[:,0],joined[:,1],joined[:,2],linewidth=0.5)
                ax.set_title(f'Number of short rods: {np.count_nonzero(length_list < 15)}')
            plt.savefig(f'/Users/yeonsu/Figures/Segmenter_testing/short_alpha200_epsilon00_{iteration}.png')
            plt.close()            
                
            length_list = np.array(length_list)
            fig,ax=plt.subplots(1,1)
            ax.hist(length_list, bins=100)
            # ax.set_ylim([0, len(rod_processing.initial_segments)/10])
            ax.set_title(f'Number of long enough rods: {np.count_nonzero(length_list > 550)}')
            plt.savefig(f'/Users/yeonsu/Figures/Segmenter_testing/hist_alpha200_epsilon00_{iteration}.png')
            plt.close()
            
            
            dump_path = rod_processing.data_dir / f'rod_processing.pkl'
            with open(dump_path, 'wb') as f:
                pickle.dump(rod_processing, f)
        
        if (iteration-1) >= 1000:  # For example, stop after 100 iterations
            break
    
    print