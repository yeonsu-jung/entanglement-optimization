# %%
# %matplotlib qt
import numpy as np
import sys
sys.path.append('/Users/yeonsu/GitHub/entanglement-optimization')
import Segments
from pathlib import Path
from scipy.io import loadmat,savemat
from matplotlib import pyplot as plt
import os
# %%
for alpha in [66,76,100,200]:
    epsilon = 0
    matfile_path = Path(f'/Users/yeonsu/GitHub/entanglement/data/alpha{alpha}_epsilon{epsilon:02d}/segments.mat')
    matobj = loadmat(f'/Users/yeonsu/GitHub/entanglement/data/alpha{alpha}_epsilon{epsilon:02d}/segments.mat',simplify_cells=True)
    segments = matobj['segments']
    print(f'alpha: {alpha}, epsilon: {epsilon}, number of segments: {len(segments)}')
    
    # if seg is 1d array it should be converted to 2d array
    for i in range(len(segments)):
        if len(segments[i].shape) == 1:
            segments[i] = segments[i].reshape(-1,3)

    save_folder = Path('perform-segmentation') / f'alpha{alpha}_epsilon{epsilon:02d}'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        
    tracker = 0
    dist_threshold = 15
    dist_threshold_inc = 1
    
    align_threshold = 0.025
    align_threshold_inc = 0.001
    
    next_round = segments.copy()    
    for _i in range(1):
        dist_threshold += dist_threshold_inc
        align_threshold += align_threshold_inc
        
        new_segm = Segments.Segments(next_round)
        new_segm.initialize_filament_processing()
        next_round = new_segm.end_to_end_clustering_cpp(number_of_endpoint_averaging=50,dist_threshold=dist_threshold,align_threshold=0.025)
        plt.close('all')
        new_segm.plot_length_histogram()
        plt.savefig(f'{save_folder}/length_histogram_{tracker}.png')
        plt.close('all')

        tracker += 1
        
        if tracker % 100 == 0:
            layered = np.zeros((len(next_round),1),dtype=object)
            layered.shape
            for i in range(len(layered)):
                layered[i] = [next_round[i].astype(np.uint16)]
            savemat(f'{save_folder}/segments_in_construction.mat',{'segments':layered},do_compression=True)
    layered = np.zeros((len(next_round),1),dtype=object)
    for i in range(len(layered)):
        layered[i] = [next_round[i].astype(np.uint16)]
    savemat(f'{save_folder}/segments_in_construction_{tracker}.mat',{'segments':layered},do_compression=True)
            
    layered = np.zeros((len(next_round),1),dtype=object)
    layered.shape
    for i in range(len(layered)):
        layered[i] = [next_round[i].astype(np.uint16)]
    savemat(f'{save_folder}/segments_{tracker}.mat',{'segments':layered},do_compression=True)

    # copy this file to the data folder for future reference
    os.system(f'cp {__file__} {save_folder}/log_segmentation_{tracker}.py')
# %%


# plt.close('all')
# new_segm.plot_large_clusters(100)
# # %%
# plt.close('all')
# new_segm.check_short_segments(100)
# # %%
# _length_list = np.array(new_segm.length_list).copy()
# plt.close('all')
# plt.hist(_length_list[(_length_list < 2000) & (_length_list > 10)],bins=100)

# # %%
# # to_show = np.where((_length_list < 100) & (_length_list > 10))[0]
# to_show = np.where((_length_list < 50))[0]
# to_show.shape

# plt.close('all')
# fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
# for i in to_show:
#     if np.linalg.norm(new_segm.segments[i] - np.array([900,700,700])) < 500:
#         ax.plot(*new_segm.segments[i].T)
# # %%
# def load_mat_file_v73(filename):
#     import h5py
#     with h5py.File(filename, 'r') as f:
#         data = {key: np.array(f[key]) for key in f.keys()}
#     return data

# alpha = 66
# epsilon = 0
# zstack_path = f'/Users/yeonsu/GitHub/entanglement/data/alpha{alpha}_epsilon{epsilon:02d}/zstack.mat'
# zstack_data = load_mat_file_v73(zstack_path)
# zstack = zstack_data['zstack']
# zstack = np.transpose(zstack, (2, 1, 0))
# # %%
# alpha = 66
# epsilon = 0

# save_folder = Path('perform-segmentation') / f'alpha{alpha}_epsilon{epsilon:02d}'
# final_save_folder = f'../data/alpha{alpha}_epsilon{epsilon}'

# f'{final_save_folder}/segments.mat'

# matobj = loadmat(f'{final_save_folder}/segments.mat',simplify_cells=True)
# # %%
# segments = matobj['segments']
# # %%
# segments[0]
# zstack[1059,976,237]

# # %%


# # %%
# shape = zstack.shape
# a_point = segments[10][1].astype(np.int64)
# ind = np.ravel_multi_index(a_point.T, shape)

# zstack.flat[ind]
# # %%
# np.where(zstack > 0)

# # %%
# # image
# plt.close('all')
# fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
# ax.voxels(crop,edgecolors='k')
        
# # %%
# Segments_instance = Segments.Segments(next_round)
# # %%
# Segments_instance.initialize_filament_processing()
# # %%
# # rod_diameter = 1/alpha
# Segments_instance.fp.calculate_svd_scores(500,0.3)
# # %%
# Segments_instance.fp.calculate_filament_distance_matrix(500,0.3)
# # %%
# ij = Segments_instance.fp.get_ij()
# scores = Segments_instance.fp.get_scores()
# # %%
# ij = np.array(ij,dtype=np.int64)
# scores = np.array(scores,dtype=np.float64)
# scores.shape
# # %%
# distance_score = scores[:,0]
# align_score = scores[:,1]
# fitting_score = scores[:,2]
# # %%
# import networkx as nx
# graph = nx.Graph()
# graph.add_nodes_from(range(len(Segments_instance.segments)))

# mask = (fitting_score < 1) & (distance_score < 100) & (align_score < 0.1)
# masked_ij = ij[mask]

# # %%
# graph.add_edges_from(masked_ij)
# conn_comps = list(nx.connected_components(graph))

# len(Segment_instance.segments)
# len(conn_comps)
# num_elements_list = [len(comp) for comp in conn_comps]

# i_max = np.argsort(num_elements_list)[-25]
# cc_max = conn_comps[i_max]

# fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
# for i in cc_max:
#     ax.plot(*Segments_instance.segments[i].T)
# ax.axis('equal')

# # %%
# length_list = []
# for cc in conn_comps:
#     joined = np.vstack([Segments_instance.segments[i] for i in cc])
#     joined = Segments.Segment.sort_curve(joined)
    
#     length_list.append(Segments.Segment.seg_len(joined))
    
# # %%
# length_list = np.array(length_list)
# TF = (length_list > 10) & (length_list < 2000)
# np.count_nonzero(TF)

# plt.close('all')
# plt.hist(length_list[TF],bins=100)
# # %%

# # %%
# fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
# ax.plot(*joined.T)

# # %%


# # %%
# # Segments_instance.initialize_filament_processing()
        

# # %%
# # continued


# for alpha in [66]:
#     epsilon = 0
#     # matfile_path = Path(f'/Users/yeonsu/GitHub/entanglement/data/alpha{alpha}_epsilon{epsilon:02d}/segments.mat')
#     # matobj = loadmat(f'/Users/yeonsu/GitHub/entanglement/data/alpha{alpha}_epsilon{epsilon:02d}/segments.mat',simplify_cells=True)
#     # segments = matobj['segments']
#     save_folder = Path('perform-segmentation') / f'alpha{alpha}_epsilon{epsilon:02d}'
#     final_save_folder = f'../data/alpha{alpha}_epsilon{epsilon}'

#     # cache_file_path = f'test/alpha{alpha}_epsilon{epsilon:02d}/segments_in_construction.mat'
#     cache_file_path = save_folder / 'segments.mat'
#     matobj = loadmat(cache_file_path,simplify_cells=True)
#     segments = matobj['segments']
#     print(f'alpha: {alpha}, epsilon: {epsilon}, number of segments: {len(segments)}')
    
#     # if seg is 1d array it should be converted to 2d array
#     for i in range(len(segments)):
#         if len(segments[i].shape) == 1:
#             segments[i] = segments[i].reshape(-1,3)

#     Segment_instance = Segments.Segments(segments)
#     # length_list, error_list = Segment_instance.inspect_segments()
#     Segment_instance.initialize_filament_processing()
#     next_round = Segment_instance.end_to_end_clustering(number_of_endpoint_averaging=50,dist_threshold=500,align_threshold=0.1)
#     # %%
    
    
#     plt.close('all')
#     Segment_instance.plot_large_clusters(100)
#     # %%
#     plt.close('all')
#     Segment_instance.check_short_segments(100)
#     # %%
#     _length_list = np.array(Segment_instance.length_list).copy()
#     plt.close('all')
#     plt.hist(_length_list[(_length_list < 2000) & (_length_list > 100)],bins=100)
#     # %%
#     to_show = np.where((_length_list < 2000) & (_length_list > 100))[0]
    
#     plt.close('all')
#     fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
#     for i in to_show:
#         ax.plot(*Segment_instance.segments[i].T)
    
#     # %%
    
#     # %%
#     save_folder = Path('test') / f'alpha{alpha}_epsilon{epsilon:02d}'
#     save_folder = Path('/Users/yeonsu/GitHub/entanglement-source-codes/segments-processing/test')                       

#     if not os.path.exists(save_folder):
#         os.makedirs(save_folder)
#     else:
#         # check the number of files in the folder
#         list(save_folder.glob('length_histogram_*.png'))
#         # get the last number
#         last_number = max([int(str(file).split('_')[-1].split('.')[0]) for file in list(save_folder.glob('length_histogram_*.png'))])
        
#     tracker = last_number + 1
#     dist_threshold = 50
#     align_threshold = 0.025
#     align_threshold_inc = 0.001
#     for _i in range(300):
#         align_threshold += align_threshold_inc
        
#         new_segm = Segments.Segments(next_round)
#         new_segm.initialize_filament_processing()
#         next_round = new_segm.end_to_end_clustering_cpp(number_of_endpoint_averaging=50,dist_threshold=dist_threshold,align_threshold=align_threshold)
#         plt.close('all')
#         new_segm.plot_length_histogram()
#         plt.savefig(f'{save_folder}/length_histogram_{tracker}.png')
#         plt.close('all')

#         tracker += 1
        
#         if tracker % 100 == 0:
#             layered = np.zeros((len(next_round),1),dtype=object)
#             layered.shape
#             for i in range(len(layered)):
#                 layered[i] = [next_round[i].astype(np.uint16)]
#             savemat(f'{save_folder}/segments_in_construction.mat',{'segments':layered},do_compression=True)

#     final_save_folder = f'../data/alpha{alpha}_epsilon{epsilon}'
#     layered = np.zeros((len(next_round),1),dtype=object)
#     layered.shape
#     for i in range(len(layered)):
#         layered[i] = [next_round[i].astype(np.uint16)]
#     savemat(f'{final_save_folder}/segments.mat',{'segments':layered},do_compression=True)

#     # copy this file to the data folder for future reference
#     os.system(f'cp {__file__} {final_save_folder}/log_segmentation_{tracker}.py')
    
    
# # %%