from pathlib import Path
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import filamentprocessing
from scipy.io import savemat

rod_data_root_dir = Path('/Users/yeonsu/Data/steel-rods-xray-data')
segments_file_path = rod_data_root_dir / 'alpha200_epsilon00' / 'pruned_segments.pkl'
# connectivity_file_path = rod_data_root_dir / 'alpha200_epsilon00' / 'ijscores_total.pkl'

if os.path.exists(segments_file_path):
    segments = pickle.load(open(segments_file_path,'rb'))
else:
    print(f'File not found: {segments_file_path}')
    pass
    exit

def unpack_segment_list(centerlines):
    unpacked = np.vstack(centerlines)
    labels = np.zeros(unpacked.shape[0],dtype=np.int64)
    start_idx = 0
    for i,cl in enumerate(centerlines):
        end_idx = start_idx + cl.shape[0]
        labels[start_idx:end_idx] = i + 1
        start_idx = end_idx
        
    return np.hstack((unpacked,labels[:,None]))

# unpacked_segments = unpack_segment_list(segments)            
# savemat(rod_data_root_dir / 'alpha200_epsilon00' / 'pruned_segments.mat', {'pruned_segments': unpacked_segments})
    
# segments = segments[:1000]
import time
start_time = time.time()
fp = filamentprocessing.FilamentProcessing(segments,2500,0.2,0.99)
print(f'--- {time.time() - start_time} seconds ---')

ij = fp.get_svd_ij()
ij = np.array(ij,dtype=np.int64)
scores = fp.get_svd_scores()    
scores = np.array(scores,dtype=np.float64)

savemat(rod_data_root_dir / 'alpha200_epsilon00' / 'ij_total.mat', {'ij': ij})
savemat(rod_data_root_dir / 'alpha200_epsilon00' / 'scores_total.mat', {'scores': scores})



