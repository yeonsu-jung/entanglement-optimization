# %%
import numpy as np
from scipy.io import loadmat


root_dir = '/Users/yeonsu/Dropbox (Harvard University)/Data/PrunedData/rod-sim-pnas-revision/'
parent_dir = f'{root_dir}/MetalNestSegmented/'
matpath = f'{parent_dir}/segments.mat'

segmented = loadmat(matpath)['segments_nanpad']

# %%
segments = []
for seg in segmented:
    seg = seg.reshape(-1,3)
    segments.append(seg[~np.isnan(seg).any(axis=1)])
# %%


# %%
all_segments = np.concatenate(segments)
import polyscope as ps

# ps.remove_all_structures()
ps.init()
ps.look_at((-5., 0., 2.), (0., 0., 2.))

for seg in segmented:
    edges = np.array([[i, i+1] for i in range(seg.shape[0]-1)])
    
ps.register_curve_network('seg', all_segments, edges, enabled=True, radius=0.01)

ps.show()
    