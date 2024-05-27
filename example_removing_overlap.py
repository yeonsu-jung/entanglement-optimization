from scipy.io import loadmat
import numpy as np
from matplotlib import pyplot as plt


mat_obj = loadmat('/Users/yeonsu/Documents/GitHub/entanglement-optimization/centerlines_alpha200_epsilon00.mat')

centerlines = mat_obj['centerlines']
centerlines = [seg[0] for seg in centerlines]
N_centerlines = len(centerlines)

print(N_centerlines)


import filamentprocessing
import time

start = time.time()
fp = filamentprocessing.FilamentProcessing(centerlines,2,1,0.99)
print(f'Elapsed time: {time.time()-start}')

svd_ij = np.array(fp.get_svd_ij())
svd_dist = np.array(fp.get_svd_scores())[:,0]
print(f'Number of pairs: {len(svd_ij)}')

fig,ax=plt.subplots(1,1)
ax.hist(svd_dist[svd_dist<3],bins=100)

np.count_nonzero(svd_dist<3)
svd_ij[svd_dist<3]

def draw_pair(ii,jj):
    fig,ax=plt.subplots(1,1,figsize=(10,10),subplot_kw={'projection':'3d'})
    rr_i = centerlines[ii]
    rr_j = centerlines[jj]

    ax.plot(rr_i[:,0],rr_i[:,1],rr_i[:,2],label='Segment i')
    ax.plot(rr_j[:,0],rr_j[:,1],rr_j[:,2],label='Segment j')
    ax.axis('equal')

draw_pair(2,1249)

nbs = svd_ij[svd_ij[:,0]==0,1]
svd_dist[svd_ij[:,0]==0]

fig,ax=plt.subplots(1,1,figsize=(10,10),subplot_kw={'projection':'3d'})
ax.plot(centerlines[0][:,0],centerlines[0][:,1],centerlines[0][:,2],linewidth=5)
for i in nbs:
    rr_i = centerlines[i]
    ax.plot(rr_i[:,0],rr_i[:,1],rr_i[:,2])

start = time.time()
fp.calculate_filament_distance_matrix(10,1)
print(f'Elapsed time: {time.time()-start}')
ij = fp.get_ij()
scores = fp.get_scores()

ij = np.array(ij)
scores = np.array(scores)

dist_score = scores[:,0]
dist_score[np.argsort(dist_score)][::-1]
fig,ax=plt.subplots(1,1)
ax.hist(dist_score,bins=100)


np.argsort(dist_score)[::-1]
idx = 326



def seg_len(seg):
    return np.sum(np.sqrt(np.sum(np.diff(seg, axis=0) ** 2, axis=1)))

centerline_length_list = [seg_len(seg) for seg in centerlines]
fig,ax=plt.subplots(1,1)
ax.hist(centerline_length_list,bins=100)


print


