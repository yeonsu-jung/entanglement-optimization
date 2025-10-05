# %%
import pickle

recent_one = '/Users/yeonsu/GitHub/entanglement-optimization/test_segmenting5/clustered_segments.pkl'

with open(recent_one, 'rb') as f:
    segments = pickle.load(f)
    
# %%
len(segments)

# %%
import numpy as np
def seg_len(seg):
    return np.sum(np.sqrt(np.sum(np.diff(seg,axis=0)**2,axis=1)))

from fitting import fit_rod
error_list = []
length_list = []
for i,seg in enumerate(segments):
    fr = fit_rod(seg,0.00001,10000)
    error_list.append(fr['err'])
    length_list.append(seg_len(seg))
    
# %%
from matplotlib import pyplot as plt
fig,ax=plt.subplots()
ax.scatter(length_list,error_list)
# %%
fig,ax=plt.subplots()
plt.hist(error_list,bins=100)
np.max(error_list)
# %%
log_bins = np.logspace(0,4,100)
fig,ax=plt.subplots()
plt.hist(length_list,bins=log_bins)
np.max(length_list)
ax.set_xscale('log')
plt.xlabel('Length')
plt.ylabel('Count')
plt.savefig('length_distribution.png')
# %%
length_list = np.array(length_list)

# %%


from scipy.io import loadmat
matobj = loadmat('/Users/yeonsu/GitHub/entanglement/data/alpha200_epsilon00/trimmed.mat')

# %%
trimmed = matobj['trimmed']
# %%
def sort_curve(segment):
    centroid = np.mean(segment,axis=0)
    segment_centered = segment - centroid        
    _,_, V = np.linalg.svd(segment_centered, full_matrices=False)
    v1 = V[0,:]
    orientation = v1 * np.sign(np.sum(v1 * (segment_centered[-1, :] - segment_centered[0, :])))
    slist = np.dot((segment - centroid), orientation)
    sorted_indices = np.argsort(slist)
    return centroid + segment_centered[sorted_indices]
# %%
tmp = []
for i,seg in enumerate(trimmed):
    seg = sort_curve(seg[0])
    tmp.append(seg)
# %%
len(tmp)

# %%
fig,ax=plt.subplots(subplot_kw={'projection':'3d'})
for seg in tmp[-100:]:
    ax.plot(seg[:,0],seg[:,1],seg[:,2])
    
# %%
all_edgelengths = []
for i,centerline in enumerate(tmp):
    # check individual edgelengths
    edgelengths = np.linalg.norm(centerline[:-1]-centerline[1:],axis=1)
    all_edgelengths.append(edgelengths)

np.mean(np.concatenate(all_edgelengths))
# %%

    
    
error_list = []
length_list = []
for i,seg in enumerate(tmp):
    fr = fit_rod(seg,0.00001,10000)
    error_list.append(fr['err'])
    length_list.append(seg_len(seg))
    
    
# %%
fig,ax=plt.subplots()
ax.scatter(length_list,error_list)
# %%
fig,ax=plt.subplots()
plt.hist(error_list,bins=100)
np.max(error_list)
# %%
log_bins = np.logspace(0,4,100)
fig,ax=plt.subplots()
plt.hist(length_list,bins=log_bins)
np.max(length_list)
ax.set_xscale('log')
plt.xlabel('Length')
plt.ylabel('Count')
plt.savefig('length_distribution_after_trim.png')
# %%
i_wrong = np.argsort(error_list)[-10:]

seg_wrong = [tmp[i] for i in i_wrong]
fig,ax=plt.subplots(subplot_kw={'projection':'3d'})
for seg in seg_wrong:
    ax.plot(seg[:,0],seg[:,1],seg[:,2],'o-')
    
ax.axis('equal')



# %%
# remove long but few-point segments
length_list = np.array(length_list)
number_list = np.array([len(seg) for seg in tmp])
fig,ax=plt.subplots(subplot_kw={'projection':'3d'})
for i in np.argsort(error_list)[-10:]:
    
    edgelength = np.linalg.norm(tmp[i][:-1]-tmp[i][1:],axis=1)
    mean_edgelength = np.mean(edgelength)
    print(mean_edgelength)
    seg = tmp[i]
    ax.plot(seg[:,0],seg[:,1],seg[:,2],'o-')
# %%

mean_edgelength_list = []
for i,seg in enumerate(tmp):    
    edgelength = np.linalg.norm(seg[:-1]-seg[1:],axis=1)
    mean_edgelength_list.append(np.mean(edgelength))
    
# %%
i_wrong = np.argwhere(np.array(mean_edgelength_list) > 10).flatten()
fig,ax=plt.subplots(subplot_kw={'projection':'3d'})
for i in i_wrong:
    seg = tmp[i]
    ax.plot(seg[:,0],seg[:,1],seg[:,2],'o-')
    ax.axis('equal')

# remove elements belong to i_wrong from tmp
tmp = [tmp[i] for i in range(len(tmp)) if i not in i_wrong]

# %%


error_list = []
length_list = []
for i,seg in enumerate(tmp):
    fr = fit_rod(seg,0.00001,10000)
    error_list.append(fr['err'])
    length_list.append(seg_len(seg))
    
# %%
fig,ax=plt.subplots()
plt.hist(error_list,bins=100)
np.max(error_list)
# %%
fig,ax=plt.subplots()
plt.hist(length_list,bins=100)


# %%
log_bins = np.logspace(0,4,100)
fig,ax=plt.subplots()
plt.hist(length_list,bins=log_bins)
np.max(length_list)
ax.set_xscale('log')
plt.xlabel('Length')
plt.ylabel('Count')
plt.savefig('length_distribution_after_trim2.png')
# %%
# this is our centerline.

# fit centerline to a polynomial?
