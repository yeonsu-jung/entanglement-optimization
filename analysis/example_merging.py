# %%
import numpy as np
from scipy.io import loadmat
from pathlib import Path
# %%
rod_data_root_dir = Path('/Users/yeonsu/Data/steel-rods-xray-data')
file_path = rod_data_root_dir / 'alpha200_epsilon00' / 'repeated_clustering.mat'

matobj = loadmat(file_path)
# %%
segments_rectangle = matobj['segments_rectangle']
# %%
from matplotlib import pyplot as plt
plt.close('all')
fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
segments = []
for segment_flat in segments_rectangle:
    segment = segment_flat[~np.isnan(segment_flat)].reshape(-1,3)
    segments.append(segment)
    # ax.plot(segment[:,0],segment[:,1],segment[:,2],'-')
    
# %%
from Segments import Segments
segm = Segments(segments)
segm.initialize_filament_processing()
# %%
segm.calculate_end_to_end_scores(200)
segm.end_to_end_clustering(300,300,0.5,0.025,shrinkage=0.95)

# %%
ab_ = segm.fp.get_end_ab()
scores_ = segm.fp.get_end_scores()

ab_ = np.array(ab_)
scores_ = np.array(scores_)

dist_scores_ = scores_[:,0]
# align_scores_ = scores_[:,1] # useless here       

distance_threshold = 100
fitting_error_threshold = 1.5

# %%
from fitting import fit_rod
test = 1
if test:
    i_min_list = np.argsort(dist_scores_[dist_scores_>0])
    i_min = i_min_list[32]
    i_min_global = np.where(dist_scores_>0)[0][i_min]
    print(scores_[i_min_global,:])                

    _ij = ab_[i_min_global,:]
    rr0 = segments[_ij[0]//2]
    rr = segments[_ij[1]//2]
    plt.close('all')
    fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
    ax.plot(rr0[:,0],rr0[:,1],rr0[:,2],'-')
    ax.plot(rr[:,0],rr[:,1],rr[:,2],'-')
    ax.axis('equal')
    joined = np.vstack([rr0,rr])
    fr = fit_rod(joined,linearity_threshold=0.0001,radius_curvature_threshold=100000)
    print(f'Error: {fr["err"]}') # good, merge it.

    ep1 = segm.endpoints[_ij[0]]
    ep2 = segm.endpoints[_ij[1]]
    dvec = ep1 - ep2
    dvec /= np.linalg.norm(dvec)

    # dist = np.linalg.norm(ep1 - ep2)
    tan1 = segm.endtangents[_ij[0]]
    tan2 = segm.endtangents[_ij[1]]
# %%
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


# %%
from Segments import Segment

distance_threshold = 200
fitting_error_threshold = 1

parent = {i:i for i in range(len(segments))}
connections = []
for i_ in np.where((dist_scores_ < distance_threshold) & (dist_scores_ > 0))[0]:    
    _ij = ab_[i_,:]
    rr0 = segments[_ij[0]//2]
    rr = segments[_ij[1]//2]
    
    joined = np.vstack([rr0,rr])
    joined = Segment.sort_curve(joined)
    
    
    fr = fit_rod(joined,linearity_threshold=0.0001,radius_curvature_threshold=100000)
    if fr['err'] < fitting_error_threshold:
        connections.append( (_ij[0]//2,_ij[1]//2) )
        print(f'Error: {fr["err"]}')

# %%
import networkx as nx
merge_graph = nx.Graph()
merge_graph.add_nodes_from(range(len(segments)))
merge_graph.add_edges_from(connections)

# %%
ccs = list(nx.connected_components(merge_graph))

ccs = [list(cc) for cc in ccs]
len(ccs)
# %%
i_max = np.argsort([len(cc) for cc in ccs])[-9]
cc_max = ccs[i_max]
plt.close('all')
fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
joined = []
for i_ in cc_max:
    joined.append(segments[i_])
    ax.plot(segments[i_][:,0],segments[i_][:,1],segments[i_][:,2],'-')
ax.axis('equal')
joined = np.vstack(joined)
joined = Segment.sort_curve(joined)
# %%
fr = fit_rod(joined,linearity_threshold=0.0001,radius_curvature_threshold=100000)
fr['err']

# %%
# plt.close('all')
# fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
merged = []
for cc in ccs:
    joined = []
    for i_ in cc:
        joined.append(segments[i_])
    joined = np.vstack(joined)
    joined = Segment.sort_curve(joined)
    merged.append(joined)
    # ax.plot(joined[:,0],joined[:,1],joined[:,2],'-')
# %%
max_cols = max([len(seg) for seg in merged])
nan_padded = np.full((len(merged),max_cols*3),np.nan)
for i,seg in enumerate(merged):
    nan_padded[i,:len(seg)*3] = seg.flatten()
# %%
from scipy.io import savemat
savemat('/Users/yeonsu/Data/steel-rods-xray-data/alpha200_epsilon00/merged.mat',{'merged_nanpad':nan_padded})

# %%
length_list = [Segment.seg_len(merged[i]) for i in range(len(merged))]

# %%
bins = np.linspace(10,1000,100)
fig,ax=plt.subplots(1,1)
ax.hist(length_list,bins=bins)
# %%
short_labels = [i for i in range(len(merged)) if Segment.seg_len(merged[i]) < 500]
fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
for i in short_labels:
    segment = merged[i]
    
    if np.all( segment[:,2] < 170 ):
        ax.plot(merged[i][:,0],merged[i][:,1],merged[i][:,2],'-')
        ax.text(segment[0,0],segment[0,1],segment[0,2],str(i),fontsize=6)    
        
ax.axis('equal')
# %%
error_list = []
for i in range(len(merged)):
    fr = fit_rod(merged[i],linearity_threshold=0.0001,radius_curvature_threshold=100000)
    error_list.append(fr['err'])
# %%
error_list = np.array(error_list)
np.count_nonzero(error_list > 1.5)
fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
for i in np.where(error_list > 1.5)[0]:
    ax.plot(merged[i][:,0],merged[i][:,1],merged[i][:,2],'-')

# %%

short_labels = [i for i in range(len(segments)) if Segment.seg_len(segments[i]) < 500]
fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
for i in short_labels:
    segment = segments[i]
    
    if np.all( segment[:,2] < 600 ) and np.all (segment[:,2] > 570):
        ax.plot(segments[i][:,0],segments[i][:,1],segments[i][:,2],'-')
        ax.text(100*np.random.rand()+segment[0,0],100*np.random.rand()+segment[0,1],segment[0,2],str(i),fontsize=6)
        
ax.axis('equal')

# %%
merged_segm = Segments(segments)
merged_segm.initialize_filament_processing(dist_threshold=400,align_threshold=1,svd_cutoff=1.)
# initialize_filament_processing(self,dist_threshold=50,align_threshold=1,svd_cutoff=1.):
merged_segm.calculate_end_to_end_scores(200)
# %%
merged_segm.end_to_end_clustering(300,1,-10,0.1,shrinkage=0.90)
# %%
merged_segm.fp.calculate_end_to_end_scores(100,-10)
# %%
merged_segm.end_ab
dmat_dict = {}

for i in range(len(merged_segm.end_ab)):
    a,b = merged_segm.end_ab[i]
    dmat_dict[a] = dmat_dict.get(a,[]) + [(b,merged_segm.end_scores[i])]
    # if np.all(merged_segm.end_scores[i] > 0):
    #     end_ab = merged_segm.end_ab[i]
    #     _ij = tuple(end_ab)
    #     dmat_dict[_ij] = merged_segm.end_scores[i]
# %%
dmat_dict[10774]
# %%
merged_segm.check_nearby_segments(5389,10)
# %%
merged = segments
i_seg = 10734//2
j_seg = 10778//2

i_ = i_seg*2
i_conj = i_ + 1

j_ = j_seg*2
j_conj = j_ + 1

e_i = merged_segm.corrected_end_points[i_]
e_i_conj = merged_segm.corrected_end_points[i_conj]

e_j = merged_segm.corrected_end_points[j_]
e_j_conj = merged_segm.corrected_end_points[j_conj]

fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
ax.plot(merged[i_seg][:,0],merged[i_seg][:,1],merged[i_seg][:,2],'-')
ax.plot(merged[j_seg][:,0],merged[j_seg][:,1],merged[j_seg][:,2],'-')
ax.plot(e_i[0],e_i[1],e_i[2],'ro')
ax.plot(e_j[0],e_j[1],e_j[2],'bo')
ax.plot(e_i_conj[0],e_i_conj[1],e_i_conj[2],'rx')
ax.plot(e_j_conj[0],e_j_conj[1],e_j_conj[2],'bx')

ax.text(e_i[0],e_i[1],e_i[2],f'{i_}',fontsize=6)
ax.text(e_j[0],e_j[1],e_j[2],f'{i_conj}',fontsize=6)
ax.text(50+e_i_conj[0],e_i_conj[1],e_i_conj[2],f'{j_}',fontsize=6)
ax.text(50+e_j_conj[0],e_j_conj[1],e_j_conj[2],f'{j_conj}',fontsize=6)
ax.axis('equal')
# %%

d1 = e_i - e_j
d2 = e_i_conj - e_j_conj
d1 /= np.linalg.norm(d1)
d2 /= np.linalg.norm(d2)

d3 = e_i - e_j_conj
d4 = e_i_conj - e_j

d3 /= np.linalg.norm(d3)
d4 /= np.linalg.norm(d4)

# int i_conj = (i % 2 == 0) ? i + 1 : i - 1;
i_min = np.argmin([d1,d2,d3,d4])
if i_min == 0:
    i_ = i_seg*2
    i_conj = i_ + 1
    j_ = j_seg*2
    j_conj = j_ + 1
    
elif i_min == 1:
    i_ = i_seg*2
    i_conj = i_ + 1
    j_conj = j_seg*2
    j_ = j_conj + 1
    
elif i_min == 2:
    i_ = i_seg*2
    i_conj = i_ + 1
    j_ = j_seg*2
    j_conj = j_ + 1
    
elif i_min == 3:
    i_conj = i_seg*2
    i_ = i_conj + 1
    j_ = j_seg*2
    j_conj = j_ + 1

# %%
tan_i = merged_segm.endtangents[i_]
tan_i_conj = merged_segm.endtangents[i_conj]

tan_j = merged_segm.endtangents[j_]
tan_j_conj = merged_segm.endtangents[j_conj]

e_i = merged_segm.corrected_end_points[i_]
e_i_conj = merged_segm.corrected_end_points[i_conj]

e_j = merged_segm.corrected_end_points[j_]
e_j_conj = merged_segm.corrected_end_points[j_conj]

dvec = e_i - e_j
dist = np.linalg.norm(dvec)

dvec_normalized = dvec / dist
alignment = (np.linalg.norm(np.cross(dvec_normalized,tan_i)) + np.linalg.norm(np.cross(dvec_normalized,tan_j))) / 2

inward_i = e_i - e_i_conj
inward_i /= np.linalg.norm(inward_i)
inward_j = e_j - e_j_conj
inward_j /= np.linalg.norm(inward_j)


np.dot(dvec_normalized,inward_i)

fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
ax.plot(merged[i_seg][:,0],merged[i_seg][:,1],merged[i_seg][:,2],'-')
ax.plot(merged[j_seg][:,0],merged[j_seg][:,1],merged[j_seg][:,2],'-')

ax.plot(e_i[0],e_i[1],e_i[2],'ro')
ax.plot(e_j[0],e_j[1],e_j[2],'bo')

scale = 100
ax.quiver(e_i[0],e_i[1],e_i[2],scale*inward_i[0],scale*inward_i[1],scale*inward_i[2],color='r')
ax.quiver(e_j[0],e_j[1],e_j[2],scale*inward_j[0],scale*inward_j[1],scale*inward_j[2],color='b')
ax.quiver(e_i[0],e_i[1],e_i[2],scale*dvec_normalized[0],scale*dvec_normalized[1],scale*dvec_normalized[2],color='k')

ax.plot(e_i_conj[0],e_i_conj[1],e_i_conj[2],'rx')
ax.plot(e_j_conj[0],e_j_conj[1],e_j_conj[2],'bx')

ax.axis('equal')

samesideness = np.dot(dvec_normalized,inward_i)
print(f'Distance: {dist}')
print(f'Alignment: {alignment}')
print(f'Same side-ness: {samesideness}')



# %%

# %%
sparse = []
for i in range(len(merged)):
    sparse.append(merged[i][::20])

max_rows = np.max([len(s) for s in sparse])
# %%
fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
for i in range(len(sparse)):
    ax.plot(sparse[i][:,0],sparse[i][:,1],sparse[i][:,2],'-')
# %%
import filamentFields
fF = filamentFields.filamentFields([],[])
# %%
fF.update_filament_nodes_list(sparse)
# %%
local_edges = fF.analyze_local_volume([300,500,500], 50, 0.01)

fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})

for edge in local_edges:
    r_i = edge[:3]
    r_j = edge[3:]
    ax.plot([r_i[0],r_j[0]],[r_i[1],r_j[1]],[r_i[2],r_j[2]],'-')
# %%
fF.return_volume_fraction()
fF.return_entanglement()
# %%
all_nodes = np.concatenate(sparse)

xlim = np.min(all_nodes[:,0]),np.max(all_nodes[:,0])
ylim = np.min(all_nodes[:,1]),np.max(all_nodes[:,1])
zlim = np.min(all_nodes[:,2]),np.max(all_nodes[:,2])

num_grid = 60
mg = np.meshgrid(np.linspace(xlim[0],xlim[1],num_grid),np.linspace(ylim[0],ylim[1],num_grid),np.linspace(zlim[0],zlim[1],num_grid))
sampling_points = np.array([mg[0].flatten(),mg[1].flatten(),mg[2].flatten()]).T
# %%
sampling_points[0]
# %%
np.linalg.norm(sampling_points[0] - sampling_points[1])

# %%
np.sqrt(750*3)
# %%
e_fields = np.zeros(len(sampling_points))
for i_ in range(len(sampling_points)):
    local_edges = fF.analyze_local_volume(sampling_points[i_], 50, 0.01)
    e_fields[i_] = fF.return_entanglement()
    
# %%
e_image = e_fields.reshape(num_grid,num_grid,num_grid)
e_image[np.isnan(e_image)] = 0
# e_image.shape
e_proj = np.sum(e_image,axis=2)
# e_proj = np.sum(e_image,axis=0)
# e_proj = np.flipud(e_proj.T)

fig,ax=plt.subplots(1,1)
ax.imshow(e_proj,extent=[xlim[0],xlim[1],ylim[0],ylim[1]],cmap='coolwarm')
ax.axis('equal')
# %%
np.nanstd(e_image[e_image>0])/np.nanmean(e_image[e_image>0])

# %%
    


# %%
# R_omega = 20
# fF.precompute(R_omega)
# local_nodes = fF.analyze_local_volume_from_precomputed([1000,1000,500], R_omega, 0.01)
# %%

    


# %%
# parallel transport


