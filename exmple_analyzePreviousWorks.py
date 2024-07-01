# %%
from pathlib import Path
from scipy.io import loadmat
import time
import polyscope as ps
import numpy as np
import filamentFields
import filamentprocessing
from matplotlib import pyplot as plt
colors = np.array([
    [76, 153, 204],   # light blue
    [204, 76, 153],   # pinkish red
    [76, 204, 153],   # mint green
    [153, 204, 76],   # light olive green
    [204, 153, 76],   # goldenrod
    [153, 76, 204],   # medium purple
    [204, 76, 102],   # crimson
    [76, 204, 204],   # cyan
    [204, 204, 76],   # sunflower yellow
    [102, 76, 204]    # indigo
    ])


# %%
parent_folder = Path('/Users/yeonsu/Downloads/tangle-science-data')

# find mat files
mat_files = list(parent_folder.glob('**/*.mat'))
print(mat_files)

# %%
def seg_len(seg):
        return np.sum(np.sqrt(np.sum(np.diff(seg,axis=0)**2,axis=1)))
    
result_dict = {}
for mat_file in mat_files:
    if 'W0' not in loadmat(mat_file).keys():
        continue
    mat_obj = loadmat(mat_file)

    # do something with the mat file    
    W0 = mat_obj['W0']
    num_worms = W0.shape[0]//3
    centerlines = []
    for i in range(num_worms):
        start_i = 3*i
        last_i = start_i + 3
        centerline_i = W0[start_i :last_i,:].T
        centerlines.append(centerline_i)
        
    fp = filamentprocessing.FilamentProcessing(centerlines,1000,1000,1)
    fp.calculate_svd_scores(1000,100)
    fp.calculate_end_to_end_properties(1000,1)
    fp.get_corrected_end_points()
    fp.calculate_filament_distance_matrix(1000,1000)
    ij = fp.get_ij()
    scores = fp.get_scores()
    ij = np.array(ij)
    scores = np.array(scores)
    distances = scores[:,0]
    
    length_list = np.array([seg_len(centerline) for centerline in centerlines])
    rod_length = np.median(length_list)
    rod_diameter = np.min(distances)*1.25
    R_omega = np.sqrt(rod_length*rod_diameter)
    
    nodes = np.vstack(centerlines).reshape((-1,3))
    num_nodes_each_rod = centerlines[0].shape[0]
    num_rods = len(centerlines)
    edges = np.array([[i, i + 1] for i in range(len(nodes) - 1) if i % num_nodes_each_rod != num_nodes_each_rod - 1])

    ps.init()
    ps.set_SSAA_factor(3)
    ps.set_navigation_style("free")

    # ps.set_ground_plane_mode("tile")
    ps.set_ground_plane_mode("none")
    ps.set_ground_plane_mode("shadow_only")  # set +Z as up direction
    ps.set_ground_plane_height_factor(-0.25) # adjust the plane height
    ps.set_shadow_darkness(0.1)              # lighter shadows
    ps.set_view_projection_mode("perspective")
    # ps.set_transparency_mode('simple')

    ps_all_nodes = ps.register_curve_network("all_nodes", nodes, edges, enabled=True)
    ps_all_nodes.set_radius(rod_diameter/2,relative=False)

    # Add color to edges
    fF = filamentFields.filamentFields(centerlines)
    fF.precompute(1000)
    fF.compute_total_linking_matrix()
    fF.compute_filament_linking_matrix()
    link_mat = fF.return_filament_linking_matrix()
    
    xlim = np.min(nodes[:,0]),np.max(nodes[:,0])
    ylim = np.min(nodes[:,1]),np.max(nodes[:,1])
    zlim = np.min(nodes[:,2]),np.max(nodes[:,2])

    num_grid = 30
    mg = np.meshgrid(np.linspace(xlim[0],xlim[1],num_grid),np.linspace(ylim[0],ylim[1],num_grid),np.linspace(zlim[0],zlim[1],num_grid))
    sampling_points = np.array([mg[0].flatten(),mg[1].flatten(),mg[2].flatten()]).T
    result = fF.analyze_local_volume_over_domain(sampling_points, R_omega, 0.01)    
    n_field = result[:,0]
    e_field = result[:,3]
    
    # clk_ij = np.argwhere( (np.abs(link_mat) > 0.5) )
    contact_ijs = ij[distances < rod_diameter]
    clk = np.full((num_rods,num_rods),np.nan)
    for i,j in contact_ijs:
        clk[i,j] = np.abs(link_mat[i,j])
        
    sum_clk = clk[~np.isnan(clk)].sum()
    avg_degrees = contact_ijs.shape[0]/num_rods
    
    # color code edges according to the entanglement field
    vals_edge = np.zeros((len(edges),3))
    e_field_sampled = []
    for i,edge in enumerate(edges):
        vert_i = edge[0]
        vert_j = edge[1]
        
        nodes_i = nodes[vert_i]
        nodes_j = nodes[vert_j]
        mid_point = (nodes_i + nodes_j)/2
        
        fF.analyze_local_volume(mid_point, R_omega, 0.01)
        rep_val = fF.return_entanglement()
        e_field_sampled.append(rep_val)
        # rep_val = fF.return_number_of_labels()
        
        # coolwarm
        vals_edge[i,:] = plt.cm.coolwarm(rep_val/20)[:3]
    

    ps_all_nodes.add_color_quantity(f"rod_colors", vals_edge, defined_on='edges', enabled=True)


    ps_all_nodes.set_material("clay")
    # ps.look_at((-5., 0., 1.), (0., 0., 0.))
    ps.set_up_dir("z_up")

    ps_all_nodes.set_transparency(1)
    ps_all_nodes.set_enabled(True)

    ps.set_screenshot_extension(".png");
    filepath = f'{mat_file.stem}_entanglement.png'
    ps.screenshot(filepath,transparent_bg=False)
    
    result = {
        'num_rods':num_rods,
        'length_list':length_list,
        'ij':ij,
        'distances': distances,
        'rod_length':rod_length,
        'rod_diameter':rod_diameter,
        'avg_degrees':avg_degrees,
        'sum_clk':sum_clk,
        'e_field':e_field,
    }
    result_dict[mat_file.stem] = result
    # ps.show()
# %%
avg_degree_list = []
sum_clk_list = []
coef_var_list = []
for result in result_dict.values():
    # print(result['num_rods'])
    # print(result['rod_length'])
    # print(result['rod_diameter'])
    
    avg_degree_list.append(result['avg_degrees'])
    sum_clk_list.append(result['sum_clk'])
    coef_var_list.append(np.nanstd(result['e_field'])/np.nanmean(result['e_field']))
# %%
fig,axs=plt.subplots(1,3)
axs[0].scatter(avg_degree_list,sum_clk_list)
axs[1].scatter(avg_degree_list,coef_var_list)
axs[2].scatter(sum_clk_list,coef_var_list)


# %%
centerline= centerlines[0]
fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
ax.plot(centerline[:,0],centerline[:,1],centerline[:,2],'o-')

def curvature_of_polygonal_curve(nodes):
    tan2 = nodes[2:,:] - nodes[1:-1,:]    
    tan1 = nodes[1:-1,:] - nodes[:-2,:]
    
    nom = np.linalg.norm(2*np.cross(tan1,tan2,axis=1),axis=1)
    den = np.sum(tan1*tan2,axis=1)
    curvature = np.sum(nom/den)
    return curvature, nom/den

_, local_curv = curvature_of_polygonal_curve(centerline)

# %%
# autocorrelation
# from scipy.signal import correlate
# from scipy.spatial.distance import pdist,squareform

# def autocorrelation(x):
#     x = x - np.mean(x)
#     result = correlate(x, x, mode='full')
#     result = result[result.size//2:]
#     result /= result[0]
#     return result

# def crosscorrelation(x,y):
#     x = x - np.mean(x)
#     y = y - np.mean(y)
#     result = correlate(x, y, mode='full')
#     result = result[result.size//2:]
#     result /= result[0]
#     return result

# gamma = autocorrelation(local_curv)
# fig,ax=plt.subplots(1,1)
# ax.plot(gamma,'o')

# %%

fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
for i in range(len(centerlines)):
    centerline = centerlines[i]
    ax.plot(centerline[:,0],centerline[:,1],centerline[:,2])

fF = filamentFields.filamentFields(centerlines)



xlim = np.min(nodes[:,0]),np.max(nodes[:,0])
ylim = np.min(nodes[:,1]),np.max(nodes[:,1])
zlim = np.min(nodes[:,2]),np.max(nodes[:,2])

num_grid = 30
mg = np.meshgrid(np.linspace(xlim[0],xlim[1],num_grid),np.linspace(ylim[0],ylim[1],num_grid),np.linspace(zlim[0],zlim[1],num_grid))
sampling_points = np.array([mg[0].flatten(),mg[1].flatten(),mg[2].flatten()]).T

start = time.time()
result = fF.analyze_local_volume_over_domain(sampling_points, 2, 0.01)
print(f'Time taken: {time.time()-start:.2f} seconds')
n_field = result[:,0]
e_field = result[:,3]

# e_image = e_field.reshape((num_grid,num_grid,num_grid))
# e_proj_z = np.sum(e_image,axis=2)
# e_proj_x = np.sum(e_image,axis=0)
# e_proj_x = np.rot90(e_proj_x)

# fig,axs=plt.subplots(1,2)
# axs[0].imshow(e_proj_z)
# axs[1].imshow(e_proj_x)



# griddata fitting?
# from scipy.interpolate import griddata

# 
# fp = filamentprocessing.FilamentProcessing(centerlines,1000,1000,1)
# fp.calculate_filament_distance_matrix(1000,1000)
# fp.calculate_svd_scores(1000,100)
# fp.calculate_end_to_end_properties(1000,1)
# fp.get_corrected_end_points()

# ij = fp.get_ij()
# scores = fp.get_scores()    
# print(ij)
# print(scores)
# scores = np.array(scores)
# distances = scores[:,0]

# color code edges according to the entanglement field
vals_edge = np.zeros((len(edges),3))
for i,edge in enumerate(edges):
    vert_i = edge[0]
    vert_j = edge[1]
    
    nodes_i = nodes[vert_i]
    nodes_j = nodes[vert_j]
    mid_point = (nodes_i + nodes_j)/2
    
    fF.analyze_local_volume(mid_point, 3, 0.01)
    rep_val = fF.return_entanglement()
    # rep_val = fF.return_number_of_labels()
    
    # coolwarm
    vals_edge[i,:] = plt.cm.coolwarm(rep_val/50)[:3]

ps_all_nodes.add_color_quantity(f"rod_colors", vals_edge, defined_on='edges', enabled=True)
ps_all_nodes.set_material("clay")
# ps.look_at((-5., 0., 1.), (0., 0., 0.))
ps.set_up_dir("z_up")

ps_all_nodes.set_transparency(1)
ps_all_nodes.set_enabled(True)

ps.set_screenshot_extension(".png");
ps.screenshot('temp.png',transparent_bg=False)

# %%
e_field[~np.isnan(e_field)].std()/e_field[~np.isnan(e_field)].mean()


