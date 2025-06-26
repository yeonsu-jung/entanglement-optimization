# %%
import numpy as np
from matplotlib import pyplot as plt
import re
from pathlib import Path
from scipy.io import loadmat
from scipy import ndimage
import polyscope as ps
from skimage import measure
from skimage.morphology import convex_hull_image
from numba import jit
from matplotlib.path import Path as mpath


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})


parent_folder = Path('/Users/yeonsu/GitHub/entanglement/data')

# %%
def points_inside_hull(points, hull):
    hull_path = mpath(hull.points[hull.vertices])
    inside_points = [p for p in points if hull_path.contains_point(p)]
    return inside_points

def unpack_centerlines(centerlines):
    unpacked = np.vstack(centerlines)
    labels = np.zeros(unpacked.shape[0],dtype=np.int64)
    start_idx = 0
    for i,cl in enumerate(centerlines):
        end_idx = start_idx + cl.shape[0]
        labels[start_idx:end_idx] = i
        start_idx = end_idx
    return unpacked,labels

def get_principal_axis_length(rr):
    if rr.size == 0:
        return {}
    
    if rr.shape[0] == 1:
        return {
            'Centroid': rr[0],
            'PrincipalAxisLength': [0, 0, 0],
            'Orientation': [0, 0, 0],
            'EigenValues': [0, 0, 0],
            'EigenVectors': np.zeros((3, 3)),
            'EffectiveSystemSize': [0, 0, 0]
        }
    
    centroid = np.mean(rr, axis=0)
    centered = rr - centroid
    mu000 = np.sum(centered**0)
    mu200 = np.sum(centered[:, 0]**2) / mu000 + 1/12
    mu020 = np.sum(centered[:, 1]**2) / mu000 + 1/12
    mu002 = np.sum(centered[:, 2]**2) / mu000 + 1/12
    mu110 = np.sum(centered[:, 0] * centered[:, 1]) / mu000
    mu011 = np.sum(centered[:, 1] * centered[:, 2]) / mu000
    mu101 = np.sum(centered[:, 2] * centered[:, 0]) / mu000
    
    num_points = rr.shape[0]
    cov_mat = np.array([
        [mu200, mu110, mu101],
        [mu110, mu020, mu011],
        [mu101, mu011, mu002]
    ]) / num_points
    
    # Calculate effective system size
    effective_system_size = 2 * np.sqrt(np.var(centered, axis=0))
    
    U, S, _ = np.linalg.svd(cov_mat)
    S = sorted(S, reverse=True)
    
    if U[0, 0] < 0:
        U = -U
        U[:, 2] = -U[:, 2]
    
    eig_values, eig_vectors = np.linalg.eig(cov_mat)
    ind = np.argsort(eig_values)[::-1]
    eig_values = eig_values[ind]
    eig_vectors = eig_vectors[:, ind]
    
    return {
        'Centroid': centroid,
        'PrincipalAxisLength': [4 * np.sqrt(S[i] * num_points) for i in range(3)],
        'Orientation': rotm2euler(U),
        'EigenValues': eig_values * num_points,
        'EigenVectors': eig_vectors,
        'EffectiveSystemSize': effective_system_size
    }

def calculate_central_moments(im, centroid, i, j, k):
    r, c, p = im.shape
    central_moments = np.outer((np.arange(1, r+1) - centroid[1])**i, (np.arange(1, c+1) - centroid[0])**j)
    z = np.reshape((np.arange(1, p+1) - centroid[2])**k, (1, 1, p))
    central_moments = central_moments[..., np.newaxis] * z * im
    return np.sum(central_moments)

def rotm2euler(rotm):
    k = 180 / np.pi
    cy = np.hypot(rotm[0, 0], rotm[1, 0])
    
    if cy > 16 * np.finfo(float).eps:
        psi = k * np.arctan2(rotm[2, 1], rotm[2, 2])
        theta = k * np.arctan2(-rotm[2, 0], cy)
        phi = k * np.arctan2(rotm[1, 0], rotm[0, 0])
    else:
        psi = k * np.arctan2(-rotm[1, 2], rotm[1, 1])
        theta = k * np.arctan2(-rotm[2, 0], cy)
        phi = 0
        
    return [phi, theta, psi]



@jit(nopython=True)
def get_local_labels(unpacked,labels,query_points,R):
    sampled_labels = []
    for i,query_point in enumerate(query_points):
        I1 = (unpacked[:,0] - query_point[0])**2
        I2 = (unpacked[:,1] - query_point[1])**2
        I3 = (unpacked[:,2] - query_point[2])**2
        
        TF = I1 + I2 + I3 < R**2
        sampled_labels.append(labels[TF])
        
        print(f'{i}')
        
    return sampled_labels

class Data:
    def __init__(self, alpha, epsilon, all_fields, centerlines):
        self.alpha = alpha
        self.epsilon = epsilon
        self.all_fields = all_fields
        self.centerlines = centerlines
        
class Fields:
    def __init__(self, all_fields):
        self.n = all_fields['n'][0][0]
        self.phi = all_fields['phi'][0][0]
        self.s = all_fields['s'][0][0]
        self.c = all_fields['c'][0][0]
        self.e = all_fields['e'][0][0]
        self.e2 = all_fields['e2'][0][0][0]
        self.cx = all_fields['cx'][0][0][0]
        self.cy = all_fields['cy'][0][0][0]
        self.cz = all_fields['cz'][0][0][0]
        self.R = all_fields['R'][0][0][0][0]
        self.h = all_fields['h'][0][0][0][0]
        

data_dict = {}
# primary: alpha, secondary: epsilon
for pth in parent_folder.iterdir():
    if not pth.is_dir():
        continue
    search_result = re.search('alpha(\d+)_epsilon(\d+)',pth.stem)
    alpha = int(search_result.group(1))
    epsilon = int(search_result.group(2))
    
    
    
    all_fields_file_name = 'all_fields_V04_1.mat'
    if not (pth / all_fields_file_name).exists():
        all_fields_file_name = 'all_fields_V04_0.mat'
        
    centerline_file_name = 'centerlines.mat'
    
    
    matobj = loadmat(pth / all_fields_file_name)
    all_fields = matobj['all_fields']
    all_fields_cleaned = Fields(all_fields)
    
    matobj = loadmat(pth / centerline_file_name)
    centerlines = matobj['centerlines']
    
    tmp = []
    for i, centerline in enumerate(centerlines):
        tmp.append(centerline[0])        
    
    dta = Data(alpha, epsilon, all_fields_cleaned, tmp)
    # data_dict[alpha][epsilon] = dta
    if alpha not in data_dict:
        data_dict[alpha] = {}
    data_dict[alpha][epsilon] = dta
    
# %%
zeta_results = {}

alpha_list = data_dict.keys()
cluster_size_list = []
for method in ['IQR','Z','MAD']:
    for alpha in alpha_list:
        if alpha == 38:
            rod_radius = 650/alpha/2
        else:
            rod_radius = 650/alpha
        
        n_field = data_dict[alpha][0].all_fields.n
        convex_hull = convex_hull_image(n_field > 0)
        convex_hull_points = np.argwhere(convex_hull)
        
            
        system_x = data_dict[alpha][0].all_fields.cx[convex_hull_points[:,0]]
        system_y = data_dict[alpha][0].all_fields.cx[convex_hull_points[:,1]]
        system_z = data_dict[alpha][0].all_fields.cx[convex_hull_points[:,2]]
        
        stat = get_principal_axis_length(np.vstack([system_x,system_y,system_z]).T)
        # system_size = stat['EffectiveSystemSize']
        # system_size = stat['PrincipalAxisLength']
        system_size = [np.max(system_x) - np.min(system_x),
        np.max(system_y) - np.min(system_y),
        np.max(system_z) - np.min(system_z)]
        
        e_field = data_dict[alpha][0].all_fields.e
        e_field_inside = e_field[convex_hull]
        R = data_dict[alpha][0].all_fields.R
        
        if method == 'IQR':
            Q1 = np.percentile(e_field_inside, 25)
            Q3 = np.percentile(e_field_inside, 75)
            IQR = Q3 - Q1
            outlier_step = 1.5 * IQR
            upper_bound = Q3 + outlier_step
        
        # upper bound by Z score (3Z)
        elif method == 'Z':
            Z = 3
            upper_bound = np.mean(e_field_inside) + Z * np.std(e_field_inside)
        
        elif method == 'MAD':
            # upper bound by MAD (Median Absolute Deviation)
            MAD = np.median(np.abs(e_field_inside - np.median(e_field_inside)))
            upper_bound = np.median(e_field_inside) + 3.5 * MAD / 0.6745
        
        mask = e_field > upper_bound
        # connected components for mask?

        # Perform connected component analysis
        labels, num_labels = ndimage.label(mask)
        mask_points = np.argwhere(mask)
        
        label_sizes = np.zeros(num_labels-1)
        label_centers = np.zeros((num_labels-1,3))
        label_xyz_size = np.zeros((num_labels-1,3))
        for i in range(num_labels-1):
            label_sizes[i] = np.sum(labels == i+1)
            label_image = labels == i+1
            
            # label_image_points = np.argwhere(label_image)
            # stat = get_principal_axis_length(label_image_points)
            # label_xyz_size[i] = stat['EffectiveSystemSize']
            # label_center[i] = stat['Centroid']            
            # label_centers[i] = np.argwhere(labels == i+1).mean(axis=0)
            
            _pts = np.argwhere(labels == i+1)
            
            cx = []
            cy = []
            cz = []
            
            for pt in _pts:
                cx.append(data_dict[alpha][0].all_fields.cx[pt[0]])
                cy.append(data_dict[alpha][0].all_fields.cy[pt[1]])
                cz.append(data_dict[alpha][0].all_fields.cy[pt[2]])
                
            pts = np.vstack([cx,cy,cz]).T                
            # zeta_x = np.max(cx) - np.min(cx)
            # zeta_y = np.max(cy) - np.min(cy)
            # zeta_z = np.max(cz) - np.min(cz)
            # label_xyz_size[i] = np.array([zeta_x,zeta_y,zeta_z])
            stat = get_principal_axis_length(pts)
            # label_xyz_size[i] = stat['EffectiveSystemSize']
            # label_xyz_size[i] = stat['PrincipalAxisLength']
            label_xyz_size[i] = [np.max(cx) - np.min(cx),
            np.max(cy) - np.min(cy),
            np.max(cz) - np.min(cz)]
            
            label_centers[i] = stat['Centroid']
        
        i_max = np.argmax(label_sizes)
        xyz_size = label_xyz_size[i_max]
            
        centerlines = data_dict[alpha][0].centerlines
        whole_centerline = np.vstack(centerlines)
        # system_size = np.array([np.max(whole_centerline[:,i]) - np.min(whole_centerline[:,i]) for i in range(3)])
        
        if alpha not in zeta_results:
            zeta_results[alpha] = {}        
        zeta_results[alpha][method] = (xyz_size/system_size)
        
        cluster_size_list.append(np.max(label_sizes))
            
        _cx = data_dict[alpha][0].all_fields.cx[mask_points[:,0]]
        _cy = data_dict[alpha][0].all_fields.cy[mask_points[:,1]]
        _cz = data_dict[alpha][0].all_fields.cz[mask_points[:,2]]
        query_points = np.vstack([_cx,_cy,_cz]).T        
        
        unpacked,centerline_labels = unpack_centerlines(centerlines)
        sampled_labels = []
        for label_center in label_centers:
            tmp = centerline_labels[np.linalg.norm(unpacked - label_center,axis=1) < R/2]
            sampled_labels.append(tmp)
                
        undersampled = [centerline[::10] for centerline in centerlines]
        # unpacked,labels = unpack_centerlines(undersampled)
        # sampled_labels = get_local_labels(unpacked,labels,label_centers,R)
        # print(sampled_labels)

        unique_labels = np.unique(np.concatenate(sampled_labels))

        edges = []
        for i in unique_labels:
            r1 = undersampled[i][0]
            r2 = undersampled[i][-1]
            
            edges.append(np.concatenate([r1,r2]))
        edges = np.array(edges)
        nodes = edges.reshape(-1,3)
        bounding_box = np.array([[np.min(nodes[:,i]),np.max(nodes[:,i])] for i in range(3)])
        # Define the box vertices from the bounding box
        box_vertices = np.array([
            [bounding_box[0, 0], bounding_box[1, 0], bounding_box[2, 0]],
            [bounding_box[0, 1], bounding_box[1, 0], bounding_box[2, 0]],
            [bounding_box[0, 1], bounding_box[1, 1], bounding_box[2, 0]],
            [bounding_box[0, 0], bounding_box[1, 1], bounding_box[2, 0]],
            [bounding_box[0, 0], bounding_box[1, 0], bounding_box[2, 1]],
            [bounding_box[0, 1], bounding_box[1, 0], bounding_box[2, 1]],
            [bounding_box[0, 1], bounding_box[1, 1], bounding_box[2, 1]],
            [bounding_box[0, 0], bounding_box[1, 1], bounding_box[2, 1]]
        ])

        # Define the box edges
        box_edges = np.array([
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 4],
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7]
        ])


        ps.init()
        ps.set_ground_plane_mode("none")
        ps.set_ground_plane_mode("shadow_only")  # set +Z as up direction

        # register a point cloud
        N = mask_points.shape[0]
        points = mask_points

        ps_cloud = ps.register_point_cloud("my points", query_points)

        clr = np.array([76, 153, 204])/255
        ps_cloud.set_color(clr)
        ps_cloud.set_radius(R/2,relative=False)

        ps.set_up_dir("z_up")

        num_nodes_each_rod = 2
        connectivities = np.array([[i, i + 1] for i in range(len(nodes) - 1) if i % num_nodes_each_rod != num_nodes_each_rod - 1])
        ps_rods = ps.register_curve_network("my rods", nodes, connectivities,transparency=0.5)
        # gold
        ps_rods.set_color((1.0, 0.71, 0.29))
        ps_rods.set_radius(rod_radius, relative=False)

        # Register the box as a curve network
        ps_box = ps.register_curve_network("box", box_vertices, box_edges,enabled=True)
        ps_box.set_radius(3, relative=False)  # Set the radius of the box edges
        ps_box.set_color((0.8, 0.8, 0.8))  # Set the color of the box edges

        ps.look_at((4000,4000,2000),(-2000,-2000,200))

        # view the point cloud with all of these quantities
        # ps.show() 

        root_dir = f'/Users/yeonsu/Dropbox (Harvard University)/Data/PrunedData/rod-sim-pnas-revision/visuals'
        output_dir = f'{root_dir}/percolation/{method}'
        import os
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        ps.set_screenshot_extension(".png");
        screenshot_path = f'{output_dir}/percolation_{alpha}.png'
        ps.screenshot(screenshot_path,transparent_bg=False)

# %%

# for method in ['IQR','Z','MAD']:
for method in ['IQR']:
    tmp = np.array([zeta_results[alpha][method] for alpha in alpha_list])

    single_column_size = (2.3,1.5)
    fig,ax=plt.subplots(figsize=single_column_size)
    plt.rcParams.update({'font.size': 8})

    ax.plot(alpha_list,tmp[:,0],'o-',mfc='none',label=r'$\xi_x^c/L_x$')
    ax.plot(alpha_list,tmp[:,1],'o-',mfc='none',label=r'$\xi_y^c/L_y$')
    ax.plot(alpha_list,tmp[:,2],'o-',mfc='none',label=r'$\xi_z^c/L_z$')
    # ax.set_xticks(list(alpha_list))
    ax.set_xlabel(r'$\alpha$',labelpad=-3)
    ax.set_ylabel(r'$\xi/L$')
    

    plt.legend(fontsize=6)
    # plt.tight_layout()
    plt.savefig(f'{root_dir}/percolation/percolation_vs_alpha_{method}.svg',dpi=300,bbox_inches='tight')

# zeta_results[alpha][method] = (xyz_size/system_size)

# %%
fig,axs=plt.subplots(1,3,figsize=(12,3))
plt.rcParams.update({'font.size': 10})

for i,method in enumerate(['IQR','Z','MAD']):
    tmp = np.array([zeta_results[alpha][method] for alpha in alpha_list])

    axs[i].plot(alpha_list,tmp[:,0],'o-',mfc='none',label=r'$\xi_x^c/L_x$')
    axs[i].plot(alpha_list,tmp[:,1],'o-',mfc='none',label=r'$\xi_y^c/L_y$')
    axs[i].plot(alpha_list,tmp[:,2],'o-',mfc='none',label=r'$\xi_z^c/L_z$')
    
axs[0].set_title(r'IQR')
axs[1].set_title(r'Z-score')
axs[2].set_title(r'MAD')

axs[0].set_ylabel(r'$\xi/L$')
axs[1].set_ylabel(r'$\xi/L$')
axs[2].set_ylabel(r'$\xi/L$')


for ax in axs:
    ax.set_xlabel(r'$\alpha$')
    
# legend outside
box = axs[2].get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
axs[2].legend(loc='center left', bbox_to_anchor=(1, 0.5))
# plt.savefig(f'{root_dir}/percolation/percolation_vs_alpha_wrt_method.png',dpi=300,bbox_inches='tight')


# %%


alpha_list = data_dict.keys()
cluster_size_list = []
zeta_results_wrt_M = {}

for M in np.linspace(0.5,3,10):
    for alpha in alpha_list:
        if alpha == 38:
            rod_radius = 650/alpha/2
        else:
            rod_radius = 650/alpha
        
        n_field = data_dict[alpha][0].all_fields.n        
        convex_hull = convex_hull_image(n_field > 0)
        convex_hull_points = np.argwhere(convex_hull)
        system_x = data_dict[alpha][0].all_fields.cx[convex_hull_points[:,0]]
        system_y = data_dict[alpha][0].all_fields.cx[convex_hull_points[:,1]]
        system_z = data_dict[alpha][0].all_fields.cx[convex_hull_points[:,2]]
        
        stat = get_principal_axis_length(np.vstack([system_x,system_y,system_z]).T)
        # system_size = stat['EffectiveSystemSize']
        # system_size = stat['PrincipalAxisLength']
        system_size = [np.max(system_x) - np.min(system_x),
        np.max(system_y) - np.min(system_y),
        np.max(system_z) - np.min(system_z)]
        
        e_field = data_dict[alpha][0].all_fields.e
        e_field_inside = e_field[convex_hull]
        R = data_dict[alpha][0].all_fields.R
        
        Q1 = np.percentile(e_field_inside, 25)
        Q3 = np.percentile(e_field_inside, 75)
        IQR = Q3 - Q1
        outlier_step = M * IQR
        upper_bound = Q3 + outlier_step
        
        # upper bound by MAD (Median Absolute Deviation)
        # MAD = np.median(np.abs(e_field_inside - np.median(e_field_inside)))
        # upper_bound = np.median(e_field_inside) + M * MAD / 0.6745
        mask = e_field > upper_bound
        

        # Perform connected component analysis
        labels, num_labels = ndimage.label(mask)
        mask_points = np.argwhere(mask)
        
        label_sizes = np.zeros(num_labels-1)
        label_centers = np.zeros((num_labels-1,3))
        label_xyz_size = np.zeros((num_labels-1,3))
        for i in range(num_labels-1):
            label_sizes[i] = np.sum(labels == i+1)
            label_image = labels == i+1
            _pts = np.argwhere(labels == i+1)
            
            cx = []
            cy = []
            cz = []
            
            for pt in _pts:
                cx.append(data_dict[alpha][0].all_fields.cx[pt[0]])
                cy.append(data_dict[alpha][0].all_fields.cy[pt[1]])
                cz.append(data_dict[alpha][0].all_fields.cy[pt[2]])
                
            pts = np.vstack([cx,cy,cz]).T                
            stat = get_principal_axis_length(pts)
            # label_xyz_size[i] = stat['EffectiveSystemSize']
            # label_xyz_size[i] = stat['PrincipalAxisLength']
            label_xyz_size[i] = [np.max(cx) - np.min(cx),
            np.max(cy) - np.min(cy),
            np.max(cz) - np.min(cz)]
            
            label_centers[i] = stat['Centroid']
        
        i_max = np.argmax(label_sizes)
        xyz_size = label_xyz_size[i_max]
            
        centerlines = data_dict[alpha][0].centerlines
        whole_centerline = np.vstack(centerlines)
        if alpha not in zeta_results_wrt_M:
            zeta_results_wrt_M[alpha] = {}
        zeta_results_wrt_M[alpha][M] = (xyz_size/system_size)
    
    
# %%

fig,axs=plt.subplots(1,3,figsize=(12,3))
plt.rcParams.update({'font.size': 10})
for M in np.linspace(0.5,3,10):
    tmp = np.array([zeta_results_wrt_M[alpha][M] for alpha in alpha_list])
    axs[0].plot(alpha_list,tmp[:,0],'o-',mfc='none',label=rf'${M:.2f}$')
    axs[1].plot(alpha_list,tmp[:,1],'o-',mfc='none',label=rf'${M:.2f}$')
    axs[2].plot(alpha_list,tmp[:,2],'o-',mfc='none',label=rf'${M:.2f}$')

axs[0].set_title(r'$\xi_x^c/L_x$')
axs[1].set_title(r'$\xi_y^c/L_y$')
axs[2].set_title(r'$\xi_z^c/L_z$')

axs[0].set_ylabel(r'$\xi_x^c/L_x$')
axs[1].set_ylabel(r'$\xi_y^c/L_y$')
axs[2].set_ylabel(r'$\xi_z^c/L_z$')


for ax in axs:
    ax.set_xlabel(r'$\alpha$')
    
# legend outside
box = axs[2].get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
axs[2].legend(loc='center left', bbox_to_anchor=(1, 0.5))
# plt.savefig(f'{root_dir}/percolation/percolation_vs_alpha_M.png',dpi=300,bbox_inches='tight')
    
# %%

num_alpha = len(alpha_list)
num_M = len(zeta_results_wrt_M[alpha])

fig,axs=plt.subplots(1,num_M,figsize=(20,20))

for k,v in zeta_results_wrt_M[alpha].items():
    
    
    

# %%
single_column_size = (2.3,1.5)
fig,ax=plt.subplots(figsize=single_column_size)
plt.rcParams.update({'font.size': 8})

ax.plot(alpha_list,tmp[:,0],'o-',mfc='none',label=r'$\xi_x^c/L_x$')
ax.plot(alpha_list,tmp[:,1],'o-',mfc='none',label=r'$\xi_y^c/L_y$')
ax.plot(alpha_list,tmp[:,2],'o-',mfc='none',label=r'$\xi_z^c/L_z$')
# ax.set_xticks(list(alpha_list))
ax.set_xlabel(r'$\alpha$',labelpad=-3)
ax.set_ylabel(r'$\xi/L$')


plt.legend(fontsize=6)
# plt.tight_layout()
plt.savefig(f'{root_dir}/percolation/percolation_vs_alpha_{method}.png',dpi=300,bbox_inches='tight')


# %%
alpha = 200
s_field = data_dict[alpha][0].all_fields.s
# %%
_ = plt.hist(s_field.flatten(),bins=100)
# %%
for alpha in alpha_list:
    if alpha == 38:
        rod_radius = 650/alpha/2
    else:
        rod_radius = 650/alpha
    
    n_field = data_dict[alpha][0].all_fields.n
    s_field = data_dict[alpha][0].all_fields.s
    e_field = data_dict[alpha][0].all_fields.e
    convex_hull = convex_hull_image(n_field > 0)
    convex_hull_points = np.argwhere(convex_hull)
    
    n_field_inside = n_field[convex_hull]
    s_field_inside = s_field[convex_hull]
    e_field_inside = e_field[convex_hull]
    
    
    
    break

# %%
fig,ax=plt.subplots()
ax.hist(s_field_inside[n_field_inside>0],bins=100)
    
# %%


