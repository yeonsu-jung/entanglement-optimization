# %%
from pathlib import Path
import os
import numpy as np
import re
from data_io import import_all_log, parse_path_string

from scipy.io import savemat, loadmat
from scipy import ndimage
from matplotlib import pyplot as plt
# %%
root_path = Path('filament_fields_output')
# %%
npz_list = list(root_path.glob('**/*.npz'))

alpha_list = []
for pth in npz_list:
    alpha_list.append(int(re.search(r'N(\d+)-AR(\d+)',str(pth)).group(2)))
# %%
# sort npz_list by AR
npz_list = [x for _,x in sorted(zip(alpha_list,npz_list))]

# %%
# structural_element = np.ones((3,3,3))
# center-connected
structural_element = np.ones((3,3,3))
# Set elements to 0 that are not connected by faces or edges
# We only need to remove corners, keeping face and edge connected elements
structural_element[0, 0, 0] = 0
structural_element[0, 0, 2] = 0
structural_element[0, 2, 0] = 0
structural_element[0, 2, 2] = 0
structural_element[2, 0, 0] = 0
structural_element[2, 0, 2] = 0
structural_element[2, 2, 0] = 0
structural_element[2, 2, 2] = 0

xi_dict = {}
vol_dict = {}
for pth in npz_list:
    npzfile = np.load(pth)
    all_fields = npzfile['all_fields']
    e_field = all_fields[:,3]
    num_grids = np.round((e_field.size)**(1/3)).astype(int)
    system_size = np.array([1,1,1])*num_grids
    
    num_rods = npzfile['num_rods']
    AR = npzfile['AR']
    AR = int(AR)
    e_volume = e_field.reshape((num_grids,num_grids,num_grids))

    e_field_inside = e_volume[e_volume > 0]
    Q1 = np.percentile(e_field_inside, 25)
    Q3 = np.percentile(e_field_inside, 75)
    IQR = Q3 - Q1
    outlier_step = 1.5 * IQR
    upper_bound = Q3 + outlier_step
    # npzfile['upper_bound']
    mask = e_volume > upper_bound
    
    if mask.sum() == 0:
        xi_dict[AR] = np.array([0,0,0])
        continue
    
    # connected components for mask?
    # Perform connected component analysis
    
    labels, num_labels = ndimage.label(mask, structural_element)
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
        label_xyz_size[i] = np.max(_pts,axis=0) - np.min(_pts,axis=0)    
    
    i_max = np.argmax(label_sizes)
    vol_size = label_sizes[i_max]
    xyz_size = label_xyz_size[i_max]
    
    xi_dict[AR] = xyz_size/num_grids
    vol_dict[AR] = vol_size/num_grids**3
    
    tmp = labels == i_max+1
    img = np.max(tmp,axis=0)
    img = np.flipud(img.T)
    plt.close('all')
    plt.imshow(img)
    plt.savefig(f'filament_fields_output/proj0_AR{AR:03d}.png')
    
    img = np.max(tmp,axis=1)
    img = np.flipud(img.T)
    plt.close('all')
    plt.imshow(img)
    plt.savefig(f'filament_fields_output/proj1_AR{AR:03d}.png')
    
    img = np.max(tmp,axis=2)
    img = np.flipud(img.T)
    plt.close('all')
    plt.imshow(img)
    plt.savefig(f'filament_fields_output/proj2_AR{AR:03d}.png')
    
    # savemat(f'filament_fields_output/{file_id}_last_nodes.mat', {'last_nodes':last_nodes})
    
    
    # savemat(f'filament_fields_output/N{num_rods:04d}_AR{AR:03d}.mat',
    #         {'e_field':e_field.reshape((num_grids,num_grids,num_grids)),
    #          'last_nodes': last_nodes})
# %%
e_volume
# %%
label_image.shape
# outlier_e_volume = e_volume > upper_bound
# outlier_e_volume = label_image
outlier_e_volume = mask
# outlier_e_volume = labels == 27
img = np.max(mask,axis=0)
# img = e_volume[:,:,2]
img = np.flipud(img.T)
plt.close('all')
plt.imshow(img)
# plt.imshow(mask[5,:,:])

        
# %%
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})
# sort xi_dict by AR
xi_dict = dict(sorted(xi_dict.items()))
alpha_list = list(xi_dict.keys())

xi_1 = [v[0] for k,v in xi_dict.items()]
xi_2 = [v[1] for k,v in xi_dict.items()]
xi_3 = [2*v[2] for k,v in xi_dict.items()]

 # %%
fig,ax=plt.subplots(1,1,figsize=(2.25,1.7))
ax.plot(alpha_list,xi_1,'o-',label=r'$\xi_x/L$')
ax.plot(alpha_list,xi_2,'o-',label=r'$\xi_y/L$')
ax.plot(alpha_list,xi_3,'o-',label=r'$\xi_z/L$')
plt.legend()
plt.xlabel(r'$\alpha$')
plt.ylabel(r'$\xi/L$')
plt.savefig('filament_fields_output/xi_AR.png',dpi=300,bbox_inches='tight')
# %%
vol_size_list = list(vol_dict.values())
fig,ax=plt.subplots(1,1,figsize=(4,3))
ax.plot(alpha_list,vol_size_list,'o-')
    
# %%
pathlist = []
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PertrubCalmEEModelo1-Fullset/20240620-1254_RUN_PerturbCalmEEModelo1_N0125_AR025_freq100')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PertrubCalmEEModelo1-Fullset/20240620-1302_RUN_PerturbCalmEEModelo1_N0250_AR050_freq100')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PertrubCalmEEModelo1-Fullset/20240620-1302_RUN_PerturbCalmEEModelo1_N0300_AR060_freq100')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PertrubCalmEEModelo1-Fullset/20240620-1302_RUN_PerturbCalmEEModelo1_N0350_AR070_freq100')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PertrubCalmEEModelo1-Fullset/20240620-1302_RUN_PerturbCalmEEModelo1_N0375_AR075_freq100')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PertrubCalmEEModelo1-Fullset/20240620-1302_RUN_PerturbCalmEEModelo1_N0400_AR080_freq100')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PertrubCalmEEModelo1-Fullset/20240620-1302_RUN_PerturbCalmEEModelo1_N0450_AR090_freq100')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PertrubCalmEEModelo1-Fullset/20240620-1302_RUN_PerturbCalmEEModelo1_N0500_AR100_freq100')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PertrubCalmEEModelo1-Fullset/20240620-1302_RUN_PerturbCalmEEModelo1_N0525_AR105_freq100')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PertrubCalmEEModelo1-Fullset/20240620-1302_RUN_PerturbCalmEEModelo1_N0550_AR110_freq100')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PertrubCalmEEModelo1-Fullset/20240620-1302_RUN_PerturbCalmEEModelo1_N0575_AR115_freq100')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PertrubCalmEEModelo1-Fullset/20240620-1302_RUN_PerturbCalmEEModelo1_N0600_AR120_freq100')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PertrubCalmEEModelo1-Fullset/20240620-1302_RUN_PerturbCalmEEModelo1_N0625_AR125_freq100')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PertrubCalmEEModelo1-Fullset/20240620-1302_RUN_PerturbCalmEEModelo1_N0750_AR150_freq100')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PertrubCalmEEModelo1-Fullset/20240620-1302_RUN_PerturbCalmEEModelo1_N0875_AR175_freq100')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PertrubCalmEEModelo1-Fullset/20240620-1302_RUN_PerturbCalmEEModelo1_N1000_AR200_freq100')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PertrubCalmEEModelo1-Fullset/20240620-1302_RUN_PerturbCalmEEModelo1_N1250_AR250_freq100')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PertrubCalmEEModelo1-Fullset/20240620-1302_RUN_PerturbCalmEEModelo1_N1500_AR300_freq100')

# %%
import time
start_time = time.time()
    
# folder_path ='/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PertrubCalmEEModelo1-Fullset/20240620-1254_RUN_PerturbCalmEEModelo1_N0125_AR025_freq100'
for folder_path in pathlist:    

    folder_path = Path(folder_path)
    possible_paths = []
    for pth in folder_path.glob('**/*.csv'):
        if 'lastFrame' in str(pth):
            continue
        else:
            possible_paths.append(pth)    
    if len(possible_paths) == 0:
        print('No csv files found in the folder')
        exit()
    elif len(possible_paths) > 1:
        print('Multiple csv files found in the folder')
        # find heaviest file
        max_size = 0
        for pth in possible_paths:
            size = os.path.getsize(pth)
            if size > max_size:
                max_size = size
                heaviest_file = pth
        possible_paths = [heaviest_file]
        
    pth = str(possible_paths[0])
    file_id,surfix,num_rods,AR,datetime_str = parse_path_string(pth)
    
    
    
    
    
    R_omega_factor = 1
    arrow_scale_factor = 100
    
    rod_length = 1
    
    xlim = [-0.5,0.5]
    ylim = [-0.5,0.5]
    zlim = [-1,1]
    
    visualize_fields = 1
    visualize_rods_contacts = 1
    skip_frames = 10
    max_rows = 100000
    overlap_factor = 2

        
    rod_diameter = rod_length/AR
    R_omega = R_omega_factor*np.sqrt(rod_length*rod_diameter)
    h_omega = R_omega/overlap_factor
    num_grids = int((xlim[1]-xlim[0])/h_omega*2)

    time_line, node_list, contact_list = import_all_log(pth,max_rows=max_rows)
    
    last_nodes = node_list[-1]
    last_nodes = last_nodes.reshape(-1,10,3)
    
    savemat(f'filament_fields_output/{file_id}_last_nodes.mat', {'last_nodes':last_nodes})
    


# %%

for pth in root_path.glob('**/*.npz'):

    npzfile = np.load(pth)
    # np.savez_compressed(f'{output_folder}/{file_id}_all_fields.npz',
    #                         all_fields=results,R_omega=R_omega,AR=AR,num_rods=num_rods,
    #                         upper_bound=upper_bound,folder_path=folder_path)
    
    all_fields = npzfile['all_fields']
    e_field = all_fields[:,3]
    num_grids = np.round((e_field.size)**(1/3)).astype(int)
    num_rods = npzfile['num_rods']
    AR = npzfile['AR']    
    data_root_path = Path('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PertrubCalmEEModelo1-Fullset/')
    
    query = f'_N{num_rods:04d}_AR{AR:03d}_freq100'
    # find _N{num_rods:04d}_AR{AR:03d}_freq100
    for _pth in data_root_path.iterdir():
        if query in str(_pth):
            folder_path = _pth
            break
    
    possible_paths = []
    for pth in folder_path.glob('**/*.csv'):
        if 'lastFrame' in str(pth):
            continue
        else:
            possible_paths.append(pth)    
    if len(possible_paths) == 0:
        print('No csv files found in the folder')
        exit()
    elif len(possible_paths) > 1:
        print('Multiple csv files found in the folder')
        # find heaviest file
        max_size = 0
        for pth in possible_paths:
            size = os.path.getsize(pth)
            if size > max_size:
                max_size = size
                heaviest_file = pth
        possible_paths = [heaviest_file]
        
    pth = str(possible_paths[0])
    file_id,surfix,num_rods,AR,datetime_str = parse_path_string(pth)
    time_line, node_list, contact_list = import_all_log(pth,max_rows=10000000)
    
    last_nodes = node_list[-1]
    last_nodes = last_nodes.reshape(-1,10,3)    
        
    savemat(f'filament_fields_output/N{num_rods:04d}_AR{AR:03d}.mat',
            {'e_field':e_field.reshape((num_grids,num_grids,num_grids)),
             'last_nodes': last_nodes})
    
    