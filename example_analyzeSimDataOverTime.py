# %%
import numpy as np
from matplotlib import pyplot as plt
import re
import os
from data_io import import_all_log
from fields import get_local_fields_at_a_point
from pathlib import Path

def sample_nodes_locally(nodes,center,R):
    I1 = np.linalg.norm((nodes[:,:3] - center), axis=1) < R
    I2 = np.linalg.norm((nodes[:,3:6] - center), axis=1) < R
    I_sphere_segments = I1 & I2
    return nodes[I_sphere_segments]

def label_nodes(nodes):
    labels = []
    for i,nodeset in enumerate(nodes):
        n = nodeset.shape[0]
        lb = np.ones(n,dtype=np.int64)*i        
        labels.append(lb)
        
    return np.hstack(np.array(labels))

def rowwise_norm(A):
    return np.sqrt(np.sum(A**2,axis=1))

def get_bbox(curves):
    x_min = np.min(curves[:,:,0])
    x_max = np.max(curves[:,:,0])
    y_min = np.min(curves[:,:,1])
    y_max = np.max(curves[:,:,1])
    z_min = np.min(curves[:,:,2])
    z_max = np.max(curves[:,:,2])
    return [x_min, x_max, y_min, y_max, z_min, z_max]

def parse_path_string(pth):    
    filename = pth.split('/')[-1]        
    file_id = filename.split('-mu')[0]
    
    surfix_match = re.search(r'\d{8}-\d{6}', filename)
    surfix = surfix_match.group(0) if surfix_match else None
    
    num_rods_match = re.search(r'-N(\d+)-', filename)
    num_rods = int(num_rods_match.group(1)) if num_rods_match else None
    
    AR_match = re.search(r'-AR(\d+)-', filename)
    AR = int(AR_match.group(1)) if AR_match else None
    
    return file_id, surfix, num_rods, AR


# '/Users/yeonsu/Data/from-cluster/NonIntersectingBox-N1000-AR200-Scale1-mu0.20-visc0.00-amp0.00_allLog_20240527-193121.csv'
folder_path ='/Users/yeonsu/Data/from_cluster/20240527-1934_RUN_CarrotCake2,N250_AR50_mu0.2_visc0_boxsize0.5_freq10_amp0.05/'
folder_path = Path(folder_path)
protocol_id = 'CarrotCake2-ExciteEntangle'



possible_paths = []
for pth in folder_path.glob('**/*.csv'):
    if 'lastFrame' in str(pth):
        continue
    possible_paths.append(pth)
    
if len(possible_paths) == 0:
    print('No csv files found in the folder')
    exit()
elif len(possible_paths) > 1:
    print('Multiple csv files found in the folder')
    exit()
    
pth = str(possible_paths[0])

file_id,surfix,num_rods,AR = parse_path_string(pth)
time_line, node_list, contact_list = import_all_log(pth,max_rows=10000)

output_folder = f'/Users/yeonsu/Videos/{protocol_id}/{file_id}_twoPlots_highResolution/'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
else:
    print(f'Folder already exists: {output_folder}')

print(f'Size of time_line: {len(time_line)}')
print(f'Number of rods: {num_rods}')
print(f'Aspect ratio: {AR}')
# %%
rod_length = 1
rod_diameter = rod_length/AR
R_omega = np.sqrt(2*rod_length*rod_diameter)
# %%
mid_y = 0
num_grids = 30
xlim = [-0.5,0.5]
zlim = [-1,1]

mg = np.meshgrid(np.linspace(xlim[0],xlim[1],num_grids),mid_y,np.linspace(zlim[0],zlim[1],num_grids))
sampling_points = np.array([mg[0].flatten(),mg[1].flatten(),mg[2].flatten()]).T

import importlib
import fields
importlib.reload(fields)
from fields import get_local_fields_at_a_point

import time
start = time.time()
import filamentFields
# %%
start = time.time()
for frame in range(0,len(sampling_points),1):
    curves = node_list[frame]
    curves = curves.reshape(num_rods, -1, 3)

    n_fields = np.zeros(len(sampling_points))
    phi_fields = np.zeros(len(sampling_points))
    S_fields = np.zeros(len(sampling_points))
    e_fields = np.zeros(len(sampling_points))
    fF = filamentFields.filamentFields(curves)

    for iterator,sampling_point in enumerate(sampling_points):
        fF.analyzeLocalVolume(sampling_point, R_omega*2, rod_diameter)
        n_fields[iterator] = fF.return_number_of_labels()
        phi_fields[iterator] = fF.return_volume_fraction()
        S_fields[iterator] = fF.return_orientational_order_parameter()
        e_fields[iterator] = fF.return_entanglement()
    
    e_fields_img = e_fields.reshape(num_grids,num_grids)
    e_fields_img = np.flipud(e_fields_img.T)
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    # axs[1].imshow(e_fields_img, extent=[-2, 2, -1, 1],vmin=0,vmax=60)
    fig.colorbar(axs[1].imshow(e_fields_img, extent=[xlim[0], xlim[1], zlim[0], zlim[1]],vmin=0,vmax=240), ax=axs[1])
    
    for curve in curves:
        axs[0].plot(curve[:, 0], curve[:, 2], alpha=1)
    for ax in axs:
        ax.set_xlim([-2, 2])
        ax.set_ylim([-1, 1])
        ax.set_aspect('equal')  # Ensure aspect ratio is equal
    axs[0].set_title('Rods')
    axs[1].set_title('Entanglement field')
    axs[0].text(-1.5, 0.8, f'Time: {time_line[frame]} sec', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{output_folder}/frames_{frame:04d}.png', dpi=300)
    plt.close('all')
    
    print(f'Elapsed time: {time.time()-start}')


# %%
