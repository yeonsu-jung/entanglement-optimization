# %%
import numpy as np
from matplotlib import pyplot as plt
import re
import os
from data_io import import_all_log, parse_path_string
from fields import get_local_fields_at_a_point
from pathlib import Path

from fields import get_local_fields_at_a_point
import time
import filamentFields


def plot_contacts(contact_info,scale_factor,ax):
    ni1 = contact_info["ni1"]
    ni2 = contact_info["ni2"]
    nj1 = contact_info["nj1"]
    nj2 = contact_info["nj2"]
    fi1 = contact_info["fi1"]
    fi2 = contact_info["fi2"]
    fi3 = contact_info["fi3"]
    fi4 = contact_info["fi4"]
    contact_point_i = contact_info["contact_point_i"]
    contact_force_i = contact_info["contact_force_i"]
    contact_point_j = contact_info["contact_point_j"]
    contact_force_j = contact_info["contact_force_j"]
    
    ax.plot([ni1[0],ni2[0]],[ni1[1],ni2[1]],[ni1[2],ni2[2]],'r',linewidth=0.5)
    ax.plot([nj1[0],nj2[0]],[nj1[1],nj2[1]],[nj1[2],nj2[2]],'r',linewidth=0.5)
    ax.plot(contact_point_i[0],contact_point_i[1],contact_point_i[2],'g.')
    ax.plot(contact_point_j[0],contact_point_j[1],contact_point_j[2],'g.')

    ax.quiver(contact_point_i[0],contact_point_i[1],contact_point_i[2],contact_force_i[0]/scale_factor,contact_force_i[1]/scale_factor,contact_force_i[2]/scale_factor,color='g',linestyle='-')
    ax.quiver(contact_point_j[0],contact_point_j[1],contact_point_j[2],contact_force_j[0]/scale_factor,contact_force_j[1]/scale_factor,contact_force_j[2]/scale_factor,color='g',linestyle='-')

    # ax.quiver(ni1[0],ni1[1],ni1[2],fi1[0]/scale_factor,fi1[1]/scale_factor,fi1[2]/scale_factor,color='r',linestyle='--')
    # ax.quiver(ni2[0],ni2[1],ni2[2],fi2[0]/scale_factor,fi2[1]/scale_factor,fi2[2]/scale_factor,color='r',linestyle='--')
    # ax.quiver(nj1[0],nj1[1],nj1[2],fi3[0]/scale_factor,fi3[1]/scale_factor,fi3[2]/scale_factor,color='b',linestyle='--')
    # ax.quiver(nj2[0],nj2[1],nj2[2],fi4[0]/scale_factor,fi4[1]/scale_factor,fi4[2]/scale_factor,color='b',linestyle='--')
    # ax.axis('equal')


def guess_contact_point(fi1,fi2):
    fi1x = fi1[0]
    fi1y = fi1[1]
    fi1z = fi1[2]
    fi2x = fi2[0]
    fi2y = fi2[1]
    fi2z = fi2[2]
    
    force_tol = 1e-6
    if ((fi1x**2 + fi1y**2 + fi1z**2) < force_tol) and ((fi2x**2 + fi2y**2 + fi2z**2) < force_tol):
        return np.nan
    
    x_frac = fi2x/(fi1x+fi2x)
    y_frac = fi2y/(fi1y+fi2y)
    z_frac = fi2z/(fi1z+fi2z)
    
    tol = 1e-1
    assert(np.abs(x_frac-y_frac) < tol and np.abs(y_frac-z_frac) < tol)   
    
    return x_frac


# at edges between (idx1, idx3) and (idx2, idx4)
# (4 * idx1 + e1, contact_gradient[e1], idx5);
# (4 * idx3 + e1, contact_gradient[e1 + 3], idx5);
# (4 * idx2 + e1, contact_gradient[e1 + 6], idx6);
# (4 * idx4 + e1, contact_gradient[e1 + 9], idx6);

def process_contact_data(single_contact_info):
    # dismech originally computes force as a gradient of potential
    # thus here we need to change its sign
    rod_i = int(single_contact_info[4])
    rod_j = int(single_contact_info[5])
    node_i1 = int(single_contact_info[0])
    node_i2 = int(single_contact_info[2])
    node_j1 = int(single_contact_info[1])
    node_j2 = int(single_contact_info[3])

    fi1 = -single_contact_info[6:9]
    fi2 = -single_contact_info[9:12]
    fi3 = -single_contact_info[12:15]
    fi4 = -single_contact_info[15:18]

    ni1 = curr_nodes[rod_i][node_i1]
    ni2 = curr_nodes[rod_i][node_i2]
    nj1 = curr_nodes[rod_j][node_j1]
    nj2 = curr_nodes[rod_j][node_j2]
    
    contact_point_i = guess_contact_point(fi1,fi2) * (ni2-ni1) + ni1
    contact_force_i = fi1 + fi2

    contact_point_j = guess_contact_point(fi3,fi4) * (nj2-nj1) + nj1
    contact_force_j = fi3 + fi4
    
    contact_info = {"rod_i":rod_i,
                    "rod_j":rod_j,
                    "node_i1":node_i1,
                    "node_i2":node_i2,
                    "node_j1":node_j1,
                    "node_j2":node_j2,
                    "contact_point_i":contact_point_i,
                    "contact_force_i":contact_force_i,
                    "contact_point_j":contact_point_j,
                    "contact_force_j":contact_force_j,
                    "ni1":ni1,
                    "ni2":ni2,
                    "nj1":nj1,
                    "nj2":nj2,
                    "fi1":fi1,
                    "fi2":fi2,
                    "fi3":fi3,
                    "fi4":fi4}
    
    return contact_info


# '/Users/yeonsu/Data/from-cluster/NonIntersectingBox-N1000-AR200-Scale1-mu0.20-visc0.00-amp0.00_allLog_20240527-193121.csv'
folder_path ='/Users/yeonsu/Data/from_cluster/20240527-1934_RUN_CarrotCake2,N250_AR50_mu0.2_visc0_boxsize0.5_freq10_amp0.05/'
folder_path = Path(folder_path)
protocol_id = 'CarrotCake2-ExciteEntangle'

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
    exit()
    
pth = str(possible_paths[0])
file_id,surfix,num_rods,AR = parse_path_string(pth)

time_line, node_list, contact_list = import_all_log(pth,max_rows=100)
output_folder = f'/Users/yeonsu/Videos/{protocol_id}/{file_id}_twoPlots_highResolution/'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
else:
    print(f'Folder already exists: {output_folder}')

print(f'Size of time_line: {len(time_line)}')
print(f'Number of rods: {num_rods}')
print(f'Aspect ratio: {AR}')
# %%
curr_time = -1
curr_nodes = node_list[curr_time].reshape((num_rods,-1,3))
curr_force = contact_list[curr_time].reshape(-1,18)

curr_force.shape

arrow_scale_factor = 1000
fig,ax=plt.subplots(1,1,figsize=(10,10),subplot_kw={'projection':'3d'})
for rod in curr_nodes:
    ax.plot(rod[:,0],rod[:,1],rod[:,2],'k',linewidth=0.2)
for query_index in range(0,len(curr_force),1):
    single_contact_info = curr_force[query_index]
    contact_info = process_contact_data(single_contact_info)
    plot_contacts(contact_info,arrow_scale_factor,ax)
    # plot contacts
    
# %%
query_index = 2
single_contact_info = curr_force[query_index]

contact_info = process_contact_data(single_contact_info)



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


# %%
n_fields_over_time = np.zeros((len(time_line),len(sampling_points)))
phi_fields_over_time = np.zeros((len(time_line),len(sampling_points)))
S_fields_over_time = np.zeros((len(time_line),len(sampling_points)))
e_fields_over_time = np.zeros((len(time_line),len(sampling_points)))
                              
start = time.time()
for frame in range(0,len(time_line),1):
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
    
    n_fields_over_time[frame] = n_fields
    phi_fields_over_time[frame] = phi_fields
    S_fields_over_time[frame] = S_fields
    e_fields_over_time[frame] = e_fields
    print(f'Evaluating frame {frame} took {time.time()-start} seconds')
    


# %%
