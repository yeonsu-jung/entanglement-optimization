# %%
from matplotlib import pyplot as plt
import numpy as np
import re
import os
from data_io import import_all_log, parse_path_string
from fields import get_local_fields_at_a_point
from pathlib import Path

from fields import get_local_fields_at_a_point
import time
import pandas as pd
import filamentFields
import argparse
import datetime

# mamba create -n simdata-analysis numpy scipy numba pandas matplotlib jax jaxlib
# use python lower version

def plot_contacts(contact_info,scale_factor,ax):
    ni1 = contact_info["ni1"]
    ni2 = contact_info["ni2"]
    nj1 = contact_info["nj1"]
    nj2 = contact_info["nj2"]
    fi1 = contact_info["fi1"]
    fi2 = contact_info["fi2"]
    fj1 = contact_info["fj1"]
    fj2 = contact_info["fj2"]
    contact_point_i = contact_info["contact_point_i"]
    contact_force_i = contact_info["contact_force_i"]
    contact_point_j = contact_info["contact_point_j"]
    contact_force_j = contact_info["contact_force_j"]
    log_contact_force_i = contact_info["log_contact_force_i"]
    log_contact_force_j = contact_info["log_contact_force_j"]
    
    if (np.isnan(contact_point_i).any() or np.isnan(contact_point_j).any()):
        return
    
    ax.plot([ni1[0],ni2[0]],[ni1[1],ni2[1]],[ni1[2],ni2[2]],'r',linewidth=0.5)
    ax.plot([nj1[0],nj2[0]],[nj1[1],nj2[1]],[nj1[2],nj2[2]],'r',linewidth=0.5)
    ax.plot(contact_point_i[0],contact_point_i[1],contact_point_i[2],'g.')
    ax.plot(contact_point_j[0],contact_point_j[1],contact_point_j[2],'g.')
    
    # log scale
    ax.quiver(contact_point_i[0],contact_point_i[1],contact_point_i[2],log_contact_force_i[0]/scale_factor,log_contact_force_i[1]/scale_factor,log_contact_force_i[2]/scale_factor,color='g',linestyle='-')
    ax.quiver(contact_point_j[0],contact_point_j[1],contact_point_j[2],log_contact_force_j[0]/scale_factor,log_contact_force_j[1]/scale_factor,log_contact_force_j[2]/scale_factor,color='g',linestyle='-')



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
    
    # tol = 1e-1
    # assert(np.abs(x_frac-y_frac) < tol and np.abs(y_frac-z_frac) < tol)
    
    return x_frac


# at edges between (idx1, idx3) and (idx2, idx4)
# (4 * idx1 + e1, contact_gradient[e1], idx5);
# (4 * idx3 + e1, contact_gradient[e1 + 3], idx5);
# (4 * idx2 + e1, contact_gradient[e1 + 6], idx6);
# (4 * idx4 + e1, contact_gradient[e1 + 9], idx6);

def process_contact_data(single_contact_info,curr_nodes):
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
    fj1 = -single_contact_info[12:15]
    fj2 = -single_contact_info[15:18]

    ni1 = curr_nodes[rod_i][node_i1]
    ni2 = curr_nodes[rod_i][node_i2]
    nj1 = curr_nodes[rod_j][node_j1]
    nj2 = curr_nodes[rod_j][node_j2]
    
    contact_point_i = guess_contact_point(fi1,fi2) * (ni2-ni1) + ni1
    contact_force_i = fi1 + fi2
    contact_point_j = guess_contact_point(fj1,fj2) * (nj2-nj1) + nj1
    contact_force_j = fj1 + fj2
    
    log_contact_force_i = np.sign(contact_force_i) * np.log(np.abs(contact_force_i) + 1e-6)
    log_contact_force_j = np.sign(contact_force_j) * np.log(np.abs(contact_force_j) + 1e-6)
    
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
                    "log_contact_force_i":log_contact_force_i,
                    "log_contact_force_j":log_contact_force_j,
                    "ni1":ni1,
                    "ni2":ni2,
                    "nj1":nj1,
                    "nj2":nj2,
                    "fi1":fi1,
                    "fi2":fi2,
                    "fj1":fj1,
                    "fj2":fj2}
    
    return contact_info

def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('protocol_id', metavar = 'protocol_id', type=str, nargs='?')
    parser.add_argument('folder_path', metavar = 'folder_path', type=str, nargs='?')
    args = parser.parse_args()
    protocol_id = args.protocol_id
    folder_path = args.folder_path
    
    if args.protocol_id is None:
        protocol_id = 'CarrotCake2-ExciteEntangle'
        folder_path ='/Users/yeonsu/Data/from_cluster/20240528-1714_RUN_EntangleCarrotCake4,N1000_AR200_mu0.2_visc0_boxsize0.5_freq10_amp0.05'
    
    # python analyze_sim_dataset.py CarrotCake2-ExciteEntangle /Users/yeonsu/Data/from_cluster/20240528-1714_RUN_EntangleCarrotCake4,N1000_AR200_mu0.2_visc0_boxsize0.5_freq10_amp0.05
    
    
    print(f'Analyzing the dataset')
    print(f'Protocol ID: {protocol_id}')
    print(f'Folder path: {folder_path}')
    
    R_omega_factor = 4
    arrow_scale_factor = 100
    
    rod_length = 1

    mid_y = 0
    num_grids = 50
    xlim = [-0.5,0.5]
    zlim = [-1,1]
    
    visualize_fields = 1
    visualize_rods_contacts = 1
    skip_frames = 1
    max_rows = 100
    
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
        
    rod_diameter = rod_length/AR
    R_omega = R_omega_factor*np.sqrt(rod_length*rod_diameter)

    time_line, node_list, contact_list = import_all_log(pth,max_rows=max_rows)
    
    # find analysis-data
    TF_found = False
    for pth in Path(os.getcwd()).parent.glob('./analysis-data'):
        if pth.is_dir():
            data_root = str(pth)
            TF_found = True
            break
        
    if not TF_found:
        for pth in Path(os.getcwd()).parent.parent.glob('./analysis-data'):
            if pth.is_dir():
                data_root = str(pth)
                TF_found = True
                break
    
    output_folder = f'{data_root}/{protocol_id}/{file_id}_{datetime_str}/'
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    else:
        print(f'Folder already exists: {output_folder}')

    current_datetime = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    field_output_folder = f'{output_folder}/fieldsPlots_{current_datetime}/'
    contact_output_folder = f'{output_folder}/rodsContactPlots_{current_datetime}/'

    if not os.path.exists(field_output_folder):
        os.makedirs(field_output_folder)
        
    if not os.path.exists(contact_output_folder):
        os.makedirs(contact_output_folder)

    print(f'Size of time_line: {len(time_line)}')
    print(f'Number of rods: {num_rods}')
    print(f'Aspect ratio: {AR}')
    print(f'Output folder: {output_folder}')

    mg = np.meshgrid(np.linspace(xlim[0],xlim[1],num_grids),mid_y,np.linspace(zlim[0],zlim[1],num_grids))
    sampling_points = np.array([mg[0].flatten(),mg[1].flatten(),mg[2].flatten()]).T
            
    n_fields_over_time = np.zeros((len(time_line),len(sampling_points)))
    phi_fields_over_time = np.zeros((len(time_line),len(sampling_points)))
    S_fields_over_time = np.zeros((len(time_line),len(sampling_points)))
    e_fields_over_time = np.zeros((len(time_line),len(sampling_points)))
    c_fields_over_time = np.zeros((len(time_line),len(sampling_points)))
    f_fields_over_time = np.zeros((len(time_line),len(sampling_points)))
    total_entanglement_over_time = np.zeros(len(time_line))
    
    fF = filamentFields.filamentFields([],[])
    
    start = time.time()
    for frame in range(0,len(time_line),skip_frames):
        curr_nodes = node_list[frame].reshape((-1,10,3))
        curr_force_all_info = contact_list[frame].reshape(-1,18)
        num_total_contacts = len(curr_force_all_info)

        curr_force_essentials = np.zeros((num_total_contacts,6))
        for query_index in range(num_total_contacts):
            single_contact_info = curr_force_all_info[query_index]
            contact_info = process_contact_data(single_contact_info,curr_nodes)        
            pi = contact_info['contact_point_i']
            pj = contact_info['contact_point_j']
            cij = (pi+pj)/2
            fij = contact_info['contact_force_i']        
            curr_force_essentials[query_index] = np.array([cij[0],cij[1],cij[2],fij[0],fij[1],fij[2]])
            
        if visualize_rods_contacts and (frame % skip_frames == 0):
            fig,ax=plt.subplots(1,1,figsize=(10,10),subplot_kw={'projection':'3d'})
            for rod in curr_nodes:
                ax.plot(rod[:,0],rod[:,1],rod[:,2],'k',linewidth=0.2)
            for query_index in range(0,len(curr_force_all_info),1):
                single_contact_info = curr_force_all_info[query_index]
                contact_info = process_contact_data(single_contact_info,curr_nodes)
                plot_contacts(contact_info,arrow_scale_factor,ax)
                # plot contacts
            ax.text(0.5,0.5,1,f'time: {time_line[frame]:.2f}')
            ax.view_init(elev=0, azim=0)
            ax.set_xlim([-0.5,0.5])
            ax.set_ylim([-0.5,0.5])
            ax.set_zlim([-1,1])            
            plt.savefig(f'{contact_output_folder}/frame_{frame:04d}.png',dpi=300)
            plt.close()

        n_fields = np.zeros(len(sampling_points))
        phi_fields = np.zeros(len(sampling_points))
        S_fields = np.zeros(len(sampling_points))
        e_fields = np.zeros(len(sampling_points))
        c_fields = np.zeros(len(sampling_points))
        f_fields = np.zeros(len(sampling_points))
        
        fF.updateFilamentNodesList(curr_nodes)
        fF.updateContactArray(curr_force_essentials)
        
        another_start = time.time()
        fF.compute_total_linking_matrix()
        print(f'Elapsed time for computing total linking matrix: {time.time()-another_start}')
        
        for iterator,sampling_point in enumerate(sampling_points):
            fF.analyzeLocalVolumeFromPrecomputed(sampling_point, R_omega, rod_diameter)
            n_fields[iterator] = fF.return_number_of_labels()
            phi_fields[iterator] = fF.return_volume_fraction()
            S_fields[iterator] = fF.return_orientational_order_parameter()
            e_fields[iterator] = fF.return_entanglement()
            c_fields[iterator] = fF.return_number_of_local_contacts()
            f_fields[iterator] = fF.return_force_sum()
        
        n_fields_over_time[frame] = n_fields
        phi_fields_over_time[frame] = phi_fields
        S_fields_over_time[frame] = S_fields
        e_fields_over_time[frame] = e_fields
        c_fields_over_time[frame] = c_fields
        f_fields_over_time[frame] = f_fields
        
        
        total_entanglement_over_time[frame] = fF.return_total_entanglement()
        
        if visualize_fields and (frame % skip_frames == 0):
            phi_image = phi_fields.reshape((num_grids,num_grids))
            phi_image = np.flipud(phi_image.T)
            S_image = S_fields.reshape((num_grids,num_grids))
            S_image = np.flipud(S_image.T)
            e_image = e_fields.reshape((num_grids,num_grids))
            e_image = np.flipud(e_image.T)
            c_image = c_fields.reshape((num_grids,num_grids))
            c_image = np.flipud(c_image.T)
            f_image = f_fields.reshape((num_grids,num_grids))
            f_image = np.flipud(f_image.T)            
        
            fig,axs=plt.subplots(1,5,figsize=(20,4))
            fig.colorbar(axs[0].imshow(phi_image, extent=[xlim[0], xlim[1], zlim[0], zlim[1]]), ax=axs[0])
            fig.colorbar(axs[1].imshow(S_image, extent=[xlim[0], xlim[1], zlim[0], zlim[1]]), ax=axs[1])
            fig.colorbar(axs[2].imshow(e_image, extent=[xlim[0], xlim[1], zlim[0], zlim[1]]), ax=axs[2])
            fig.colorbar(axs[3].imshow(c_image, extent=[xlim[0], xlim[1], zlim[0], zlim[1]]), ax=axs[3])
            fig.colorbar(axs[4].imshow(f_image, extent=[xlim[0], xlim[1], zlim[0], zlim[1]]), ax=axs[4])
            
            axs[0].set_title('Volume fraction')
            axs[1].set_title('Orientational order parameter')
            axs[2].set_title('Entanglement')
            axs[3].set_title('Number of contacts')
            axs[4].set_title('Force sum')
            axs[0].text(-2,1,f'time: {time_line[frame]:.2f} sec')
            
            plt.tight_layout()
            plt.savefig(f'{field_output_folder}/frame_{frame:04d}.png',dpi=300)
            plt.close()
        
        if (frame % 10 == 0):
            print(f'Evaluated {frame} frames so far. Elapsed time: {time.time()-start:.2f} sec')
            
    
    np.savez_compressed(f'{output_folder}/all_fields_over_time.npz',
                        n_fields_over_time=n_fields_over_time,
                        phi_fields_over_time=phi_fields_over_time,
                        S_fields_over_time=S_fields_over_time,
                        e_fields_over_time=e_fields_over_time,
                        c_fields_over_time=c_fields_over_time,
                        f_fields_over_time=f_fields_over_time,
                        total_entanglement_over_time=total_entanglement_over_time)
    
if __name__ == "__main__":
    main()