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

from visualizations import plot_contacts

# mamba create -n simdata-analysis numpy scipy numba pandas matplotlib jax jaxlib
# use python lower version

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
def get_curr_force_essentials(curr_force_all_info,curr_nodes):
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
            return curr_force_essentials

def main(pth):
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('protocol_id', metavar = 'protocol_id', type=str, nargs='?')
    parser.add_argument('folder_path', metavar = 'folder_path', type=str, nargs='?')
    args = parser.parse_args()
    protocol_id = args.protocol_id
    folder_path = args.folder_path
    
    if args.protocol_id is None:
        protocol_id = 'EntangleCarrotCake5_intermediate'
        folder_path = pth
    
    # python analyze_sim_dataset.py CarrotCake2-ExciteEntangle /Users/yeonsu/Data/from_cluster/20240528-1714_RUN_EntangleCarrotCake4,N1000_AR200_mu0.2_visc0_boxsize0.5_freq10_amp0.05
    print(f'Analyzing the dataset')
    print(f'Protocol ID: {protocol_id}')
    print(f'Folder path: {folder_path}')
    
    R_omega_factor = 4
    arrow_scale_factor = 100
    
    rod_length = 1
    
    xlim = [-0.5,0.5]
    ylim = [-0.5,0.5]
    zlim = [-1,1]
    
    visualize_fields = 1
    visualize_rods_contacts = 1
    skip_frames = 10
    max_rows = 100000
    overlap_factor = 5
    
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
    h_omega = R_omega/overlap_factor
    num_grids = int((xlim[1]-xlim[0])/h_omega*2)

    time_line, node_list, contact_list = import_all_log(pth,max_rows=max_rows)
    time_line0 = time_line
    time_line = np.array(time_line)
    time_line = time_line[time_line <= 10]
        
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
            
    data_root = '../../analysis-data'    
    output_folder = f'{data_root}/{protocol_id}/{file_id}_{datetime_str}/'
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    else:
        print(f'Folder already exists: {output_folder}')

    current_datetime = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    data_output_folder = f'{output_folder}/data_{current_datetime}/'
    field_output_folder = f'{output_folder}/fieldsPlots_{current_datetime}/'
    contact_output_folder = f'{output_folder}/rodsContactPlots_{current_datetime}/'    
    if not os.path.exists(data_output_folder):
        os.makedirs(data_output_folder)
    if not os.path.exists(field_output_folder):
        os.makedirs(field_output_folder)        
    if not os.path.exists(contact_output_folder):
        os.makedirs(contact_output_folder)

    print(f'Size of time_line: {len(time_line)}')
    print(f'Number of rods: {num_rods}')
    print(f'Aspect ratio: {AR}')
    print(f'Output folder: {output_folder}')
    
    last_curve = node_list[time_line0.index(time_line[-1])].reshape((-1,10,3))
    fig,ax=plt.subplots(1,1,figsize=(10,10),subplot_kw={'projection':'3d'})
    for rod in last_curve:
        ax.plot(rod[:,0],rod[:,1],rod[:,2],linewidth=0.5)
    ax.view_init(elev=0, azim=0)
    plt.savefig(f'{data_output_folder}/lastFrame.png',dpi=300)
    np.savetxt(f'{data_output_folder}/lastFrame.txt',last_curve.reshape(-1,30))
        
    # mg = np.meshgrid(np.linspace(xlim[0],xlim[1],num_grids),mid_y,np.linspace(zlim[0],zlim[1],num_grids))
    mg = np.meshgrid(np.linspace(xlim[0],xlim[1],num_grids),np.linspace(-ylim[0],ylim[1],num_grids),np.linspace(zlim[0],zlim[1],num_grids))
    sampling_points = np.array([mg[0].flatten(),mg[1].flatten(),mg[2].flatten()]).T
            
    
    # total_entanglement_over_time = np.zeros(len(time_line))
    total_number_of_contacts = np.zeros(len(time_line))
    total_force_sum = np.zeros(len(time_line))
    
    fF = filamentFields.filamentFields([],[])    
    start = time.time()
    last_frame = len(time_line)-1
    print(f'Last frame: {last_frame}')
    for frame in range(0,len(time_line),1):
        curr_nodes = node_list[frame].reshape((-1,10,3))
        curr_force_all_info = contact_list[frame].reshape(-1,18)
        curr_force_essentials = get_curr_force_essentials(curr_force_all_info,curr_nodes)
        total_number_of_contacts[frame] = len(curr_force_essentials)
        total_force_sum[frame] = np.sum(np.linalg.norm(curr_force_essentials[:,3:6],axis=1))
        
    fig,ax=plt.subplots(1,1,figsize=(10,5))
    ax.plot(time_line,total_number_of_contacts)
    ax.set_xlabel('Time')
    ax.set_ylabel('Number of contacts')
    plt.savefig(f'{data_output_folder}/number_of_contacts.png',dpi=300)
    
    fig,ax=plt.subplots(1,1,figsize=(10,5))
    ax.plot(time_line,total_force_sum)
    ax.set_xlabel('Time')
    ax.set_ylabel('Total force sum')
    plt.savefig(f'{data_output_folder}/total_force_sum.png',dpi=300)
        
    
if __name__ == "__main__":
    pathlist = []
    pathlist.append('/Users/yeonsu/Data/from_cluster/20240531-2224_RUN_EntangleCarrotCake5_N0625-AR125')
    pathlist.append('/Users/yeonsu/Data/from_cluster/20240531-2224_RUN_EntangleCarrotCake5_N0125-AR025')
    pathlist.append('/Users/yeonsu/Data/from_cluster/20240531-2224_RUN_EntangleCarrotCake5_N0250-AR050')
    pathlist.append('/Users/yeonsu/Data/from_cluster/20240531-2224_RUN_EntangleCarrotCake5_N0375-AR075')
    pathlist.append('/Users/yeonsu/Data/from_cluster/20240531-2224_RUN_EntangleCarrotCake5_N0500-AR100')
    for pth in pathlist:
        main(pth)