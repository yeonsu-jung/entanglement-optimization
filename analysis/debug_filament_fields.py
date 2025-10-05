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
from scipy.io import savemat
from visualizations import plot_contacts
# %%
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
# %%
def main():
# %%
    output_folder = 'filament_fields_output2'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    pathlist = []
    # pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/CalmModelo1/20240620-0039_RUN_CalmEEModelo1_N0625_AR125')
    pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/CalmModelo1/20240619-2353_RUN_CalmEEModelo1_N0125_AR025')
    # pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/CalmModelo1/20240620-0039_RUN_CalmEEModelo1_N0250_AR050')
    # pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/CalmModelo1/20240620-0039_RUN_CalmEEModelo1_N0300_AR060')
    # pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/CalmModelo1/20240620-0039_RUN_CalmEEModelo1_N0350_AR070')
    # pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/CalmModelo1/20240620-0039_RUN_CalmEEModelo1_N0375_AR075')
    # pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/CalmModelo1/20240620-0039_RUN_CalmEEModelo1_N0400_AR080')
    # pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/CalmModelo1/20240620-0039_RUN_CalmEEModelo1_N0450_AR090')
    # pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/CalmModelo1/20240620-0039_RUN_CalmEEModelo1_N0500_AR100')
    # pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/CalmModelo1/20240620-0039_RUN_CalmEEModelo1_N0525_AR105')
    # pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/CalmModelo1/20240620-0039_RUN_CalmEEModelo1_N0550_AR110')
    # pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/CalmModelo1/20240620-0039_RUN_CalmEEModelo1_N0575_AR115')
    # pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/CalmModelo1/20240620-0039_RUN_CalmEEModelo1_N0600_AR120')
    # pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/CalmModelo1/20240620-0039_RUN_CalmEEModelo1_N0750_AR150')
    # pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/CalmModelo1/20240620-0039_RUN_CalmEEModelo1_N0875_AR175')
    # pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/CalmModelo1/20240620-0039_RUN_CalmEEModelo1_N1000_AR200')
    # pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/CalmModelo1/20240620-0039_RUN_CalmEEModelo1_N1250_AR250')
    # pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/CalmModelo1/20240620-0039_RUN_CalmEEModelo1_N1500_AR300')
    # %%
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
        time_line0 = time_line
        time_line = np.array(time_line)
        time_line = time_line[time_line <= 10]
        
    # %%
        # last_curve = node_list[time_line0.index(time_line[-1])].reshape((-1,10,3))
        # mg = np.meshgrid(np.linspace(xlim[0],xlim[1],num_grids),mid_y,np.linspace(zlim[0],zlim[1],num_grids))
        mg = np.meshgrid(np.linspace(xlim[0],xlim[1],num_grids),np.linspace(ylim[0],ylim[1],num_grids),np.linspace(zlim[0],zlim[1],num_grids))
        sampling_points = np.array([mg[0].flatten(),mg[1].flatten(),mg[2].flatten()]).T
        
        fF = filamentFields.filamentFields([],[])    
        last_nodes = node_list[time_line0.index(time_line[-1])].reshape((-1,10,3))
        last_force_all_info = contact_list[-1].reshape(-1,18)
        last_force_essentials = get_curr_force_essentials(last_force_all_info,last_nodes)
        
        tmp = []
        for i in range(len(last_nodes)):
            rr = last_nodes[i]
            xx = np.interp(np.linspace(0,1,60),np.linspace(0,1,10),rr[:,0])
            yy = np.interp(np.linspace(0,1,60),np.linspace(0,1,10),rr[:,1])
            zz = np.interp(np.linspace(0,1,60),np.linspace(0,1,10),rr[:,2])
            rr = np.array([xx,yy,zz]).T
            tmp.append(rr)
            
        last_nodes = tmp
        plt.close('all')
        fig,ax=plt.subplots(1,1,figsize=(10,10),subplot_kw={'projection':'3d'})
        for rod in last_nodes:
            ax.plot(rod[:,0],rod[:,1],rod[:,2],linewidth=0.5)
        plt.savefig(f'{output_folder}/{file_id}_rods_last_frame.png')
        
        
        # %%
        fF.update_filament_nodes_list(last_nodes)
        fF.update_contact_array(last_force_essentials)
        
        fF.precompute(R_omega)
        results = fF.analyze_local_volume_over_domain_from_precomputed(sampling_points, R_omega, rod_diameter)
        n_volume = results[:,0].reshape((num_grids,num_grids,num_grids))        
        e_volume = results[:,3].reshape((num_grids,num_grids,num_grids))
        e_volume[np.isnan(e_volume)] = 0
        
        savemat(f'{output_folder}/{file_id}_e_field.mat',{'e_volume':e_volume},do_compression=True)
        # %%
        from skimage.morphology import convex_hull_image
        # convex_hull = convex_hull_image(n_volume > 0)
        e_field_inside = e_volume[e_volume > 0]
            
        plt.close('all')
        plt.hist(e_field_inside,bins=100)
        plt.savefig(f'{output_folder}/{file_id}_e_field_hist.png')
        
        Q1 = np.percentile(e_field_inside, 25)
        Q3 = np.percentile(e_field_inside, 75)
        IQR = Q3 - Q1
        outlier_step = 1.5 * IQR
        upper_bound = Q3 + outlier_step
        
        outlier_e_volume = e_volume > upper_bound        
        img = np.max(outlier_e_volume,axis=0)
        img = np.flipud(img.T)
        plt.close('all')
        plt.imshow(img)
        plt.savefig(f'{output_folder}/{file_id}_outlier_e_field.png')
        # %%
        
        np.savez_compressed(f'{output_folder}/{file_id}_all_fields.npz',
                            all_fields=results,R_omega=R_omega,AR=AR,num_rods=num_rods,
                            upper_bound=upper_bound,folder_path=folder_path)
        
        print(f'Elapsed time: {time.time()-start_time}')
        print(f'Finished {folder_path}')
        
        
if __name__ == "__main__":
    main()