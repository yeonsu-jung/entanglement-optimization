import numpy as np
import jax.numpy as jnp

from data_io import import_all_log, save_as_mat
from scipy.io import loadmat
import matplotlib.pyplot as plt
from potentials import acn_over_ij, dist_lin_seg_over_ij

import os
import re
from pathlib import Path

import time
import pickle
import pandas as pd



def postprocess_cache_369():
    pathlist = []
    pathlist.append('/Users/yeonsu/GitHub/dismech-rods-main/data/MaxEntFinal/37,178,56')
    pathlist.append('/Users/yeonsu/GitHub/dismech-rods-main/data/MaxEntFinal/6,7,8')
    pathlist.append('/Users/yeonsu/GitHub/dismech-rods-main/data/MaxEntFinal/919,461,568')

    # post-process and cache the data
    random_keys_list = []
    N_list = []
    AR_list = []
    Scale_list = []
    rr_list = []

    for pth in pathlist:
        for f in os.listdir(pth):
            if f.endswith('.txt'):
                file_path = os.path.join(pth, f)
                print(file_path)

                # parse filename                
                pattern = re.compile(r'MaxEnt(\d+),(\d+),(\d+)-N(\d+)-AR(\d+)-Scale(\d+)')
                match = pattern.search(Path(file_path).stem)
                random_keys = [int(match.group(1)), int(match.group(2)), int(match.group(3))]
                N = int(match.group(4))
                AR = int(match.group(5))
                Scale = int(match.group(6))

                # load data
                rr = np.loadtxt(file_path, delimiter=' ')
                
                random_keys_list.append(random_keys)
                N_list.append(N)
                AR_list.append(AR)
                Scale_list.append(Scale)
                rr_list.append(rr)

    # save the data
    data = {
        'random_keys': random_keys_list,
        'N': N_list,
        'AR': AR_list,
        'Scale': Scale_list,
        'rr': rr_list
    }

    df = pd.DataFrame(data)
    df.to_pickle('dataframe_results/data_packings369.pkl')


    return

if __name__ == "__main__":

    df = pd.read_pickle('dataframe_results/data_packings369.pkl')
    df.shape
    print(df)

    df['N'].unique()

    local_df = df[df['AR'] == 500]


    # AR_list = []
    # angle_list = []
    # num_contacts_list = []
    # distances_list = []
    # skewness_list = []
    # total_entanglement_list = []
    # for pth in pathlist:
    #     dt_string, AR, num_rods, random_keys = parse_pathname(pth)

    #     AR = float(re.search('AR(\d+)',pth).group(1))
    #     AR_list.append(AR)
    #     diameter = 1/AR

    #     # qq = np.load(pth)
    #     # qq_reshaped = qq.reshape(-1,num_rods,5)
    #     # q = qq_reshaped[-1]
    #     q = np.loadtxt(pth)
    #     x = q_to_x(q)
    #     q_pairs = create_pairs(q.reshape(-1,5))
    #     distances = all_pairwise_distances(q_pairs)

    #     angles = all_pairwise_angles(q_pairs)
    #     angle_list.append(angles)
        
    #     num_contacts = np.count_nonzero(distances < diameter*1.05)
    #     num_contacts_list.append(num_contacts)
    #     distances_list.append(distances)

    #     skewness = all_pairwise_skewness(q_pairs)
    #     skewness_list.append(skewness)

    #     final_e = total_effective_potential(q)
    #     total_entanglement_list.append(final_e)


    # rr = np.loadtxt(file_path, delimiter=' ')
    # num_rods = rr.shape[0]
    # r1 = rr[:,:3]
    # r2 = rr[:,3:]    

    # i_indices, j_indices = jnp.triu_indices(num_rods, k=1)
    # pairwise_acn = acn_over_ij(r1, r2, i_indices, j_indices)
    # pairwise_dist = dist_lin_seg_over_ij(r1, r2, i_indices, j_indices)
    # dist = np.min( pairwise_dist )
    # print(dist)

    # entanglement = np.sum(np.abs(pairwise_acn)) / (num_rods*(num_rods-1)/2)
    # print(entanglement)