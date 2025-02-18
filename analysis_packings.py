import numpy as np
import jax.numpy as jnp

from data_io import import_all_log, save_as_mat
from scipy.io import loadmat
import matplotlib.pyplot as plt
from potentials import acn_over_ij, dist_lin_seg_over_ij, skewness_over_ij, angle_over_ij

import os
import re
from pathlib import Path

import time
import pickle
import pandas as pd
from scipy.optimize import curve_fit


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
    ARs = df['AR'].unique()
    ARs = np.sort(ARs)

    # Add new columns to the DataFrame
    df['Distance'] = None
    df['Angle'] = None
    df['Skewness'] = None
    df['Entanglement'] = None
    df['NumContacts'] = None

    # Iterate over rows and compute the desired quantities
    for idx, row in df.iterrows():
        rr = row['rr']
        num_rods = row['N']
        AR = row['AR']
        rod_diameter = 1 / AR

        # Split the coordinates into r1 and r2 (assuming rr is a 2D array)
        r1 = rr[:, :3]
        r2 = rr[:, 3:]
        
        # Get the indices for the upper triangle (excluding the diagonal)
        i_idx, j_idx = jnp.triu_indices(num_rods, k=1)

        # Compute various quantities for all pairs (i,j)
        local_acn = acn_over_ij(r1, r2, i_idx, j_idx)
        local_dist = dist_lin_seg_over_ij(r1, r2, i_idx, j_idx)
        local_angle = angle_over_ij(r1, r2, i_idx, j_idx)
        local_skewness = skewness_over_ij(r1, r2, i_idx, j_idx)
        local_contacts = np.count_nonzero(local_dist < 1.05 * rod_diameter) / num_rods

        # If local_dist, local_angle, etc. are arrays and you want a scalar summary
        # (e.g., an average), you might use np.mean(local_dist), etc.
        # For example:
        # avg_dist = np.mean(local_dist)

        # Update the DataFrame using df.at for clarity
        df.at[idx, 'Distance'] = local_dist
        df.at[idx, 'Angle'] = local_angle
        df.at[idx, 'Skewness'] = local_skewness
        df.at[idx, 'Entanglement'] = np.sum(np.abs(local_acn)) / (num_rods * (num_rods - 1) / 2)
        df.at[idx, 'NumContacts'] = local_contacts


        # print(row['Distance'], row['Angle'], row['Skewness'], row['Entanglement'])


    num_rods = 500
    plt.figure(figsize=(2,1.8))
    for AR in ARs:
        df_AR = df[ (df['AR'] == AR) & (df['N'] == num_rods) ]        
        dist = np.concatenate(df_AR['Distance'].to_numpy())
        plt.hist(dist, bins=100, alpha=0.5, label=f'{AR}', density=True)
    plt.legend(title=r'$\alpha$',loc='center left', bbox_to_anchor=(1, 0.5),fontsize=8, title_fontsize=8,)
    plt.xlabel('Distance, $d_{ij}/l$')
    plt.ylabel('PDF, $P(d_{ij}/l)$')
    plt.savefig('junkyard/packings369_distance_pdf.svg', dpi=300, bbox_inches='tight')
    
    plt.figure(figsize=(2,1.8))
    for AR in ARs:
        df_AR = df[ (df['AR'] == AR) & (df['N'] == num_rods) ]        
        angle = np.concatenate(df_AR['Angle'].to_numpy())
        plt.hist(angle, bins=100, alpha=0.5, label=f'{AR}', density=True)
    plt.legend(title=r'$\alpha$',loc='center left', bbox_to_anchor=(1, 0.5),fontsize=8, title_fontsize=8,)
    plt.xlabel('Angle, $\\theta_{ij}$')
    plt.ylabel('PDF, $P(\\theta_{ij})$')
    plt.savefig('junkyard/packings369_angle_pdf.svg', dpi=300, bbox_inches='tight')


    # contacts
    plt.figure(figsize=(2,1.8))
    num_contacts_wrt_AR = []
    for AR in ARs:
        df_AR = df[ (df['AR'] == AR) & (df['N'] == num_rods) ]        
        num_contacts = df_AR['NumContacts'].to_numpy()
        avg = np.mean(num_contacts)
        std = np.std(num_contacts)
        num_contacts_wrt_AR.append([avg,std])
    num_contacts_wrt_AR = np.array(num_contacts_wrt_AR)

    plt.errorbar(ARs,num_contacts_wrt_AR[:,0],yerr=num_contacts_wrt_AR[:,1],fmt='o-',label='Data')
    plt.xlabel('Aspect Ratio, $\\alpha$')
    plt.ylabel('Avg. no. contacts')
    plt.savefig('junkyard/packings369_num_contacts.svg', dpi=300, bbox_inches='tight')

    # entanglement
    plt.figure(figsize=(2,1.8))
    entanglement_wrt_AR = []
    for AR in ARs:
        df_AR = df[ (df['AR'] == AR) & (df['N'] == num_rods) ]        
        entanglement = df_AR['Entanglement'].to_numpy()
        avg = np.mean(entanglement)
        std = np.std(entanglement)
        entanglement_wrt_AR.append([avg,std])
    entanglement_wrt_AR = np.array(entanglement_wrt_AR)

    def exponential_hill(x,a,b):
        return a*np.exp(-x/b)
    
    popt,pcov = curve_fit(exponential_hill,ARs,entanglement_wrt_AR[:,0],p0=[0.45,100])

    plt.errorbar(ARs,entanglement_wrt_AR[:,0],yerr=entanglement_wrt_AR[:,1],fmt='o-',label='Data')
    x_fit = np.linspace(20,500,100)
    y_fit = exponential_hill(x_fit,*popt)
    plt.plot(x_fit,y_fit,label=f'$y={popt[0]:.2f}e^{{-x/{popt[1]:.2f}}}$')

    plt.xlabel('Aspect Ratio, $\\alpha$')
    plt.ylabel('Entanglement')
    plt.legend()
    plt.savefig('junkyard/packings369_entanglement.svg', dpi=300, bbox_inches='tight')


    plt.figure(figsize=(2,1.8))
    ns = []
    x_bins = np.linspace(0,2,100)
    for AR in ARs:
        df_AR = df[ (df['AR'] == AR) & (df['N'] == num_rods) ]
        num_rep = len(df_AR)

        # data = np.abs(np.array(df_AR['Skewness'].tolist()).flatten()) - 0.5
        data = np.abs(np.array(df_AR['Skewness'].tolist()).reshape(3,-1) ) - 0.5

        tmp = []
        for i in range(num_rep):
            n,_,_ = plt.hist(data[i], bins=x_bins, alpha=0.5, label=f'{AR}', density=True)
            tmp.append(n)
        # n,_,_ = plt.hist(data, bins=x_bins, alpha=0.5, label=f'{AR}', density=True)

        ns.append(tmp)
        # skewness = np.concatenate(df_AR['Skewness'].to_numpy())
        # plt.hist(skewness, bins=100, alpha=0.5, label=f'{AR}', density=True)

    def gaussian(x, a, b):
        return a * np.exp(-b * (x)**2)
    
    # def func(x, a, b):
    #     return a * np.exp(-b * (x)**2)
    
    plt.close('all')
    plt.figure(figsize=(2,1.8))
    sigma_list = []
    for n in ns:
        num_rep = len(n)
        avg = np.mean(n,axis=0)
        std = np.std(n,axis=0)
        plt.plot(x_bins[:-1],avg,'-',label=f'{AR}')
        plt.fill_between(x_bins[:-1], avg-std, avg+std, alpha=0.5)
        
        tmp = []
        for i in range(num_rep):
            popt, _ = curve_fit(gaussian, x_bins[:-1], avg, p0=[1, 1])
            b = popt[1]
            sigma = 1/np.sqrt(2*b)
            tmp.append(sigma)

        sigma_list.append(tmp)
    
    plt.legend(title=r'$\alpha$',loc='center left', bbox_to_anchor=(1, 0.5),fontsize=8, title_fontsize=8,)
    plt.xlabel('Skewness, $\\Delta_{ij}$')
    plt.ylabel('PDF, $P(\\Delta_{ij})$')
    plt.savefig('junkyard/packings369_skewness_pdf.svg', dpi=300, bbox_inches='tight')

    def power_law(x,a):
        return a*x**(-1)
    
    sigma_list = np.array(sigma_list)
    sigmas = np.mean(sigma_list,axis=1)


    popt,pcov = curve_fit(power_law,ARs,sigmas)
    x_fit = np.linspace(20,500,100)
    y_fit = power_law(x_fit,*popt)

    plt.figure(figsize=(2,1.8))
    # plt.loglog(ARs,sigmas,'o-',label='Data')
    plt.errorbar(ARs,sigmas,yerr=np.std(sigma_list,axis=1),fmt='.-',label='Data')
    plt.xscale('log')
    plt.loglog(x_fit,y_fit,label=f'$y={popt[0]:.2f}x^{{-1}}$')
    plt.xlabel('Aspect Ratio, $\\alpha$')
    plt.ylabel('Skewness width, $\\sigma_\Delta$')
    plt.legend(fontsize=8)
    plt.savefig('junkyard/packings369_skewness_width.svg', dpi=300, bbox_inches='tight')