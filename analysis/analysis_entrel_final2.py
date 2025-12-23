# %%
import sys
sys.path.append('../core')
sys.path.append('core')

# /Users/yeonsu/GitHub/filamentFields/filamentFields.cpython-312-darwin.so
sys.path.append('/Users/yeonsu/GitHub/filamentFields')

this_file = __file__
from pathlib import Path
# get the name
file_name = Path(this_file).stem
# make an output folder
output_folder = f'../results/{file_name}'

import numpy as np
from analysis import orientational_statistics,compute_nematic_order
from transforms import q_to_x, x_to_q
# %%
pathlist = []

pathlist.append('/Users/yeonsu/Downloads/entangled_config_data/6,7,8/2025-02-16_17_EntangledRelaxedPacking-N0200-AR0050-Scale1/x_relaxed.txt')
pathlist.append('/Users/yeonsu/Downloads/entangled_config_data/6,7,8/2025-02-16_17_EntangledRelaxedPacking-N0200-AR0100-Scale1/x_relaxed.txt')
pathlist.append('/Users/yeonsu/Downloads/entangled_config_data/6,7,8/2025-02-16_17_EntangledRelaxedPacking-N0200-AR0150-Scale1/x_relaxed.txt')
pathlist.append('/Users/yeonsu/Downloads/entangled_config_data/6,7,8/2025-02-16_17_EntangledRelaxedPacking-N0200-AR0300-Scale1/x_relaxed.txt')
pathlist.append('/Users/yeonsu/Downloads/entangled_config_data/6,7,8/2025-02-16_17_EntangledRelaxedPacking-N0200-AR0500-Scale1/x_relaxed.txt')
pathlist.append('/Users/yeonsu/Downloads/entangled_config_data/6,7,8/2025-02-18_18_EntangledRelaxedPacking-N0200-AR0010-Scale1/x_relaxed.txt')
pathlist.append('/Users/yeonsu/Downloads/entangled_config_data/6,7,8/2025-02-18_18_EntangledRelaxedPacking-N0200-AR0020-Scale1/x_relaxed.txt')
pathlist.append('/Users/yeonsu/Downloads/entangled_config_data/6,7,8/2025-02-18_18_EntangledRelaxedPacking-N0200-AR0075-Scale1/x_relaxed.txt')
pathlist.append('/Users/yeonsu/Downloads/entangled_config_data/6,7,8/2025-02-18_18_EntangledRelaxedPacking-N0200-AR0200-Scale1/x_relaxed.txt')
pathlist.append('/Users/yeonsu/Downloads/entangled_config_data/6,7,8/2025-11-05_21_EntangledRelaxedPacking-N0200-AR0010-Scale1/x_relaxed.txt')
pathlist.append('/Users/yeonsu/Downloads/entangled_config_data/6,7,8/2025-11-05_21_EntangledRelaxedPacking-N0200-AR1000-Scale1/x_relaxed.txt')

pathlist.append('/Users/yeonsu/Downloads/entangled_config_data/37,178,56/2025-11-05_21_EntangledRelaxedPacking-N0200-AR1000-Scale1/x_relaxed.txt')
pathlist.append('/Users/yeonsu/Downloads/entangled_config_data/37,178,56/2025-11-05_21_EntangledRelaxedPacking-N0200-AR0010-Scale1/x_relaxed.txt')
pathlist.append('/Users/yeonsu/Downloads/entangled_config_data/37,178,56/2025-02-18_18_EntangledRelaxedPacking-N0200-AR0200-Scale1/x_relaxed.txt')
pathlist.append('/Users/yeonsu/Downloads/entangled_config_data/37,178,56/2025-02-18_18_EntangledRelaxedPacking-N0200-AR0075-Scale1/x_relaxed.txt')
pathlist.append('/Users/yeonsu/Downloads/entangled_config_data/37,178,56/2025-02-18_18_EntangledRelaxedPacking-N0200-AR0020-Scale1/x_relaxed.txt')
pathlist.append('/Users/yeonsu/Downloads/entangled_config_data/37,178,56/2025-02-18_18_EntangledRelaxedPacking-N0200-AR0010-Scale1/x_relaxed.txt')
pathlist.append('/Users/yeonsu/Downloads/entangled_config_data/37,178,56/2025-02-16_17_EntangledRelaxedPacking-N0200-AR0500-Scale1/x_relaxed.txt')
pathlist.append('/Users/yeonsu/Downloads/entangled_config_data/37,178,56/2025-02-16_17_EntangledRelaxedPacking-N0200-AR0300-Scale1/x_relaxed.txt')
pathlist.append('/Users/yeonsu/Downloads/entangled_config_data/37,178,56/2025-02-16_17_EntangledRelaxedPacking-N0200-AR0150-Scale1/x_relaxed.txt')
pathlist.append('/Users/yeonsu/Downloads/entangled_config_data/37,178,56/2025-02-16_17_EntangledRelaxedPacking-N0200-AR0100-Scale1/x_relaxed.txt')
pathlist.append('/Users/yeonsu/Downloads/entangled_config_data/37,178,56/2025-02-16_17_EntangledRelaxedPacking-N0200-AR0050-Scale1/x_relaxed.txt')

pathlist.append('/Users/yeonsu/Downloads/entangled_config_data/919,461,568/2025-02-16_17_EntangledRelaxedPacking-N0200-AR0050-Scale1/x_relaxed.txt')
pathlist.append('/Users/yeonsu/Downloads/entangled_config_data/919,461,568/2025-02-16_17_EntangledRelaxedPacking-N0200-AR0100-Scale1/x_relaxed.txt')
pathlist.append('/Users/yeonsu/Downloads/entangled_config_data/919,461,568/2025-02-16_17_EntangledRelaxedPacking-N0200-AR0150-Scale1/x_relaxed.txt')
pathlist.append('/Users/yeonsu/Downloads/entangled_config_data/919,461,568/2025-02-16_17_EntangledRelaxedPacking-N0200-AR0300-Scale1/x_relaxed.txt')
pathlist.append('/Users/yeonsu/Downloads/entangled_config_data/919,461,568/2025-02-16_17_EntangledRelaxedPacking-N0200-AR0500-Scale1/x_relaxed.txt')
pathlist.append('/Users/yeonsu/Downloads/entangled_config_data/919,461,568/2025-02-18_18_EntangledRelaxedPacking-N0200-AR0010-Scale1/x_relaxed.txt')
pathlist.append('/Users/yeonsu/Downloads/entangled_config_data/919,461,568/2025-02-18_18_EntangledRelaxedPacking-N0200-AR0020-Scale1/x_relaxed.txt')
pathlist.append('/Users/yeonsu/Downloads/entangled_config_data/919,461,568/2025-02-18_18_EntangledRelaxedPacking-N0200-AR0075-Scale1/x_relaxed.txt')
pathlist.append('/Users/yeonsu/Downloads/entangled_config_data/919,461,568/2025-02-18_18_EntangledRelaxedPacking-N0200-AR0200-Scale1/x_relaxed.txt')
pathlist.append('/Users/yeonsu/Downloads/entangled_config_data/919,461,568/2025-11-05_21_EntangledRelaxedPacking-N0200-AR0010-Scale1/x_relaxed.txt')
pathlist.append('/Users/yeonsu/Downloads/entangled_config_data/919,461,568/2025-11-05_21_EntangledRelaxedPacking-N0200-AR1000-Scale1/x_relaxed.txt')



# pathlist.append("/Users/yeonsu/Dropbox (Harvard University)/Data/maximum-entanglement/3_1_2/3_1_2_separately.../2024-10-16_11_EntangledRelaxedPacking-N0500-AR0010-Scale1/q_relaxed.txt")
# pathlist.append("/Users/yeonsu/Dropbox (Harvard University)/Data/maximum-entanglement/3_1_2/3_1_2_separately.../2024-10-15_11_EntangledRelaxedPacking-N0500-AR0020-Scale1/q_relaxed.txt")
# pathlist.append("/Users/yeonsu/Dropbox (Harvard University)/Data/maximum-entanglement/3_1_2/3_1_2_separately.../2024-10-15_11_EntangledRelaxedPacking-N0500-AR0050-Scale1/q_relaxed.txt")
# pathlist.append("/Users/yeonsu/Dropbox (Harvard University)/Data/maximum-entanglement/3_1_2/3_1_2_separately.../2024-10-15_11_EntangledRelaxedPacking-N0500-AR0075-Scale1/q_relaxed.txt")
# pathlist.append("/Users/yeonsu/Dropbox (Harvard University)/Data/maximum-entanglement/3_1_2/3_1_2_separately.../2024-10-15_11_EntangledRelaxedPacking-N0500-AR0100-Scale1/q_relaxed.txt")
# pathlist.append("/Users/yeonsu/Dropbox (Harvard University)/Data/maximum-entanglement/3_1_2/3_1_2_separately.../2024-10-16_00_EntangledRelaxedPacking-N0500-AR0200-Scale1/q_relaxed.txt")
# pathlist.append("/Users/yeonsu/Dropbox (Harvard University)/Data/maximum-entanglement/3_1_2/3_1_2_separately.../2024-10-16_00_EntangledRelaxedPacking-N0500-AR0300-Scale1/q_relaxed.txt")
# pathlist.append("/Users/yeonsu/Dropbox (Harvard University)/Data/maximum-entanglement/3_1_2/3_1_2_separately.../2024-10-16_00_EntangledRelaxedPacking-N0500-AR0500-Scale1/q_relaxed.txt")

# pathlist.append("/Users/yeonsu/GitHub/entanglement-optimization/results/5,7,9/2024-10-20_23_EntangledRelaxedPacking-N0500-AR0500-Scale1/q_relaxed.txt")
# pathlist.append("/Users/yeonsu/GitHub/entanglement-optimization/results/5,7,9/2024-10-20_23_EntangledRelaxedPacking-N0500-AR0300-Scale1/q_relaxed.txt")
# pathlist.append("/Users/yeonsu/GitHub/entanglement-optimization/results/5,7,9/2024-10-20_23_EntangledRelaxedPacking-N0500-AR0200-Scale1/q_relaxed.txt")
# pathlist.append("/Users/yeonsu/GitHub/entanglement-optimization/results/5,7,9/2024-10-20_23_EntangledRelaxedPacking-N0500-AR0100-Scale1/q_relaxed.txt")
# pathlist.append("/Users/yeonsu/GitHub/entanglement-optimization/results/5,7,9/2024-10-20_23_EntangledRelaxedPacking-N0500-AR0075-Scale1/q_relaxed.txt")
# pathlist.append("/Users/yeonsu/GitHub/entanglement-optimization/results/5,7,9/2024-10-20_23_EntangledRelaxedPacking-N0500-AR0050-Scale1/q_relaxed.txt")
# pathlist.append("/Users/yeonsu/GitHub/entanglement-optimization/results/5,7,9/2024-10-20_23_EntangledRelaxedPacking-N0500-AR0020-Scale1/q_relaxed.txt")
# pathlist.append("/Users/yeonsu/GitHub/entanglement-optimization/results/5,7,9/2024-10-20_23_EntangledRelaxedPacking-N0500-AR0010-Scale1/q_relaxed.txt")

# pathlist.append("/Users/yeonsu/GitHub/entanglement-optimization/results/65,72,99/2024-10-21_02_EntangledRelaxedPacking-N0500-AR0010-Scale1/q_relaxed.txt")
# pathlist.append("/Users/yeonsu/GitHub/entanglement-optimization/results/65,72,99/2024-10-21_02_EntangledRelaxedPacking-N0500-AR0020-Scale1/q_relaxed.txt")
# pathlist.append("/Users/yeonsu/GitHub/entanglement-optimization/results/65,72,99/2024-10-21_02_EntangledRelaxedPacking-N0500-AR0050-Scale1/q_relaxed.txt")
# pathlist.append("/Users/yeonsu/GitHub/entanglement-optimization/results/65,72,99/2024-10-21_02_EntangledRelaxedPacking-N0500-AR0075-Scale1/q_relaxed.txt")
# pathlist.append("/Users/yeonsu/GitHub/entanglement-optimization/results/65,72,99/2024-10-21_02_EntangledRelaxedPacking-N0500-AR0100-Scale1/q_relaxed.txt")
# pathlist.append("/Users/yeonsu/GitHub/entanglement-optimization/results/65,72,99/2024-10-21_02_EntangledRelaxedPacking-N0500-AR0200-Scale1/q_relaxed.txt")
# pathlist.append("/Users/yeonsu/GitHub/entanglement-optimization/results/65,72,99/2024-10-21_02_EntangledRelaxedPacking-N0500-AR0300-Scale1/q_relaxed.txt")
# pathlist.append("/Users/yeonsu/GitHub/entanglement-optimization/results/65,72,99/2024-10-21_02_EntangledRelaxedPacking-N0500-AR0500-Scale1/q_relaxed.txt")

# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/20241210-1240_RUN_protocol_AR500_N200_randomkeys29,19,70/q_relaxed.txt')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/20241210-1240_RUN_protocol_AR300_N200_randomkeys29,19,70/q_relaxed.txt')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/20241210-1240_RUN_protocol_AR200_N200_randomkeys29,19,70/q_relaxed.txt')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/20241210-1240_RUN_protocol_AR150_N200_randomkeys29,19,70/q_relaxed.txt')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/20241210-1240_RUN_protocol_AR100_N200_randomkeys29,19,70/q_relaxed.txt')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/20241210-1240_RUN_protocol_AR50_N200_randomkeys29,19,70/q_relaxed.txt')



# pathlist.append('/Users/yeonsu/GitHub/entanglement-optimization/results/65,72,99/2024-10-21_02_EntangledRelaxedPacking-N0500-AR0100-Scale1/q_relaxed.txt')
# pathlist.append('/Users/yeonsu/GitHub/entanglement-optimization/results/65,72,99/2024-10-21_02_EntangledRelaxedPacking-N0500-AR0050-Scale1/q_relaxed.txt')
# pathlist.append('/Users/yeonsu/GitHub/entanglement-optimization/results/65,72,99/2024-10-21_02_EntangledRelaxedPacking-N0500-AR0010-Scale1/q_relaxed.txt')
# pathlist.append('/Users/yeonsu/GitHub/entanglement-optimization/results/65,72,99/2024-10-21_02_EntangledRelaxedPacking-N0500-AR0500-Scale1/q_relaxed.txt')
# pathlist.append('/Users/yeonsu/GitHub/entanglement-optimization/results/65,72,99/2024-10-21_02_EntangledRelaxedPacking-N0500-AR0300-Scale1/q_relaxed.txt')
# pathlist.append('/Users/yeonsu/GitHub/entanglement-optimization/results/65,72,99/2024-10-21_02_EntangledRelaxedPacking-N0500-AR0200-Scale1/q_relaxed.txt')
# pathlist.append('/Users/yeonsu/GitHub/entanglement-optimization/results/65,72,99/2024-10-21_02_EntangledRelaxedPacking-N0500-AR0075-Scale1/q_relaxed.txt')
# pathlist.append('/Users/yeonsu/GitHub/entanglement-optimization/results/65,72,99/2024-10-21_02_EntangledRelaxedPacking-N0500-AR0020-Scale1/q_relaxed.txt')





# pathlist.append('/Users/yeonsu/GitHub/entanglement-optimization/results/85,32,12/2024-10-24_14_EntangledRelaxedPacking-N0050-AR0500-Scale1/q_relaxed.txt')
# %%
# sort by AR
pathlist = sorted(pathlist,key=lambda x: float(x.split('-AR')[1].split('-')[0]))
# pathlist = sorted(pathlist,key=lambda x: float(x.split('AR')[1].split('_')[0]))

# copy to local

# pathlist.append("/Users/yeonsu/GitHub/entanglement-optimization/results/2024-10-16_10_EntangledRelaxedPacking-N0100-AR0010-Scale1/q_relaxed.txt")
# pathlist.append("/Users/yeonsu/GitHub/entanglement-optimization/results/2024-10-16_10_EntangledRelaxedPacking-N0100-AR0020-Scale1/q_relaxed.txt")
# pathlist.append("/Users/yeonsu/GitHub/entanglement-optimization/results/2024-10-16_10_EntangledRelaxedPacking-N0100-AR0050-Scale1/q_relaxed.txt")
# pathlist.append("/Users/yeonsu/GitHub/entanglement-optimization/results/2024-10-16_10_EntangledRelaxedPacking-N0100-AR0075-Scale1/q_relaxed.txt")
# pathlist.append("/Users/yeonsu/GitHub/entanglement-optimization/results/2024-10-16_10_EntangledRelaxedPacking-N0100-AR0100-Scale1/q_relaxed.txt")
# pathlist.append("/Users/yeonsu/GitHub/entanglement-optimization/results/2024-10-16_10_EntangledRelaxedPacking-N0100-AR0200-Scale1/q_relaxed.txt")
# pathlist.append("/Users/yeonsu/GitHub/entanglement-optimization/results/2024-10-16_10_EntangledRelaxedPacking-N0100-AR0300-Scale1/q_relaxed.txt")
# pathlist.append("/Users/yeonsu/GitHub/entanglement-optimization/results/2024-10-16_10_EntangledRelaxedPacking-N0100-AR0500-Scale1/q_relaxed.txt")
# %%

# %%
import re
def parse_pathname(pathname):
    dt_string = re.search(r'(\d{4}-\d{2}-\d{2}_\d{2})',pathname).group(1)
    # dt_string = re.search(r'(\d{8}-\d{4})',pathname).group(1)
    AR = float(re.search('AR(\d+)',pathname).group(1))
    num_rods = int(re.search('N(\d+)',pathname).group(1))
    random_keys = re.search('(\d+),(\d+),(\d+)',pathname).group(0)

    return dt_string, AR, num_rods,random_keys

_,_,num_rods,random_keys = parse_pathname(pathlist[0])
# %%
import os
common_folder = f"/Users/yeonsu/Harvard University Dropbox/Yeonsu Jung/Data/maximum-entanglement/{random_keys}"
if not os.path.exists(common_folder):
    os.makedirs(common_folder)

from pathlib import Path
for pth in pathlist:
    dt_string, AR, num_rods,random_keys = parse_pathname(pth)
    print(dt_string, AR, num_rods,random_keys)

    # Load x_relaxed.txt
    x = np.loadtxt(pth)
    filename = f"MaxEnt_{random_keys}_N{num_rods}-AR{int(AR):04d}-Scale1.txt"
    np.savetxt(f"{common_folder}/{filename}",x)

# %%
import matplotlib.pyplot as plt
from potentials import all_pairwise_angles,all_pairwise_distances,all_pairwise_skewness,total_effective_potential
from transforms import x_to_q
from potentials import create_pairs

def compute_nematic_order(q):
    q = np.reshape(q, (-1, 5))
    phi =   q[:,3]
    theta = q[:,4]

    u = np.array([np.sin(phi)*np.cos(theta), np.sin(phi)*np.sin(theta), np.cos(phi)]).T
    outer_products = np.einsum('ni,nj->nij', u, u)  # Shape (N, 3, 3)
    S = np.mean(outer_products, axis=0)  # Shape (3, 3)

    S = S - np.eye(3)/3
    Q = 1.5*S
    return Q
    # outer product

import re
AR_list = []
angle_list = []
num_contacts_list = []
distances_list = []
skewness_list = []
total_entanglement_list = []
total_nematic_order_list = []
for pth in pathlist:
    dt_string, AR, num_rods, random_keys = parse_pathname(pth)
    AR = float(re.search('AR(\d+)',pth).group(1))
    AR_list.append(AR)
    diameter = 1/AR
    # qq = np.load(pth)
    # qq_reshaped = qq.reshape(-1,num_rods,5)
    # q = qq_reshaped[-1]

    # Load x_relaxed.txt and convert to q
    x = np.loadtxt(pth)
    q = x_to_q(x)
    q_pairs = create_pairs(q.reshape(-1,5))
    distances = all_pairwise_distances(q_pairs)

    angles = all_pairwise_angles(q_pairs)
    angle_list.append(angles)
    
    num_contacts = np.count_nonzero(distances < diameter*1.1)
    num_contacts_list.append(num_contacts)
    distances_list.append(distances)

    skewness = all_pairwise_skewness(q_pairs)
    skewness_list.append(skewness)

    final_e = total_effective_potential(q)
    total_entanglement_list.append(final_e)

    S = compute_nematic_order(q)
    total_nematic_order_list.append(S)

AR_list = np.array(AR_list)
num_contacts_list = np.array(num_contacts_list)*2
total_entanglement_list = np.array(total_entanglement_list)
# %%
plt.figure(figsize=(2.5,2))
for i,S in enumerate(total_nematic_order_list):
    # get the first eigenvalue
    w,v = np.linalg.eig(S)
    w = np.sort(w)
    plt.scatter(AR_list[i],w[2])

plt.ylim([0,1])
plt.xlabel('Aspect Ratio, $\\alpha$')
plt.ylabel('Nematic order, $S$')
# plt.savefig(f'{common_folder}/nematic_order_{dt_string}_N{num_rods}.png',dpi=300, bbox_inches='tight')

# %%
plt.figure(figsize=(2.5,2))
ns = []
x_bins = np.linspace(0,2,100)
for skewness in skewness_list:
    all_skewnewss= np.concatenate(skewness)
    n,_,_ = plt.hist(np.abs(all_skewnewss - 0.5), bins=x_bins, density=True)
    ns.append(n)

    # plt.hist(skewness[0], bins=np.linspace(0.5-2,0.5+2,100), density=True)
    # plt.hist(skewness[1], bins=np.linspace(0.5-2,0.5+2,100), density=True)
    # for skewness in skewness_list[0]:
    #     plt.hist(skewness, bins=np.linspace(0.5-2,0.5+2,200), density=True)

# plt.xlim([0.5-2,0.5+2])
plt.xlabel('$a_i$')
plt.ylabel('Probability density, $P(a_i)$')
# plt.legend(np.array(AR_list).astype(int))
# plt.savefig(f'{common_folder}/skewness_histogram_{dt_string}_N{num_rods}.png',dpi=300, bbox_inches='tight')
# %%
plt.figure(figsize=(2.5,2))
popt_list = []
for n in ns:
    plt.plot(x_bins[:-1],n,'-')
    x_fit = x_bins[:-1]
    y_fit = n

    from scipy.optimize import curve_fit
    def func(x, a, b):
        return a * np.exp(-b * (x)**2)

    popt, pcov = curve_fit(func, x_fit, y_fit)
    # plt.plot(x_fit, func(x_fit, *popt), 'r-')
    popt_list.append(popt)

plt.xlabel('$\Delta_i$')
plt.ylabel('Probability density, $P(\Delta_i)$')
# plt.legend(np.array(AR_list).astype(int))
plt.savefig(f'{common_folder}/skewness_histogram_{dt_string}_N{num_rods}.png',dpi=300, bbox_inches='tight')
# %%
# b to sigma
bs = np.array(popt_list)[:,1]
sigmas = 1/np.sqrt(2*bs)

def power_law(x,a):
    return a*x**(-3/4)

from scipy.optimize import curve_fit
popt,pcov = curve_fit(power_law,AR_list,sigmas)
x_fit = np.linspace(0,1000,100)
y_fit = power_law(x_fit,*popt)

plt.figure(figsize=(2.5,2))
plt.loglog(AR_list,sigmas,'o-',label='Data')
plt.loglog(x_fit,y_fit,label=f'$y={popt[0]:.2f}x^{{-3/4}}$')
plt.xlabel('Aspect Ratio, $\\alpha$')
plt.ylabel('Skewness width, $\\sigma_a$')
plt.legend(fontsize=8)
plt.savefig(f'{common_folder}/skewness_width_{dt_string}_N{num_rods}.png',dpi=300, bbox_inches='tight')

# %%
plt.figure(figsize=(2.5,2))
for angles in angle_list:
    plt.hist(angles, bins=100, density=True)
plt.xlabel('Pairwise angle, $\\theta$')
plt.ylabel('Probability density, $P(\\theta)$')
# plt.legend(np.array(AR_list).astype(int))
# plt.savefig(f'{common_folder}/angle_histogram_{dt_string}_N{num_rods}.png',dpi=300, bbox_inches='tight')

# %%
plt.figure(figsize=(2.5,2))
ax=plt.gca()
for distances in distances_list:
    plt.hist(distances, bins=100, density=True, alpha=0.5)
plt.xlabel('Distance, $d$')
plt.ylabel('Probability density, $P(d)$')
ax.set_yscale('log')
ax.set_xscale('log')
plt.legend(np.array(AR_list).astype(int))
# plt.savefig(f'{common_folder}/distance_histogram_{dt_string}_N{num_rods}.pdf',bbox_inches='tight')
# plt.savefig(f'{common_folder}/distance_histogram_{dt_string}_N{num_rods}.png',dpi=300, bbox_inches='tight')
    
# %%
plt.figure(figsize=(2.5,2))
plt.plot(AR_list,num_contacts_list/500,'o-')
plt.xlabel('Aspect Ratio, $\\alpha$')
plt.ylabel('Avg. no. of contacts')
plt.savefig(f'{common_folder}/num_contacts_{dt_string}_N{num_rods}.png',dpi=300, bbox_inches='tight')

# %%
normalized = -total_entanglement_list/(num_rods*(num_rods-1)/2)
plt.figure(figsize=(2.5,2))
plt.plot(AR_list,normalized,'o-')
plt.xlabel('Aspect Ratio, $\\alpha$')
plt.ylabel(r'$e/n_p$')
plt.savefig(f'{common_folder}/total_entanglement_{dt_string}_N{num_rods}.png',dpi=300, bbox_inches='tight')

def exp_hill(x,a,b):
    return a*(1 - np.exp(-x/b))

popt,pcov=curve_fit(exp_hill,AR_list,normalized,p0=[1,100])
y_fit = exp_hill(x_fit,*popt)
plt.plot(x_fit,y_fit,label=f'$y={popt[0]:.2f}\exp(-x/{popt[1]:.2f})$')
plt.legend()
plt.savefig(f'{common_folder}/total_entanglement_{dt_string}_N{num_rods}_fit.png',dpi=300, bbox_inches='tight')
# %%

# %%



# %%
pth = pathlist[0]
x = np.loadtxt(pth)
q = x_to_q(x)
_,AR,_,_ = parse_pathname(pth)

import polyscope as ps
from visualizations import prep_for_polyscope

rod_diameter = 1/AR
ps.init()
ps.set_autoscale_structures(False)
ps.set_automatically_compute_scene_extents(False)
ps.set_ground_plane_mode("none")
# Create a camera view from parameters
# intrinsics = ps.CameraIntrinsics(fov_vertical_deg=60, aspect=2)
# extrinsics = ps.CameraExtrinsics(root=(2., 2., 2.), look_dir=(-1., -1.,-1.), up_dir=(0.,1.,0.))
# params = ps.CameraParameters(intrinsics, extrinsics)
# cam = ps.register_camera_view("cam", params)


a_list_of_curves = q_to_x(q).reshape(num_rods,-1,3)
# a_list_of_curves = node_list[_t].reshape(num_rods,-1,3)
nodes,edges,edge_colors = prep_for_polyscope(a_list_of_curves,num_rods)
min_z = np.min(nodes[:,2])
ps_curves = ps.register_curve_network("filaments",nodes,edges)
ps_curves.add_color_quantity("edge_colors",edge_colors,defined_on='edges',enabled=True)
ps_curves.set_radius(rod_diameter/2,relative=False)

ps.set_length_scale(2.)
sz = 2.
low = np.array((-sz, -sz, -sz))
high = np.array((sz, sz, sz))
ps.set_bounding_box(low, high)
ps.set_up_dir("z_up")
ps.screenshot(f'EntRel_{AR}.png',transparent_bg=False)

# %%
def foo(x,n):
    return x**n/(1+x**n)

popt,pcov=curve_fit(foo,AR_list,normalized,p0=[5])

x_fit = np.linspace(0,500,100)
y_fit = foo(x_fit,*popt)
plt.plot(x_fit,y_fit)
plt.plot(AR_list,normalized,'o-')



# %%

# %%
# -----------------------------
# Aggregate analysis across datasets (6,7,8), (37,178,56), (919,461,568)
# Produces errorbar plots (mean ± std across datasets) and fits the same models
# -----------------------------

import glob
from collections import defaultdict
from typing import Dict, List, Tuple

# Reuse parse_pathname defined above

def find_latest_files_for_dataset(random_keys: str,
                                  base_dir: str = '/Users/yeonsu/Downloads/entangled_config_data',
                                  expected_N: int = 200,
                                  file_name: str = 'x_relaxed.txt') -> List[str]:
    """Find one path per AR for a given dataset random_keys, selecting the latest by timestamp in path.

    Returns a list of absolute file paths (one per AR), sorted by AR ascending.
    """
    dataset_dir = f"{base_dir}/{random_keys}"
    candidates = glob.glob(f"{dataset_dir}/**/{file_name}", recursive=True)
    per_AR: Dict[float, Tuple[str, str]] = {}  # AR -> (dt, path)
    for p in candidates:
        try:
            dt_string, AR, N, rk = parse_pathname(p)
        except Exception:
            continue
        if N != expected_N:
            continue
        # keep latest dt per AR
        prev = per_AR.get(AR)
        if (prev is None) or (dt_string > prev[0]):
            per_AR[AR] = (dt_string, p)
    # sort by AR
    sorted_items = sorted(per_AR.items(), key=lambda kv: kv[0])
    return [p for _, (dt, p) in sorted_items]


def compute_dataset_metrics(paths: List[str]):
    """Compute per-AR metrics for a list of x_relaxed.txt paths.

    Returns:
    - ARs: array of AR values
    - S_max: largest eigenvalue of nematic order tensor
    - sigma: skewness width from Gaussian fit a*exp(-b x^2), sigma = 1/sqrt(2b)
    - avg_contacts: 2 * num_contacts / 500 (to mirror existing normalization)
    - ent_per_pair: -E / (n*(n-1)/2)
    - num_rods: inferred N
    """
    import re as _re
    from scipy.optimize import curve_fit as _curve_fit

    ARs: List[float] = []
    S_max: List[float] = []
    sigmas: List[float] = []
    avg_contacts: List[float] = []
    ent_per_pair: List[float] = []
    num_rods = None

    # histogram bins and fit function for skewness width
    x_bins = np.linspace(0, 2, 100)

    def _gauss0(x, a, b):
        return a * np.exp(-b * (x) ** 2)

    for pth in sorted(paths, key=lambda x: float(_re.search('AR(\d+)', x).group(1))):
        dt_string, AR, N, rk = parse_pathname(pth)
        if num_rods is None:
            num_rods = N
        # load and convert
        x = np.loadtxt(pth)
        q = x_to_q(x)
        q_pairs = create_pairs(q.reshape(-1, 5))

        # contacts
        distances = all_pairwise_distances(q_pairs)
        diameter = 1.0 / AR
        n_contacts = np.count_nonzero(distances < diameter * 1.1)

        # skewness width
        skewness = all_pairwise_skewness(q_pairs)
        all_sk = np.concatenate(skewness)
        deltas = np.abs(all_sk - 0.5)
        n_hist, _ = np.histogram(deltas, bins=x_bins, density=True)
        x_fit = x_bins[:-1]
        y_fit = n_hist
        # robust fit with fallbacks
        try:
            popt, _ = _curve_fit(_gauss0, x_fit, y_fit, maxfev=20000)
            b = popt[1]
            sigma = 1 / np.sqrt(2 * b) if b > 0 else np.nan
        except Exception:
            sigma = np.nan

        # energy per pair
        E = total_effective_potential(q)
        e_pp = -E / (N * (N - 1) / 2)

        # nematic order largest eigenvalue
        S = compute_nematic_order(q)
        w, v = np.linalg.eig(S)
        w = np.sort(w)
        Smax = float(w[2])

        # append
        ARs.append(float(AR))
        avg_contacts.append(2.0 * n_contacts / 500.0)
        sigmas.append(sigma)
        ent_per_pair.append(e_pp)
        S_max.append(Smax)

    return {
        'ARs': np.array(ARs, dtype=float),
        'S_max': np.array(S_max, dtype=float),
        'sigma': np.array(sigmas, dtype=float),
        'avg_contacts': np.array(avg_contacts, dtype=float),
        'ent_per_pair': np.array(ent_per_pair, dtype=float),
        'num_rods': num_rods if num_rods is not None else 0,
    }


def aggregate_metrics(datasets: List[dict]):
    """Intersect ARs across datasets and compute mean/std for each metric."""
    # intersect ARs
    common_ARs = set(datasets[0]['ARs'])
    for ds in datasets[1:]:
        common_ARs &= set(ds['ARs'])
    common_ARs = np.array(sorted(list(common_ARs)), dtype=float)

    def collect(metric: str):
        values = []
        for ds in datasets:
            # map AR->value for this dataset
            mapping = {float(a): float(v) for a, v in zip(ds['ARs'], ds[metric])}
            values.append(np.array([mapping[a] for a in common_ARs], dtype=float))
        values = np.stack(values, axis=0)  # (num_datasets, num_AR)
        return values.mean(axis=0), values.std(axis=0)

    means = {}
    stds = {}
    for metric in ['S_max', 'sigma', 'avg_contacts', 'ent_per_pair']:
        m, s = collect(metric)
        means[metric] = m
        stds[metric] = s

    return common_ARs, means, stds


def ensure_dir(p: str):
    if not os.path.exists(p):
        os.makedirs(p)


# Execute aggregation
datasets_keys = ['6,7,8', '37,178,56', '919,461,568']
base_dir = '/Users/yeonsu/Downloads/entangled_config_data'

all_ds = []
for rk in datasets_keys:
    paths = find_latest_files_for_dataset(rk, base_dir=base_dir, expected_N=200, file_name='x_relaxed.txt')
    if len(paths) == 0:
        print(f"Warning: no files found for dataset {rk} in {base_dir}")
    ds_metrics = compute_dataset_metrics(paths)
    all_ds.append(ds_metrics)

agg_ARs, agg_means, agg_stds = aggregate_metrics(all_ds)

# output folder
agg_folder = f"/Users/yeonsu/Harvard University Dropbox/Yeonsu Jung/Data/maximum-entanglement/aggregate_{'+'.join([rk.replace(',', '-') for rk in datasets_keys])}"
ensure_dir(agg_folder)

# Save CSV of aggregated stats
csv_path = f"{agg_folder}/aggregate_stats_N{all_ds[0]['num_rods']}.csv"
header = 'AR,S_max_mean,S_max_std,sigma_mean,sigma_std,contacts_mean,contacts_std,ent_mean,ent_std'
data_mat = np.column_stack([
    agg_ARs,
    agg_means['S_max'], agg_stds['S_max'],
    agg_means['sigma'], agg_stds['sigma'],
    agg_means['avg_contacts'], agg_stds['avg_contacts'],
    agg_means['ent_per_pair'], agg_stds['ent_per_pair'],
])
np.savetxt(csv_path, data_mat, delimiter=',', header=header, comments='')

# Plotting helpers
def _errorbar_plot(x, y, yerr, xlabel, ylabel, fname, logx=False, logy=False):
    plt.figure(figsize=(2.5, 2))
    plt.errorbar(
        x, y, yerr=yerr,
        fmt='o-',
        markersize=2,
        lw=0.8,
        elinewidth=1.0,
        capsize=3,
        capthick=1.0,
        markerfacecolor='white',
        markeredgewidth=0.5,
    )
    ax = plt.gca()
    if logx:
        ax.set_xscale('log')
    if logy:
        ax.set_yscale('log')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(f"{agg_folder}/{fname}", dpi=300, bbox_inches='tight')


# 1) Nematic order (no fit before, so errorbar only)
_errorbar_plot(
    agg_ARs, agg_means['S_max'], agg_stds['S_max'],
    'Aspect Ratio, $\\alpha$', 'Nematic order, $S$',
    f'nematic_order_aggregate_N{all_ds[0]["num_rods"]}.png',
)

# 2) Skewness width sigma with power-law fit y = a * x^{-3/4}
from scipy.optimize import curve_fit as _curve_fit

def power_law(x, a):
    return a * x ** (-3 / 4)

try:
    popt_sig, pcov_sig = _curve_fit(power_law, agg_ARs, agg_means['sigma'], p0=[agg_means['sigma'][0] if len(agg_means['sigma'])>0 else 1.0])
except Exception:
    popt_sig = [1.0]

x_fit = np.linspace(max(1e-3, agg_ARs.min()), max(agg_ARs.max(), 1.0), 200)
y_fit_sig = power_law(x_fit, *popt_sig)

plt.figure(figsize=(2.5, 2))
plt.errorbar(
    agg_ARs, agg_means['sigma'], yerr=agg_stds['sigma'],
    fmt='o',
    markersize=2,
    lw=0.8,
    elinewidth=1.0,
    capsize=3,
    capthick=1.0,
    markerfacecolor='white',
    markeredgewidth=0.5,
    label='Data (mean ± std)'
)
plt.plot(x_fit, y_fit_sig, '-', label=f'$y={popt_sig[0]:.2f}x^{{-3/4}}$')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Aspect Ratio, $\\alpha$')
plt.ylabel('Skewness width, $\\sigma_a$')
plt.legend(fontsize=8)
plt.savefig(f"{agg_folder}/skewness_width_aggregate_N{all_ds[0]['num_rods']}.png", dpi=300, bbox_inches='tight')

# 3) Avg. contacts (no explicit fit originally)
_errorbar_plot(
    agg_ARs, agg_means['avg_contacts'], agg_stds['avg_contacts'],
    'Aspect Ratio, $\\alpha$', 'Avg. no. of contacts',
    f'num_contacts_aggregate_N{all_ds[0]["num_rods"]}.png',
)

# 4) Total entanglement per pair with exponential hill fit a*(1-exp(-x/b))
def exp_hill(x, a, b):
    return a * (1 - np.exp(-x / b))

try:
    popt_ent, pcov_ent = _curve_fit(exp_hill, agg_ARs, agg_means['ent_per_pair'], p0=[1.0, 100.0])
except Exception:
    popt_ent = [np.nan, np.nan]

x_fit_ent = np.linspace(0, max(agg_ARs.max(), 1000.0), 300)
y_fit_ent = exp_hill(x_fit_ent, *popt_ent) if not np.isnan(popt_ent[0]) else np.full_like(x_fit_ent, np.nan)

plt.figure(figsize=(2.5, 2))
plt.errorbar(
    agg_ARs, agg_means['ent_per_pair'], yerr=agg_stds['ent_per_pair'],
    fmt='o',
    markersize=2,
    lw=0.8,
    elinewidth=1.0,
    capsize=3,
    capthick=1.0,
    markerfacecolor='white',
    markeredgewidth=0.5,
    label='Data (mean ± std)'
)
if not np.isnan(popt_ent[0]):
    plt.plot(x_fit_ent, y_fit_ent, '-', label=f'$y={popt_ent[0]:.2f}(1-e^{{-x/{popt_ent[1]:.2f}}})$')
plt.xlabel('Aspect Ratio, $\\alpha$')
plt.ylabel(r'$e/n_p$')
plt.legend(fontsize=8)
plt.savefig(f"{agg_folder}/total_entanglement_aggregate_N{all_ds[0]['num_rods']}.png", dpi=300, bbox_inches='tight')

# Optional: also fit the alternative sigmoidal model y = x^n / (1 + x^n) used at the end of the script
def foo(x, n):
    return x ** n / (1 + x ** n)

try:
    popt_alt, _ = _curve_fit(foo, agg_ARs, agg_means['ent_per_pair'], p0=[5.0], maxfev=20000)
    y_fit_alt = foo(x_fit_ent, *popt_alt)
    plt.figure(figsize=(2.5, 2))
    plt.errorbar(
        agg_ARs, agg_means['ent_per_pair'], yerr=agg_stds['ent_per_pair'],
        fmt='o',
        markersize=2,
        lw=0.8,
        elinewidth=1.0,
        capsize=3,
        capthick=1.0,
        markerfacecolor='white',
        markeredgewidth=0.5,
        label='Data (mean ± std)'
    )
    plt.plot(x_fit_ent, y_fit_alt, '-', label=f'$y=x^{{{popt_alt[0]:.2f}}}/(1+x^{{{popt_alt[0]:.2f}}})$')
    plt.xlabel('Aspect Ratio, $\\alpha$')
    plt.ylabel(r'$e/n_p$')
    plt.legend(fontsize=8)
    plt.savefig(f"{agg_folder}/total_entanglement_aggregate_altfit_N{all_ds[0]['num_rods']}.png", dpi=300, bbox_inches='tight')
except Exception:
    pass

# --- Log-log plot for aggregated total entanglement per pair ---
# Create a log-log errorbar plot (masking any non-positive values which can't be shown on log scale)
_ent_x = agg_ARs
_ent_y = agg_means['ent_per_pair']
_ent_err = agg_stds['ent_per_pair']
mask = (_ent_x > 0) & (_ent_y > 0)
if np.count_nonzero(mask) >= 2:
    x_log = _ent_x[mask]
    y_log = _ent_y[mask]
    err_log = _ent_err[mask]

    # Optional simple power-law fit y = c * x^k on masked data
    try:
        coeffs = np.polyfit(np.log(x_log), np.log(y_log), 1)  # log y = k log x + log c
        k, log_c = coeffs
        c = np.exp(log_c)
        x_fit_pw = np.linspace(x_log.min(), x_log.max(), 200)
        y_fit_pw = c * x_fit_pw**k
        fit_label = f'$y={c:.2e}x^{{{k:.2f}}}$'
    except Exception:
        x_fit_pw = None
        y_fit_pw = None
        fit_label = None

    plt.figure(figsize=(2.5,2))
    plt.errorbar(
        x_log, y_log, yerr=err_log,
        fmt='o',
        markersize=2,
        lw=0.8,
        elinewidth=1.0,
        capsize=3,
        capthick=1.0,
        markerfacecolor='white',
        markeredgewidth=0.5,
        label='Data (mean ± std)'
    )
    if x_fit_pw is not None:
        plt.plot(x_fit_pw, y_fit_pw, '-', label=fit_label)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Aspect Ratio, $\\alpha$')
    plt.ylabel(r'$e/n_p$')
    plt.legend(fontsize=8)
    plt.savefig(f"{agg_folder}/total_entanglement_aggregate_loglog_N{all_ds[0]['num_rods']}.png", dpi=300, bbox_inches='tight')
else:
    print("Insufficient positive data points for log-log entanglement plot.")

# --- Semilog-X plot (log x, linear y) for aggregated total entanglement per pair ---
if np.count_nonzero(_ent_x > 0) >= 2:
    x_pos = _ent_x[_ent_x > 0]
    y_pos = _ent_y[_ent_x > 0]
    err_pos = _ent_err[_ent_x > 0]

    plt.figure(figsize=(2.5,2))
    plt.errorbar(
        x_pos, y_pos, yerr=err_pos,
        fmt='o',
        markersize=2,
        lw=0.8,
        elinewidth=1.0,
        capsize=3,
        capthick=1.0,
        markerfacecolor='white',
        markeredgewidth=0.5,
        label='Data (mean ± std)'
    )
    if not np.isnan(popt_ent[0]):
        plt.plot(x_fit_ent, y_fit_ent, '-', label=f'$y={popt_ent[0]:.2f}(1-e^{{-x/{popt_ent[1]:.2f}}})$')
    plt.xscale('log')
    plt.xlabel('Aspect Ratio, $\\alpha$')
    plt.ylabel(r'$e/n_p$')
    plt.legend(fontsize=8)
    plt.savefig(f"{agg_folder}/total_entanglement_aggregate_semilogx_N{all_ds[0]['num_rods']}.png", dpi=300, bbox_inches='tight')

# --- Semilog-Y plot (linear x, log y) for aggregated total entanglement per pair ---
if np.count_nonzero(_ent_y > 0) >= 2:
    x_pos = _ent_x[_ent_y > 0]
    y_pos = _ent_y[_ent_y > 0]
    err_pos = _ent_err[_ent_y > 0]

    plt.figure(figsize=(2.5,2))
    plt.errorbar(
        x_pos, y_pos, yerr=err_pos,
        fmt='o',
        markersize=2,
        lw=0.8,
        elinewidth=1.0,
        capsize=3,
        capthick=1.0,
        markerfacecolor='white',
        markeredgewidth=0.5,
        label='Data (mean ± std)'
    )
    if not np.isnan(popt_ent[0]):
        plt.plot(x_fit_ent, y_fit_ent, '-', label=f'$y={popt_ent[0]:.2f}(1-e^{{-x/{popt_ent[1]:.2f}}})$')
    plt.yscale('log')
    plt.xlabel('Aspect Ratio, $\\alpha$')
    plt.ylabel(r'$e/n_p$')
    plt.legend(fontsize=8)
    plt.savefig(f"{agg_folder}/total_entanglement_aggregate_semilogy_N{all_ds[0]['num_rods']}.png", dpi=300, bbox_inches='tight')

