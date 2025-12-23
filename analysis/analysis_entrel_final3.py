# %%
import sys
sys.path.append('../core')
sys.path.append('core')

# /Users/yeonsu/GitHub/filamentFields/filamentFields.cpython-312-darwin.so
sys.path.append('/Users/yeonsu/GitHub/filamentFields')

this_file = __file__
from pathlib import Path
# Script filename stem
file_name = Path(this_file).stem
# Use script directory to build results path inside repo: <repo_root>/results/<script_stem>
script_dir = Path(this_file).parent.resolve()
repo_root = script_dir.parent  # one level up from analysis/
output_folder = repo_root / 'results' / file_name
output_folder.mkdir(parents=True, exist_ok=True)
print(f"[analysis_entrel_final3] Saving copies of aggregate figures to: {output_folder}")

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
    plt.scatter(AR_list[i], w[2], facecolors='none', edgecolors='C0')

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

plt.xlim([0.5-2,0.5+2])
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
plt.savefig(f'{common_folder}/skewness_histogram_{dt_string}_N{num_rods}.svg', bbox_inches='tight')
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
plt.loglog(AR_list, sigmas, 'o-', label='Data', markerfacecolor='none')
plt.loglog(x_fit,y_fit,label=f'$y={popt[0]:.2f}x^{{-3/4}}$')
plt.xlabel('Aspect Ratio, $\\alpha$')
plt.ylabel('Skewness width, $\\sigma_a$')
plt.legend(fontsize=8)
plt.savefig(f'{common_folder}/skewness_width_{dt_string}_N{num_rods}.png',dpi=300, bbox_inches='tight')
plt.savefig(f'{common_folder}/skewness_width_{dt_string}_N{num_rods}.svg', bbox_inches='tight')

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
plt.plot(AR_list, num_contacts_list/500, 'o-', markerfacecolor='none')
plt.xlabel('Aspect Ratio, $\\alpha$')
plt.ylabel('Avg. no. of contacts')
plt.savefig(f'{common_folder}/num_contacts_{dt_string}_N{num_rods}.png',dpi=300, bbox_inches='tight')
plt.savefig(f'{common_folder}/num_contacts_{dt_string}_N{num_rods}.svg', bbox_inches='tight')

# %%
normalized = -total_entanglement_list/(num_rods*(num_rods-1)/2)
plt.figure(figsize=(2.5,2))
plt.plot(AR_list, normalized, 'o-', markerfacecolor='none')
plt.xlabel('Aspect Ratio, $\\alpha$')
plt.ylabel(r'$e/n_p$')
plt.savefig(f'{common_folder}/total_entanglement_{dt_string}_N{num_rods}.png',dpi=300, bbox_inches='tight')
plt.savefig(f'{common_folder}/total_entanglement_{dt_string}_N{num_rods}.svg', bbox_inches='tight')

def exp_hill(x,a,b):
    return a*(1 - np.exp(-x/b))

popt,pcov=curve_fit(exp_hill,AR_list,normalized,p0=[max(0.1,float(np.nanmax(normalized))),100.0],bounds=(0,np.inf))
y_fit = exp_hill(x_fit,*popt)
plt.plot(x_fit,y_fit,label=f'$a={popt[0]:.2f},\\ b={popt[1]:.1f}$')
plt.legend()
plt.savefig(f'{common_folder}/total_entanglement_{dt_string}_N{num_rods}_fit.png',dpi=300, bbox_inches='tight')
plt.savefig(f'{common_folder}/total_entanglement_{dt_string}_N{num_rods}_fit.svg', bbox_inches='tight')
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

    # store raw distributions per AR for later aggregation of PDFs
    dist_raw: List[np.ndarray] = []
    angle_raw: List[np.ndarray] = []
    skew_delta_raw: List[np.ndarray] = []

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

        # angles for PDF storage
        angles = all_pairwise_angles(q_pairs)

        # append
        ARs.append(float(AR))
        avg_contacts.append(2.0 * n_contacts / 500.0)
        sigmas.append(sigma)
        ent_per_pair.append(e_pp)
        S_max.append(Smax)
        # store distributions
        dist_raw.append(distances)
        angle_raw.append(angles)
        skew_delta_raw.append(deltas)

    return {
        'ARs': np.array(ARs, dtype=float),
        'S_max': np.array(S_max, dtype=float),
        'sigma': np.array(sigmas, dtype=float),
        'avg_contacts': np.array(avg_contacts, dtype=float),
        'ent_per_pair': np.array(ent_per_pair, dtype=float),
        'num_rods': num_rods if num_rods is not None else 0,
        'dist_raw': dist_raw,
        'angle_raw': angle_raw,
        'skew_delta_raw': skew_delta_raw,
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

# Build unified histograms (distance, angle, skewness delta) averaged across datasets for each AR
common_ARs_list = list(agg_ARs)

# Flatten raw arrays to determine global bin ranges
all_distances = []
all_angles = []
all_skew_deltas = []
for ds in all_ds:
    for d in ds['dist_raw']:
        all_distances.append(d)
    for a in ds['angle_raw']:
        all_angles.append(a)
    for s in ds['skew_delta_raw']:
        all_skew_deltas.append(s)
if len(all_distances) > 0:
    dist_min = max(1e-6, np.min([d.min() for d in all_distances]))
    dist_max = np.max([d.max() for d in all_distances])
else:
    dist_min, dist_max = 1e-6, 1
if len(all_angles) > 0:
    angle_min = 0.0
    angle_max = np.max([a.max() for a in all_angles])
else:
    angle_min, angle_max = 0, np.pi
# For skewness metric a in [0,1] (segment parameter), visualize Δ=|a-0.5| ∈ [0,0.5]
skew_min, skew_max = 0.0, 0.5

# Use logarithmic binning for distance PDF
dist_bins = np.logspace(np.log10(dist_min), np.log10(dist_max), 100)
angle_bins = np.linspace(angle_min, angle_max, 100)
skew_bins = np.linspace(skew_min, skew_max, 100)

# Per AR store mean and std histograms
dist_hist_mean = {}
dist_hist_std = {}
angle_hist_mean = {}
angle_hist_std = {}
skew_hist_mean = {}
skew_hist_std = {}

for AR in common_ARs_list:
    # collect hist arrays for each dataset for this AR
    dists_H = []
    angles_H = []
    skew_H = []
    for ds in all_ds:
        if AR in ds['ARs']:
            idx = list(ds['ARs']).index(AR)
            d_raw = ds['dist_raw'][idx]
            a_raw = ds['angle_raw'][idx]
            s_raw = ds['skew_delta_raw'][idx]
            d_hist,_ = np.histogram(d_raw, bins=dist_bins, density=True)
            a_hist,_ = np.histogram(a_raw, bins=angle_bins, density=True)
            s_hist,_ = np.histogram(s_raw, bins=skew_bins, density=True)
            dists_H.append(d_hist)
            angles_H.append(a_hist)
            skew_H.append(s_hist)
    if dists_H:
        dist_stack = np.vstack(dists_H)
        dist_hist_mean[AR] = dist_stack.mean(axis=0)
        dist_hist_std[AR] = dist_stack.std(axis=0)
    if angles_H:
        angle_stack = np.vstack(angles_H)
        angle_hist_mean[AR] = angle_stack.mean(axis=0)
        angle_hist_std[AR] = angle_stack.std(axis=0)
    if skew_H:
        skew_stack = np.vstack(skew_H)
        skew_hist_mean[AR] = skew_stack.mean(axis=0)
        skew_hist_std[AR] = skew_stack.std(axis=0)

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
        markerfacecolor='none',
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
    # Also export SVG
    try:
        stem = fname.rsplit('.', 1)[0]
        plt.savefig(f"{agg_folder}/{stem}.svg", bbox_inches='tight')
    except Exception:
        pass


# Separate plot: Skewness delta PDFs on log-log axes (per-AR mean ± std)
try:
    AR_sorted = sorted(common_ARs_list)
    default_colors = plt.rcParams['axes.prop_cycle'].by_key().get('color', ['C0','C1','C2','C3','C4','C5','C6','C7','C8','C9'])
    ar_color_map = {ar: default_colors[i % len(default_colors)] for i, ar in enumerate(AR_sorted)}

    centers_sk = 0.5 * (skew_bins[:-1] + skew_bins[1:])
    plt.figure(figsize=(2.5, 2))
    ax_sk = plt.gca()
    legend_lines_sk = []
    for AR in AR_sorted:
        if AR in skew_hist_mean:
            mean_v = skew_hist_mean[AR]
            std_v = skew_hist_std.get(AR, np.zeros_like(mean_v))
            color = ar_color_map[AR]
            ln, = ax_sk.plot(centers_sk, np.maximum(mean_v, 1e-16), color=color, lw=1, label=f'{int(AR):d}')
            ax_sk.fill_between(centers_sk, np.maximum(mean_v - std_v, 1e-16), np.maximum(mean_v + std_v, 1e-16), color=color, alpha=0.25, linewidth=0)
            legend_lines_sk.append(ln)
    ax_sk.set_xscale('log')
    ax_sk.set_yscale('log')
    # Determine positive lower bound
    x_lower = max(centers_sk[0], 1e-4)
    ax_sk.set_xlim([x_lower, 0.5])
    ax_sk.set_xlabel('$\\Delta_i = |a_i-0.5|$')
    ax_sk.set_ylabel('PDF')
    # Add reference slope lines y ~ x, y ~ x^{-2}, y ~ x^{-3} scaled at pivot
    try:
        # Aggregate a representative PDF (average over ARs)
        if legend_lines_sk:
            # Build average mean over ARs
            means_stack = []
            for AR in AR_sorted:
                if AR in skew_hist_mean:
                    means_stack.append(skew_hist_mean[AR])
            if means_stack:
                avg_mean = np.mean(np.vstack(means_stack), axis=0)
                # Choose pivot near middle of x range
                pivot_x = centers_sk[len(centers_sk)//2]
                pivot_y = np.interp(pivot_x, centers_sk, np.maximum(avg_mean,1e-16))
                ref_specs = [ (-1, 'x^{-1}'), (-3, 'x^{-3}'), (-4, 'x^{-4}') ]
                for p, lbl in ref_specs:
                    C = pivot_y / (pivot_x**p)
                    y_ref = C * centers_sk**p
                    ax_sk.plot(centers_sk, y_ref, linestyle='--', linewidth=0.8,
                               color='k', alpha=0.6, label=f'$\propto {lbl}$')
    except Exception as _eref:
        print(f"[warn] Could not add reference slopes: {_eref}")
    # Full legend with all ARs plus reference slopes
    if legend_lines_sk:
        handles, existing_labels = ax_sk.get_legend_handles_labels()
        ar_handles = [h for h,l in zip(handles, existing_labels) if not l.startswith('$\\propto')]
        ar_labels  = [l for l in existing_labels if not l.startswith('$\\propto')]
        slope_handles = [h for h,l in zip(handles, existing_labels) if l.startswith('$\\propto')]
        slope_labels  = [l for l in existing_labels if l.startswith('$\\propto')]
        # Order: all ARs then slope references
        ax_sk.legend(ar_handles + slope_handles, ar_labels + slope_labels, title='AR & refs', loc='lower left', frameon=False, fontsize=7, ncol=1)
    plt.tight_layout()
    ax_sk.set_ylim([1e-4,1e2])
    plt.savefig(f"{agg_folder}/skewness_delta_pdf_loglog_aggregate_N{all_ds[0]['num_rods']}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{agg_folder}/skewness_delta_pdf_loglog_aggregate_N{all_ds[0]['num_rods']}.svg", bbox_inches='tight')
except Exception as _e:
    print(f"[warn] Failed to generate log-log skewness delta PDF plot: {_e}")


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
    markerfacecolor='none',
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
plt.savefig(f"{agg_folder}/skewness_width_aggregate_N{all_ds[0]['num_rods']}.svg", bbox_inches='tight')

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
    popt_ent, pcov_ent = _curve_fit(
        exp_hill,
        agg_ARs,
        agg_means['ent_per_pair'],
        p0=[max(0.1, float(np.nanmax(agg_means['ent_per_pair']))), 100.0],
        bounds=(0, np.inf)
    )
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
    markerfacecolor='none',
    markeredgewidth=0.5,
    label='Data (mean ± std)'
)
if not np.isnan(popt_ent[0]):
    # plt.plot(x_fit_ent, y_fit_ent, '-', label=f'$a={popt_ent[0]:.2f},\\ b={popt_ent[1]:.1f}$')
    plt.plot(x_fit_ent, y_fit_ent, '-', label=f'$y={popt_ent[0]:.2f}(1 - e^{{-x/{popt_ent[1]:.1f}}})$')
plt.xlabel('Aspect Ratio, $\\alpha$')
plt.ylabel(r'$e/n_p$')
plt.legend(fontsize=8)
plt.savefig(f"{agg_folder}/total_entanglement_aggregate_N{all_ds[0]['num_rods']}.png", dpi=300, bbox_inches='tight')
plt.savefig(f"{agg_folder}/total_entanglement_aggregate_N{all_ds[0]['num_rods']}.svg", bbox_inches='tight')

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
        markerfacecolor='none',
        markeredgewidth=0.5,
        label='Data (mean ± std)'
    )
    plt.plot(x_fit_ent, y_fit_alt, '-', label=f'$y=x^{{{popt_alt[0]:.2f}}}/(1+x^{{{popt_alt[0]:.2f}}})$')
    plt.xlabel('Aspect Ratio, $\\alpha$')
    plt.ylabel(r'$e/n_p$')
    plt.legend(fontsize=8)
    plt.savefig(f"{agg_folder}/total_entanglement_aggregate_altfit_N{all_ds[0]['num_rods']}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{agg_folder}/total_entanglement_aggregate_altfit_N{all_ds[0]['num_rods']}.svg", bbox_inches='tight')
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
        markerfacecolor='none',
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
    plt.savefig(f"{agg_folder}/total_entanglement_aggregate_loglog_N{all_ds[0]['num_rods']}.svg", bbox_inches='tight')
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
        markerfacecolor='none',
        markeredgewidth=0.5,
        label='Data (mean ± std)'
    )
    if not np.isnan(popt_ent[0]):
        plt.plot(x_fit_ent, y_fit_ent, '-', label=f'$a={popt_ent[0]:.2f},\\ b={popt_ent[1]:.1f}$')
    plt.xscale('log')
    plt.xlabel('Aspect Ratio, $\\alpha$')
    plt.ylabel(r'$e/n_p$')
    plt.legend(fontsize=8)
    plt.savefig(f"{agg_folder}/total_entanglement_aggregate_semilogx_N{all_ds[0]['num_rods']}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{agg_folder}/total_entanglement_aggregate_semilogx_N{all_ds[0]['num_rods']}.svg", bbox_inches='tight')

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
        markerfacecolor='none',
        markeredgewidth=0.5,
        label='Data (mean ± std)'
    )
    if not np.isnan(popt_ent[0]):
        plt.plot(x_fit_ent, y_fit_ent, '-', label=f'$a={popt_ent[0]:.2f},\\ b={popt_ent[1]:.1f}$')
    plt.yscale('log')
    plt.xlabel('Aspect Ratio, $\\alpha$')
    plt.ylabel(r'$e/n_p$')
    plt.legend(fontsize=8)
    plt.savefig(f"{agg_folder}/total_entanglement_aggregate_semilogy_N{all_ds[0]['num_rods']}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{agg_folder}/total_entanglement_aggregate_semilogy_N{all_ds[0]['num_rods']}.svg", bbox_inches='tight')

    # ------------------------------------------------------
    # Combined 2x3 panel figure (A-F)
    # A: distance pdf
    # B: angle pdf
    # C: skewness PDF (delta)
    # D: skewness width vs AR (mean ± std already computed)
    # E: avg contacts vs AR (mean ± std)
    # F: entanglement per pair vs AR (mean ± std and exp hill fit)
    # ------------------------------------------------------

    import matplotlib as mpl
    # Use Matplotlib's default discrete color cycle for diverse AR colors
    AR_sorted = sorted(common_ARs_list)
    default_colors = plt.rcParams['axes.prop_cycle'].by_key().get('color', ['C0','C1','C2','C3','C4','C5','C6','C7','C8','C9'])
    ar_color_map = {ar: default_colors[i % len(default_colors)] for i, ar in enumerate(AR_sorted)}

    fig, axes = plt.subplots(2,3, figsize=(9,6))
    axA, axB, axC, axD, axE, axF = axes.flatten()

    # Panel A: distance PDFs (mean ± std shaded)
    legend_lines = []
    for AR in AR_sorted:
        if AR in dist_hist_mean:
            # Geometric centers for log-spaced bins
            centers = np.sqrt(dist_bins[:-1] * dist_bins[1:])
            mean_v = dist_hist_mean[AR]
            std_v = dist_hist_std.get(AR, np.zeros_like(mean_v))
            color = ar_color_map[AR]
            ln, = axA.plot(centers, mean_v, color=color, lw=1, label=f'{int(AR):d}')
            axA.fill_between(centers, np.maximum(mean_v-std_v, 1e-16), mean_v+std_v, color=color, alpha=0.25, linewidth=0)
            legend_lines.append(ln)
    axA.set_xscale('log')
    axA.set_yscale('log')
    axA.set_ylim([1e-5, 1e3])
    axA.set_xlabel('Distance, $d$')
    axA.set_ylabel('PDF')
    axA.text(0.02, 0.98, 'A', transform=axA.transAxes, fontsize=12, fontweight='bold', va='top', ha='left')

    # Panel B: angle PDFs (mean ± std shaded)
    for AR in AR_sorted:
        if AR in angle_hist_mean:
            centers = 0.5*(angle_bins[:-1] + angle_bins[1:])
            mean_v = angle_hist_mean[AR]
            std_v = angle_hist_std.get(AR, np.zeros_like(mean_v))
            color = ar_color_map[AR]
            axB.plot(centers, mean_v, color=color, lw=1)
            axB.fill_between(centers, np.maximum(mean_v-std_v, 0.0), mean_v+std_v, color=color, alpha=0.25, linewidth=0)
    axB.set_xlabel('Angle, $\\theta$')
    axB.set_ylabel('PDF')
    axB.text(0.02, 0.98, 'B', transform=axB.transAxes, fontsize=12, fontweight='bold', va='top', ha='left')
    # Isotropic reference for undirected axes (θ ∈ [0, π/2]): p(θ) = sin θ
    angle_centers = 0.5*(angle_bins[:-1] + angle_bins[1:])
    iso_pdf = np.sin(angle_centers)
    iso_ln, = axB.plot(angle_centers, iso_pdf, 'k--', lw=1.5, label='$p(\\theta)=\\sin(\\theta)$')
    axB.legend([iso_ln], [iso_ln.get_label()], loc='upper right', frameon=False, fontsize=8)

    # Panel C: skewness delta PDFs (mean ± std shaded)
    for AR in AR_sorted:
        if AR in skew_hist_mean:
            centers = 0.5*(skew_bins[:-1] + skew_bins[1:])
            mean_v = skew_hist_mean[AR]
            std_v = skew_hist_std.get(AR, np.zeros_like(mean_v))
            color = ar_color_map[AR]
            axC.plot(centers, mean_v, color=color, lw=1)
            axC.fill_between(centers, np.maximum(mean_v-std_v, 0.0), mean_v+std_v, color=color, alpha=0.25, linewidth=0)
    axC.set_xlabel('$\\Delta_i = |a_i-0.5|$')
    axC.set_xlim([0, 0.5])
    axC.set_ylabel('PDF')
    axC.text(0.02, 0.98, 'C', transform=axC.transAxes, fontsize=12, fontweight='bold', va='top', ha='left')

    # Panel D: skewness width vs AR
    axD.errorbar(agg_ARs, agg_means['sigma'], yerr=agg_stds['sigma'], fmt='o', markersize=3, lw=0.8, capsize=3, label='mean ± std')
    # Add power-law fit y = a * α^{-3/4}
    try:
        axD.plot(x_fit, y_fit_sig, '-', color='k', lw=1.2, label=f'fit $a={popt_sig[0]:.2f}\\,\\alpha^{{-3/4}}$')
    except Exception:
        pass
    axD.set_xscale('log')
    axD.set_yscale('log')
    axD.set_xlabel('Aspect Ratio, $\\alpha$')
    axD.set_ylabel('$\\sigma_a$')
    axD.text(0.02, 0.98, 'D', transform=axD.transAxes, fontsize=12, fontweight='bold', va='top', ha='left')
    axD.legend(fontsize=8, frameon=False, loc='lower left')

    # Panel E: avg contacts vs AR (connect points with a line; ensure x is sorted)
    order = np.argsort(agg_ARs)
    xE = agg_ARs[order]
    yE = agg_means['avg_contacts'][order]
    yEerr = agg_stds['avg_contacts'][order]
    axE.errorbar(xE, yE, yerr=yEerr, fmt='o-', markersize=3, lw=0.8, capsize=3)
    axE.set_xlabel('Aspect Ratio, $\\alpha$')
    axE.set_ylabel('Avg contacts')
    axE.text(0.02, 0.98, 'E', transform=axE.transAxes, fontsize=12, fontweight='bold', va='top', ha='left')

    # Panel F: ent per pair vs AR
    axF.errorbar(agg_ARs, agg_means['ent_per_pair'], yerr=agg_stds['ent_per_pair'], fmt='o', markersize=3, lw=0.8, capsize=3, label='Data')
    if not np.isnan(popt_ent[0]):
        axF.plot(x_fit_ent, y_fit_ent, '-', label=f'$a={popt_ent[0]:.2f},\\ b={popt_ent[1]:.1f}$')
    axF.set_xlabel('Aspect Ratio, $\\alpha$')
    axF.set_ylabel('$e/n_p$')
    # Panel F requested to be on log-log scale (both axes). Ensure positive values only are displayed.
    axF.set_xscale('log')
    axF.set_yscale('log')
    axF.text(0.02, 0.98, 'F', transform=axF.transAxes, fontsize=12, fontweight='bold', va='top', ha='left')
    axF.legend(fontsize=8)

    # No colorbar when using discrete default color cycle; rely on legend

    # External legend with all AR entries
    if legend_lines:
        labels = [ln.get_label() for ln in legend_lines]
        fig.legend(legend_lines, labels, title='AR', loc='center right', bbox_to_anchor=(1.02, 0.5), frameon=False, ncol=1)

    fig.tight_layout()
    combined_name = f"combined_panels_N{all_ds[0]['num_rods']}"
    fig.savefig(f"{agg_folder}/{combined_name}.png", dpi=300)
    fig.savefig(f"{agg_folder}/{combined_name}.pdf")
    fig.savefig(f"{agg_folder}/{combined_name}.svg")

    # Duplicate key aggregate outputs to local output folder using file_name
    import shutil
    def _copy_if_exists(fname: str):
        src = Path(agg_folder)/fname
        if src.exists():
            shutil.copy2(src, Path(output_folder)/fname)

    for f in [
        f'nematic_order_aggregate_N{all_ds[0]["num_rods"]}.png',
        f'nematic_order_aggregate_N{all_ds[0]["num_rods"]}.svg',
        f'skewness_width_aggregate_N{all_ds[0]["num_rods"]}.png',
        f'skewness_width_aggregate_N{all_ds[0]["num_rods"]}.svg',
        f'num_contacts_aggregate_N{all_ds[0]["num_rods"]}.png',
        f'num_contacts_aggregate_N{all_ds[0]["num_rods"]}.svg',
        f'total_entanglement_aggregate_N{all_ds[0]["num_rods"]}.png',
        f'total_entanglement_aggregate_N{all_ds[0]["num_rods"]}.svg',
        f'total_entanglement_aggregate_loglog_N{all_ds[0]["num_rods"]}.png',
        f'total_entanglement_aggregate_loglog_N{all_ds[0]["num_rods"]}.svg',
        f'total_entanglement_aggregate_semilogx_N{all_ds[0]["num_rods"]}.png',
        f'total_entanglement_aggregate_semilogx_N{all_ds[0]["num_rods"]}.svg',
        f'total_entanglement_aggregate_semilogy_N{all_ds[0]["num_rods"]}.png',
        f'total_entanglement_aggregate_semilogy_N{all_ds[0]["num_rods"]}.svg',
        f'total_entanglement_aggregate_altfit_N{all_ds[0]["num_rods"]}.png',
        f'total_entanglement_aggregate_altfit_N{all_ds[0]["num_rods"]}.svg',
        f'skewness_delta_pdf_loglog_aggregate_N{all_ds[0]["num_rods"]}.png',
        f'skewness_delta_pdf_loglog_aggregate_N{all_ds[0]["num_rods"]}.svg',
        f'{combined_name}.png',
        f'{combined_name}.svg'
    ]:
        _copy_if_exists(f)


