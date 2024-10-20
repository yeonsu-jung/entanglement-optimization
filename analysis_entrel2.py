# %%
import numpy as np
from analysis import orientational_statistics,compute_nematic_order
from transforms import q_to_x
# %%
pathlist = []

pathlist.append("/Users/yeonsu/Dropbox (Harvard University)/Data/maximum-entanglement/2024-10-16_11_EntangledRelaxedPacking-N0500-AR0010-Scale1/q_relaxed.txt")
pathlist.append("/Users/yeonsu/Dropbox (Harvard University)/Data/maximum-entanglement/2024-10-15_11_EntangledRelaxedPacking-N0500-AR0020-Scale1/q_relaxed.txt")
pathlist.append("/Users/yeonsu/Dropbox (Harvard University)/Data/maximum-entanglement/2024-10-15_11_EntangledRelaxedPacking-N0500-AR0050-Scale1/q_relaxed.txt")
pathlist.append("/Users/yeonsu/Dropbox (Harvard University)/Data/maximum-entanglement/2024-10-15_11_EntangledRelaxedPacking-N0500-AR0075-Scale1/q_relaxed.txt")
pathlist.append("/Users/yeonsu/Dropbox (Harvard University)/Data/maximum-entanglement/2024-10-15_11_EntangledRelaxedPacking-N0500-AR0100-Scale1/q_relaxed.txt")
pathlist.append("/Users/yeonsu/Dropbox (Harvard University)/Data/maximum-entanglement/2024-10-16_00_EntangledRelaxedPacking-N0500-AR0200-Scale1/q_relaxed.txt")
pathlist.append("/Users/yeonsu/Dropbox (Harvard University)/Data/maximum-entanglement/2024-10-16_00_EntangledRelaxedPacking-N0500-AR0300-Scale1/q_relaxed.txt")
pathlist.append("/Users/yeonsu/Dropbox (Harvard University)/Data/maximum-entanglement/2024-10-16_00_EntangledRelaxedPacking-N0500-AR0500-Scale1/q_relaxed.txt")


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
    AR = float(re.search('AR(\d+)',pathname).group(1))
    num_rods = int(re.search('N(\d+)',pathname).group(1))

    return dt_string, AR, num_rods

_,_,num_rods=parse_pathname(pathlist[0])
# %%
import os
random_keys = "3_1_2"
common_folder = f"/Users/yeonsu/Dropbox (Harvard University)/Data/maximum-entanglement/{random_keys}"
if not os.path.exists(common_folder):
    os.makedirs(common_folder)

for pth in pathlist:
    dt_string, AR, num_rods = parse_pathname(pth)
    print(dt_string, AR, num_rods)
    qf = np.loadtxt(pth)
    x = q_to_x(qf)
    filename = f"MaxEnt_{random_keys}_N{num_rods}-AR{int(AR):04d}-Scake1.txt"
    np.savetxt(f"{common_folder}/{filename}",x)

# %%
import matplotlib.pyplot as plt
from potentials import all_pairwise_angles,all_pairwise_distances,all_pairwise_skewness,total_effective_potential
from transforms import x_to_q
from potentials import create_pairs

import re
AR_list = []
angle_list = []
num_contacts_list = []
distances_list = []
skewness_list = []
total_entanglement_list = []
for pth in pathlist:
    dt_string, AR, num_rods = parse_pathname(pth)

    AR = float(re.search('AR(\d+)',pth).group(1))
    AR_list.append(AR)
    diameter = 1/AR

    # qq = np.load(pth)
    # qq_reshaped = qq.reshape(-1,num_rods,5)
    # q = qq_reshaped[-1]
    q = np.loadtxt(pth)
    x = q_to_x(q)
    q_pairs = create_pairs(q.reshape(-1,5))
    distances = all_pairwise_distances(q_pairs)

    angles = all_pairwise_angles(q_pairs)
    angle_list.append(angles)
    
    num_contacts = np.count_nonzero(distances < diameter*1.05)
    num_contacts_list.append(num_contacts)
    distances_list.append(distances)

    skewness = all_pairwise_skewness(q_pairs)
    skewness_list.append(skewness)

    final_e = total_effective_potential(q)
    total_entanglement_list.append(final_e)

AR_list = np.array(AR_list)
num_contacts_list = np.array(num_contacts_list)*2
total_entanglement_list = np.array(total_entanglement_list)

# %%
plt.figure(figsize=(2.5,2))
for skewness in skewness_list:
    plt.hist(skewness, bins=np.linspace(0,1,100), density=True)
plt.xlim([0,1])
plt.xlabel('Skewness')
plt.ylabel('Probability Density')
# plt.legend(np.array(AR_list).astype(int))
plt.savefig(f'{common_folder}/skewness_histogram_{dt_string}_N{num_rods}.png',dpi=300, bbox_inches='tight')

# %%
plt.figure(figsize=(2.5,2))
for angles in angle_list:
    plt.hist(angles, bins=100, density=True)
plt.xlabel('Pairwise angle, $\\theta$')
plt.ylabel('Probability Density, $P(\\theta)$')
# plt.legend(np.array(AR_list).astype(int))
plt.savefig(f'{common_folder}/angle_histogram_{dt_string}_N{num_rods}.png',dpi=300, bbox_inches='tight')
# %%
plt.figure(figsize=(2.5,2))
for distances in distances_list:
    plt.hist(distances, bins=100, density=True)
plt.xlabel('Distance, $d$')
plt.ylabel('Probability Density, $P(d)$')
plt.legend(np.array(AR_list).astype(int))
plt.savefig(f'{common_folder}/distance_histogram_{dt_string}_N{num_rods}.png',dpi=300, bbox_inches='tight')



    
# %%
plt.figure(figsize=(2.5,2))
plt.plot(AR_list,num_contacts_list/500,'o-')
plt.xlabel('Aspect Ratio, $\\alpha$')
plt.ylabel('Number of contacts')
plt.savefig(f'{common_folder}/num_contacts_{dt_string}_N{num_rods}.png',dpi=300, bbox_inches='tight')

# %%
plt.figure(figsize=(2.5,2))
plt.plot(AR_list,-total_entanglement_list/(num_rods*(num_rods-1)/2),'o-')
plt.xlabel('Aspect Ratio, $\\alpha$')
plt.ylabel(r'$e/n_p$')
plt.savefig(f'{common_folder}/total_entanglement_{dt_string}_N{num_rods}.png',dpi=300, bbox_inches='tight')
