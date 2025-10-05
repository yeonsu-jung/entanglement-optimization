# %%
import numpy as np
from matplotlib import pyplot as plt
import re

# %%
pathlist = []
pathlist.append("/Users/yeonsu/GitHub/entanglement-optimization/results/65,72,99/2024-10-21_02_EntangledRelaxedPacking-N0500-AR0010-Scale1/q_relaxed.txt")
pathlist.append("/Users/yeonsu/GitHub/entanglement-optimization/results/65,72,99/2024-10-21_02_EntangledRelaxedPacking-N0500-AR0020-Scale1/q_relaxed.txt")
pathlist.append("/Users/yeonsu/GitHub/entanglement-optimization/results/65,72,99/2024-10-21_02_EntangledRelaxedPacking-N0500-AR0050-Scale1/q_relaxed.txt")
pathlist.append("/Users/yeonsu/GitHub/entanglement-optimization/results/65,72,99/2024-10-21_02_EntangledRelaxedPacking-N0500-AR0075-Scale1/q_relaxed.txt")
pathlist.append("/Users/yeonsu/GitHub/entanglement-optimization/results/65,72,99/2024-10-21_02_EntangledRelaxedPacking-N0500-AR0100-Scale1/q_relaxed.txt")
pathlist.append("/Users/yeonsu/GitHub/entanglement-optimization/results/65,72,99/2024-10-21_02_EntangledRelaxedPacking-N0500-AR0200-Scale1/q_relaxed.txt")
pathlist.append("/Users/yeonsu/GitHub/entanglement-optimization/results/65,72,99/2024-10-21_02_EntangledRelaxedPacking-N0500-AR0300-Scale1/q_relaxed.txt")
pathlist.append("/Users/yeonsu/GitHub/entanglement-optimization/results/65,72,99/2024-10-21_02_EntangledRelaxedPacking-N0500-AR0500-Scale1/q_relaxed.txt")

# %%





# %%
# x = np.loadtxt(pathlist[2])

# from transforms import cart2sph

def orientational_statistics(_x):
    _u = _x[:,:3] - _x[:,3:]
    r,theta,phi = cart2sph(_u)
    return r,theta,phi

def compute_nematic_order(_x):
    # Calculate the direction vectors between two sets of points
    _u = _x[:, :3] - _x[:, 3:]
    norms = np.linalg.norm(_u, axis=1, keepdims=True)
    _u = np.divide(_u, norms, where=(norms != 0))  # Avoid division by zero
    outer_products = np.einsum('ni,nj->nij', _u, _u)  # Shape (N, 3, 3)
    S = np.mean(outer_products, axis=0)  # Shape (3, 3)
    Q = (3 * S - np.eye(3)) / 2
    eigvals = np.linalg.eigvals(Q)
    return eigvals


# Q_evals = compute_nematic_order(x.reshape(-1,5))
# r,theta,phi=orientational_statistics(x)
# %%
AR_list = []
nematic_order_list = []
for pth in pathlist:
    x = np.loadtxt(pth)

    AR = float(re.search('AR(\d+)',pth).group(1))
    AR_list.append(AR)

    Q_evals = compute_nematic_order(x)
    r,theta,phi = orientational_statistics(x)
    print(Q_evals)

    nematic_order_list.append(Q_evals[0])
    
# %%
plt.figure(figsize=(2.5,2))
plt.plot(AR_list,nematic_order_list,'o-')
plt.xlabel(r'Aspect Ratio, $\alpha$')
plt.ylabel(r'Nematic Order Parameter, $S$')
plt.savefig('figures/nematic_order.png',dpi=300, bbox_inches='tight')

# %%
for pth in pathlist:
    x = np.loadtxt(pth)
    print(x.shape)
# %%
for pth in pathlist:
    x = np.loadtxt(pth)
    r,theta,phi=orientational_statistics(x)
    theta = theta % np.pi/2
    plt.hist(theta, bins=100, density=True)

# %%
fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
_u = x[:,:3] - x[:,3:]
norms = np.linalg.norm(_u, axis=1, keepdims=True)
_u = np.divide(_u, norms, where=(norms != 0))  # Avoid division by zero

ax.scatter(_u[:,0],_u[:,1],_u[:,2],s=1)
plt.show()
# %%
from potentials import create_pairs
from transforms import x_to_q

q = x_to_q(x)
q_pairs = create_pairs(q)

# %%
from potentials import all_pairwise_angles,all_pairwise_distances,all_pairwise_skewness,total_effective_potential



# %%
import re
AR_list = []
angle_list = []
num_contacts_list = []
distances_list = []
skewness_list = []
total_entanglement_list = []
for pth in pathlist:
    AR = float(re.search('AR(\d+)',pth).group(1))
    AR_list.append(AR)
    diameter = 1/AR

    x = np.loadtxt(pth)
    q = x_to_q(x)
    q_pairs = create_pairs(q)
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
for angles in angle_list:
    plt.hist(angles, bins=100, density=True)
plt.xlabel('Pairwise angle, $\\theta$')
plt.ylabel('Probability Density, $P(\\theta)$')
# plt.legend(np.array(AR_list).astype(int))
plt.savefig('figures/angle_histogram.png',dpi=300, bbox_inches='tight')
# %%
plt.figure(figsize=(2.5,2))
for distances in distances_list:
    plt.hist(distances, bins=100, density=True)
plt.xlabel('Distance, $d$')
plt.ylabel('Probability Density, $P(d)$')
plt.legend(np.array(AR_list).astype(int))
plt.savefig('figures/distance_histogram.png',dpi=300, bbox_inches='tight')


# %%
plt.figure(figsize=(2.5,2))
for skewness in skewness_list:
    plt.hist(skewness, bins=100, density=True)
plt.xlabel('Skewness')
plt.ylabel('Probability Density')
plt.legend(np.array(AR_list).astype(int))
plt.savefig('figures/skewness_histogram.png',dpi=300, bbox_inches='tight')
    
# %%
plt.figure(figsize=(2.5,2))
plt.plot(AR_list,num_contacts_list/500,'o-')
plt.xlabel('Aspect Ratio, $\\alpha$')
plt.ylabel('Number of contacts')
plt.savefig('figures/num_contacts.png',dpi=300, bbox_inches='tight')

# %%
plt.figure(figsize=(2.5,2))
plt.plot(AR_list,-total_entanglement_list/(500*499/2),'o-')
plt.xlabel('Aspect Ratio, $\\alpha$')
plt.ylabel(r'$e/n_p$')
plt.savefig('figures/total_entanglement.png',dpi=300, bbox_inches='tight')


# %%
pathlist = []
pathlist.append("/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/RandomInitialKick/20241013-0158_RUN_RandomInitialKick0001_N0500-AR0050")
pathlist.append("/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/RandomInitialKick/20241013-0202_RUN_RandomInitialKick0001_N0500-AR0075")
pathlist.append("/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/RandomInitialKick/20241013-0202_RUN_RandomInitialKick0001_N0500-AR0100")
pathlist.append("/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/RandomInitialKick/20241013-0202_RUN_RandomInitialKick0001_N0500-AR0200")
pathlist.append("/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/RandomInitialKick/20241013-0202_RUN_RandomInitialKick0001_N0500-AR0300")
pathlist.append("/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/RandomInitialKick/20241013-0202_RUN_RandomInitialKick0001_N0500-AR0500")
# %%
def find_csv_file(folder_path):
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
    return pth

from pathlib import Path
for pth in pathlist:
    folder_path = Path(pth)
    pth = find_csv_file(folder_path)
    print(pth)
# %%
from data_io import import_all_log
time_line, node_list, force_list, contact_list, box_contact_list = import_all_log(pth,max_rows=10000000)

# %%
node_list[0].shape