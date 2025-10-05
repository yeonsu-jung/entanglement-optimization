# %%
import numpy as np
old_312 = '/Users/yeonsu/GitHub/dismech-rods-main/data/MaxEnt/3_1_2/MaxEnt3,1,2-N500-AR0500-Scale1.txt'
new_312 = '/Users/yeonsu/GitHub/dismech-rods-main/data/MaxEnt2/3,1,2/MaxEnt3,1,2-N500-AR0500-Scale1.txt'


# %%
from matplotlib import pyplot as plt
from visualizations import plot_many_rods

# fig,ax=plt.subplots(1,2,figsize=(10,5),subplot_kw={'projection':'3d'})
# for i in range(500):
#     ax[0].plot(*old_312[i].reshape(-1,3).T)
#     ax[1].plot(*new_312[i].reshape(-1,3).T)

# %%
def parse_pathname(pathname):
    dt_string = re.search(r'(\d{4}-\d{2}-\d{2}_\d{2})',pathname).group(1)
    # dt_string = re.search(r'(\d{8}-\d{4})',pathname).group(1)
    AR = float(re.search('AR(\d+)',pathname).group(1))
    num_rods = int(re.search('N(\d+)',pathname).group(1))
    random_keys = re.search('(\d+),(\d+),(\d+)',pathname).group(0)

    return dt_string, AR, num_rods,random_keys

pathlist = []
pathlist.append(old_312)
pathlist.append(new_312)
# %%
from transforms import q_to_x, x_to_q
import matplotlib.pyplot as plt
from potentials import all_pairwise_angles,all_pairwise_distances,all_pairwise_skewness,total_effective_potential
from transforms import x_to_q
from potentials import create_pairs
import re

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

# old = np.loadtxt(old_312)
# new = np.loadtxt(new_312)

# q = np.loadtxt(pth)
# x = q_to_x(q)

# %%
AR_list = []
angle_list = []
num_contacts_list = []
distances_list = []
skewness_list = []
total_entanglement_list = []
total_nematic_order_list = []

AR = 500
diameter = 1/AR

distances_list = []
for pth in pathlist:
    x = np.loadtxt(pth)
    q = x_to_q(x)

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

    S = compute_nematic_order(q)
    total_nematic_order_list.append(S)
# %%
for d in distances_list:
    plt.hist(d.flatten(),bins=100)

# %%
# np.mean(distances_list[0]), np.mean(distances_list[1])
np.min(distances_list[0]), np.min(distances_list[1])
    
# %%
sk_1 = np.concatenate([np.abs(skewness_list[0][0] - 0.5),np.abs(skewness_list[0][1] - 0.5)])
sk_2 = np.concatenate([np.abs(skewness_list[1][0] - 0.5),np.abs(skewness_list[1][1] - 0.5)])
np.mean(sk_1), np.mean(sk_2)

# %%
# higher skewness,, somehow creates more entanglement

# %%


