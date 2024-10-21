# %%
import numpy as np
from analysis import orientational_statistics,compute_nematic_order
from transforms import q_to_x
# %%
pathlist = []

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

pathlist.append("/Users/yeonsu/GitHub/entanglement-optimization/results/65,72,99/2024-10-21_02_EntangledRelaxedPacking-N0500-AR0010-Scale1/q_relaxed.txt")
pathlist.append("/Users/yeonsu/GitHub/entanglement-optimization/results/65,72,99/2024-10-21_02_EntangledRelaxedPacking-N0500-AR0020-Scale1/q_relaxed.txt")
pathlist.append("/Users/yeonsu/GitHub/entanglement-optimization/results/65,72,99/2024-10-21_02_EntangledRelaxedPacking-N0500-AR0050-Scale1/q_relaxed.txt")
pathlist.append("/Users/yeonsu/GitHub/entanglement-optimization/results/65,72,99/2024-10-21_02_EntangledRelaxedPacking-N0500-AR0075-Scale1/q_relaxed.txt")
pathlist.append("/Users/yeonsu/GitHub/entanglement-optimization/results/65,72,99/2024-10-21_02_EntangledRelaxedPacking-N0500-AR0100-Scale1/q_relaxed.txt")
pathlist.append("/Users/yeonsu/GitHub/entanglement-optimization/results/65,72,99/2024-10-21_02_EntangledRelaxedPacking-N0500-AR0200-Scale1/q_relaxed.txt")
pathlist.append("/Users/yeonsu/GitHub/entanglement-optimization/results/65,72,99/2024-10-21_02_EntangledRelaxedPacking-N0500-AR0300-Scale1/q_relaxed.txt")
pathlist.append("/Users/yeonsu/GitHub/entanglement-optimization/results/65,72,99/2024-10-21_02_EntangledRelaxedPacking-N0500-AR0500-Scale1/q_relaxed.txt")

# %%
# sort by AR
pathlist = sorted(pathlist,key=lambda x: float(x.split('-AR')[1].split('-')[0]))

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
    random_keys = re.search('(\d+),(\d+),(\d+)',pathname).group(0)

    return dt_string, AR, num_rods,random_keys

_,_,num_rods,random_keys = parse_pathname(pathlist[0])
# %%
import os
common_folder = f"/Users/yeonsu/Dropbox (Harvard University)/Data/maximum-entanglement/{random_keys}"
if not os.path.exists(common_folder):
    os.makedirs(common_folder)

for pth in pathlist:
    dt_string, AR, num_rods,random_keys = parse_pathname(pth)
    print(dt_string, AR, num_rods,random_keys)

    qf = np.loadtxt(pth)
    x = q_to_x(qf)
    filename = f"MaxEnt_{random_keys}_N{num_rods}-AR{int(AR):04d}-Scale1.txt"
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
    dt_string, AR, num_rods, random_keys = parse_pathname(pth)

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
    plt.hist(skewness, bins=np.linspace(0.5-2,0.5+2,200), density=True)
plt.xlim([0.5-2,0.5+2])
plt.axvline(0.5,linestyle='--',color='k')
plt.xlabel('$a_i$')
plt.ylabel('Probability density, $P(a_i)$')
# plt.legend(np.array(AR_list).astype(int))
plt.savefig(f'{common_folder}/skewness_histogram_{dt_string}_N{num_rods}.png',dpi=300, bbox_inches='tight')

# %%
plt.figure(figsize=(2.5,2))
for angles in angle_list:
    plt.hist(angles, bins=100, density=True)
plt.xlabel('Pairwise angle, $\\theta$')
plt.ylabel('Probability density, $P(\\theta)$')
# plt.legend(np.array(AR_list).astype(int))
plt.savefig(f'{common_folder}/angle_histogram_{dt_string}_N{num_rods}.png',dpi=300, bbox_inches='tight')
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
plt.savefig(f'{common_folder}/distance_histogram_{dt_string}_N{num_rods}.pdf',bbox_inches='tight')
# plt.savefig(f'{common_folder}/distance_histogram_{dt_string}_N{num_rods}.png',dpi=300, bbox_inches='tight')
    
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
# %%
pth = pathlist[7]
q = np.loadtxt(pth)

_,AR,_=parse_pathname(pth)

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
