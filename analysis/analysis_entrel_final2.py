# %%
import sys
sys.path.append('../core')
sys.path.append('core')

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
common_folder = f"/Users/yeonsu/Dropbox (Harvard University)/Data/maximum-entanglement/{random_keys}"
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
x_fit = np.linspace(0,500,100)
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
