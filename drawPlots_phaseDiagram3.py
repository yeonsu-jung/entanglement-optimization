# %%
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import glob
import re
import pickle
import networkx as nx
output_dir = '/Users/yeonsu/Dropbox (Harvard University)/Data/PrunedData/rod-sim-pnas-revision/entanglement'
# %%
# colors = np.array([
#     [76, 153, 204],   # light blue
#     [204, 76, 153],   # pinkish red
#     [76, 204, 153],   # mint green
#     [153, 204, 76],   # light olive green
#     [204, 153, 76],   # goldenrod
#     [153, 76, 204],   # medium purple
#     [204, 76, 102],   # crimson
#     [76, 204, 204],   # cyan
#     [204, 204, 76],   # sunflower yellow
#     [102, 76, 204]    # indigo
#     ])/255

# # %%
# import polyscope as ps
# import numpy as np

# ps.init()
# ps.set_SSAA_factor(3)
# ps.set_ground_plane_mode('shadow_only')

# # xy-plane
# vertices_xy = np.array([
#     [0, 0, 0],
#     [1, 0, 0],
#     [1, 1, 0],
#     [0, 1, 0]
# ])
# faces_xy = np.array([
#     [0, 1, 2],
#     [0, 2, 3]
# ])

# # yz-plane
# vertices_yz = np.array([
#     [0, 0, 0],
#     [0, 1, 0],
#     [0, 1, 1],
#     [0, 0, 1]
# ])
# faces_yz = np.array([
#     [0, 1, 2],
#     [0, 2, 3]
# ])

# # zx-plane
# vertices_zx = np.array([
#     [0, 0, 0],
#     [0, 0, 1],
#     [1, 0, 1],
#     [1, 0, 0]
# ])
# faces_zx = np.array([
#     [0, 1, 2],
#     [0, 2, 3]
# ])

# ps_xy = ps.register_surface_mesh("xy-plane", vertices_xy, faces_xy, smooth_shade=False, enabled=True,transparency=0.3)
# ps_yz = ps.register_surface_mesh("yz-plane", vertices_yz, faces_yz, smooth_shade=False, enabled=True,transparency=0.3)
# ps_zx = ps.register_surface_mesh("zx-plane", vertices_zx, faces_zx, smooth_shade=False, enabled=True,transparency=0.3)

# ps_xy.set_color((0.8, 0.8, 0.8))
# ps_yz.set_color((0.8, 0.8, 0.8))
# ps_zx.set_color((0.8, 0.8, 0.8))

# # Add a random point within the box (0,0,0) to (1,1,1)
# random_point = np.random.rand(10,3)
# ps_point = ps.register_point_cloud("random_point", random_point)
# ps_point.set_radius(0.01, relative=False)


# # Project the random point onto the xy-plane (z=0)
# shadow_point_xy = random_point.copy()
# shadow_point_xy[:,2] = 0
# ps_shadow_xy = ps.register_point_cloud("shadow_point_xy", shadow_point_xy, color=(1, 0, 0),enabled=False)

# ps.set_up_dir("z_up")

# # ps.show()



# # %%
# from data_io import import_all_log, parse_path_string
# from scipy.optimize import curve_fit
# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "Helvetica"
# })

# # %%
# class data_container:
#     def __init__(self,dataphat,start_row=0,max_rows=100000,skip_rows=1):
#         self.path = Path(dataphat)
#         out = parse_path_string(self.path)
#         # self.file_id,self.surfix,self.num_rods,self.AR,self.datetime_string
#         self.file_id = out[0]
#         self.surfix = out[1]
#         self.num_rods = out[2]
#         self.AR = out[3]
#         self.datetime_string = out[4]
#         self.time_line, self.node_list, self.contact_list = import_all_log(self.path,start_row=start_row,max_rows=max_rows,skip_rows=skip_rows)


# root_dir = Path('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PhaseDiagramStudy-Modelo1')


# pathlist = []
# for pth in root_dir.rglob('*.csv'):
#     if str(pth.stem).endswith('lastFrame'):
#         continue
#     pathlist.append(pth.parent)


# # %%        
# max_rows = 1000000
# data_container_list = []

# # (g, freq)
# data_container_dict = {}

# for pth in pathlist:
#     # find csv file
#     data_path = None
#     for file in Path(pth).rglob('*.csv'):
#         if str(file.stem).endswith('lastFrame'):
#             continue
        
#         # data_container_list.append(data_container(file,max_rows=max_rows))
#         pth = str(pth)
#         exp_id = pth.split('/')[-1]
#         search_result = re.search(r'N(\d+)[-_]AR(\d+)_g(\d+(\.\d+)?)_freq(\d+(\.\d+)?)', exp_id)
        
#         N = int(search_result.group(1))
#         AR = int(search_result.group(2))
#         g = float(search_result.group(3))
#         freq = float(search_result.group(5))
        
#         data_container_list.append(data_container(file,start_row=480,max_rows=10000))
        
#         print(N,AR,g,freq)
        
        
# # %%








# # %%
# i_ = 2
# data_entry = data_container_list[i_]

# num_rods = data_entry.num_rods
# tt = data_entry.time_line
# vv = data_entry.contact_list[i_].reshape(-1,18)

# contact_ij = vv[:,4:6].astype(int)


# # contact_ij_next_frame = next_force_all_info[:,4:6].astype(int)            
# curr_nodes = data_entry.node_list[i_]
# graph = nx.Graph()
# graph.add_nodes_from(range(len(curr_nodes)))
# graph.add_edges_from(contact_ij)
# clusters = list(nx.connected_components(graph))
# len(clusters)

# # largest clusters
# largest_cluster = max(clusters,key=len)
# f = len(largest_cluster)/num_rods

# print(f'f = {f}')
# print(f'AR = {data_entry.AR}')
# print(f'{len(contact_ij)}')
# # %%
# nodes_in_shape = curr_nodes.reshape(-1,10,3)

# fig,ax=plt.subplots(subplot_kw={'projection': '3d'})
# for rr in nodes_in_shape:
#     ax.plot(rr[:,0],rr[:,1],rr[:,2],alpha=0.2)
# # # %%
# # fig,ax=plt.subplots(subplot_kw={'projection': '3d'})
# for i_ in largest_cluster:
#     ax.plot(nodes_in_shape[i_][:,0],nodes_in_shape[i_][:,1],nodes_in_shape[i_][:,2],'k')
# # %%


# # %%
# single_column_size = (1.8,1.5)
# import pandas as pd
# # initialize df
# df = pd.DataFrame(columns=['static','dynamic','f','AR','sigma_over_mu'])

# ARs = np.array([25,50,75,100,125,200,300])
# cve = np.array([0.9483138 , 1.15187444, 1.21259957, 1.24966556, 1.25542284,1.34128365, 1.41815353])
# cve_dict = {25: 0.9483138 , 50: 1.15187444, 75: 1.21259957, 100: 1.24966556, 125: 1.25542284, 150: 1.3124823, 200: 1.34128365, 300: 1.41815353}
# mu_over_sigma_dict = {'25': 1/0.95, '100': 1/1.25, '300': 1/1.42}

# fig,ax=plt.subplots(subplot_kw={'projection': '3d'},figsize=np.array(single_column_size)*1.5)
# data_table = []

# entangled_data = []
# untangled_data = []
# for dc in data_container_list:
#     num_rods = dc.num_rods
#     contact_ij = dc.contact_list[-1].reshape(-1,18)[:,4:6].astype(int)
#     curr_nodes = dc.node_list[-1]
    
#     contact_graph = nx.Graph()
#     contact_graph.add_nodes_from(range(len(curr_nodes)))
#     contact_graph.add_edges_from(contact_ij)
#     clusters = list(nx.connected_components(contact_graph))
#     # largest clusters
#     largest_cluster = max(clusters,key=len)
#     f = len(largest_cluster)/num_rods
    
#     exp_id = str(dc.path).split('/')[-2]
#     search_result = re.search(r'N(\d+)[-_]AR(\d+)_g(\d+(\.\d+)?)_freq(\d+(\.\d+)?)', exp_id)        
#     N = int(search_result.group(1))
#     AR = int(search_result.group(2))
#     g = float(search_result.group(3))
#     freq = float(search_result.group(5))
    
#     # if freq == 100:
#     #     continue
    
#     # if AR == 150:
#     #     AR = 125
    
#     sigma_over_mu = cve_dict[AR]
#     mu_over_sigma = 1/sigma_over_mu
    
#     static = g/0.5
#     dynamic = 4*np.pi*freq**2*0.001/0.5
#     # data_table.append(np.array([static,dynamic,f,AR,sigma_over_mu]))
#     df.loc[len(df)] = [static,dynamic,f,AR,sigma_over_mu]
    
#     # ax.scatter(static,dynamic,f)
#     # foots
#     # ax.plot([static,static],[dynamic,dynamic],[0,f],'k--',linewidth=0.5)
    
#     if f > 0.9:
#         ax.scatter(static,mu_over_sigma,dynamic,color='g')
#         entangled_data.append([static,mu_over_sigma,dynamic])
#     else:
#         ax.scatter(static,mu_over_sigma,dynamic,color='k',alpha=0.5,marker='x')
#         untangled_data.append([static,mu_over_sigma,dynamic])


# ax.set_xlabel(r'$F/(\rho_s g d^2 l)$',labelpad=-3)
# ax.set_ylabel(r'$\mu/\sigma$',labelpad=-3)
# ax.set_zlabel(r'$a/g$',labelpad=-8, rotation=45)

# # ax.set_xlim([0,8])
# # ax.set_ylim([0,1.6])
# # ax.set_zlim([0,3])
# ax.invert_yaxis()
# ax.zaxis.loc='top'

# # ax.zaxis.labelpad=-6 # <- change the value here

# ax.tick_params(axis='x', pad=-3)
# ax.tick_params(axis='y', pad=-3)
# ax.tick_params(axis='z', pad=-3)
# # plt.zticks(rotation=90)

# # plt.tight_layout()
# output_dir = '/Users/yeonsu/Dropbox (Harvard University)/Data/PrunedData/rod-sim-pnas-revision/entanglement'
# # plt.savefig(f'{output_dir}/entanglement_vs_a-g-cv-3d-with-data2.png',dpi=300)
# # plt.savefig(f'{output_dir}/entanglement_vs_a-g-cv-3d-with-data2.pdf',bbox_inches='tight')

# # %%


# # %%    
# # ax.set_xlabel(r'$F/(\rho_s g d^2 l)$')
# # ax.set_ylabel(r'$a/g$')
# # ax.set_zlabel(r'$f$')
    
    
# # %%
# data_container_list[2].AR
# data_container_list[2].path.parent.stem
    
    
#     # for i_ in range(len(_list)):
#     #     dta = _list[i_][0]
#     #     f = _list[i_][1]
        
#     #     if f > 0.9:
#     #         ax.scatter(1, mu_over_sigma_dict[k], dta, color='b', s=20)
#     #     else:
#     #         ax.scatter(1, mu_over_sigma_dict[k], dta, color='k', alpha = 0.1)
# # %%
# # plot f for static = 1 and AR = 100
# default_color_list = plt.rcParams['axes.prop_cycle'].by_key()['color']
# single_column_size = (1.5,1.3)
# fig,ax=plt.subplots(figsize=single_column_size)
# #fontsize
# plt.rcParams.update({'font.size': 8})
# df_ = df[(df['static'] == 1) & (df['AR'] == 25)]
# ax.plot(df_['dynamic'],df_['f'],'o-',label=r'$\alpha=25, \tilde{F} = 1$',markersize=3,color=default_color_list[0],alpha=0.5)

# df_ = df[(df['static'] == 1) & (df['AR'] == 100)]
# xx = df_['dynamic'].to_numpy()
# i_sort = np.argsort(xx)
# xx = xx[i_sort]
# yy = df_['f'].to_numpy()
# yy = yy[i_sort]

# ax.plot(xx,yy,'o-',label=r'$\alpha=100, \tilde{F} = 1$',markersize=3,color=default_color_list[1],alpha=0.5)

# df_ = df[(df['static'] == 1) & (df['AR'] == 125)]
# ax.plot(df_['dynamic'],df_['f'],'o-',label=r'$\alpha=125, \tilde{F} = 1$',markersize=3,color=default_color_list[2],alpha=0.5)

# # small legend
# # legend = plt.legend(loc='lower right',fontsize=6)
# # plt.xlabel(r'$a/g$',labelpad=-3)
# # plt.ylabel(r'$f$',labelpad=2,rotation=90)
# # # plt.tight_layout()
# # plt.savefig(f'{output_dir}/f_vs_a-g.png',dpi=300,bbox_inches='tight')


# # single_column_size = (1.5,1.3)
# # fig,ax=plt.subplots(figsize=single_column_size)
# #fontsize

# plt.rcParams.update({'font.size': 8})
# df_ = df[(df['static'] == 20) & (df['AR'] == 25)]
# ax.plot(df_['dynamic'],df_['f'],'s--',label=r'$\alpha=25, \tilde{F} = 20$',markersize=3,color=default_color_list[0],alpha=0.5)

# df_ = df[(df['static'] == 20) & (df['AR'] == 100)]
# xx = df_['dynamic'].to_numpy()
# i_sort = np.argsort(xx)
# xx = xx[i_sort]
# yy = df_['f'].to_numpy()
# yy = yy[i_sort]

# ax.plot(xx,yy,'s--',label=r'$\alpha=100, \tilde{F} = 20$',markersize=3,color=default_color_list[1],alpha=0.5)

# df_ = df[(df['static'] == 20) & (df['AR'] == 125)]
# ax.plot(df_['dynamic'],df_['f'],'s--',label=r'$\alpha=125, \tilde{F} = 20$',markersize=3,color=default_color_list[2],alpha=0.5)

# ax.set_xscale('log')
# ax.set_yscale('log')


# plt.yticks([0.1,1],rotation=90)

# plt.xlabel(r'$a/g$',labelpad=-3)
# plt.ylabel(r'$f$',labelpad=2)
# # plt.tight_layout()
# # plt.savefig(f'{output_dir}/f_vs_a-g.png',dpi=300,bbox_inches='tight')
# plt.legend()
# plt.savefig(f'{output_dir}/f_vs_a-g.eps',bbox_inches='tight')




# # %%

# np.savez(f'{output_dir}/entangled_data.npz',entangled_data=entangled_data,untangled_data=untangled_data)
# np.savez(f'{output_dir}/untangled_data.npz',entangled_data=entangled_data,untangled_data=untangled_data)

# %%

entangled_data = np.load(f'{output_dir}/entangled_data.npz',allow_pickle=True)['entangled_data']
untangled_data = np.load(f'{output_dir}/entangled_data.npz',allow_pickle=True)['untangled_data']


# static,mu_over_sigma,dynamic ==> mu_over_sigma, static, dynamic
entangled_data = np.array(entangled_data)
# permute 2,1,3
entangled_data = entangled_data[:,[1,0,2]]

untangled_data = np.array(untangled_data)
untangled_data = untangled_data[:,[1,0,2]]

total_data = np.concatenate([entangled_data,untangled_data])
data_max = np.max(total_data,axis=0)
data_min = np.min(total_data,axis=0)
data_max = np.array([1.2,25,300])
data_min = np.array([0.7,-5,-30])

mg = np.meshgrid(np.linspace(data_min[0],data_max[0],10),np.linspace(data_min[1],data_max[1],10),np.linspace(data_min[2],data_max[2],10))
# np.exp(-x**2-y**2)
surf = np.exp(-mg[0]**2-mg[1]**2-mg[2]**2)



# normalized_entangled_data = entangled_data/data_max
# normalized_untangled_data = untangled_data/data_max

normalized_entangled_data = (entangled_data - data_min)/(data_max - data_min)
normalized_untangled_data = (untangled_data - data_min)/(data_max - data_min)

import polyscope as ps
import numpy as np

ps.init()
ps.set_SSAA_factor(3)
ps.set_ground_plane_mode('shadow_only')

# xy-plane
vertices_xy = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [1, 1, 0],
    [0, 1, 0]
])
faces_xy = np.array([
    [0, 1, 2],
    [0, 2, 3]
])

# yz-plane
vertices_yz = np.array([
    [0, 0, 0],
    [0, 1, 0],
    [0, 1, 1],
    [0, 0, 1]
])
faces_yz = np.array([
    [0, 1, 2],
    [0, 2, 3]
])

# zx-plane
vertices_zx = np.array([
    [0, 0, 0],
    [0, 0, 1],
    [1, 0, 1],
    [1, 0, 0]
])
faces_zx = np.array([
    [0, 1, 2],
    [0, 2, 3]
])

ps_xy = ps.register_surface_mesh("xy-plane", vertices_xy, faces_xy, smooth_shade=False, enabled=True, transparency=0.3)
ps_yz = ps.register_surface_mesh("yz-plane", vertices_yz, faces_yz, smooth_shade=False, enabled=True, transparency=0.3)
ps_zx = ps.register_surface_mesh("zx-plane", vertices_zx, faces_zx, smooth_shade=False, enabled=True, transparency=0.3)

ps_xy.set_color((0.8, 0.8, 0.8))
ps_yz.set_color((0.8, 0.8, 0.8))
ps_zx.set_color((0.8, 0.8, 0.8))

# Add random points within the box (0,0,0) to (1,1,1)
# random_points = np.random.rand(10, 3)

# random_points = np.array([[1,0.5,0.5],[1,0.5,0.1]])
random_points = normalized_entangled_data
ps_point = ps.register_point_cloud("random_point", random_points)
ps_point.set_radius(0.015, relative=False)

points2 = normalized_untangled_data
ps_point2 = ps.register_point_cloud("random_point2", points2,point_render_mode='quad')
ps_point2.set_radius(0.015, relative=False)

# Parameters for circular shadows
num_circle_points = 20  # Number of points to represent each circle
shadow_radius = 0.015   # Radius of each shadow
shadow_offset = 0.001    # Offset from the original points

def create_circular_shadow(points, plane):
    shadow_vertices = []
    shadow_faces = []

    for i, point in enumerate(points):
        # Create vertices for the circle
        angles = np.linspace(0, 2 * np.pi, num_circle_points, endpoint=False)
        
        if plane == "xy":
            circle_vertices = np.stack([point[0] + shadow_radius * np.cos(angles), 
                                        point[1] + shadow_radius * np.sin(angles), 
                                        np.full(num_circle_points,shadow_offset)], axis=1)
            center_vertex = np.array([[point[0], point[1], shadow_offset]])
        
        elif plane == "yz":
            circle_vertices = np.stack([np.full(num_circle_points,shadow_offset), 
                                        point[1] + shadow_radius * np.cos(angles), 
                                        point[2] + shadow_radius * np.sin(angles)], axis=1)
            center_vertex = np.array([[shadow_offset, point[1], point[2]]])
        
        elif plane == "zx":
            circle_vertices = np.stack([point[0] + shadow_radius * np.cos(angles), 
                                        np.full(num_circle_points,shadow_offset), 
                                        point[2] + shadow_radius * np.sin(angles)], axis=1)
            center_vertex = np.array([[point[0], shadow_offset, point[2]]])
        
        # Append vertices
        idx = len(shadow_vertices)
        shadow_vertices.extend(center_vertex)
        shadow_vertices.extend(circle_vertices)
        
        # Create faces for the circle (triangles fan around the center vertex)
        for j in range(num_circle_points):
            shadow_faces.append([idx, idx + 1 + j, idx + 1 + (j + 1) % num_circle_points])
    
    return np.array(shadow_vertices), np.array(shadow_faces)

# Project points onto xy-plane (z=0)
shadow_points_xy = random_points.copy()
shadow_points_xy[:, 2] = 0
shadow_vertices_xy, shadow_faces_xy = create_circular_shadow(shadow_points_xy, "xy")
ps_shadow_xy = ps.register_surface_mesh("shadow_points_xy", shadow_vertices_xy, shadow_faces_xy, smooth_shade=False, enabled=True, transparency=0.8)
ps_shadow_xy.set_color((0,0,0))

# Project points onto yz-plane (x=0)
shadow_points_yz = random_points.copy()
shadow_points_yz[:, 0] = 0
shadow_vertices_yz, shadow_faces_yz = create_circular_shadow(shadow_points_yz, "yz")
ps_shadow_yz = ps.register_surface_mesh("shadow_points_yz", shadow_vertices_yz, shadow_faces_yz, smooth_shade=False, enabled=True, transparency=0.8)
ps_shadow_yz.set_color((0,0,0))

# Project points onto zx-plane (y=0)
shadow_points_zx = random_points.copy()
shadow_points_zx[:, 1] = 0
shadow_vertices_zx, shadow_faces_zx = create_circular_shadow(shadow_points_zx, "zx")
ps_shadow_zx = ps.register_surface_mesh("shadow_points_zx", shadow_vertices_zx, shadow_faces_zx, smooth_shade=False, enabled=True, transparency=0.8)
ps_shadow_zx.set_color((0,0,0))



# Function to create a cylinder
def create_cylinder(start, end, radius, num_segments=20):
    vec = end - start
    length = np.linalg.norm(vec)
    vec /= length
    orthogonal = np.array([vec[1], -vec[0], 0])
    if np.linalg.norm(orthogonal) == 0:
        orthogonal = np.array([vec[2], 0, -vec[0]])
    orthogonal /= np.linalg.norm(orthogonal)
    perp = np.cross(vec, orthogonal)

    angles = np.linspace(0, 2 * np.pi, num_segments, endpoint=False)
    circle = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)
    basis = np.stack([orthogonal, perp, vec], axis=1)  # Fix: create a proper 3x3 basis matrix
    bottom_circle = start + circle @ basis[:, :2].T  # Fix: use the first two columns for 2D circle
    top_circle = end + circle @ basis[:, :2].T  # Fix: use the first two columns for 2D circle

    vertices = np.vstack([bottom_circle, top_circle])
    faces = []
    for i in range(num_segments):
        next_i = (i + 1) % num_segments
        faces.append([i, next_i, num_segments + i])
        faces.append([next_i, num_segments + next_i, num_segments + i])
    
    return vertices, np.array(faces)

# Function to create a cone
def create_cone(base, tip, base_radius, num_segments=20):
    vec = tip - base
    length = np.linalg.norm(vec)
    vec /= length
    orthogonal = np.array([vec[1], -vec[0], 0])
    if np.linalg.norm(orthogonal) == 0:
        orthogonal = np.array([vec[2], 0, -vec[0]])
    orthogonal /= np.linalg.norm(orthogonal)
    perp = np.cross(vec, orthogonal)

    angles = np.linspace(0, 2 * np.pi, num_segments, endpoint=False)
    basis = np.stack([orthogonal, perp, vec], axis=1)  # Fix: create a proper 3x3 basis matrix
    base_circle = base + base_radius * np.stack([np.cos(angles), np.sin(angles)], axis=1) @ basis[:, :2].T  # Fix: use the first two columns

    vertices = np.vstack([base_circle, tip])
    faces = []
    for i in range(num_segments):
        next_i = (i + 1) % num_segments
        faces.append([i, next_i, num_segments])
    
    return vertices, np.array(faces)

# Axes definitions
axes_length = 1.2
axes_radius = 0.02
num_segments = 20

# X-axis
cylinder_vertices_x, cylinder_faces_x = create_cylinder(np.array([0, 0, 0]), np.array([axes_length, 0, 0]), axes_radius, num_segments)
cone_vertices_x, cone_faces_x = create_cone(np.array([axes_length, 0, 0]), np.array([axes_length + 0.1, 0, 0]), axes_radius * 2, num_segments)
ps_x_axis_cylinder = ps.register_surface_mesh("x_axis_cylinder", cylinder_vertices_x, cylinder_faces_x, smooth_shade=False, enabled=True)
ps_x_axis_cylinder.set_color((1, 0, 0))
ps_x_axis_cone = ps.register_surface_mesh("x_axis_cone", cone_vertices_x, cone_faces_x, smooth_shade=False, enabled=True)
ps_x_axis_cone.set_color((1, 0, 0))

# Y-axis
cylinder_vertices_y, cylinder_faces_y = create_cylinder(np.array([0, 0, 0]), np.array([0, axes_length, 0]), axes_radius, num_segments)
cone_vertices_y, cone_faces_y = create_cone(np.array([0, axes_length, 0]), np.array([0, axes_length + 0.1, 0]), axes_radius * 2, num_segments)
ps_y_axis_cylinder = ps.register_surface_mesh("y_axis_cylinder", cylinder_vertices_y, cylinder_faces_y, smooth_shade=False, enabled=True)
ps_y_axis_cylinder.set_color((0, 1, 0))
ps_y_axis_cone = ps.register_surface_mesh("y_axis_cone", cone_vertices_y, cone_faces_y, smooth_shade=False, enabled=True)
ps_y_axis_cone.set_color((0, 1, 0))

# Z-axis
cylinder_vertices_z, cylinder_faces_z = create_cylinder(np.array([0, 0, 0]), np.array([0, 0, axes_length]), axes_radius, num_segments)
cone_vertices_z, cone_faces_z = create_cone(np.array([0, 0, axes_length]), np.array([0, 0, axes_length + 0.1]), axes_radius * 2, num_segments)
ps_z_axis_cylinder = ps.register_surface_mesh("z_axis_cylinder", cylinder_vertices_z, cylinder_faces_z, smooth_shade=False, enabled=True)
ps_z_axis_cylinder.set_color((0, 0, 1))
ps_z_axis_cone = ps.register_surface_mesh("z_axis_cone", cone_vertices_z, cone_faces_z, smooth_shade=False, enabled=True)
ps_z_axis_cone.set_color((0, 0, 1))

ps.set_up_dir("z_up")
# ps.set_front_dir("x_front")





# Create mesh grid
x = np.linspace(data_min[0], data_max[0], 20)
y = np.linspace(data_min[1], data_max[1], 20)

mg = np.meshgrid(x, y, indexing='ij')

# Compute surface values; quite arbitrary
surf = 1200*np.exp(-4.5*mg[0]**4 - mg[1]**2/1000)

# Flatten the mesh grid and surface values
points = np.vstack([mg[0].ravel(), mg[1].ravel(), surf.ravel()]).T

# Create surface vertices and faces
def create_surface_mesh(points):
    nx, ny = mg[0].shape
    faces = []

    # Create faces by connecting adjacent vertices
    for i in range(nx - 1):
        for j in range(ny - 1):
            idx = i * ny + j
            faces.append([idx, idx + 1, idx + ny])
            faces.append([idx + 1, idx + ny, idx + ny + 1])

    return points, np.array(faces)

# Create the surface mesh
surface_vertices, surface_faces = create_surface_mesh(points)
surface_vertices = (surface_vertices - data_min)/(data_max - data_min)

# Register the surface mesh with Polyscope
ps_surface = ps.register_surface_mesh("surface", surface_vertices, surface_faces, smooth_shade=False, enabled=True, transparency=0.5)
ps_surface.set_color((0.2, 0.7, 0.2))

ps.set_up_dir("z_up")


ps.show()
# %%
