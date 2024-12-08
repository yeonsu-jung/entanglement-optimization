import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
from jax import lax
from optimization import optimize_fire2,optimize_fire_debug
from potentials import effective_potential,total_effective_potential,total_effective_potential_ref
import numpy as np
from matplotlib import pyplot as plt
import time


def plot_contacts(contact_info,scale_factor,ax):
    ni1 = contact_info["ni1"]
    ni2 = contact_info["ni2"]
    nj1 = contact_info["nj1"]
    nj2 = contact_info["nj2"]
    fi1 = contact_info["fi1"]
    fi2 = contact_info["fi2"]
    fj1 = contact_info["fj1"]
    fj2 = contact_info["fj2"]
    contact_point_i = contact_info["contact_point_i"]
    contact_force_i = contact_info["contact_force_i"]
    contact_point_j = contact_info["contact_point_j"]
    contact_force_j = contact_info["contact_force_j"]
    log_contact_force_i = contact_info["log_contact_force_i"]
    log_contact_force_j = contact_info["log_contact_force_j"]
    
    if (np.isnan(contact_point_i).any() or np.isnan(contact_point_j).any()):
        return
    
    ax.plot([ni1[0],ni2[0]],[ni1[1],ni2[1]],[ni1[2],ni2[2]],'r',linewidth=0.5)
    ax.plot([nj1[0],nj2[0]],[nj1[1],nj2[1]],[nj1[2],nj2[2]],'r',linewidth=0.5)
    ax.plot(contact_point_i[0],contact_point_i[1],contact_point_i[2],'g.')
    ax.plot(contact_point_j[0],contact_point_j[1],contact_point_j[2],'g.')
    
    # log scale
    ax.quiver(contact_point_i[0],contact_point_i[1],contact_point_i[2],log_contact_force_i[0]/scale_factor,log_contact_force_i[1]/scale_factor,log_contact_force_i[2]/scale_factor,color='g',linestyle='-')
    ax.quiver(contact_point_j[0],contact_point_j[1],contact_point_j[2],log_contact_force_j[0]/scale_factor,log_contact_force_j[1]/scale_factor,log_contact_force_j[2]/scale_factor,color='g',linestyle='-')




def set_3d_plot():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    return fig,ax

# def plot_contacts(q,i,neighbors):
#     qs = jnp.reshape(q,(-1,5))
#     # 3d plots    
#     plot_many_rods(jnp.reshape(qs[i,:],(-1,5)),opt_dict={"color":'r','linewidth':2})
#     plot_many_rods(qs[neighbors,:])
    
#     return 1

def plot_many_rods(q,ax=None,opt_dict={}):
    if ax is None:
        fig,ax=set_3d_plot()
        
    N = q.shape[0]
    for i in range(N):        
        plot_rod(q[i,:],opt_dict)
        
    return ax

def plot_rod(q_single,opt_dict):
    q_np = np.array(q_single)
    x1 = q_np[0]
    y1 = q_np[1]
    z1 = q_np[2]
    phi1 = q_np[3]
    theta1 = q_np[4]
    rod_length = 1.

    x11 = x1 + rod_length*jnp.sin(phi1)*jnp.cos(theta1)
    y11 = y1 + rod_length*jnp.sin(phi1)*jnp.sin(theta1)
    z11 = z1 + rod_length*jnp.cos(phi1)
    plt.plot([x1, x11], [y1, y11], [z1, z11],**opt_dict)
    
def plot_rods(q):
    q_np = np.array(q)
    x1 = q_np[0]
    y1 = q_np[1]
    z1 = q_np[2]
    phi1 = q_np[3]
    theta1 = q_np[4]
    rod_length = 1.

    x11 = x1 + rod_length*jnp.sin(phi1)*jnp.cos(theta1)
    y11 = y1 + rod_length*jnp.sin(phi1)*jnp.sin(theta1)
    z11 = z1 + rod_length*jnp.cos(phi1)

    x2 = q_np[5]
    y2 = q_np[6]
    z2 = q_np[7]
    phi2 = q_np[8]
    theta2 = q_np[9]

    x22 = x2 + rod_length*jnp.sin(phi2)*jnp.cos(theta2)
    y22 = y2 + rod_length*jnp.sin(phi2)*jnp.sin(theta2)
    z22 = z2 + rod_length*jnp.cos(phi2)

    # 3d plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.plot([x1, x11], [y1, y11], [z1, z11])
    ax.plot([x2, x22], [y2, y22], [z2, z22])
    
def plot_single_rod(single_rod, *args, ax=None, **kwargs):
    if ax is None:
        fig,ax = set_3d_plot()
    ax.plot(single_rod[:,0],single_rod[:,1],single_rod[:,2],*args,**kwargs)
    return ax
    
def plot_curves(curve,ax,params={}):
    ax.plot(curve[:,0],curve[:,1],curve[:,2],**params)    
    return 1

def plot_many_curves(curves,num_rods,ax,params={}):
    if curves.ndim == 1:
        curves = curves.reshape(num_rods,-1,3)
    elif curves.ndim == 3:
        num_vertices = curves.shape[1]//3//num_rods    
        curves = curves.reshape((num_rods,-1))
    else:
        print('Input must be 1d or 3d array')
        return -1
    
    for i in range(num_rods):
        plot_curves(curves[i,:],ax,params)
    return 1

def plot_edges(edges,ax=None,params={}):
    N = edges.shape[0]
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
    for i in range(N):
        ax.plot([edges[i,0],edges[i,3]],[edges[i,1],edges[i,4]],[edges[i,2],edges[i,5]],**params)    
    

def plot_centerline_with_container(centerlines,svd_cylinders,i,ax):
    cl = centerlines[i]
    cyl = svd_cylinders[i,:]

    cyl_diam = cyl[6]
    cyl_e1 = cyl[0:3]
    cyl_e2 = cyl[3:6]
    cyl_cen = (cyl_e1+cyl_e2)/2
    cyl_len = np.linalg.norm(cyl_e1-cyl_e2)
    cyl_axis = (cyl_e2-cyl_e1)/cyl_len

    Xc, Yc, Zc = data_for_cylinder_along_z(0, 0, cyl_diam, cyl_len/2)
    # Compute the rotation matrix
    rotation_matrix = rotation_matrix_from_vectors(np.array([0, 0, 1]), cyl_axis) 
    # Rotate the cylinder
    Xc_rot, Yc_rot, Zc_rot = rotate_grid(Xc, Yc, Zc, rotation_matrix)
    Xc_rot = Xc_rot + cyl_cen[0]
    Yc_rot = Yc_rot + cyl_cen[1]
    Zc_rot = Zc_rot + cyl_cen[2]
    
    bounding_box = np.array([np.min(cl, axis=0), np.max(cl, axis=0)])
    ax.plot_surface(Xc_rot, Yc_rot, Zc_rot, alpha=0.5)
    ax.plot(cl[:,0], cl[:,1], cl[:,2], color='r')
    # ax.scatter(cyl_e1[0], cyl_e1[1], cyl_e1[2], color='g')
    # zoom in
    # ax.set_xlim(bounding_box[:,0])
    # ax.set_ylim(bounding_box[:,1])
    # ax.set_zlim(bounding_box[:,2])
    
def draw_sphere():
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color='b', alpha=0.1)

def data_for_cylinder_along_z(center_x, center_y, radius, height_z):
    z = np.linspace(-height_z, height_z, 50)
    theta = np.linspace(0, 2 * np.pi, 50)
    theta_grid, z_grid = np.meshgrid(theta, z)
    x_grid = radius * np.cos(theta_grid) + center_x
    y_grid = radius * np.sin(theta_grid) + center_y
    return x_grid, y_grid, z_grid

def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2 """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat @ kmat * ((1 - c) / (s ** 2))
    return rotation_matrix

def rotate_grid(X, Y, Z, rotation_matrix):
    shape = X.shape
    grid = np.vstack([X.ravel(), Y.ravel(), Z.ravel()])
    rotated_grid = rotation_matrix @ grid
    X_rot, Y_rot, Z_rot = rotated_grid.reshape(3, *shape)
    return X_rot, Y_rot, Z_rot

def show_rods_and_fields():
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    # axs[1].imshow(e_fields_img, extent=[-2, 2, -1, 1],vmin=0,vmax=60)
    fig.colorbar(axs[1].imshow(e_fields_img, extent=[xlim[0], xlim[1], zlim[0], zlim[1]],vmin=0,vmax=240), ax=axs[1])
    
    for curve in curves:
        axs[0].plot(curve[:, 0], curve[:, 2], alpha=1)
    for ax in axs:
        ax.set_xlim([-2, 2])
        ax.set_ylim([-1, 1])
        ax.set_aspect('equal')  # Ensure aspect ratio is equal
    axs[0].set_title('Rods')
    axs[1].set_title('Entanglement field')
    axs[0].text(-1.5, 0.8, f'Time: {time_line[frame]} sec', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{output_folder}/frames_{frame:04d}.png', dpi=300)
    plt.close('all')
    
    print(f'Elapsed time: {time.time()-start}')


#  for polyscope


def edge_for_polyscope(segments):
    # segments: list of segments
    edges = []
    for i,seg in enumerate(segments):
        edges += [[i, i + 1] for i in range(len(seg) - 1)]
    return np.array(edges)

def color_for_polyscope(segments):
    colors = np.array([
        [76, 153, 204],   # light blue
        [204, 76, 153],   # pinkish red
        [76, 204, 153],   # mint green
        [153, 204, 76],   # light olive green
        [204, 153, 76],   # goldenrod
        [153, 76, 204],   # medium purple
        [204, 76, 102],   # crimson
        [76, 204, 204],   # cyan
        [204, 204, 76],   # sunflower yellow
        [102, 76, 204]    # indigo
    ])
    
    # segments: list of segments
    colors = []
    for i,seg in enumerate(segments):
        colors += [np.array([i/len(segments), 0, 0]) for i in range(len(seg))]
        
    return np.array(colors)


def prep_for_polyscope(r_list,num_nodes):
    nodes = np.vstack(r_list)
    colors = np.array([
        [76, 153, 204],   # light blue
        [204, 76, 153],   # pinkish red
        [76, 204, 153],   # mint green
        [153, 204, 76],   # light olive green
        [204, 153, 76],   # goldenrod
        [153, 76, 204],   # medium purple
        [204, 76, 102],   # crimson
        [76, 204, 204],   # cyan
        [204, 204, 76],   # sunflower yellow
        [102, 76, 204]    # indigo
        ])/255
    
    colors_interpolated = np.zeros((len(r_list), 3))
    for i in range(3):
        colors_interpolated[:, i] = np.interp(
            np.linspace(0, 1, num_nodes),
            np.linspace(0, 1, colors.shape[0]),
            colors[:, i]
        )

    edge_colors = []
    edges = []

    starting_index = 0
    for i in range(len(r_list)):
        num_edges = len(r_list[i])-1
        to_add = [[starting_index + j,starting_index + j+1] for j in range(num_edges)]
        edges.append(to_add)
        edge_colors.append(np.array([colors_interpolated[i] for j in range(num_edges)]))
        starting_index += num_edges+1
    edges = np.vstack(edges)
    edge_colors = np.vstack(edge_colors)
    
    return nodes,edges,edge_colors

# %%
def live_view_with_polyscope(a_list_of_curves):
    nodes,edges,edge_colors = prep_for_polyscope(a_list_of_curves)
    
    import polyscope as ps
    ps.init()

    ps_curves = ps.register_curve_network("filaments",nodes,edges)
    ps_curves.add_color_quantity("edge_colors",edge_colors,defined_on='edges')

    ps.set_up_dir("z_up")
    ps.show()
    
# def screenshot_with_polyscope(a_list_of_curves):
#     nodes,edges,edge_colors = prep_for_polyscope(a_list_of_curves)
    
#     import polyscope as ps
#     ps.init()

#     ps_curves = ps.register_curve_network("filaments",nodes,edges)
#     ps_curves.add_color_quantity("edge_colors",edge_colors,defined_on='edges')

#     ps.set_up_dir("z_up")
#     ps.show()
    
#     return ps_curves
    
# %%
if __name__ == '__main__':
    from data_io import import_all_log
    from pathlib import Path

    pth = '/Users/yeonsu/GitHub/dismech-rods-main/runs/20240711-0258_COMPILE_metal_nest_relaxation/log_files/metal_nest_allLog_20240711-025850.csv'
    time_line, node_list, contact_list = import_all_log(pth,max_rows=1000000)

    folder_path = Path(pth).parent
    subfolder_name = 'Inbox'

    file_id = pth.split('/')[-1].split('.')[0].split('_allLog_')[-2]
    surfix = pth.split('.')[-2].split('allLog_')[-1]
    file_id = f'worm_1_{file_id}'

    # worm 1
    num_rods = 1451
    AR = 0.3/0.002/2
    rod_diameter = 0.25
        
    output_path = f'/Users/yeonsu/Videos/{subfolder_name}/{file_id}_{surfix}'
    import os

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # live_view_with_polyscope(a_list_of_curves)
    len(node_list)
    # %%
    import polyscope as ps
    ps.init()
    a_list_of_curves = node_list[-1].reshape(1451,-1,3)
    nodes,edges,edge_colors = prep_for_polyscope(a_list_of_curves,1451)
    ps_curves = ps.register_curve_network("filaments",nodes,edges)
    ps_curves.add_color_quantity("edge_colors",edge_colors,defined_on='edges',enabled=True)
    ps.set_up_dir("z_up")
    file_path = f'{output_path}/frame_{0:04d}.png'
    ps.screenshot(file_path)
    # %%
    time_line, node_list, contact_list = import_all_log(pth,max_rows=1000000)
    skip_factor = 5
    for i,a_list_of_curves in enumerate(node_list[::skip_factor]):
        a_list_of_curves = a_list_of_curves.reshape(12,-1,3)
        nodes = np.vstack(a_list_of_curves)
        ps_curves.update_node_positions(nodes)
        file_path = f'{output_path}/frame_{i:04d}.png'
        ps.screenshot(file_path)
        




    # %%

    pth = '/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/20240710-1839_RUN_StandModelo1_9999_9999_9999/cruved_rod_test_allLog_20240710-183926.csv'
    time_line, node_list, contact_list = import_all_log(pth)

    folder_path = Path(pth).parent
    subfolder_name = 'Inbox'

    file_id = pth.split('/')[-1].split('.')[0].split('_allLog_')[-2]
    surfix = pth.split('.')[-2].split('allLog_')[-1]
    file_id = f'worm_1_{file_id}'

    # worm 1
    num_rods = 12
    AR = 0.3/0.002/2
    rod_diameter = 0.25
        
    output_path = f'/Users/yeonsu/Videos/{subfolder_name}/{file_id}_{surfix}'
    import os

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # live_view_with_polyscope(a_list_of_curves)
    len(node_list)
    # %%
    import polyscope as ps
    ps.init()
    a_list_of_curves = node_list[0].reshape(12,-1,3)
    nodes,edges,edge_colors = prep_for_polyscope(a_list_of_curves)
    ps_curves = ps.register_curve_network("filaments",nodes,edges)
    ps_curves.add_color_quantity("edge_colors",edge_colors,defined_on='edges',enabled=True)
    ps.set_up_dir("z_up")
    file_path = f'{output_path}/frame_{0:04d}.png'
    ps.screenshot(file_path)
    # %%
    time_line, node_list, contact_list = import_all_log(pth,max_rows=1000000)
    skip_factor = 5
    for i,a_list_of_curves in enumerate(node_list[::skip_factor]):
        a_list_of_curves = a_list_of_curves.reshape(12,-1,3)
        nodes = np.vstack(a_list_of_curves)
        ps_curves.update_node_positions(nodes)
        file_path = f'{output_path}/frame_{i:04d}.png'
        ps.screenshot(file_path)