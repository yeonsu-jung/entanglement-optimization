# %%
import numpy as np
from pathlib import Path
import polyscope as ps

# %%
pth = Path('qs')
# %%
# check number of files in pth
num_snapshots = len(list(pth.glob('*.npy')))

# %%
# load the first snapshot
q0 = np.load(pth / 'q_0.npy')
q0.shape
# %%
from transforms import q_to_x

x0 = q_to_x(q0)
# %%
from matplotlib import pyplot as plt

centroid = np.mean(x0[:,:3],axis=0)

fig,ax=plt.subplots(subplot_kw={'projection':'3d'})
for edge in x0:
    r1 = edge[:3]-centroid
    r2 = edge[3:]-centroid
    
    ax.plot([r1[0],r2[0]],[r1[1],r2[1]],[r1[2],r2[2]])
# %%
nodes = x0.reshape(-1,3)
num_nodes_each_rod = 2
edges = np.array([[i, i + 1] for i in range(len(nodes) - 1) if i % num_nodes_each_rod != num_nodes_each_rod - 1])
# %%
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
# %%
num_rods = len(nodes) // num_nodes_each_rod
# %%
ps.init()

ps.set_SSAA_factor(3)
ps.set_navigation_style("free")

# ps.set_ground_plane_mode("none") 
ps.set_ground_plane_mode("shadow_only")  # set +Z as up direction
ps.set_ground_plane_height_factor(-0.25) # adjust the plane height
ps.set_shadow_darkness(0.1)              # lighter shadows

ps_all_nodes = ps.register_curve_network("all_nodes", nodes, edges, enabled=True)
vals_edge = np.ones((len(edges),3))
for i in range(num_rods):
    vals_edge[i*num_nodes_each_rod:(i+1)*num_nodes_each_rod] = colors[i%10]/255

ps_all_nodes.add_color_quantity(f"rod_colors", vals_edge, defined_on='edges', enabled=True)

# ps.show()
ps.screenshot('temp.png',transparent_bg=False)

# %%
num_snapshots = 1000
iterations = 0
for i in range(0, num_snapshots, 10):
    q = np.load(pth / f'q_{i}.npy')
    x = q_to_x(q)
    ps_all_nodes.update_node_positions(x.reshape(-1,3))
    ps.screenshot(f'qs/images/q_{iterations:04d}.png',transparent_bg=False)
    iterations += 1
# %%




# # for random rods

# from protocols import create_random_rods
# num_rods = 100

# ps.init()

# ps.set_SSAA_factor(3)
# ps.set_navigation_style("free")

# # ps.set_ground_plane_mode("none") 
# ps.set_ground_plane_mode("shadow_only")  # set +Z as up direction
# ps.set_ground_plane_height_factor(-0.25) # adjust the plane height
# ps.set_shadow_darkness(0.1)              # lighter shadows

# # q0 = np.loadtxt('/Users/yeonsu/Data/cache/EntangledPacking_N300_AR100.txt')
# # q0 = create_random_rods(300)
# x0 = q_to_x(q0)

# nodes = x0.reshape(-1,3)
# num_nodes_each_rod = 2
# edges = np.array([[i, i + 1] for i in range(len(nodes) - 1) if i % num_nodes_each_rod != num_nodes_each_rod - 1])


# ps_all_nodes = ps.register_curve_network("all_nodes", nodes, edges, enabled=True)
# vals_edge = np.ones((len(edges),3))
# for i in range(num_rods):
#     vals_edge[i*num_nodes_each_rod:(i+1)*num_nodes_each_rod] = colors[i%10]/255

# ps_all_nodes.add_color_quantity(f"rod_colors", vals_edge, defined_on='edges', enabled=True)
# ps.look_at((-3,-3,-3),(0,0,0))
# # ps.show()
# ps.screenshot('temp.png',transparent_bg=False)
# # %%
# # x0 = x0.reshape(-1,3)
# num_snapshots = x0.shape[0]//2
# for i_frame in range(0,num_snapshots,100):
#     nodes = x0[:i_frame,:].reshape(-1,3)
#     edges = np.array([[i, i + 1] for i in range(len(nodes) - 1) if i % num_nodes_each_rod != num_nodes_each_rod - 1])    
    
#     vals_edge = np.ones((len(edges),3))
#     _num_rods = len(edges)
#     for i in range(_num_rods):
#         vals_edge[i*num_nodes_each_rod:(i+1)*num_nodes_each_rod] = colors[i%10]/255
    
#     ps_each_node = ps.register_curve_network("all_nodes", nodes, edges, enabled=True)    
#     ps_each_node.add_color_quantity(f"rod_colors", vals_edge, defined_on='edges', enabled=True)
    
#     ps.look_at((-3,-3,-3),(0,0,0))
#     ps.screenshot(f'qs/images/q_{i_frame:04d}.png',transparent_bg=False)




