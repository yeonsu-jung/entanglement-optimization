# %%
import numpy as np
from visualizations import plot_many_rods
import os
from pathlib import Path



import polyscope as ps
from transforms import q_to_x
from visualizations import prep_for_polyscope
import re



# current filename
current_file = os.path.basename(__file__)
filename = os.path.splitext(current_file)[0]
output_folder = f'{filename}'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f'Output folder: {output_folder}')



output_folder = Path(output_folder)
MOVIE_DIR = f"{output_folder}/movie"
if not os.path.exists(MOVIE_DIR):
    os.makedirs(MOVIE_DIR)
    print(f'Movie folder: {MOVIE_DIR}')


pth = '/Users/yeonsu/GitHub/entanglement-optimization/protocols5/results/6,7,8/2025-10-05_03_EntangledRelaxedPacking-N0100-AR0010-Scale1/qq.npy'
q = np.load(pth)


# Extract aspect ratio (AR) from the file path

match = re.search(r'AR(\d+)', pth)
if match:
    AR = int(match.group(1)) 
    print(f"Aspect Ratio (AR): {AR}")
else:
    raise ValueError("Aspect Ratio (AR) not found in the file path.")

rod_diameter = 1/AR

# %%
num_rods = 100
q = q.reshape(-1,num_rods,5)

num_timesteps = q.shape[0]
print(f"num_timesteps: {num_timesteps}")

# %%
# q0 = q[0]
qf = q[-1]
plot_many_rods(qf)
# %%


ii,jj= np.triu_indices(num_rods,k=1)
from potentials import dist_lin_seg_over_ij, acn_over_ij

x = q_to_x(qf).reshape(num_rods,-1,3)
r1 = x[:,0,:]
r2 = x[:,-1,:]

dij = dist_lin_seg_over_ij(r1,r2,ii,jj)
acn = acn_over_ij(r1,r2,ii,jj)

ent = np.sum( np.abs(acn) )
# %%




# %%

ps.init()
ps.set_autoscale_structures(False)
ps.set_automatically_compute_scene_extents(False)
ps.set_ground_plane_mode("none")

a_list_of_curves = q_to_x(q[0]).reshape(num_rods, -1, 3)
nodes, edges, edge_colors = prep_for_polyscope(a_list_of_curves, num_rods)

min_z = np.min(nodes[:, 2])
ps_curves = ps.register_curve_network( "filaments", nodes, edges )
ps_curves.add_color_quantity( "edge_colors", edge_colors, defined_on='edges', enabled=True )
ps_curves.set_radius( rod_diameter / 2, relative=False )

ps.set_length_scale(2.)
sz = 2.
low = np.array((-sz, -sz, -sz))
high = np.array((sz, sz, sz))
ps.set_bounding_box(low, high)
ps.set_up_dir("z_up")

nodes, edges, edge_colors = prep_for_polyscope(a_list_of_curves, num_rods)
ps.screenshot(f"{MOVIE_DIR}/step-0000.png")

# %%
for i in range(num_timesteps):
    qi = q[i]    
    a_list_of_curves = q_to_x(qi).reshape(num_rods, -1, 3)
    nodes = np.vstack(a_list_of_curves)
    ps_curves.update_node_positions(nodes)
    

    ps.screenshot(f"{MOVIE_DIR}/step-{i:04d}.png")
    print(f"Saved step-{i:04d}.png")

# %%
# create movie from python
# exec shell command
os.system(f'ffmpeg -framerate 10 -i {MOVIE_DIR}/step-%04d.png -c:v libx264 -pix_fmt yuv420p -y {output_folder}/output.mp4')

