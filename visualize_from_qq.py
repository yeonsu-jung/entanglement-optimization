# %%
import numpy as np
from pathlib import Path
import polyscope as ps
# %%
import numpy as np 
x = np.loadtxt('/Users/yeonsu/Data/export/EntangledRelaxedPackings/EntangledRelaxedPacking-N0500-AR0100-Scale1.txt')

from transforms import q_to_x, x_to_q
from potentials import create_pairs, all_pairwise_distances
q = x_to_q(x)
# %%
q_pairs = create_pairs(q)
d = all_pairwise_distances(q_pairs)
np.min(d)


# %%
pth = Path('/Users/yeonsu/GitHub/entanglement-optimization/results/2024-10-15_11_EntangledRelaxedPacking-N0500-AR0050-Scale1/qq.npy')

video_path = pth.parent/'images'
video_path.mkdir(exist_ok=True)

# %%
qq = np.load(pth)
qq_reshaped = qq.reshape(-1,100,5)
q0 = qq_reshaped[0]
num_snapshots = qq_reshaped.shape[0]
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
ps_all_nodes.set_radius(1/100,relative=False)
# ps.show()
ps.screenshot('temp.png',transparent_bg=False)

# %%
# num_snapshots = 1000
iterations = 0
for i in range(0, num_snapshots, 100):
    q = qq_reshaped[i]
    x = q_to_x(q)
    ps_all_nodes.update_node_positions(x.reshape(-1,3))
    ps.screenshot(f'{video_path}/q_{iterations:05d}.png',transparent_bg=False)
    iterations += 1

# %%
# ffmpeg -i frame_%04d.png -vf "scale=2*trunc(iw/2):2*trunc(ih/2)" -c:v libx264 -crf 18 -framerate 60 -profile:v main -pix_fmt yuv420p -c:a aac -ac 2 -b:a 128k -movflags faststart ./output.mp4

import subprocess

# Define the ffmpeg command as a list of strings
command = [
    "ffmpeg", 
    "-i", f"{video_path}/q_%05d.png", 
    "-vf", "scale=2*trunc(iw/2):2*trunc(ih/2)", 
    "-c:v", "libx264", 
    "-crf", "18", 
    "-framerate", "60", 
    "-profile:v", "main", 
    "-pix_fmt", "yuv420p", 
    "-c:a", "aac", 
    "-ac", "2", 
    "-b:a", "128k", 
    "-movflags", "faststart", 
    f"{video_path}/output.mp4"
]

# Execute the command
try:
    subprocess.run(command, check=True)
    print("Video creation successful.")
except subprocess.CalledProcessError as e:
    print(f"Error occurred: {e}")

# %%
