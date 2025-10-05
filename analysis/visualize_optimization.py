# %%
import numpy as np
# pth = '/Users/yeonsu/GitHub/entanglement-optimization/results/85,32,12/2024-10-22_12_EntangledRelaxedPacking-N0100-AR0050-Scale1/qq.npy'
# pth = '/Users/yeonsu/GitHub/entanglement-optimization/results/85,32,12/2024-10-22_12_EntangledRelaxedPacking-N0100-AR0100-Scale1/qq.npy'
# pth = '/Users/yeonsu/GitHub/entanglement-optimization/results/85,32,12/2024-10-22_12_EntangledRelaxedPacking-N0100-AR0200-Scale1/qq.npy'
# pth = '/Users/yeonsu/GitHub/entanglement-optimization/results/85,32,12/2024-10-22_12_EntangledRelaxedPacking-N0100-AR0300-Scale1/qq.npy'
pth = '/Users/yeonsu/GitHub/entanglement-optimization/results/85,32,12/2024-10-22_12_EntangledRelaxedPacking-N0100-AR0500-Scale1/qq.npy'

from pathlib import Path
output_path = Path(pth).parent / 'polyscope'

if not output_path.exists():
    output_path.mkdir()

# %%
import re
def parse_pathname(pathname):
    dt_string = re.search(r'(\d{4}-\d{2}-\d{2}_\d{2})',pathname).group(1)
    AR = float(re.search('AR(\d+)',pathname).group(1))
    num_rods = int(re.search('N(\d+)',pathname).group(1))
    return dt_string, AR, num_rods

_,AR,num_rods=parse_pathname(pth)
# %%
qq = np.load(pth)
qq = qq.reshape(-1,num_rods,5)
num_frames = qq.shape[0]
qq.shape
# %%
import polyscope as ps
from visualizations import prep_for_polyscope
from transforms import q_to_x

rod_diameter = 1/AR
_t = num_frames-1
_t = 0
q = qq[_t]
x = q_to_x(q)

node_list = []
for i in range(1,num_frames):
    q = qq[i].reshape(-1,5)
    x = q_to_x(q)
    node_list.append(x)

from matplotlib import pyplot as plt
from visualizations import plot_many_rods,set_3d_plot
fig,ax = set_3d_plot()
plot_many_rods(q.reshape(-1,5),ax=ax)
# %%
from potentials import all_pairwise_distances
q = qq[0]
np.min(all_pairwise_distances(q))
# %%
ps.init()

a_list_of_curves = x.reshape(num_rods,-1,3)
nodes,edges,edge_colors = prep_for_polyscope(a_list_of_curves,num_rods)
min_z = np.min(nodes[:,2])
# ps.set_ground_plane_height_factor(-min_z)
               
ps_curves = ps.register_curve_network("filaments",nodes,edges)
ps_curves.add_color_quantity("edge_colors",edge_colors,defined_on='edges',enabled=True)
ps_curves.set_radius(rod_diameter/2,relative=False)

ps.set_up_dir("z_up")
ps.screenshot('temp.png',transparent_bg=False)
# ps.show()
# exit()
# %%
num_files_already = len(list(Path(output_path).glob('frame_*')))
print(f'Number of frames: {num_files_already}')

# %%
skip_factor = 1
for i,a_list_of_curves in enumerate(node_list[num_files_already::skip_factor]):
    a_list_of_curves = a_list_of_curves.reshape(num_rods,-1,3)
    # num_rods = len(a_list_of_curves)
    nodes = np.vstack(a_list_of_curves)
    ps_curves.update_node_positions(nodes)
    file_path = f'{output_path}/frame_{i+num_files_already:04d}.png'
    ps.screenshot(file_path)

import subprocess
subprocess.run(['ffmpeg', '-framerate', '10', '-i', f'{output_path}/frame_%04d.png', '-r', '30', '-pix_fmt', 'yuv420p', f'{output_path}/output.mp4'])

# %%
