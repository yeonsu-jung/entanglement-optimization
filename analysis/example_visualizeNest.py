# %%
from scipy.io import loadmat
import numpy as np
matobj = loadmat('/Users/yeonsu/Dropbox (Harvard University)/Data/PrunedData/rod-sim-pnas-revision/MetalNestSegmented/segments.mat')

# %%
segments_nanpad = matobj['segments_nanpad']
segments = [seg[~np.isnan(seg)].reshape(-1,3) for seg in segments_nanpad]
# %%
rod_diameter = 15
nodes = np.concatenate(segments)
edges = []

last_i = 0
for i in range(len(segments)):
    segment = segments[i]    
    num_nodes = len(segment)
    edges.append([(last_i+i, last_i+i + 1) for i in range(len(segment) - 1)])
    last_i += num_nodes
edges = np.vstack(edges)
    

import polyscope as ps

ps.init()
ps.set_SSAA_factor(3)
ps.set_navigation_style("free")

# ps.set_ground_plane_mode("tile")
ps.set_ground_plane_mode("none")
ps.set_ground_plane_mode("shadow_only")  # set +Z as up direction
ps.set_ground_plane_height_factor(0.35) # adjust the plane height
ps.set_shadow_darkness(0.1)              # lighter shadows
ps.set_view_projection_mode("perspective")
# ps.set_transparency_mode('simple')

ps_all_nodes = ps.register_curve_network("all_nodes", nodes, edges, enabled=True)
ps_all_nodes.set_radius(rod_diameter/2,relative=False)
# ps_all_nodes.add_color_quantity(f"rod_colors", vals_edge, defined_on='edges', enabled=True)
ps_all_nodes.set_material("clay")
# ps.look_at((-5., 0., 1.), (0., 0., 0.))
ps.set_up_dir("z_up")

ps.set_ground_plane_mode("tile_reflection")
ps.set_shadow_darkness(0.5)              # lighter shadows
# ps.screenshot('test.png',transparent_bg=False)
ps.show()
    
# %%

