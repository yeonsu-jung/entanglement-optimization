import sys
import os
from pathlib import Path
import numpy as np
import polyscope as ps



sys.path.append('/Users/yeonsu/GitHub/entanglement-optimization-combined/entanglement-optimization/core')

from visualizations import prep_for_polyscope


# test

# pth = '/Users/yeonsu/GitHub/entanglement-optimization-combined/entanglement-optimization/scripts/outputs/entangled_packings/56,321,194/2025-12-23_13_EntangledRelaxedPacking-N0050-AR0100-Scale1/x_entangled_packing.txt'
# pth = '/Users/yeonsu/GitHub/entanglement-optimization-combined/entanglement-optimization/scripts/outputs/entangled_packings/56,321,194/N20/x_entangled_packing.txt'
pth = '/Users/yeonsu/GitHub/entanglement-optimization-combined/entanglement-optimization-cpp/examples/entangled_packings/N500/517,862,750/x_relaxed_AR10.txt'
# pth = '/Users/yeonsu/GitHub/entanglement-optimization-combined/entanglement-optimization/nonintersecting_packings_contained/dataset1/x_nonintersecting_packing_100.txt'

x = np.loadtxt(pth)
x = x.reshape(-1,6)

num_rods = x.shape[0]

x = x.reshape(num_rods,-1,3)

ps.init()

rod_diameter = 0.01

nodes,edges,edge_colors = prep_for_polyscope(x,num_rods)
min_z = np.min(nodes[:,2])
ps_curves = ps.register_curve_network("filaments",nodes,edges)

ps_curves.add_color_quantity("edge_colors",edge_colors,defined_on='edges',enabled=True)

ps_curves.set_radius(rod_diameter/2,relative=False)

ps.set_length_scale(2.)
ps.show()





