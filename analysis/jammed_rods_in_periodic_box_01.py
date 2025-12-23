# %%
pth = '/Users/yeonsu/GitHub/rod-dynamics-3d/initial-configs/rods932.csv'

# read csv
import numpy as np

# get first rows with # as header
metadata = []
with open(pth, 'r') as f:
    for line in f:
        if line.startswith('#'):
            metadata.append(line.strip())
        else:
            break

# get
# rod_length
# rod_diameter
# box_size
# placed

rod_length = float(metadata[0].split('=')[1])
rod_diameter = float(metadata[1].split('=')[1])
box_size = float(metadata[4].split('=')[1])
placed = int(metadata[7].split('=')[1])
    
# %%
# skip the rows starting with # and x0
rods_in_shape = np.loadtxt(pth, delimiter=',', skiprows=10)
rods_in_shape = rods_in_shape.reshape(932,-1,3)

# %%
import sys

sys.path.append('/Users/yeonsu/GitHub/entanglement-optimization-combined/entanglement-optimization/core')

from visualizations import prep_for_polyscope

import polyscope as ps
ps.init()
ps.set_up_dir("z_up")
nodes0, edges0, _ = prep_for_polyscope(rods_in_shape,932)
ps_rods = ps.register_curve_network("rods", nodes0, edges0)
ps.show()