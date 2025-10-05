# %%
import numpy as np
from visualizations import prep_for_polyscope
from pathlib import Path
from data_io import import_all_log


# %%
pth = '/Users/yeonsu/GitHub/dismech-rods-main/runs/20240711-1634_COMPILE_metal_nest/log_files/metal_nest_allLog_20240711-163458.csv'
time_line, node_list, contact_list = import_all_log(pth,max_rows=100)

num_rods = 1265
rod_diameter = 0.025

a_list_of_curves = node_list[19].reshape(num_rods,-1,3)
# %%
# remove element having nan
nan_idx = []
for i,curve in enumerate(a_list_of_curves):
    nan_idx.append(np.isnan(curve).sum() > 0)    
np.array(nan_idx).sum()
new_node_list = np.delete(a_list_of_curves,nan_idx,axis=0)
num_rods = len(new_node_list)

# %%
import polyscope as ps
ps.init()
a_list_of_curves = new_node_list
nodes,edges,edge_colors = prep_for_polyscope(a_list_of_curves,num_rods)
ps_curves = ps.register_curve_network("filaments",nodes,edges)
ps_curves.add_color_quantity("edge_colors",edge_colors,defined_on='edges',enabled=True)
ps.set_up_dir("z_up")

file_path = 'temp.png'
ps.screenshot(file_path)

# %%
