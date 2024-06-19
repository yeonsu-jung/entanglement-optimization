# %%
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import glob
import re
import pickle

from data_io import import_all_log, parse_path_string    
# %%
pathlist = []
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/HangModelo1/20240610-1802_RUN_HangEEModelo1_N0250_AR050')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/HangModelo1/20240611-1139_RUN_HangEEModelo1_N0450_AR090')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/HangModelo1/20240610-1802_RUN_HangEEModelo1_N0500_AR100')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/HangModelo1/20240610-1755_RUN_HangEEModelo1_N1500_AR300')

# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/Modelo1_SlowExcitation/20240611-1241_RUN_WeakPerturbEEModelo1_N0250_AR050')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/Modelo1_SlowExcitation/20240611-1247_RUN_WeakPerturbEEModelo1_N0450_AR090')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/Modelo1_SlowExcitation/20240611-1241_RUN_WeakPerturbEEModelo1_N0500_AR100')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/Modelo1_SlowExcitation/20240611-1241_RUN_WeakPerturbEEModelo1_N1500_AR300')

pathlist.append("/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/Modelo1_FineExcitation/20240609-1052_RUN_PerturbEEModelo1_N0250_AR050")
pathlist.append("/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/Modelo1_FineExcitation/20240611-0134_RUN_PerturbEEModelo1_N0450_AR090")
pathlist.append("/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/Modelo1_FineExcitation/20240609-1052_RUN_PerturbEEModelo1_N0500_AR100")
pathlist.append("/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/Modelo1_FineExcitation/20240609-1052_RUN_PerturbEEModelo1_N1500_AR300")
    
# %%
output_root = '/Users/yeonsu/Dropbox (Harvard University)/Data/PrunedData/rod-sim-pnas-revision'
plot_id = f'PlotHangModelos_{Path(pathlist[0]).parent.stem}'

# %%

class data_container:
    def __init__(self,dataphat,max_rows=100000):
        self.path = Path(dataphat)
        out = parse_path_string(self.path)
        # self.file_id,self.surfix,self.num_rods,self.AR,self.datetime_string
        self.file_id = out[0]
        self.surfix = out[1]
        self.num_rods = out[2]
        self.AR = out[3]
        self.datetime_string = out[4]
        
        self.time_line, self.node_list, self.contact_list = import_all_log(self.path,max_rows=max_rows)
        
max_rows = 1000000
data_container_list = []
for pth in pathlist:
    # find csv file
    data_path = None
    for file in Path(pth).rglob('*.csv'):
        if str(file.stem).endswith('lastFrame'):
            continue
        data_container_list.append(data_container(file,max_rows=max_rows))
        break
# memory pressure: nothing

# %%
# check time_line are the same
for _i in range(1,len(data_container_list)):
    if not np.all(data_container_list[0].time_line == data_container_list[_i].time_line):
        print(f'Error: {data_container_list[0].path} and {data_container_list[_i].path} have different time_line')
        break
time_line = data_container_list[0].time_line
# %%
for _j in range(len(data_container_list)):
    nodes_in_matrix = data_container_list[_j].node_list[0].reshape((-1,30))
    packing_center = np.mean(np.mean(nodes_in_matrix.reshape((-1,10,3)),axis=1),axis=0)    
    locked_nodes = []
    for rr in nodes_in_matrix.reshape((-1,10,3)):
        I = np.linalg.norm(rr - packing_center,axis=1) < 0.1        
        locked_nodes.append(rr[I,:])
        
    data_container_list[_j].locked_nodes = locked_nodes

# %%
savefolder = f'{output_root}/{plot_id}'
if not os.path.exists(savefolder):
    os.makedirs(savefolder)

fig,axs=plt.subplots(2,2,subplot_kw={'projection':'3d'})
axs = axs.flatten()

for _i in range(len(time_line)):
    for _j in range(len(data_container_list)):
        AR = data_container_list[_j].AR
        rod_diameter = 1/float(AR)*100
        line_thickness = rod_diameter    
        
        nodes_in_matrix = data_container_list[_j].node_list[_i].reshape((-1,30))
        for node in nodes_in_matrix:
            rr = node.reshape((-1,3))
            axs[_j].plot(rr[:,0],rr[:,1],rr[:,2],linewidth=line_thickness)
            
        # for rr in data_container_list[_j].locked_nodes:
        #     axs[_j].plot(rr[:,0],rr[:,1],rr[:,2],'k-',linewidth=line_thickness)
            
        axs[_j].set_xlim(-2,2)
        axs[_j].set_ylim(-2,2)
        axs[_j].set_zlim(-2,2)
        axs[_j].view_init(elev=0,azim=0)
        
        axs[_j].set_title(f'{data_container_list[_j].num_rods} rods, AR={data_container_list[_j].AR}')
        axs[_j].set_xticklabels([])
        axs[_j].set_yticklabels([])
        axs[_j].set_zticklabels([])
        
    axs[0].text(1,0.5,3,f'time: {time_line[_i]}')
    
    # plt.tight_layout()
    plt.savefig(f'{output_root}/{plot_id}/frames_{_i:04d}.png', dpi=300, bbox_inches='tight', pad_inches=0, transparent=False)
    for _j in range(len(data_container_list)):
        axs[_j].clear()
        
