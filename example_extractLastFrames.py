# %%
from matplotlib import pyplot as plt
import numpy as np
import re
import os
from data_io import import_all_log, parse_path_string
from fields import get_local_fields_at_a_point
from pathlib import Path

from fields import get_local_fields_at_a_point
import time
import pandas as pd
import filamentFields
import argparse
import datetime

from visualizations import plot_contacts


pathlist = []
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/EntangleCarrotCake5/20240607-2035_RUN_EntangleCarrotCake5_N525_AR105')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/EntangleCarrotCake5/20240607-2035_RUN_EntangleCarrotCake5_N550_AR110')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/EntangleCarrotCake5/20240607-2035_RUN_EntangleCarrotCake5_N575_AR115')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/EntangleCarrotCake5/20240607-2035_RUN_EntangleCarrotCake5_N600_AR120')
# pathlist.append('/Users/yeonsu/Data/from_cluster/20240531-2224_RUN_EntangleCarrotCake5_N0625-AR125')
# pathlist.append('/Users/yeonsu/Data/from_cluster/20240531-2224_RUN_EntangleCarrotCake5_N0125-AR025')
# pathlist.append('/Users/yeonsu/Data/from_cluster/20240531-2224_RUN_EntangleCarrotCake5_N0250-AR050')
# pathlist.append('/Users/yeonsu/Data/from_cluster/20240531-2224_RUN_EntangleCarrotCake5_N0375-AR075')
# pathlist.append('/Users/yeonsu/Data/from_cluster/20240531-2224_RUN_EntangleCarrotCake5_N0500-AR100')

# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/EntangleCarrotCake5/20240531-2224_RUN_EntangleCarrotCake5_N1000-AR200')


# /Users/yeonsu/Data/from_cluster/20240531-2227_RUN_EntangleCarrotCake5_N0125_AR025_g0.5
# /Users/yeonsu/Data/from_cluster/20240531-2227_RUN_EntangleCarrotCake5_N0250_AR050_g0.5
# /Users/yeonsu/Data/from_cluster/20240531-2227_RUN_EntangleCarrotCake5_N0375_AR075_g0.5
# /Users/yeonsu/Data/from_cluster/20240531-2227_RUN_EntangleCarrotCake5_N0500_AR100_g0.5
# /Users/yeonsu/Data/from_cluster/20240531-2227_RUN_EntangleCarrotCake5_N0625_AR125_g0.5

# /Users/yeonsu/Data/from_cluster/20240531-2228_RUN_JostleCarrotCake5_N0125_AR025_g0.5
# /Users/yeonsu/Data/from_cluster/20240531-2228_RUN_JostleCarrotCake5_N0250_AR050_g0.5
# /Users/yeonsu/Data/from_cluster/20240531-2228_RUN_JostleCarrotCake5_N0375_AR075_g0.5
# /Users/yeonsu/Data/from_cluster/20240531-2228_RUN_JostleCarrotCake5_N0500_AR100_g0.5
# /Users/yeonsu/Data/from_cluster/20240531-2228_RUN_JostleCarrotCake5_N0625_AR125_g0.5


# %%
output_folder = f'/Users/yeonsu/Data/export/EntangledCarrotCake5_20240531-2224'
os.makedirs(output_folder, exist_ok=True)
import glob
for pth in pathlist:
    # find the last frame
    globpth = f'{pth}/*lastFrame.csv'
    last_frame = glob.glob(globpth)
    if len(last_frame) == 0:
        continue
    pth = last_frame[0]
    dta = np.loadtxt(pth, delimiter=',',skiprows=1)
    # np.savetxt(f'{output_folder}/{os.path.basename(pth)}', dta, delimiter=' ')
    
    fig,ax=plt.subplots(subplot_kw={'projection':'3d'})
    for i in range(dta.shape[0]):
        rr = dta[i,:].reshape(-1,3)
        ax.plot(rr[:,0],rr[:,1],rr[:,2])
    plt.savefig(f'{output_folder}/{os.path.basename(pth).replace(".csv",".png")}',dpi=300)
    ax.view_init(elev=0, azim=0)
    plt.savefig(f'{output_folder}/{os.path.basename(pth).replace(".csv","_side.png")}',dpi=300)

    
# %%
dta.shape

    
# %%
file_id,surfix,num_rods = parse_path_string(str(pth))

dta = np.loadtxt(pth, delimiter=',',skiprows=1)

fig,ax=plt.subplots(subplot_kw={'projection':'3d'})
for i in range(dta.shape[0]):
    rr = dta[i,:].reshape(-1,3)
    ax.plot(rr[:,0],rr[:,1],rr[:,2])
plt.savefig(outpth_root / f'{file_id}-{surfix}-{end_time:.2f}.png')
plt.close()

savepth = outpth_root / f'{file_id}-{surfix}-{end_time:.2f}.txt'
np.savetxt(savepth, dta, delimiter=' ')


last_curve = node_list[time_line0.index(time_line[-1])].reshape((-1,10,3))
fig,ax=plt.subplots(1,1,figsize=(10,10),subplot_kw={'projection':'3d'})
for rod in last_curve:
    ax.plot(rod[:,0],rod[:,1],rod[:,2],linewidth=0.5)
ax.view_init(elev=0, azim=0)
plt.savefig(f'{data_output_folder}/lastFrame.png',dpi=300)
np.savetxt(f'{data_output_folder}/lastFrame.txt',last_curve.reshape(-1,30))