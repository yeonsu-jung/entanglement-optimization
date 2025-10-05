# %%
import os
from pathlib import Path
import numpy as np

root_folder_list = []
# root_folder_list.append('/Users/yeonsu/GitHub/dismech-rods-main/data/MaxEntFinal/6,7,8')
# root_folder_list.append('/Users/yeonsu/GitHub/dismech-rods-main/data/MaxEntFinal/37,178,56')
# root_folder_list.append('/Users/yeonsu/GitHub/dismech-rods-main/data/MaxEntFinal/919,461,568')

# root_folder_list.append('/Users/yeonsu/GitHub/entanglement-optimization/results/919,461,568')
# root_folder_list.append('/Users/yeonsu/GitHub/entanglement-optimization/results/37,178,56')

root_folder_list.append('/Users/yeonsu/GitHub/entanglement-optimization/results/6,7,8')
root_folder_list.append('/Users/yeonsu/GitHub/entanglement-optimization/results/919,461,568')
root_folder_list.append('/Users/yeonsu/GitHub/entanglement-optimization/results/37,178,56')



# %%

# root_folder = '/Users/yeonsu/GitHub/entanglement-optimization/results/89,32,178'
root_folder = root_folder_list[0]
out_folder = '/Users/yeonsu/GitHub/dismech-rods-main/data/MaxEntFinal/' + root_folder.split('/')[-1] 

from analysis_functions import create_folder
create_folder(out_folder)
if not os.path.exists(out_folder):
    os.makedirs(out_folder)
# %%
# list subfolders
pathlist = root_folder_list


# %%
from analysis_functions import parse_pathname
from transforms import x_to_q, q_to_x
# pathlist
for pth in pathlist:
    out_folder = '/Users/yeonsu/GitHub/dismech-rods-main/data/MaxEntFinal/' + pth.split('/')[-1] 
    # find subfolders
    subfolders = [f.path for f in os.scandir(pth) if f.is_dir()]

    for subfolder in subfolders:
        try:
            dt_string, AR, num_rods, random_keys = parse_pathname(subfolder)
        except:
            continue
        tmp = np.loadtxt(subfolder + '/q_relaxed.txt')
        out = q_to_x(tmp.reshape(-1,5))
        np.savetxt(out_folder + f'/MaxEnt{random_keys}-N{num_rods:03d}-AR{int(AR):04d}-Scale1.txt', out, delimiter=' ')



# %%


# %%

