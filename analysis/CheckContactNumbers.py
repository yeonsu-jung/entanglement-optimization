# %%
import numpy as np

# %%
pathlist = []
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/CalmRigidModelo1/20240721-2055_RUN_CalmRigidModelo1_N0125_AR025')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/CalmRigidModelo1/20240721-2055_RUN_CalmRigidModelo1_N0250_AR050')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/CalmRigidModelo1/20240721-2055_RUN_CalmRigidModelo1_N0300_AR060')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/CalmRigidModelo1/20240721-2055_RUN_CalmRigidModelo1_N0350_AR070')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/CalmRigidModelo1/20240721-2055_RUN_CalmRigidModelo1_N0375_AR075')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/CalmRigidModelo1/20240721-2055_RUN_CalmRigidModelo1_N0400_AR080')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/CalmRigidModelo1/20240721-2055_RUN_CalmRigidModelo1_N0450_AR090')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/CalmRigidModelo1/20240721-2055_RUN_CalmRigidModelo1_N0500_AR100')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/CalmRigidModelo1/20240721-2055_RUN_CalmRigidModelo1_N0525_AR105')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/CalmRigidModelo1/20240721-2055_RUN_CalmRigidModelo1_N0550_AR110')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/CalmRigidModelo1/20240721-2055_RUN_CalmRigidModelo1_N0575_AR115')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/CalmRigidModelo1/20240721-2055_RUN_CalmRigidModelo1_N0600_AR120')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/CalmRigidModelo1/20240721-2055_RUN_CalmRigidModelo1_N0625_AR125')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/CalmRigidModelo1/20240721-2055_RUN_CalmRigidModelo1_N0750_AR150')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/CalmRigidModelo1/20240721-2055_RUN_CalmRigidModelo1_N0875_AR175')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/CalmRigidModelo1/20240721-2055_RUN_CalmRigidModelo1_N1000_AR200')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/CalmRigidModelo1/20240721-2055_RUN_CalmRigidModelo1_N1250_AR250')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/CalmRigidModelo1/20240721-2055_RUN_CalmRigidModelo1_N1500_AR300')

# %%
from pathlib import Path
from data_io import parse_path_string,import_all_log

coordination_number_dict = {}
for folder_path in pathlist:
    folder_path = Path(folder_path)
    possible_paths = []
    for pth in folder_path.glob('**/*.csv'):
        if 'lastFrame' in str(pth):
            continue
        else:
            possible_paths.append(pth)    
    if len(possible_paths) == 0:
        print('No csv files found in the folder')
        exit()
    elif len(possible_paths) > 1:
        print('Multiple csv files found in the folder')
        # find heaviest file
        max_size = 0
        for pth in possible_paths:
            size = os.path.getsize(pth)
            if size > max_size:
                max_size = size
                heaviest_file = pth
        possible_paths = [heaviest_file]
        
    pth = str(possible_paths[0])
    file_id,surfix,num_rods,AR,datetime_str = parse_path_string(pth)
    time_points,node_list,contact_list = import_all_log(pth,max_rows = 10000000)
    
    Nc = len(contact_list[-1].reshape(-1,18))/num_rods*2
    
    print(AR,Nc)
    coordination_number_dict[AR] = Nc
    
    
# %%
AR_list = list(coordination_number_dict.keys())
Nc_list = list(coordination_number_dict.values())

import matplotlib.pyplot as plt
plt.figure(figsize=(4,3))
plt.plot(AR_list,Nc_list,'o-')
plt.xlabel('Aspect Ratio')
plt.ylabel('Coordination Number')
plt.savefig('coordination_number_vs_AR.png',dpi=300,bbox_inches='tight')

# %%

np.savetxt('coordination_number_vs_AR.csv',np.array([AR_list,Nc_list]).T,delimiter=',',header='Aspect Ratio,Coordination Number',comments='')
# %%
tmp = np.loadtxt('coordination_number_vs_AR.csv',delimiter=',',skiprows=1)