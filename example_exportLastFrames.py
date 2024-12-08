# %%
import numpy as np
from matplotlib import pyplot as plt
import re
import os
from data_io import import_all_log, parse_path_string
from fields import get_local_fields_at_a_point
from pathlib import Path



last_frame_out_path = f'/Users/yeonsu/GitHub/dismech-rods-main/data/CalmRigidModelo1/'

pathlist = []
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PerturbRigidModelo1/20240722-0331_RUN_PerturbCalmRigidModelo1_N0125_AR025_freq10')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PerturbRigidModelo1/20240722-0331_RUN_PerturbCalmRigidModelo1_N0250_AR050_freq10')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PerturbRigidModelo1/20240722-0331_RUN_PerturbCalmRigidModelo1_N0300_AR060_freq10')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PerturbRigidModelo1/20240722-0331_RUN_PerturbCalmRigidModelo1_N0350_AR070_freq10')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PerturbRigidModelo1/20240722-0331_RUN_PerturbCalmRigidModelo1_N0375_AR075_freq10')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PerturbRigidModelo1/20240722-0331_RUN_PerturbCalmRigidModelo1_N0400_AR080_freq10')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PerturbRigidModelo1/20240722-0331_RUN_PerturbCalmRigidModelo1_N0450_AR090_freq10')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PerturbRigidModelo1/20240722-0331_RUN_PerturbCalmRigidModelo1_N0500_AR100_freq10')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PerturbRigidModelo1/20240722-0331_RUN_PerturbCalmRigidModelo1_N0525_AR105_freq10')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PerturbRigidModelo1/20240722-0331_RUN_PerturbCalmRigidModelo1_N0550_AR110_freq10')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PerturbRigidModelo1/20240722-0331_RUN_PerturbCalmRigidModelo1_N0575_AR115_freq10')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PerturbRigidModelo1/20240722-0331_RUN_PerturbCalmRigidModelo1_N0600_AR120_freq10')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PerturbRigidModelo1/20240722-0331_RUN_PerturbCalmRigidModelo1_N0625_AR125_freq10')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PerturbRigidModelo1/20240722-0331_RUN_PerturbCalmRigidModelo1_N0750_AR150_freq10')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PerturbRigidModelo1/20240722-0331_RUN_PerturbCalmRigidModelo1_N0875_AR175_freq10')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PerturbRigidModelo1/20240722-0331_RUN_PerturbCalmRigidModelo1_N1000_AR200_freq10')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PerturbRigidModelo1/20240722-0331_RUN_PerturbCalmRigidModelo1_N1250_AR250_freq10')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/PerturbRigidModelo1/20240722-0331_RUN_PerturbCalmRigidModelo1_N1500_AR300_freq10')




for folder_path in pathlist:
    folder_path = Path(folder_path)

    last_frame_path = []
    possible_paths = []
    for pth in folder_path.glob('**/*.csv'):
        if 'lastFrame' in str(pth):
            last_frame_path.append(pth)
        else:
            possible_paths.append(pth)

    with open(last_frame_path[0],'r') as f:
        last_frame = float(f.readline())

    dta = np.loadtxt(last_frame_path[0],delimiter=',',skiprows=1)
    if len(possible_paths) == 0:
        print('No csv files found in the folder')        
        exit()
    elif len(possible_paths) > 1:
        print('Multiple csv files found in the folder')
        print(possible_paths)
        exit()
        
    pth = str(last_frame_path[0])
    file_id,surfix,num_rods,AR,datetime_str = parse_path_string(pth)
    
    fig,ax=plt.subplots(subplot_kw={'projection':'3d'})
    for i in range(num_rods):
        rr = dta[i,:].reshape(-1,3)
        ax.plot(rr[:,0],rr[:,1],rr[:,2])

    if not os.path.exists(last_frame_out_path):
        os.makedirs(last_frame_out_path)
    
    np.savetxt(f'{last_frame_out_path}/{file_id}_{surfix}_lastFrame{last_frame:.2f}.csv',dta,delimiter=' ')