# %%
import numpy as np
from matplotlib import pyplot as plt
import re
import os
from data_io import import_all_log, parse_path_string
from fields import get_local_fields_at_a_point
from pathlib import Path



last_frame_out_path = f'/Users/yeonsu/GitHub/dismech-rods-main/data/EntangleSoftModelo1/'

pathlist = []
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/20240628-1333_RUN_EntangleSoftModelo1_N0125_AR025')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/20240628-1416_RUN_EntangleSoftModelo1_N0375_AR075')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/20240628-1416_RUN_EntangleSoftModelo1_N0500_AR100')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/20240628-1416_RUN_EntangleSoftModelo1_N0625_AR125')
pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/20240628-1416_RUN_EntangleSoftModelo1_N1000_AR200')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/20240620-0039_RUN_CalmEEModelo1_N1500_AR300')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/20240620-0039_RUN_CalmEEModelo1_N1250_AR250')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/20240620-0039_RUN_CalmEEModelo1_N1000_AR200')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/20240620-0039_RUN_CalmEEModelo1_N0875_AR175')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/20240620-0039_RUN_CalmEEModelo1_N0750_AR150')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/20240620-0039_RUN_CalmEEModelo1_N0625_AR125')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/20240620-0039_RUN_CalmEEModelo1_N0600_AR120')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/20240620-0039_RUN_CalmEEModelo1_N0575_AR115')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/20240620-0039_RUN_CalmEEModelo1_N0550_AR110')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/20240620-0039_RUN_CalmEEModelo1_N0525_AR105')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/20240620-0039_RUN_CalmEEModelo1_N0500_AR100')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/20240620-0039_RUN_CalmEEModelo1_N0450_AR090')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/20240620-0039_RUN_CalmEEModelo1_N0400_AR080')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/20240620-0039_RUN_CalmEEModelo1_N0375_AR075')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/20240620-0039_RUN_CalmEEModelo1_N0350_AR070')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/20240620-0039_RUN_CalmEEModelo1_N0300_AR060')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/20240620-0039_RUN_CalmEEModelo1_N0250_AR050')
# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/20240619-2353_RUN_CalmEEModelo1_N0125_AR025')

# pathlist.append('/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/CalmModelo1/20240619-2353_RUN_CalmEEModelo1_N0125_AR025')



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