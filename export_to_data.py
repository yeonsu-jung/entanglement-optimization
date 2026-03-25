# %%
import os
from pathlib import Path
import numpy as np


pathlist = []
pathlist.append('/n/home01/yjung/Github/entanglement-optimization/runs/20241210-1240_RUN_protocol_AR50_N200_randomkeys29,19,70')
pathlist.append('/n/home01/yjung/Github/entanglement-optimization/runs/20241210-1240_RUN_protocol_AR100_N200_randomkeys29,19,70')
pathlist.append('/n/home01/yjung/Github/entanglement-optimization/runs/20241210-1240_RUN_protocol_AR100_N200_randomkeys33,54,36')
pathlist.append('/n/home01/yjung/Github/entanglement-optimization/runs/20241210-1240_RUN_protocol_AR150_N200_randomkeys29,19,70')
pathlist.append('/n/home01/yjung/Github/entanglement-optimization/runs/20241210-1240_RUN_protocol_AR150_N200_randomkeys33,54,36')
pathlist.append('/n/home01/yjung/Github/entanglement-optimization/runs/20241210-1240_RUN_protocol_AR200_N200_randomkeys29,19,70')
pathlist.append('/n/home01/yjung/Github/entanglement-optimization/runs/20241210-1240_RUN_protocol_AR200_N200_randomkeys33,54,36')
pathlist.append('/n/home01/yjung/Github/entanglement-optimization/runs/20241210-1240_RUN_protocol_AR300_N200_randomkeys29,19,70')
pathlist.append('/n/home01/yjung/Github/entanglement-optimization/runs/20241210-1240_RUN_protocol_AR300_N200_randomkeys33,54,36')
pathlist.append('/n/home01/yjung/Github/entanglement-optimization/runs/20241210-1240_RUN_protocol_AR500_N200_randomkeys29,19,70')
pathlist.append('/n/home01/yjung/Github/entanglement-optimization/runs/20241210-1240_RUN_protocol_AR500_N200_randomkeys33,54,36')
pathlist.append('/n/home01/yjung/Github/entanglement-optimization/runs/20241210-1239_RUN_protocol_AR200_N200_randomkeys33,31,94')
pathlist.append('/n/home01/yjung/Github/entanglement-optimization/runs/20241210-1239_RUN_protocol_AR50_N200_randomkeys33,31,94')
pathlist.append('/n/home01/yjung/Github/entanglement-optimization/runs/20241210-1239_RUN_protocol_AR100_N200_randomkeys33,31,94')
pathlist.append('/n/home01/yjung/Github/entanglement-optimization/runs/20241210-1239_RUN_protocol_AR150_N200_randomkeys33,31,94')
pathlist.append('/n/home01/yjung/Github/entanglement-optimization/runs/20241210-1239_RUN_protocol_AR200_N200_randomkeys46,1,59')
pathlist.append('/n/home01/yjung/Github/entanglement-optimization/runs/20241210-1239_RUN_protocol_AR300_N200_randomkeys33,31,94')
pathlist.append('/n/home01/yjung/Github/entanglement-optimization/runs/20241210-1239_RUN_protocol_AR300_N200_randomkeys46,1,59')
pathlist.append('/n/home01/yjung/Github/entanglement-optimization/runs/20241210-1239_RUN_protocol_AR500_N200_randomkeys33,31,94')
pathlist.append('/n/home01/yjung/Github/entanglement-optimization/runs/20241210-1239_RUN_protocol_AR500_N200_randomkeys46,1,59')
pathlist.append('/n/home01/yjung/Github/entanglement-optimization/runs/20241210-1240_RUN_protocol_AR50_N200_randomkeys33,54,36')
pathlist.append('/n/home01/yjung/Github/entanglement-optimization/runs/20241210-1239_RUN_protocol_AR100_N200_randomkeys46,1,59')
pathlist.append('/n/home01/yjung/Github/entanglement-optimization/runs/20241210-1239_RUN_protocol_AR150_N200_randomkeys46,1,59')
pathlist.append('/n/home01/yjung/Github/entanglement-optimization/runs/20241210-1239_RUN_protocol_AR50_N200_randomkeys46,1,59')
pathlist.append('/n/home01/yjung/Github/entanglement-optimization/runs/20241210-1239_RUN_protocol_AR500_N200_randomkeys26,27,6')
pathlist.append('/n/home01/yjung/Github/entanglement-optimization/runs/20241210-1239_RUN_protocol_AR100_N200_randomkeys26,27,6')
pathlist.append('/n/home01/yjung/Github/entanglement-optimization/runs/20241210-1239_RUN_protocol_AR150_N200_randomkeys26,27,6')
pathlist.append('/n/home01/yjung/Github/entanglement-optimization/runs/20241210-1239_RUN_protocol_AR200_N200_randomkeys26,27,6')
pathlist.append('/n/home01/yjung/Github/entanglement-optimization/runs/20241210-1239_RUN_protocol_AR50_N200_randomkeys26,27,6')
pathlist.append('/n/home01/yjung/Github/entanglement-optimization/runs/20241210-1239_RUN_protocol_AR300_N200_randomkeys26,27,6')

from transforms import x_to_q, q_to_x
import re
# pth = pathlist[0]
for pth in pathlist:
    


    # random_keys = re.findall(r'\d+,\d+,\d+', pth.split('/')[-1])
    num_rods = re.findall(r'N\d+', pth.split('/')[-1])[0].split('N')[1]
    AR = re.findall(r'AR\d+', pth.split('/')[-1])[0].split('AR')[1]
    random_keys = re.search(r'(\d+),(\d+),(\d+)',pth).group(0)
    print(num_rods, AR, random_keys)
    

    # root_folder = f'/Users/yeonsu/GitHub/entanglement-optimization/results/{random_keys}'
    out_folder = f'data/{random_keys}'

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    # find q_relaxed.txt in the subfolder
    file_path = Path(pth).rglob('q_relaxed.txt')
    file_path = list(file_path)
    if len(file_path) == 0:
        print('No q_relaxed.txt found in the subfolder')
        continue
    else:
        file_path = file_path[0]
    
    tmp = np.loadtxt(file_path)
    out = q_to_x(tmp.reshape(-1,5))
    np.savetxt(out_folder + f'/MaxEnt{random_keys}-N{int(num_rods):03d}-AR{int(AR):04d}-Scale1.txt', out, delimiter=' ')

# # %%
# # list subfolders
# pathlist = []



# # %%
# from analysis_functions import parse_pathname
# from transforms import x_to_q, q_to_x
# # pathlist
# for pth in pathlist:
#     dt_string, AR, num_rods,random_keys = parse_pathname(pth)
#     tmp = np.loadtxt(pth + '/q_relaxed.txt')
#     out = q_to_x(tmp.reshape(-1,5))
#     np.savetxt(out_folder + f'/MaxEnt{random_keys}-N{num_rods:03d}-AR{int(AR):04d}-Scale1.txt', out, delimiter=' ')

# # %%


# # %%

