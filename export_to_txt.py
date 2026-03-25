# %%
import numpy as np

pathlist = []
pathlist.append('/n/home01/yjung/Github/entanglement-optimization/runs/20241209-1138_RUN_protocol_AR500_N200_randomkeys3,1,2')
pathlist.append('/n/home01/yjung/Github/entanglement-optimization/runs/20241209-1138_RUN_protocol_AR500_N200_randomkeys5,9,17')
pathlist.append('/n/home01/yjung/Github/entanglement-optimization/runs/20241209-1139_RUN_protocol_AR500_N200_randomkeys8,7,56')

# %%
# find data file



import os

def find_file_in_subfolders(root_folder, filename):
    for dirpath, dirnames, filenames in os.walk(root_folder):
        if filename in filenames:
            return os.path.join(dirpath, filename)
    return None

for root_folder in pathlist:
    filename = "q_relaxed.txt"
    file_path = find_file_in_subfolders(root_folder, filename)

    if file_path:
        print(f"File found: {file_path}")
    else:
        print("File not found.")
# %%


