import os

pathlist = []
pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20250129-0419_COMPILE_logjam_test_N700_L1.5_R0.046875_AR16_mu0.0_U0.1_T100_G1.5')
pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20250129-0418_COMPILE_logjam_test_N500_L1.0_R0.03125_AR16_mu0.0_U0.1_T100_G1.5')
pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20250129-0408_COMPILE_logjam_test_N240_L3.0_R0.09375_AR16_mu0.2_U10.0_T100_G1.5')
pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20250129-0408_COMPILE_logjam_test_N240_L3.0_R0.09375_AR16_mu0.0_U1.0_T100_G1.5')
pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20250129-0409_COMPILE_logjam_test_N240_L3.0_R0.09375_AR16_mu0.0_U0.1_T100_G1.5')
pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20250129-0408_COMPILE_logjam_test_N240_L3.0_R0.09375_AR16_mu0.0_U10.0_T100_G1.5')

with open('downloaded_list_dismech.txt','a') as f:
    for pth in pathlist:
        f.write(f"{pth}\n") # TO DO: remove duplicates

datalist = []
with open('download.sh', 'w') as f:
    f.write(f"#!/bin/bash\n")
    f.write("sftp sftp://yjung@odyssey.rc.fas.harvard.edu <<EOT\n")
    for pth in pathlist:
        folder_name = pth.split('/')[-1]
        data_root = "/Users/yeonsu/Dropbox (Harvard University)/Data/from-cluster/"
        local_path = f"{data_root}/{folder_name}"
        print(local_path)
        os.makedirs(local_path, exist_ok=True)
        # f.write(f"get -r {pth}/log_files/*.csv {local_path}/.\n")
        # f.write(f"get {pth}/log_files/*.csv '{local_path}/.'\n")
        # f.write(f"get {pth}/params.yml '{local_path}/.'\n")
        # f.write(f"get {pth}/logjam_params.yml '{local_path}/.'\n")
        f.write(f"get {pth}/run_logjam_existing_bin.py '{local_path}/.'\n")
        
    f.write('quit\n')
    f.write('EOT\n')
    