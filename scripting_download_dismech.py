import os


pathlist = []
pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240515-2032_RUN_xray_AR100_mu0.2_g1')
pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240515-0233_RUN_xray38_mu0.2')
pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240514-2131_RUN_xray38')
pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240514-2113_COMPILE_debug')
pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240514-2109_RUN_xray38')
pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240514-2053_COMPILE_')
pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240514-2048_RUN_XRAY_AR38')
pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240514-2045_RUN_XRAY_AR38')
pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240513-2045_COMPILE_')
pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240513-1338_RUN_mu0_g10_density70_floorStanding_XRAY100')
pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240513-1216_RUN_mu0_g10_density70_floorStanding_XRAY100')
pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240513-0429_RUN_mu0_g10_floorStanding_XRAY100')
pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240513-0429_RUN_mu0.2_g10_floorStanding_XRAY100')
pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240513-0353_COMPILE_random-kick-with-floor')
pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240513-0350_COMPILE_random-kick-with-floor')
pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240512-2315_RUN_Brownie_N300_AR200_mu1_amp0.1')
pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240512-2249_COMPILE_')

with open('downloaded_list.txt','a') as f:
    for pth in pathlist:
        f.write(f"{pth}\n") # TO DO: remove duplicates

datalist = []

with open('download.sh', 'w') as f:
    f.write(f"#!/bin/bash\n")
    f.write("sftp sftp://yjung@odyssey.rc.fas.harvard.edu <<EOT\n")

    for pth in pathlist:
            
        folder_name = pth.split('/')[-1]
        data_root = "/Users/yeonsu/Data/from_cluster"
        local_path = f"{data_root}/{folder_name}"
        
        print(local_path)        
        os.makedirs(local_path, exist_ok=True)
        # f.write(f"get -r {pth}/log_files/*.csv {local_path}/.\n")
        
        f.write(f"get {pth}/log_files/*.csv {local_path}/.\n")
        
        
    f.write('quit\n')
    f.write('EOT\n')