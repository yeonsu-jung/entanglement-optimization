import os

pathlist = []
pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20241208-1505_RUN_RandomKeys_3,1,2,720_Kick1.00_Friction0.18_N500_AR0500')
pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20241208-1506_RUN_RandomKeys_3,1,2,720_Kick1.00_Friction0.20_N500_AR0500')
pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20241208-1505_RUN_RandomKeys_3,1,2,720_Kick1.00_Friction0.16_N500_AR0500')
pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20241208-1505_RUN_RandomKeys_3,1,2,720_Kick1.00_Friction0.14_N500_AR0500')
pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20241208-1505_RUN_RandomKeys_3,1,2,720_Kick1.00_Friction0.12_N500_AR0500')
pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20241208-1505_RUN_RandomKeys_3,1,2,720_Kick1.00_Friction0.10_N500_AR0500')
pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20241208-1505_RUN_RandomKeys_456,514,148,720_Kick1.00_Friction0.20_N500_AR0500')
pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20241208-1505_RUN_RandomKeys_456,514,148,720_Kick1.00_Friction0.18_N500_AR0500')
pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20241208-1505_RUN_RandomKeys_456,514,148,720_Kick1.00_Friction0.16_N500_AR0500')
pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20241208-1505_RUN_RandomKeys_456,514,148,720_Kick1.00_Friction0.14_N500_AR0500')
pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20241208-1505_RUN_RandomKeys_456,514,148,720_Kick1.00_Friction0.12_N500_AR0500')
pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20241208-1505_RUN_RandomKeys_456,514,148,720_Kick1.00_Friction0.10_N500_AR0500')
pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20241208-1504_RUN_RandomKeys_95,78,32,720_Kick1.00_Friction0.20_N500_AR0500')
pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20241208-1504_RUN_RandomKeys_95,78,32,720_Kick1.00_Friction0.18_N500_AR0500')
pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20241208-1504_RUN_RandomKeys_95,78,32,720_Kick1.00_Friction0.16_N500_AR0500')
pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20241208-1504_RUN_RandomKeys_95,78,32,720_Kick1.00_Friction0.14_N500_AR0500')
pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20241208-1504_RUN_RandomKeys_95,78,32,720_Kick1.00_Friction0.12_N500_AR0500')
pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20241208-1504_RUN_RandomKeys_95,78,32,720_Kick1.00_Friction0.10_N500_AR0500')

# pathlist.append('/n/home01/yjung/Github/entanglement-optimization/analysis-data')

with open('downloaded_list.txt','a') as f:
    for pth in pathlist:
        f.write(f"{pth}\n") # TO DO: remove duplicates

datalist = []

with open('download.sh', 'w') as f:
    f.write(f"#!/bin/bash\n")
    f.write("sftp sftp://yjung@odyssey.rc.fas.harvard.edu <<EOT\n")

    for pth in pathlist:        
        folder_name = pth.split('/')[-1]
        data_root = "/Users/yeonsu/Data/steel-rods-xray-data"
        local_path = f"{data_root}/{folder_name}"
        print(local_path)

        f.write(f"get {pth}/adjacency_distance_scale0p98_threshold0p3_ij_score.pkl {local_path}/.\n")
        
        
    f.write('quit\n')
    f.write('EOT\n')