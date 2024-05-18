import os

pathlist = []
pathlist.append('/n/home01/yjung/good-bye-matlab/steel-rods-xray-data/alpha38_epsilon10')
pathlist.append('/n/home01/yjung/good-bye-matlab/steel-rods-xray-data/alpha38_epsilon05')
pathlist.append('/n/home01/yjung/good-bye-matlab/steel-rods-xray-data/alpha200_epsilon10')
pathlist.append('/n/home01/yjung/good-bye-matlab/steel-rods-xray-data/alpha200_epsilon15')
pathlist.append('/n/home01/yjung/good-bye-matlab/steel-rods-xray-data/alpha200_epsilon05')
pathlist.append('/n/home01/yjung/good-bye-matlab/steel-rods-xray-data/alpha38_epsilon00')
pathlist.append('/n/home01/yjung/good-bye-matlab/steel-rods-xray-data/alpha200_epsilon00')
pathlist.append('/n/home01/yjung/good-bye-matlab/steel-rods-xray-data/alpha38_epsilon15')
pathlist.append('/n/home01/yjung/good-bye-matlab/steel-rods-xray-data/alpha100_epsilon15')
pathlist.append('/n/home01/yjung/good-bye-matlab/steel-rods-xray-data/alpha100_epsilon10')
pathlist.append('/n/home01/yjung/good-bye-matlab/steel-rods-xray-data/alpha100_epsilon05')
pathlist.append('/n/home01/yjung/good-bye-matlab/steel-rods-xray-data/alpha100_epsilon00')
pathlist.append('/n/home01/yjung/good-bye-matlab/steel-rods-xray-data/alpha66_epsilon10')
pathlist.append('/n/home01/yjung/good-bye-matlab/steel-rods-xray-data/alpha66_epsilon15')
pathlist.append('/n/home01/yjung/good-bye-matlab/steel-rods-xray-data/alpha66_epsilon05')
pathlist.append('/n/home01/yjung/good-bye-matlab/steel-rods-xray-data/alpha66_epsilon00')
pathlist.append('/n/home01/yjung/good-bye-matlab/steel-rods-xray-data/alpha76_epsilon15')
pathlist.append('/n/home01/yjung/good-bye-matlab/steel-rods-xray-data/alpha76_epsilon10')
pathlist.append('/n/home01/yjung/good-bye-matlab/steel-rods-xray-data/alpha76_epsilon05')
pathlist.append('/n/home01/yjung/good-bye-matlab/steel-rods-xray-data/alpha76_epsilon00')

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