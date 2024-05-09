import os


pathlist = []
pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240507-1614_RUN_Flynn,N300_AR300_mu0.5_visc0.0_amp1.0')
pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240507-1613_RUN_Flynn,N300_AR300_mu0.4_visc0.0_amp1.0')
pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240507-1613_RUN_Flynn,N300_AR300_mu0.3_visc0.0_amp1.0')
pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240507-1613_RUN_Flynn,N300_AR300_mu0.2_visc0.0_amp1.0')
pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240507-1613_RUN_Flynn,N300_AR200_mu0.5_visc0.0_amp1.0')
pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240507-1613_RUN_Flynn,N300_AR200_mu0.3_visc0.0_amp1.0')
pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240507-1613_RUN_Flynn,N300_AR200_mu0.4_visc0.0_amp1.0')
pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240507-1613_RUN_Flynn,N300_AR125_mu0.5_visc0.0_amp1.0')
pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240507-1613_RUN_Flynn,N300_AR200_mu0.2_visc0.0_amp1.0')
pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240507-1613_RUN_Flynn,N300_AR125_mu0.4_visc0.0_amp1.0')
pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240507-1613_RUN_Flynn,N300_AR125_mu0.2_visc0.0_amp1.0')
pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240507-1613_RUN_Flynn,N300_AR125_mu0.3_visc0.0_amp1.0')
pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240507-1613_RUN_Flynn,N300_AR100_mu0.5_visc0.0_amp1.0')
pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240507-1613_RUN_Flynn,N300_AR100_mu0.4_visc0.0_amp1.0')
pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240507-1613_RUN_Flynn,N300_AR100_mu0.3_visc0.0_amp1.0')
pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240507-1613_RUN_Flynn,N300_AR100_mu0.2_visc0.0_amp1.0')
pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240507-1613_RUN_Flynn,N300_AR75_mu0.5_visc0.0_amp1.0')
pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240507-1613_RUN_Flynn,N300_AR75_mu0.4_visc0.0_amp1.0')
pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240507-1613_RUN_Flynn,N300_AR75_mu0.3_visc0.0_amp1.0')
pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240507-1613_RUN_Flynn,N300_AR75_mu0.2_visc0.0_amp1.0')
pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240507-1613_RUN_Flynn,N300_AR50_mu0.4_visc0.0_amp1.0')
pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240507-1613_RUN_Flynn,N300_AR50_mu0.5_visc0.0_amp1.0')
pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240507-1613_RUN_Flynn,N300_AR20_mu0.4_visc0.0_amp1.0')
pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240507-1613_RUN_Flynn,N300_AR50_mu0.3_visc0.0_amp1.0')
pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240507-1613_RUN_Flynn,N300_AR20_mu0.3_visc0.0_amp1.0')
pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240507-1613_RUN_Flynn,N300_AR20_mu0.2_visc0.0_amp1.0')
pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240507-1613_RUN_Flynn,N300_AR50_mu0.2_visc0.0_amp1.0')
pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240507-1613_RUN_Flynn,N300_AR20_mu0.5_visc0.0_amp1.0')

with open('downloaded_list.txt','a') as f:
    for pth in pathlist:
        f.write(f"{pth}\n") # TO DO: remove duplicates

datalist = []

with open('download.sh', 'w') as f:
    f.write(f"#!/bin/bash\n")
    f.write("sftp sftp://yjung@odyssey.rc.fas.harvard.edu <<EOT\n")

    for pth in pathlist:
            
        batch_info = pth.split('/')[-1].split(',')[0]
        date_time = batch_info.split('_')[0]
        batch_id = batch_info.split('_')[-1]
        
        sim_params_string = pth.split('/')[-1].split(',')[-1]
        datalist.append({'date_time': date_time, 'batch_id': batch_id, 'sim_params_string': sim_params_string})        
        
        data_root = "/Users/yeonsu/Data"
        local_path = f"{data_root}/{batch_id},/{date_time}/{sim_params_string}"
        
        print(local_path)
        
        os.makedirs(local_path, exist_ok=True)
        f.write(f"get -r {pth}/log_files/*.csv {local_path}/.\n")
        
    f.write('quit\n')
    f.write('EOT\n')