import os

pathlist = []
pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240629-1605_RUN_EntangleSoftModelo1_N0125_AR025')
pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240629-1608_RUN_EntangleSoftModelo1_N0375_AR075')
pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240629-1608_RUN_EntangleSoftModelo1_N0500_AR100')
pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240629-1608_RUN_EntangleSoftModelo1_N1000_AR200')
pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240629-1608_RUN_EntangleSoftModelo1_N1500_AR300')

# /n/home01/yjung/Github/dismech-rods-main/runs/20240611-1139_RUN_HangEEModelo1_N0300_AR060
# /n/home01/yjung/Github/dismech-rods-main/runs/20240611-1139_RUN_HangEEModelo1_N0350_AR070
# /n/home01/yjung/Github/dismech-rods-main/runs/20240611-1139_RUN_HangEEModelo1_N0400_AR080
# /n/home01/yjung/Github/dismech-rods-main/runs/20240611-1139_RUN_HangEEModelo1_N0450_AR090

# /n/holylabs/LABS/mahadevan_lab/Users/yjung/rod-packing-data-archive/20240531-2228_RUN_JostleCarrotCake5_N1000_AR200_g0.5
# /n/holylabs/LABS/mahadevan_lab/Users/yjung/rod-packing-data-archive/20240531-2228_RUN_JostleCarrotCake5_N1500_AR300_g0.5

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
        
        f.write(f"get {pth}/log_files/*.csv '{local_path}/.'\n")
        
        
    f.write('quit\n')
    f.write('EOT\n')
    
    
    


# pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240526-2125_RUN_Chuck,N1000_AR500_mu0.2_visc0_boxsize4_freq10_amp0.1')
# pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240526-2125_RUN_Chuck,N1000_AR300_mu0.2_visc0_boxsize4_freq10_amp0.1')
# pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240526-2125_RUN_Chuck,N1000_AR200_mu0.2_visc0_boxsize4_freq10_amp0.1')
# pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240526-2125_RUN_Chuck,N1000_AR20_mu0.2_visc0_boxsize4_freq10_amp0.1')
# pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240526-2125_RUN_Chuck,N1000_AR100_mu0.2_visc0_boxsize4_freq10_amp0.1')
# pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240526-2125_RUN_Chuck,N1000_AR50_mu0.2_visc0_boxsize4_freq10_amp0.1')
# pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240526-2121_RUN_Chuck,N1000_AR300_mu0.2_visc0.5_boxsize4_freq10_amp0.1')
# pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240526-2121_RUN_Chuck,N1000_AR200_mu0.2_visc0.5_boxsize4_freq10_amp0.1')
# pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240526-2121_RUN_Chuck,N1000_AR100_mu0.2_visc0.5_boxsize4_freq10_amp0.1')
# pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240526-2121_RUN_Chuck,N1000_AR500_mu0.2_visc0.5_boxsize4_freq10_amp0.1')
# pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240526-2000_RUN_Chuck,N1000_AR300_mu0.2_visc0.5_boxsize4_freq10_amp0.1')
# pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240526-2000_RUN_Chuck,N1000_AR500_mu0.2_visc0.5_boxsize4_freq10_amp0.1')
# pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240526-2000_RUN_Chuck,N1000_AR200_mu0.2_visc0.5_boxsize4_freq10_amp0.1')
# pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240526-2000_RUN_Chuck,N1000_AR100_mu0.2_visc0.5_boxsize4_freq10_amp0.1')
# pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240526-1956_RUN_Chuck,N1000_AR50_mu0.2_visc0.5_boxsize4_freq10_amp0.1')
# pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240526-1945_RUN_Chuck,N1000_AR20_mu0.2_visc0.5_amp0.0')
# pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240526-1535_RUN_McFlurry_4_10_0.1_AR20_viscosity5')
# pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240526-1535_RUN_McFlurry_4_10_0.1_AR50_viscosity5')
# pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240526-1535_RUN_McFlurry_4_10_0.1_AR100_viscosity5')
# pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240526-1535_RUN_McFlurry_4_10_0.1_AR200_viscosity5')
# pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240526-1534_RUN_McFlurry_4_10_0.1_AR300_viscosity5')
# pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240526-1534_RUN_McFlurry_4_10_0.1_AR500_viscosity5')
# pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240526-1340_RUN_McFlurry_2_10_0.1_AR100_viscosity5')
# pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240526-1340_RUN_McFlurry_2_10_0.1_AR500_viscosity5')
# pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240526-1340_RUN_McFlurry_2_10_0.1_AR20_viscosity5')
# pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240526-0605_RUN_McFlurry_2_10_0.1_AR20')
# pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240526-0605_RUN_McFlurry_2_10_0.1_AR50')
# pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240526-0605_RUN_McFlurry_2_10_0.1_AR100')
# pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240526-0603_RUN_McFlurry_2_10_0.1_AR200')
# pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240526-0603_RUN_McFlurry_2_10_0.1_AR300')
# pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240526-0603_RUN_McFlurry_2_10_0.1_AR500')

# pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240527-0210_RUN_,N500_AR50_mu0.2_visc0_boxsize1.05_freq10_amp1e-4')
# pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240527-0209_RUN_,N500_AR300_mu0.2_visc0_boxsize1.05_freq10_amp1e-4')
# pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240527-0209_RUN_,N500_AR200_mu0.2_visc0_boxsize1.05_freq10_amp1e-4')
# pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240527-0209_RUN_,N500_AR500_mu0.2_visc0_boxsize1.05_freq10_amp1e-4')
# pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240527-0209_RUN_,N500_AR100_mu0.2_visc0_boxsize1.05_freq10_amp1e-4')

# pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240528-1557_RUN_CarrotCake2,N1500_AR300_mu0.2_visc0_boxsize4_freq10_amp0.05')
# pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240528-1557_RUN_CarrotCake2,N1000_AR200_mu0.2_visc0_boxsize4_freq10_amp0.05')
# pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240528-1557_RUN_CarrotCake2,N500_AR100_mu0.2_visc0_boxsize4_freq10_amp0.05')
# pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240528-1557_RUN_CarrotCake2,N250_AR50_mu0.2_visc0_boxsize4_freq10_amp0.05')

# pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240527-1934_RUN_CarrotCake2,N1500_AR300_mu0.2_visc0_boxsize0.5_freq10_amp0.05')
# pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240527-1934_RUN_CarrotCake2,N1000_AR200_mu0.2_visc0_boxsize0.5_freq10_amp0.05')
# pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240527-1934_RUN_CarrotCake2,N500_AR100_mu0.2_visc0_boxsize0.5_freq10_amp0.05')
# pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240527-1934_RUN_CarrotCake2,N250_AR50_mu0.2_visc0_boxsize0.5_freq10_amp0.05')


# pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240528-1716_RUN_JostleCarrotCake4,N1500_AR300_mu0.2_visc0_boxsize4_freq10_amp0.05')
# pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240528-1716_RUN_JostleCarrotCake4,N1000_AR200_mu0.2_visc0_boxsize4_freq10_amp0.05')
# pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240528-1716_RUN_JostleCarrotCake4,N500_AR100_mu0.2_visc0_boxsize4_freq10_amp0.05')
# pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240528-1716_RUN_JostleCarrotCake4,N250_AR50_mu0.2_visc0_boxsize4_freq10_amp0.05')

# pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240528-1714_RUN_EntangleCarrotCake4,N1500_AR300_mu0.2_visc0_boxsize0.5_freq10_amp0.05')
# pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240528-1714_RUN_EntangleCarrotCake4,N1000_AR200_mu0.2_visc0_boxsize0.5_freq10_amp0.05')
# pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240528-1714_RUN_EntangleCarrotCake4,N500_AR100_mu0.2_visc0_boxsize0.5_freq10_amp0.05')
# pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240528-1714_RUN_EntangleCarrotCake4,N250_AR50_mu0.2_visc0_boxsize0.5_freq10_amp0.05')


# pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240528-1714_RUN_EntangleCarrotCake4,N1500_AR300_mu0.2_visc0_boxsize0.5_freq10_amp0.05')
# pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240528-1714_RUN_EntangleCarrotCake4,N1000_AR200_mu0.2_visc0_boxsize0.5_freq10_amp0.05')
# pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240528-1714_RUN_EntangleCarrotCake4,N500_AR100_mu0.2_visc0_boxsize0.5_freq10_amp0.05')
# pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240528-1714_RUN_EntangleCarrotCake4,N250_AR50_mu0.2_visc0_boxsize0.5_freq10_amp0.05')

# pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240531-2228_RUN_JostleCarrotCake5_N0625_AR125_g0.5')
# pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240531-2228_RUN_JostleCarrotCake5_N0500_AR100_g0.5')
# pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240531-2228_RUN_JostleCarrotCake5_N0375_AR075_g0.5')
# pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240531-2228_RUN_JostleCarrotCake5_N0250_AR050_g0.5')
# pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240531-2228_RUN_JostleCarrotCake5_N0125_AR025_g0.5')

# pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240531-2227_RUN_EntangleCarrotCake5_N0625_AR125_g0.5')
# pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240531-2227_RUN_EntangleCarrotCake5_N0500_AR100_g0.5')
# pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240531-2227_RUN_EntangleCarrotCake5_N0375_AR075_g0.5')
# pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240531-2227_RUN_EntangleCarrotCake5_N0250_AR050_g0.5')
# pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240531-2227_RUN_EntangleCarrotCake5_N0125_AR025_g0.5')

# pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240531-2224_RUN_EntangleCarrotCake5_N0500-AR100')
# pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240531-2224_RUN_EntangleCarrotCake5_N0375-AR075')
# pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240531-2224_RUN_EntangleCarrotCake5_N0250-AR050')
# pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240531-2224_RUN_EntangleCarrotCake5_N0125-AR025')
# pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240531-2224_RUN_EntangleCarrotCake5_N0625-AR125')