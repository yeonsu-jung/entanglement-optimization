import os

pathlist = []
# pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240711-0441_COMPILE_gripper_2')
# pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240711-1321_COMPILE_fric_worm_1')
# pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240711-1326_COMPILE_fric_worm_2')
# pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240711-1329_COMPILE_fric_worm_3')
# pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240711-1330_COMPILE_fric_worm_4')
# pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240711-1331_COMPILE_fric_worm_5')
# pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240711-2058_COMPILE_metal_nest')
# pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240711-0104_COMPILE_gripper_1')

# pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240714-1524_RUN_worm_3_20x_scale_relaxed.txt')
# pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240714-1524_RUN_worm_4_20x_scale_relaxed.txt')
# pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240714-1524_RUN_worm_2_20x_scale_relaxed.txt')
# pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240714-1524_RUN_worm_1_20x_scale_relaxed.txt')

# pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240712-0127_COMPILE_gripper_1')
# pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240712-0133_COMPILE_gripper_2')

pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240722-0331_RUN_PerturbCalmRigidModelo1_N0125_AR025_freq10')
pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240722-0331_RUN_PerturbCalmRigidModelo1_N0250_AR050_freq10')
pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240722-0331_RUN_PerturbCalmRigidModelo1_N0300_AR060_freq10')
pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240722-0331_RUN_PerturbCalmRigidModelo1_N0350_AR070_freq10')
pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240722-0331_RUN_PerturbCalmRigidModelo1_N0375_AR075_freq10')
pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240722-0331_RUN_PerturbCalmRigidModelo1_N0400_AR080_freq10')
pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240722-0331_RUN_PerturbCalmRigidModelo1_N0450_AR090_freq10')
pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240722-0331_RUN_PerturbCalmRigidModelo1_N0500_AR100_freq10')
pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240722-0331_RUN_PerturbCalmRigidModelo1_N0525_AR105_freq10')
pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240722-0331_RUN_PerturbCalmRigidModelo1_N0550_AR110_freq10')
pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240722-0331_RUN_PerturbCalmRigidModelo1_N0575_AR115_freq10')
pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240722-0331_RUN_PerturbCalmRigidModelo1_N0600_AR120_freq10')
pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240722-0331_RUN_PerturbCalmRigidModelo1_N0875_AR175_freq10')
pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240722-0331_RUN_PerturbCalmRigidModelo1_N0625_AR125_freq10')
pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240722-0331_RUN_PerturbCalmRigidModelo1_N0750_AR150_freq10')
pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240722-0331_RUN_PerturbCalmRigidModelo1_N1250_AR250_freq10')
pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240722-0331_RUN_PerturbCalmRigidModelo1_N1000_AR200_freq10')
pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240722-0331_RUN_PerturbCalmRigidModelo1_N1500_AR300_freq10')
pathlist.append('/n/home01/yjung/Github/dismech-rods-main/runs/20240721-2233_RUN_PerturbCalmRigidModelo1_N0300_AR060_freq100')






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