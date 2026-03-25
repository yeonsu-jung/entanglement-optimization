import os
from pathlib import Path
import re
# import numpy as np

def generate_sbatch(random_keys, AR):
    
    txt = f"""#!/bin/bash
#SBATCH -n 1                # Number of cores (-n)
#SBATCH -c 1                # Number of threads per core (-c)
#SBATCH -N 1                # Ensure that all cores are on one Node (-N)
#SBATCH -t 0-04:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p seas_compute             # Partition to submit to
#SBATCH --mem=1000           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o output_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e errors_%j.err  # File to which STDERR will be written, %j inserts jobid
#SBATCH --mail-type=END
#SBATCH --mail-user=jung@seas.harvard.edu

module load python
mamba activate simdata-analysis
# pip install jax

python protocols.py {random_keys[0]} {random_keys[1]} {random_keys[2]} {AR}

    """
    return txt

pathlist = []
pathlist.append('/n/home01/yjung/Github/dismech-rods-main/data/MaxEnt/91,22,12/MaxEnt91,22,12-N500-AR0500-Scale1.txt')

def parse_pathname(pathname):
    # dt_string = re.search(r'(\d{4}-\d{2}-\d{2}_\d{2})',pathname).group(1)
    AR = float(re.search(r'AR(\d+)',pathname).group(1))
    num_rods = int(re.search(r'N(\d+)',pathname).group(1))
    random_keys = re.search(r'(\d+),(\d+),(\d+)',pathname).group(0)
    return AR, num_rods, random_keys
exe_path = "/n/home01/yjung/Github/dismech-rods-main/runs/20241021-1137_COMPILE_initial_kick/disMech"

kick_amplitude = 1.
random_seed = 720
friction = 0.2
friction_list = [0.1,0.12,0.14,0.16,0.18,0.2]
root_dir = Path(__file__).parent.absolute()

import sys
# Check if enough arguments are provided
# if len(sys.argv) != 4:
#     print("Usage: python script.py <num1> <num2> <num3>")
#     sys.exit(1)
# Get the three numbers
# try:
#     num1 = int(sys.argv[1])  # Convert to float (or int if you need integers)
#     num2 = int(sys.argv[2])
#     num3 = int(sys.argv[3])
# except ValueError:
#     print("Please provide valid numbers as arguments.")
#     sys.exit(1)
# random_keys = [num1,num2,num3]

# generate random keys, random way
num_rods = 200
import random

for _ in range(1):
    # random_keys = [np.random.randint(0,100),np.random.randint(0,100),np.random.randint(0,100)]
    # random_keys without numpy
    # random_keys = [random.randint(0,100),random.randint(0,100),random.randint(0,100)]
    # random_keys = [29,19,70]
    # random_keys = [29,19,70]
    # random_keys = [37,178,56]
    # random_keys = [919,461,568]
    random_keys = [36,298,312]
    

    timestamp = os.popen('date +"%Y%m%d-%H%M"').read().strip()

    for AR in [10,25,50,100,150,200,300,500,1000]:
    # for AR in [10,1000]:
        run_id = f"{timestamp}_RUN_protocol_AR{AR}_N{num_rods}_randomkeys{random_keys[0]},{random_keys[1]},{random_keys[2]}"

        # create a folder
        os.makedirs(f"runs/{run_id}", exist_ok=False)

        # copy exe file to the folder
        os.system(f"cp protocols.py runs/{run_id}")
        os.system(f"cp optimization.py runs/{run_id}")
        os.system(f"cp visualizations.py runs/{run_id}")
        os.system(f"cp transforms.py runs/{run_id}")
        os.system(f"cp utils.py runs/{run_id}")
        os.system(f"cp potentials.py runs/{run_id}")

        # copy this file to "log_files"
        os.system(f"cp run.py log_files/{run_id}_run.py")
        sbatch_txt = generate_sbatch(random_keys,AR)

        # write the sbatch file
        with open(f"runs/{run_id}/Sbatch.sh",'w') as f:
            f.write(sbatch_txt)

        # cd to the folder and submit the job
        os.chdir(f"runs/{run_id}")
        os.system("sbatch Sbatch.sh")

        os.chdir(root_dir)
        print(f"Created run: {run_id}")