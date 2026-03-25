import os
from pathlib import Path
import re
# import numpy as np

def generate_sbatch():
    
    txt = f"""#!/bin/bash
#SBATCH -n 1                # Number of cores (-n)
#SBATCH -c 1                # Number of threads per core (-c)
#SBATCH -N 1                # Ensure that all cores are on one Node (-N)
#SBATCH -t 0-24:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p seas_compute             # Partition to submit to
#SBATCH --mem=1000           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o output_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e errors_%j.err  # File to which STDERR will be written, %j inserts jobid
#SBATCH --mail-type=END
#SBATCH --mail-user=jung@seas.harvard.edu

module load python
mamba activate simdata-analysis
# pip install jax

python run_packing.py {ASPECT_RATIO} {RANDOM_SEED}

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
# friction_list = [0.1,0.12,0.14,0.16,0.18,0.2]
root_dir = Path(__file__).parent.absolute()

import sys


# NUM_RODS = 200
# ASPECT_RATIO = 100
# RANDOM_KEYS = 0
# NUM_RODS = int(sys.argv[1])
# TOTAL_STEPS = int(sys.argv[4])
ASPECT_RATIO = int(sys.argv[1])
RANDOM_SEED = int(sys.argv[2])


import random

for _ in range(1):    

    timestamp = os.popen('date +"%Y%m%d-%H%M"').read().strip()

    for _ in [0]:
        # run_id = f"{timestamp}_RUN_nudgeprotocol_test"
        run_id = f"{timestamp}_RUN_fast_packing_AR{ASPECT_RATIO}_{RANDOM_SEED}"


        # create a folder
        os.makedirs(f"runs/{run_id}", exist_ok=True)

        # copy exe file to the folder
        os.system(f"cp core/protocols.py runs/{run_id}")
        os.system(f"cp core/optimization.py runs/{run_id}")
        os.system(f"cp core/visualizations.py runs/{run_id}")
        os.system(f"cp core/transforms.py runs/{run_id}")
        os.system(f"cp core/utils.py runs/{run_id}")
        os.system(f"cp core/potentials.py runs/{run_id}")
        os.system(f"cp core/potentials.py runs/{run_id}")
        os.system(f"cp core/fast_packing02.py runs/{run_id}/run_packing.py")

        # copy this file to "log_files"
        os.system(f"cp run.py log_files/{run_id}_run.py")
        sbatch_txt = generate_sbatch()

        # write the sbatch file
        with open(f"runs/{run_id}/Sbatch.sh",'w') as f:
            f.write(sbatch_txt)

        # cd to the folder and submit the job
        os.chdir(f"runs/{run_id}")
        os.system("sbatch Sbatch.sh")

        os.chdir(root_dir)
        print(f"Created run: {run_id}")