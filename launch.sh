#!/bin/bash
# ==== ENTER INFO HERE =====

BATCH_ID=$1
FILENAME=$2

sleep 1

RUNS_FOLDER="runs"
STAMP=$(date +"%Y%m%d-%H%M%S")
RESULT_FOLDER="${RUNS_FOLDER}/${STAMP}_RUN_${BATCH_ID}"
# exit if the folder exists
if [ -d "${RESULT_FOLDER}" ]; then
    echo "Folder ${RESULT_FOLDER} already exists. Exiting."
    exit 1
fi
mkdir ${RESULT_FOLDER}

cp analyze_sim_dataset.py "${RESULT_FOLDER}/analyze_sim_dataset.py"
cp -r *.py "${RESULT_FOLDER}/"
cp /n/home01/yjung/Github/entanglement-optimization/filamentFields.cpython-312-x86_64-linux-gnu.so "${RESULT_FOLDER}/"

cp run.sh "${RESULT_FOLDER}/run.sh"

echo "#!/bin/bash
#SBATCH -n 4                # Number of cores (-n)
#SBATCH -N 1                # Ensure that all cores are on one Node (-N)
#SBATCH -t 0-24:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p intermediate      # Partition to submit to
#SBATCH --mem=16000           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o output_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e errors_%j.err  # File to which STDERR will be written, %j inserts jobid
#SBATCH --mail-type=END
#SBATCH --mail-user=jung@seas.harvard.edu

module load python
mamba activate simdata-analysis
time python analyze_sim_dataset.py ${BATCH_ID} ${FILENAME}" > "${RESULT_FOLDER}/Sbatch.sh"


cd "${RESULT_FOLDER}"
sbatch Sbatch.sh
