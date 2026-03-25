#!/bin/bash
#SBATCH -p gpu_test         # partition. Remember to change to a desired partition
#SBATCH --mem=8g        # memory in GB
#SBATCH --time=04:00:00 # time in HH:MM:SS
#SBATCH -c 1            # number of cores
#SBATCH --gres=gpu:1

set -o errexit -o nounset -o pipefail
MY_SCRATCH=$(TMPDIR=/scratch mktemp -d)
echo $MY_SCRATCH

#Obtain the tarball and untar it in $MY_SCRATCH location to obtain the
#executable, code, and run it using the provider of your choice
curl -Lk 'https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-x64' | tar -C $MY_SCRATCH -xzf -

#VSCODE_CLI_DISABLE_KEYCHAIN_ENCRYPT=1 $MY_SCRATCH/code tunnel user login --provider github
VSCODE_CLI_DISABLE_KEYCHAIN_ENCRYPT=1 $MY_SCRATCH/code tunnel user login --provider github

#Accept the license terms & launch the tunnel
$MY_SCRATCH/code tunnel --accept-server-license-terms --name cannon_gpu