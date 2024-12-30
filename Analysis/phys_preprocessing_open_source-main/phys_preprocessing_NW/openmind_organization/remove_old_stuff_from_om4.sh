#!/bin/sh

#SBATCH -o ./slurm_logs/%A.out
#SBATCH -t 04:00:00
#SBATCH -n 1
#SBATCH --mem-per-cpu 16G
#SBATCH --mail-type=NONE
#SBATCH --mail-user=nwatters@mit.edu
#SBATCH --partition=jazayeri

################################################################################
################################################################################

# This script removes old preprocessed data from om4.

################################################################################
################################################################################

echo -e '\n\n'
echo '#############################################'
echo '##  STARTING remove_old_stuff_from_om4.sh  ##'
echo '#############################################'
echo -e '\n\n'

################################################################################
#### Preliminaries: Defining directory paths
################################################################################

BASE_PWD=`pwd`

# Base directories
OM4_BASE_DIR=/om4/group/jazlab/nwatters/multi_prediction/phys_data/Elgar

rm -rf $OM4_BASE_DIR/**/old_stuff

echo 'DONE'
