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

# This script processes open ephys data on om2.

# Usage:  Run sbatch with argument for spikes data folder from
# get_spike_times.sh, e.g.
#   $ sbatch copy_open_ephys_data_to_om4.sh Perle/2022-06-01

################################################################################
################################################################################

echo -e '\n\n'
echo '###############################################'
echo '##  STARTING copy_open_ephys_data_to_om4.sh  ##'
echo '###############################################'
echo -e '\n\n'

################################################################################
#### Preliminaries: Defining directory paths
################################################################################

BASE_PWD=`pwd`

# Base directories
OM4_BASE_DIR=/om4/group/jazlab/nwatters/multi_prediction
OM2_BASE_DIR=/om2/user/nwatters/multi_prediction

# Find raw data folder on /om4
SESSION=$1  # Argument passed in by user
echo "SESSION: $SESSION"

# Get om4 raw data folder
OM4_RAW_DATA_FOLDER=$OM4_BASE_DIR/phys_data/$SESSION/raw_data
echo "OM4_RAW_DATA_FOLDER: $OM4_RAW_DATA_FOLDER"

# Get open ephys raw data folder on /om2
OM2_SESSION_FOLDER=$OM2_BASE_DIR/phys_data/$SESSION
OM2_RAW_DATA_FOLDER=$OM2_SESSION_FOLDER/raw_data
OM2_RAW_OPEN_EPHYS_DATA_FOLDER=$OM2_RAW_DATA_FOLDER/open_ephys
echo "OM2_RAW_OPEN_EPHYS_DATA_FOLDER: $OM2_RAW_OPEN_EPHYS_DATA_FOLDER"

################################################################################
#### Copy v_probe data back to om4 raw data folder
################################################################################

echo -e '\n'

echo "Copying $OM2_RAW_DATA_FOLDER/v_probe* to $OM4_RAW_DATA_FOLDER"
cp -r $OM2_RAW_DATA_FOLDER/v_probe* $OM4_RAW_DATA_FOLDER
echo 'Done copying'
