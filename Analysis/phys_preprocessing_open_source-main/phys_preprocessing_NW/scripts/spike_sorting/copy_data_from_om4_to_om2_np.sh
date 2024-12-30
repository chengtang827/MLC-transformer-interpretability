#!/bin/sh

#SBATCH -o ./slurm_logs/%A.out
#SBATCH -t 03:00:00
#SBATCH -n 1
#SBATCH --mem-per-cpu 128G
#SBATCH --mail-type=NONE
#SBATCH --mail-user=nwatters@mit.edu
#SBATCH --partition=jazayeri

################################################################################
################################################################################

# This script copies data from om4 to om2 for trial structure computation.

# Usage:  Run sbatch with argument for spikes data folder from
# get_spike_times.sh, e.g.
#   $ sbatch copy_data_from_om4_to_om2.sh Perle/2022-06-01

################################################################################
################################################################################

echo -e '\n\n'
echo '#############################################'
echo '##  STARTING copy_data_from_om4_to_om2_np.sh  ##'
echo '#############################################'
echo -e '\n\n'

################################################################################
#### Priliminaries: Defining directory paths
################################################################################

BASE_PWD=`pwd`

# Base directories
OM4_BASE_DIR=/om4/group/jazlab/nwatters/multi_prediction
OM2_BASE_DIR=/om2/user/nwatters/multi_prediction

# Find raw data folder on /om4
SESSION=$1  # Argument passed in by user
echo "SESSION: $SESSION"

OM4_RAW_DATA_FOLDER=$OM4_BASE_DIR/phys_data/$SESSION/raw_data
echo "OM4_RAW_DATA_FOLDER: $OM4_RAW_DATA_FOLDER"

# Make raw data folder on /om2
OM2_SESSION_FOLDER=$OM2_BASE_DIR/phys_data/$SESSION
OM2_RAW_DATA_FOLDER=$OM2_SESSION_FOLDER/raw_data
rm -r $OM2_RAW_DATA_FOLDER
mkdir $OM2_RAW_DATA_FOLDER
echo "OM2_RAW_DATA_FOLDER: $OM2_RAW_DATA_FOLDER"

################################################################################
#### Copy relevant raw data from /om4 to /om2
################################################################################

echo -e '\n'

# copy spikeglx data to /om2
SPIKEGLX_RAW_DATA_PATH_FILE=$OM4_RAW_DATA_FOLDER/paths_to_task_data/spikeglx
if [ -z "$SPIKEGLX_RAW_DATA_PATH_FILE" ]; then
    echo "$SPIKEGLX_RAW_DATA_PATH_FILE does not exist"
else
    SPIKEGLX_RAW_DATA_PATH=$(head -n 1 $SPIKEGLX_RAW_DATA_PATH_FILE)
    echo "SPIKEGLX_RAW_DATA_PATH: $SPIKEGLX_RAW_DATA_PATH"
    OM2_SPIKEGLX_DIR=$OM2_RAW_DATA_FOLDER/spikeglx
    echo "Copying SpikeGLX action potential data to $OM2_SPIKEGLX_DIR"
    mkdir $OM2_SPIKEGLX_DIR
    cp -r $SPIKEGLX_RAW_DATA_PATH/*imec*/*.ap.* $OM2_SPIKEGLX_DIR
fi

echo "done"
