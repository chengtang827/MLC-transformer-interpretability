#!/bin/sh

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
echo '##  STARTING copy_data_from_om4_to_om2.sh  ##'
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
OM4_RAW_DATA_DIR=$OM4_BASE_DIR/phys_data/$SESSION/raw_data
echo "OM4_RAW_DATA_DIR: $OM4_RAW_DATA_DIR"

# Make raw data folder on /om2
OM2_SESSION_FOLDER=$OM2_BASE_DIR/phys_data/$SESSION
mkdir $OM2_SESSION_FOLDER
OM2_RAW_DATA_FOLDER=$OM2_SESSION_FOLDER/raw_data
rm -r $OM2_RAW_DATA_FOLDER
mkdir $OM2_RAW_DATA_FOLDER
echo "OM2_RAW_DATA_FOLDER: $OM2_RAW_DATA_FOLDER"

################################################################################
#### Copy relevant raw data from /om4 to /om2
################################################################################

echo -e '\n'

# copy behavior to /om2
echo "Copying behavior data $OM4_RAW_DATA_DIR/behavior to om2 \
    $OM2_RAW_DATA_FOLDER"
mkdir $OM2_RAW_DATA_FOLDER/behavior
cp -r $OM4_RAW_DATA_DIR/behavior/* $OM2_RAW_DATA_FOLDER/behavior/

OM4_PATHS_TO_TASK_DATA=$OM4_RAW_DATA_DIR/paths_to_task_data

# copy open_ephys event data to /om2
OPEN_EPHYS_RAW_DATA_PATH_FILE=$OM4_PATHS_TO_TASK_DATA/open_ephys
if [ -f "$OPEN_EPHYS_RAW_DATA_PATH_FILE" ]; then
    OPEN_EPHYS=true
    OPEN_EPHYS_RAW_DATA_PATH=$(head -n 1 $OPEN_EPHYS_RAW_DATA_PATH_FILE)
    echo "OPEN_EPHYS_RAW_DATA_PATH: $OPEN_EPHYS_RAW_DATA_PATH"
    OM2_OPEN_EPHYS_DIR=$OM2_RAW_DATA_FOLDER/open_ephys
    echo "Copying Open Ephys data to $OM2_OPEN_EPHYS_DIR"
    mkdir $OM2_OPEN_EPHYS_DIR
    cp -r "$OPEN_EPHYS_RAW_DATA_PATH"/all_channels* $OM2_OPEN_EPHYS_DIR
else
    OPEN_EPHYS=false
    echo "$OPEN_EPHYS_RAW_DATA_PATH_FILE does not exist"
fi

# copy open_ephys photodiode data to /om2
if $OPEN_EPHYS; then
    OPEN_EPHYS_PHOTODIODE_PATH_FILE=$OM4_PATHS_TO_TASK_DATA/open_ephys_photodiode_filename
    echo "OPEN_EPHYS_PHOTODIODE_PATH_FILE: $OPEN_EPHYS_PHOTODIODE_PATH_FILE"
    OPEN_EPHYS_PHOTODIODE_FILENAME=$(head -n 1 $OPEN_EPHYS_PHOTODIODE_PATH_FILE)
    OPEN_EPHYS_PHOTODIODE_FILE=$OPEN_EPHYS_RAW_DATA_PATH/$OPEN_EPHYS_PHOTODIODE_FILENAME
    echo "OPEN_EPHYS_PHOTODIODE_FILE: $OPEN_EPHYS_PHOTODIODE_FILE"
    OM2_OPEN_EPHYS_DIR=$OM2_RAW_DATA_FOLDER/open_ephys
    echo "Copying Open Ephys photodiode to $OM2_OPEN_EPHYS_DIR"
    cp -r "$OPEN_EPHYS_PHOTODIODE_FILE" $OM2_OPEN_EPHYS_DIR
    mv $OM2_OPEN_EPHYS_DIR/$OPEN_EPHYS_PHOTODIODE_FILENAME \
        $OM2_OPEN_EPHYS_DIR/photodiode.continuous
fi

# copy spikeglx event data to /om2
SPIKEGLX_RAW_DATA_PATH_FILE=$OM4_RAW_DATA_DIR/paths_to_task_data/spikeglx
if [ -f "$SPIKEGLX_RAW_DATA_PATH_FILE" ]; then
    SPIKEGLX_RAW_DATA_PATH=$(head -n 1 $SPIKEGLX_RAW_DATA_PATH_FILE)
    echo "SPIKEGLX_RAW_DATA_PATH: $SPIKEGLX_RAW_DATA_PATH"
    OM2_SPIKEGLX_DIR=$OM2_RAW_DATA_FOLDER/spikeglx
    echo "Copying SpikeGLX data to $OM2_SPIKEGLX_DIR"
    mkdir $OM2_SPIKEGLX_DIR
    cp -r $SPIKEGLX_RAW_DATA_PATH/*nidq* $OM2_SPIKEGLX_DIR
    cp -r $SPIKEGLX_RAW_DATA_PATH/*imec*/*ap.meta* $OM2_SPIKEGLX_DIR
else
    echo "$SPIKEGLX_RAW_DATA_PATH_FILE does not exist"
fi
