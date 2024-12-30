#!/bin/sh

################################################################################
################################################################################

# This script finds the paths to the raw OpenEphys and SpikeGLX data for a given
# session. It writes these paths to files in a directory named
# `paths_to_task_data` in the om4 session data.

# Usage:  Run sbatch with argument for spikes data folder from
# get_spike_times.sh, e.g.
#   $ sbatch get_raw_data_folders.sh Perle/2022-06-01

################################################################################
################################################################################

echo -e '\n\n'
echo '########################################'
echo '##  STARTING get_raw_data_folders.sh  ##'
echo '########################################'
echo -e '\n\n'

# Base direcetories
OM4_BASE_DIR=/om4/group/jazlab/nwatters/multi_prediction
OM2_BASE_DIR=/om2/user/nwatters/multi_prediction
PHYS_PREPROCESSING_DIR=$OM2_BASE_DIR/phys_preprocessing

# Find raw data folder on /om4
SESSION=$1  # Argument passed in by user
echo "SESSION: $SESSION"
OM4_RAW_DATA_FOLDER=$OM4_BASE_DIR/phys_data/$SESSION/raw_data
echo "OM4_RAW_DATA_FOLDER: $OM4_RAW_DATA_FOLDER"

PATHS_DIR=$OM4_RAW_DATA_FOLDER/paths_to_task_data
mkdir $PATHS_DIR

################################################################################
#### OpenEphys
################################################################################

echo -e '\n'

# Find raw Open Ephys directory
RAW_OPEN_EPHYS_DIR=$OM4_RAW_DATA_FOLDER/open_ephys
echo "RAW_OPEN_EPHYS_DIR: $RAW_OPEN_EPHYS_DIR"
RAW_TASK_OPEN_EPHYS_DIR=$(find $RAW_OPEN_EPHYS_DIR/* -maxdepth 0 -name \
    '*-*-*_*-*-*')
echo "RAW_TASK_OPEN_EPHYS_DIR: $RAW_TASK_OPEN_EPHYS_DIR"

# Figure out unfiltered data directory. For many sessions, there were two record
# nodes, one recording raw (unfiltered) data and the other recording
# bandpass-filtered data. We use the unfiltered data for spike sorting, so this
# section of code finds the path to the unfiltered data for use later.
NO_OPEN_EPHYS_DATA=true
NODE_FILES_COMMAND="find $RAW_TASK_OPEN_EPHYS_DIR -name 'Record_Node_*'"
NUM_NODE_FILES=$(eval $NODE_FILES_COMMAND | wc -l)
if [ $NUM_NODE_FILES -eq 0 ]; then
    echo "Did not find record node files."
    if test -f "$RAW_TASK_OPEN_EPHYS_DIR/messages.events"; then
        echo 'Top-level data exists, so using that'
        UNFILTERED_DATA_DIR="$RAW_TASK_OPEN_EPHYS_DIR"
        NO_OPEN_EPHYS_DATA=false
    fi
else
    echo "Found record node files. Figuring out which has the unfiltered data."

    # Delete files not in record node directories
    echo "Deleting all files in $RAW_TASK_OPEN_EPHYS_DIR that are not in node \
        directories."
    find $RAW_TASK_OPEN_EPHYS_DIR -maxdepth 1 -type f -delete

    # Make tmp directory on om2
    echo "Making /om2 temporary directory."
    OM2_SESSION_DIR=$OM2_BASE_DIR/phys_data/$SESSION
    mkdir $OM2_SESSION_DIR
    OM2_TMP_DIR=$OM2_SESSION_DIR/tmp
    mkdir $OM2_TMP_DIR

    # Iterate through node directories and copy sample channels to om2
    find $RAW_TASK_OPEN_EPHYS_DIR -name 'Record_Node_*' -print0 | \
    while read -d $'\0' NODE_DIR; do
        NODE_NAME=$(basename "$NODE_DIR")
        OM2_NODE_DIR=$OM2_TMP_DIR/$NODE_NAME
        echo "Copying $NODE_DIR/100_15.continuous to $OM2_NODE_DIR"
        mkdir "$OM2_NODE_DIR"
        cp -r "$NODE_DIR"/100_15.continuous "$OM2_NODE_DIR"/
    done

    # Run Matlab find_pre_filter_data function
    echo "Running matlab find_pre_filter_data on $OM2_TMP_DIR"
    BASE_PWD=`pwd`
    module add mit/matlab/2021a
    cd $PHYS_PREPROCESSING_DIR/open_ephys
    MATLAB_COMMAND="find_pre_filter_data('${OM2_TMP_DIR}'); exit;"
    matlab -nodisplay -r $MATLAB_COMMAND
    cd $BASE_PWD

    # Find path to unfiltered data
    UNFILTERED_NODE_CMD="find $OM2_TMP_DIR -name 'unfiltered_node.txt'"
    NUM_UNFILTERED_NODES=$(eval $UNFILTERED_NODE_CMD | wc -l)
    if [ $NUM_UNFILTERED_NODES -eq 0 ]; then
        echo "No results for UNFILTERED_NODE_CMD $UNFILTERED_NODE_CMD"
    else
        UNFILTERED_NODE_FILE=$(eval $UNFILTERED_NODE_CMD)
        UNFILTERED_NODE=$(head -n 1 $UNFILTERED_NODE_FILE)
        echo "UNFILTERED_NODE: $UNFILTERED_NODE"
        UNFILTERED_DATA_DIR="$RAW_TASK_OPEN_EPHYS_DIR/$UNFILTERED_NODE"
        NO_OPEN_EPHYS_DATA=false
    fi

    # Remove om2 tmp directory
    echo "Removing $OM2_TMP_DIR"
    rm -r $OM2_TMP_DIR
fi

if $NO_OPEN_EPHYS_DATA; then
    echo "Found no Open Ephys data."
else
    OPEN_EPHYS_PATH_FILE=$PATHS_DIR/open_ephys
    rm $OPEN_EPHYS_PATH_FILE
    echo "writing $UNFILTERED_DATA_DIR to $PATHS_DIR/open_ephys"
    echo "$UNFILTERED_DATA_DIR" >> $OPEN_EPHYS_PATH_FILE
fi

################################################################################
#### OpenEphys Photodiode
################################################################################

echo -e '\n'

if ! $NO_OPEN_EPHYS_DATA; then
    # Run Matlab find_photodiode_file function
    echo "Running matlab find_photodiode_file in $UNFILTERED_DATA_DIR"
    BASE_PWD=`pwd`
    module add mit/matlab/2021a
    cd $PHYS_PREPROCESSING_DIR/open_ephys_utils
    MATLAB_COMMAND="find_photodiode_file('${UNFILTERED_DATA_DIR}'); exit;"
    matlab -nodisplay -r $MATLAB_COMMAND
    cd $BASE_PWD
fi

################################################################################
#### SpikeGLX
################################################################################

echo -e '\n'

# Get SpikeGLX data dir
echo 'GETTING SPIKEGLX DATA DIR'
BASE_PWD=`pwd`
RAW_SPIKEGLX_DATA_DIR=$(find $OM4_RAW_DATA_FOLDER/spikeglx -maxdepth 1 -name \
    '*elgar_task_g0' -o -name '*perle_g0')
echo "RAW_SPIKEGLX_DATA_DIR: ${RAW_SPIKEGLX_DATA_DIR}"
if [ -z "$RAW_SPIKEGLX_DATA_DIR" ]; then
    echo "Unable to find neuropixel data."
else
    echo "writing $RAW_SPIKEGLX_DATA_DIR to $PATHS_DIR/spikeglx"
    echo "$RAW_SPIKEGLX_DATA_DIR" >> $PATHS_DIR/spikeglx
fi

echo "DONE"