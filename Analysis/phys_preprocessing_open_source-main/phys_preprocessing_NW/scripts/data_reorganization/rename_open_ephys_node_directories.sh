#!/bin/sh

################################################################################
################################################################################

# This script removes spaces from raw OpenEphys data paths.

# Usage:  Run sbatch with argument for spikes data folder from
# get_spike_times.sh, e.g.
#   $ sbatch rename_open_ephys_node_directories.sh Perle/2022-06-01

################################################################################
################################################################################

echo -e '\n\n'
echo '#############################################'
echo '##  STARTING rename_open_ephys_node_directories.sh  ##'
echo '#############################################'
echo -e '\n\n'

# Base direcetories
OM4_BASE_DIR=/om4/group/jazlab/nwatters/multi_prediction

# Find raw data folder on /om4
SESSION=$1  # Argument passed in by user
echo "SESSION: $SESSION"
OM4_RAW_DATA_FOLDER=$OM4_BASE_DIR/phys_data/$SESSION/raw_data
echo "OM4_RAW_DATA_FOLDER: $OM4_RAW_DATA_FOLDER"

################################################################################
#### OpenEphys
################################################################################

echo -e '\n'

# Find raw Open Ephys directory
RAW_OPEN_EPHYS_DIR=$OM4_RAW_DATA_FOLDER/open_ephys
echo "RAW_OPEN_EPHYS_DIR: $RAW_OPEN_EPHYS_DIR"

# Iterate through raw data directories
find $RAW_OPEN_EPHYS_DIR -mindepth 1 -maxdepth 1 -print0 |\
while read -d $'\0' DATA_DIR; do
    echo "DATA_DIR: $DATA_DIR"

    # Iterate through node directories and copy sample channels to om2
    find $DATA_DIR -name 'Record Node *' -mindepth 1 -maxdepth 1 -print0 | \
    while read -d $'\0' NODE_DIR; do
        echo "Found NODE_DIR $NODE_DIR"
        NODE_DIR=$(basename "$NODE_DIR")
        NODE_NUMBER=${NODE_DIR#"Record Node "}
        echo "NODE_NUMBER: $NODE_NUMBER"
        NEW_NODE_DIR='Record_Node_'$NODE_NUMBER
        OLD_NODE_PATH=$DATA_DIR/$NODE_DIR
        NEW_NODE_PATH=$DATA_DIR/$NEW_NODE_DIR
        echo "Moving $OLD_NODE_PATH to $NEW_NODE_PATH"
        mv "$OLD_NODE_PATH" "$NEW_NODE_PATH"
    done
done

echo "DONE"