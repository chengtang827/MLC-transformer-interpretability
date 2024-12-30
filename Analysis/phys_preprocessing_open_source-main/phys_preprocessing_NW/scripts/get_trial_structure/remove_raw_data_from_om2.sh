#!/bin/sh

################################################################################
################################################################################

# This script removes raw data from om2 after trial extraction computation.

# Usage:  Run sbatch with argument for spikes data folder from
# get_spike_times.sh, e.g.
#   $ sbatch remove_raw_data_from_om2.sh Perle/2022-06-01

################################################################################
################################################################################

echo -e '\n\n'
echo '#############################################'
echo '##  STARTING remove_raw_data_from_om2.sh  ##'
echo '#############################################'
echo -e '\n\n'

################################################################################
#### Copy data
################################################################################

# Base directories
OM2_BASE_DIR=/om2/user/nwatters/multi_prediction

# Get session
SESSION=$1  # Argument passed in by user
echo "SESSION: $SESSION"

# Get session folder on /om2
OM2_SESSION_FOLDER=$OM2_BASE_DIR/phys_data/$SESSION

# Copy trial_structure data from /om2 to /om4
echo "Removing raw_data from $OM2_SESSION_FOLDER"
rm -r $OM2_SESSION_FOLDER/raw_data