#!/bin/sh

################################################################################
################################################################################

# This script copies data from om2 to om4 after trial extraction computation.

# Usage:  Run sbatch with argument for spikes data folder from
# get_spike_times.sh, e.g.
#   $ sbatch copy_data_from_om2_to_om4.sh Perle/2022-06-01

################################################################################
################################################################################

echo -e '\n\n'
echo '#############################################'
echo '##  STARTING copy_data_from_om2_to_om4.sh  ##'
echo '#############################################'
echo -e '\n\n'

################################################################################
#### Copy data
################################################################################

# Base directories
OM2_BASE_DIR=/om2/user/nwatters/multi_prediction
OM4_BASE_DIR=/om4/group/jazlab/nwatters/multi_prediction

# Get session
SESSION=$1  # Argument passed in by user
echo "SESSION: $SESSION"

# Find session folders on /om2 and /om4
OM2_SESSION_FOLDER=$OM2_BASE_DIR/phys_data/$SESSION
OM4_SESSION_FOLDER=$OM4_BASE_DIR/phys_data/$SESSION

# Copy trial_structure data from /om2 to /om4
echo "Copying trial_structure from $OM2_SESSION_FOLDER to $OM4_SESSION_FOLDER"
cp -r $OM2_SESSION_FOLDER/trial_structure $OM4_SESSION_FOLDER/