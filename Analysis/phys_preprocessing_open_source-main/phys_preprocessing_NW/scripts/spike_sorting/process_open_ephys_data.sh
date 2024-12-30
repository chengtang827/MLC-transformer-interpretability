#!/bin/sh

################################################################################
################################################################################

# This script processes open ephys data on om2.

# Usage:  Run sbatch with argument for spikes data folder from
# get_spike_times.sh, e.g.
#   $ sbatch process_open_ephys_data.sh Perle/2022-06-01

################################################################################
################################################################################

echo -e '\n\n'
echo '#############################################'
echo '##  STARTING process_open_ephys_data.sh  ##'
echo '#############################################'
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
#### Run Open Ephys processing
################################################################################

echo -e '\n'

# Create temporary matlab file with processing steps
echo 'CREATING MATLAB FILE'
NEW_UUID=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 8 | head -n 1)
tmp_fn=tmp$NEW_UUID
echo "cd ../open_ephys_utils" >> $tmp_fn.m
echo "open_ephys_dir = '$OM2_RAW_OPEN_EPHYS_DATA_FOLDER'" >> $tmp_fn.m
echo "prep_open_ephys(open_ephys_dir);" >> $tmp_fn.m
echo "exit;" >> $tmp_fn.m

# Run and then remove temporary matlab file
echo 'STARTING MATLAB'
module add mit/matlab/2021a
matlab -nodisplay -r "$tmp_fn"
rm -f $tmp_fn.m

################################################################################
#### Copy v_probe data back to om4 raw data folder
################################################################################

echo -e '\n'

echo "Copying $OM2_RAW_DATA_FOLDER/v_probe* to $OM4_RAW_DATA_FOLDER"
cp -r $OM2_RAW_DATA_FOLDER/v_probe* $OM4_RAW_DATA_FOLDER
echo 'Done copying'
