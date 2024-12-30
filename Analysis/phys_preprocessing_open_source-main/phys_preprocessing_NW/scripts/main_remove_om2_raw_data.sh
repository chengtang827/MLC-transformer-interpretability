#!/bin/sh

# Remove raw data from om2

#SBATCH -o ./slurm_logs/%A.out
#SBATCH -t 02:00:00
#SBATCH -n 1
#SBATCH --mem-per-cpu 1G
#SBATCH --mail-type=NONE
#SBATCH --mail-user=nwatters@mit.edu
#SBATCH --partition=jazayeri

echo -e '\n\n'
echo '############################################'
echo '##  STARTING main_remove_om2_raw_data.sh  ##'
echo '############################################'
echo -e '\n\n'

SESSION=$1  # Argument passed in by user. Should be a date string.
echo "SESSION: $SESSION"
if [ -z "$SESSION" ]; then
    echo "No session specified, exiting."
    exit
fi

OM2_BASE_DIR=/om2/user/nwatters/multi_prediction
OM4_BASE_DIR=/om4/group/jazlab/nwatters/multi_prediction

################################################################################
#### Copy spikes to om4
################################################################################

OM4_SESSION_DIR=$OM4_BASE_DIR/phys_data/$SESSION
OM2_SESSION_DIR=$OM2_BASE_DIR/phys_data/$SESSION
OM2_RAW_DATA_DIR=$OM2_SESSION_DIR/raw_data
OM2_SPIKE_SORTING_DIR=$OM2_SESSION_DIR/spike_sorting

echo "Copying $OM2_SPIKE_SORTING_DIR to $OM4_SESSION_DIR/"
rm -r $OM4_SESSION_DIR/spike_sorting
cp -r $OM2_SPIKE_SORTING_DIR $OM4_SESSION_DIR/

echo "Removing $OM2_RAW_DATA_DIR"
rm -r $OM2_RAW_DATA_DIR

echo "DONE with main_remove_om2_raw_data.sh"

