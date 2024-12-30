#!/bin/sh

# Copy spike data from om2 to om4 and clean up session directories

#SBATCH -o ./slurm_logs/%A.out
#SBATCH -t 06:00:00
# # SBATCH -t 00:20:00
#SBATCH -n 1
#SBATCH --mem-per-cpu 64G
#SBATCH --mail-type=NONE
#SBATCH --mail-user=nwatters@mit.edu
#SBATCH --partition=jazayeri

echo -e '\n\n'
echo '###########################################'
echo '##  STARTING main_copy_spikes_to_om4.sh  ##'
echo '###########################################'
echo -e '\n\n'

SESSION=$1  # Argument passed in by user. Should be a date string.
echo "SESSION: $SESSION"
if [ -z "$SESSION" ]; then
    echo "No session specified, exiting."
    exit
fi

OM4_BASE_DIR=/om4/group/jazlab/nwatters/multi_prediction
OM2_BASE_DIR=/om2/user/nwatters/multi_prediction

################################################################################
#### Copy spikes to om4
################################################################################

OM4_SESSION_DIR=$OM4_BASE_DIR/phys_data/$SESSION
OM2_SESSION_DIR=$OM2_BASE_DIR/phys_data/$SESSION
OM2_SPIKES_DIR=$OM2_SESSION_DIR/spikes

echo "Copying $OM2_SPIKES_DIR to $OM4_SESSION_DIR/"
rm -r $OM4_BASE_DIR/phys_data/$SESSION/spikes
cp -r $OM2_SPIKES_DIR $OM4_SESSION_DIR/

echo "Removing $OM4_SESSION_DIR/old_stuff"
rm -r $OM4_SESSION_DIR/old_stuff

echo "Removing $OM2_SESSION_DIR/old_stuff"
rm -r $OM2_SESSION_DIR/old_stuff

echo "DONE with main_copy_spikes_to_om4.sh"