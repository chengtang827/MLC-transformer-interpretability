#!/bin/sh

#SBATCH -o ./slurm_logs/%A.out
#SBATCH -t 04:00:00
#SBATCH -n 1
#SBATCH --mem-per-cpu 16G
#SBATCH --mail-type=NONE
#SBATCH --mail-user=nwatters@mit.edu
#SBATCH --partition=normal

################################################################################
################################################################################

# This script copies preprocessed data from om2 to om4.

################################################################################
################################################################################

echo -e '\n\n'
echo '##############################################'
echo '##  STARTING copy_processed_data_to_om4.sh  ##'
echo '##############################################'
echo -e '\n\n'

# Base directories
OM4_BASE_DIR=/om4/group/jazlab/nwatters/multi_prediction/phys_data/Perle
OM2_BASE_DIR=/om2/user/nwatters/multi_prediction/phys_data/Perle

SESSIONS=$(find $OM2_BASE_DIR -maxdepth 1)
echo "SESSIONS: ${SESSIONS}"

for x in $SESSIONS; do

    # Extract session date, set up directories
    echo -e '\n'
    echo "Processing session $x"
    SESSION_DATE=$(basename $x)
    echo "SESSION_DATE: $SESSION_DATE"
    OM4_SESSION_DIR=$OM4_BASE_DIR/$SESSION_DATE
    OM2_SESSION_DIR=$OM2_BASE_DIR/$SESSION_DATE

    # Move spikes, spike_sorting, and trial_structure to old_stuff in om4
    echo "Making directory $OM4_SESSION_DIR/old_stuff"
    mkdir -p $OM4_SESSION_DIR/old_stuff
    echo "Moving spikes to old_stuff"
    mv $OM4_SESSION_DIR/spikes $OM4_SESSION_DIR/old_stuff/
    echo "Moving spike_sorting to old_stuff"
    mv $OM4_SESSION_DIR/spike_sorting $OM4_SESSION_DIR/old_stuff/
    echo "Moving trial_structure to old_stuff"
    mv $OM4_SESSION_DIR/trial_structure $OM4_SESSION_DIR/old_stuff/

    # Copy spikes, spike_sorting, trial_structure, and physiology_metadata.json
    # from om2
    echo "Copying $OM2_SESSION_DIR/spikes to $OM4_SESSION_DIR/"
    cp -r $OM2_SESSION_DIR/spikes $OM4_SESSION_DIR/
    echo "Copying $OM2_SESSION_DIR/spike_sorting to $OM4_SESSION_DIR/"
    cp -r $OM2_SESSION_DIR/spike_sorting $OM4_SESSION_DIR/
    echo "Copying $OM2_SESSION_DIR/trial_structure to $OM4_SESSION_DIR/"
    cp -r $OM2_SESSION_DIR/trial_structure $OM4_SESSION_DIR/
    echo "Copying $OM2_SESSION_DIR/physiology_metadata.json to $OM4_SESSION_DIR/"
    cp -r $OM2_SESSION_DIR/physiology_metadata.json $OM4_SESSION_DIR/

    echo "Done with session $x"
done

echo -e '\n\n'
echo 'DONE WITH EVERYTHING'
