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

# This script copies preprocessed data from om4/phys_data to om4/data_processed.

################################################################################
################################################################################

echo -e '\n\n'
echo '###############################################'
echo '##  STARTING copy_data_to_data_processed.sh  ##'
echo '###############################################'
echo -e '\n\n'

# Base directories
SOURCE_DIR=/om4/group/jazlab/nwatters/multi_prediction/phys_data/Elgar
TARGET_DIR=/om4/group/jazlab/nwatters/multi_prediction/data_processed/Elgar

# mkdir -p $TARGET_DIR

# SESSIONS=$(find $SOURCE_DIR -maxdepth 1)

SESSIONS=$(find $SOURCE_DIR -maxdepth 1 \
    -name '**2022-10-08' \
    -o -name '**2022-10-1?' \
)

echo "SESSIONS: ${SESSIONS}"

for x in $SESSIONS; do

    # Extract session date, set up directories
    echo -e '\n'
    echo "Processing session $x"
    SESSION_DATE=$(basename $x)
    echo "SESSION_DATE: $SESSION_DATE"
    SOURCE_SESSION_DIR=$SOURCE_DIR/$SESSION_DATE
    TARGET_SESSION_DIR=$TARGET_DIR/$SESSION_DATE
    # mkdir $TARGET_SESSION_DIR

    # Copy physiology_metadata.json
    echo "Copying $SOURCE_SESSION_DIR/physiology_metadata.json to $TARGET_SESSION_DIR/"
    cp -r $SOURCE_SESSION_DIR/physiology_metadata.json $TARGET_SESSION_DIR/
    
    # Copy trial_structure
    echo "Copying $SOURCE_SESSION_DIR/trial_structure to $TARGET_SESSION_DIR/"
    cp -r $SOURCE_SESSION_DIR/trial_structure $TARGET_SESSION_DIR/

    # Copy spike_sorting
    echo "Copying $SOURCE_SESSION_DIR/spike_sorting to $TARGET_SESSION_DIR/"
    SOURCE_SPIKE_SORTING=$SOURCE_SESSION_DIR/spike_sorting
    for spike_sorting_probe in $(find $SOURCE_SPIKE_SORTING -maxdepth 1); do
        PROBE_NAME=$(basename $spike_sorting_probe)
        TARGET_SPIKE_SORTING_PROBE=$TARGET_SESSION_DIR/spike_sorting/$PROBE_NAME
        mkdir -p $TARGET_SPIKE_SORTING_PROBE
        cp $spike_sorting_probe/sample_rate $TARGET_SPIKE_SORTING_PROBE/
        if [[ $PROBE_NAME == *"v_probe"* ]]; then
            echo "V-Probe"
            cp -r $spike_sorting_probe/*_curated $TARGET_SPIKE_SORTING_PROBE/
        fi
        if [[ $PROBE_NAME == *"np"* ]]; then
            echo "Neuropixel"
            cp -r $spike_sorting_probe/ks_3_output_* $TARGET_SPIKE_SORTING_PROBE/
        fi
        # Remove cluster_plots
        rm -r $TARGET_SPIKE_SORTING_PROBE/**/cluster_plots
    done

    echo "Done with session $x"
done

rm -r $TARGET_DIR/Elgar

echo -e '\n\n'
echo 'DONE WITH EVERYTHING'
