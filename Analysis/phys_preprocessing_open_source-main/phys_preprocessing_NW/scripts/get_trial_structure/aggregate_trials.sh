#!/bin/sh

################################################################################
################################################################################

# This script extracts trials from MOOG, MWorks, Open Ephys, and SpikeGLX. It
# infers the trial number, start time, and time of each task phase for every
# trial, using the sync variable logs.

# Usage:  Run sbatch with argument for spikes data folder from
# get_spike_times.sh, e.g.
#   $ sbatch aggregate_trials.sh Perle/2022-06-01

################################################################################
################################################################################

echo -e '\n\n'
echo '####################################'
echo '##  STARTING aggregate_trials.sh  ##'
echo '####################################'
echo -e '\n\n'

################################################################################
#### Priliminaries: Defining directory paths
################################################################################

BASE_PWD=`pwd`

# Base directories
OM2_BASE_DIR=/om2/user/nwatters/multi_prediction
PHYS_PREPROCESSING_DIR=$OM2_BASE_DIR/phys_preprocessing

# Get session
SESSION=$1  # Argument passed in by user
echo "SESSION: $SESSION"

OM2_SESSION_FOLDER=$OM2_BASE_DIR/phys_data/$SESSION
TRIALS_DIR=$OM2_SESSION_FOLDER/trial_structure
echo "TRIALS_DIR: $TRIALS_DIR"

################################################################################
#### Aggregate trials across data aquisition streams
################################################################################

echo -e '\n'

# Activate conda to run python
source /home/nwatters/.bashrc
conda activate phys_analysis

### Run trial aggregation

echo -e '\n'
echo 'RUNNING TRIAL AGGREGATION'
echo "PHYS_PREPROCESSING_DIR: $PHYS_PREPROCESSING_DIR"
cd $PHYS_PREPROCESSING_DIR/compute_trials
python aggregate_trials.py "$TRIALS_DIR"
cd $BASE_PWD