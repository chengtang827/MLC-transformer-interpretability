#!/bin/sh

################################################################################
################################################################################

# This script extracts trials from MOOG, MWorks, Open Ephys, and SpikeGLX. It
# infers the trial number, start time, and time of each task phase for every
# trial, using the sync variable logs. Finally, this script also collates the
# trials from all acquisitions into one file.

# Usage:  Run sbatch with argument for spikes data folder from
# get_spike_times.sh, e.g.
#   $ sbatch get_trial_structure.sh Perle/2022-06-01

################################################################################
################################################################################

echo -e '\n\n'
echo '######################################'
echo '##  STARTING extract_raw_trials.sh  ##'
echo '######################################'
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

# Find raw data folder on /om2
OM2_SESSION_FOLDER=$OM2_BASE_DIR/phys_data/$SESSION
OM2_RAW_DATA_FOLDER=$OM2_SESSION_FOLDER/raw_data
echo "OM2_RAW_DATA_FOLDER: $OM2_RAW_DATA_FOLDER"

################################################################################
#### Extract trials from MWorks, MOOG, Open Ephys, and SpikeGLX
################################################################################

echo -e '\n'

# Create directories where we're going to write the sync variables and trials
TRIALS_DIR=$OM2_SESSION_FOLDER/trial_structure
echo "TRIALS_DIR: $TRIALS_DIR"
mkdir $TRIALS_DIR
SYNC_EVENTS_DIR=$TRIALS_DIR/sync_events
echo "SYNC_EVENTS_DIR: $SYNC_EVENTS_DIR"
mkdir $SYNC_EVENTS_DIR
RAW_TRIALS_DIR=$TRIALS_DIR/raw_trials
echo "RAW_TRIALS_DIR: $RAW_TRIALS_DIR"
mkdir $RAW_TRIALS_DIR

# Activate conda to run python
source /home/nwatters/.bashrc
conda activate phys_analysis

#### Run MOOG trial-extractor

echo -e '\n'
echo 'RUNNING MOOG TRIAL EXTRACTION'
cd $PHYS_PREPROCESSING_DIR/compute_trials
python compute_moog_trials.py \
    "$OM2_RAW_DATA_FOLDER/behavior/moog" "$RAW_TRIALS_DIR"
cd $BASE_PWD

### Run MWorks trial-extractor

echo -e '\n'
echo 'RUNNING MWORKS TRIAL EXTRACTION'
MW_SYNC_EVENTS_DIR=$SYNC_EVENTS_DIR/mworks
mkdir $MW_SYNC_EVENTS_DIR
cd $PHYS_PREPROCESSING_DIR/compute_trials
python compute_mworks_trials.py \
    "$OM2_RAW_DATA_FOLDER/behavior/mworks" \
    "$RAW_TRIALS_DIR" \
    "$MW_SYNC_EVENTS_DIR"
cd $BASE_PWD

#### Run OpenEphys trial-extractor

echo -e '\n'

# Extract sync signal events
OPEN_EPHYS_RAW_DIR=$OM2_RAW_DATA_FOLDER/open_ephys
if [ -d "$OPEN_EPHYS_RAW_DIR" ]; then
    echo 'EXTRACTING OPEN EPHYS SYNC EVENTS'
    OPEN_EPHYS_SYNC_EVENTS_DIR=$SYNC_EVENTS_DIR/open_ephys
    mkdir $OPEN_EPHYS_SYNC_EVENTS_DIR
    cd $OM2_BASE_DIR/phys_preprocessing/open_ephys_utils
    module add mit/matlab/2021a
    MATLAB_COMMAND_SYNC_EVENTS="extract_sync_events('$OPEN_EPHYS_RAW_DIR'); exit;"
    matlab -nodisplay -r $MATLAB_COMMAND_SYNC_EVENTS
    MATLAB_COMMAND_PHOTODIODE="extract_photodiode('$OPEN_EPHYS_RAW_DIR'); exit;"
    matlab -nodisplay -r $MATLAB_COMMAND_PHOTODIODE
    cd $BASE_PWD
fi

# Get open-ephys trials
if [ -d "$OPEN_EPHYS_SYNC_EVENTS_DIR" ]; then
    echo -e '\n'
    echo 'RUNNING OPEN EPHYS TRIAL EXTRACTION'
    
    # Write time offset
    RECORD_START_TIME=$(head -n 1 \
        $OPEN_EPHYS_SYNC_EVENTS_DIR/photodiode_times.csv)
    echo "Writing RECORD_START_TIME $RECORD_START_TIME to \
        $OPEN_EPHYS_SYNC_EVENTS_DIR/record_start_time"
    echo $RECORD_START_TIME > $OPEN_EPHYS_SYNC_EVENTS_DIR/record_start_time

    cd $PHYS_PREPROCESSING_DIR/compute_trials
    python compute_open_ephys_trials.py \
        "$OPEN_EPHYS_SYNC_EVENTS_DIR" "$RAW_TRIALS_DIR"
    cd $BASE_PWD
fi

#### Run SpikeGLX trial-extractor

echo -e '\n'

SPIKEGLX_RAW_DIR=$OM2_RAW_DATA_FOLDER/spikeglx
if [ -d "$SPIKEGLX_RAW_DIR" ]; then
    echo -e '\n'
    echo 'RUNNING SPIKEGLX TRIAL EXTRACTION'
    SPIKEGLX_SYNC_EVENTS_DIR=$SYNC_EVENTS_DIR/spikeglx
    mkdir $SPIKEGLX_SYNC_EVENTS_DIR

    # Write record time start
    echo "Writing 0. to $SPIKEGLX_SYNC_EVENTS_DIR/record_start_time"
    echo "0." > $SPIKEGLX_SYNC_EVENTS_DIR/record_start_time

    cd $PHYS_PREPROCESSING_DIR/compute_trials
    python compute_spikeglx_trials.py \
        "$SPIKEGLX_RAW_DIR" "$RAW_TRIALS_DIR" "$SPIKEGLX_SYNC_EVENTS_DIR"
    cd $BASE_PWD
fi
