#!/usr/bin/env bash

# Re-run spike processing after V-Probe spike sorting curating

#SBATCH -o ./slurm_logs/%A.out
#SBATCH -t 01:00:00
#SBATCH -n 1
#SBATCH --mem-per-cpu 64G
#SBATCH --mail-type=NONE
#SBATCH --mail-user=nwatters@mit.edu
#SBATCH --partition=jazayeri

################################################################################
#### Preamble
################################################################################

echo -e '\n\n'
echo '########################################################'
echo '##  STARTING main_spike_processing_after_curating.sh  ##'
echo '########################################################'
echo -e '\n\n'

OM4_BASE_DIR=/om4/group/jazlab/nwatters/multi_prediction
OM2_BASE_DIR=/om2/user/nwatters/multi_prediction

################################################################################
#### Fetch arguments
################################################################################

SESSION=$1  # Argument passed in by user. Should be a date string.
echo "SESSION: $SESSION"
if [ -z "$SESSION" ]; then
    echo "No session specified, exiting."
    exit
fi

################################################################################
#### Get directories
################################################################################

OM4_SESSION_DIR=$OM4_BASE_DIR/phys_data/$SESSION
OM4_RAW_DATA_DIR=$OM4_SESSION_DIR/raw_data
echo "OM4_RAW_DATA_DIR: $OM4_RAW_DATA_DIR"
OM4_SPIKE_SORTING_DIR=$OM4_SESSION_DIR/spike_sorting

OM2_SESSION_DIR=$OM2_BASE_DIR/phys_data/$SESSION
OM2_RAW_DATA_DIR=$OM2_SESSION_DIR/raw_data
SPIKE_SORTING_DIR=$OM2_SESSION_DIR/spike_sorting
echo "SPIKE_SORTING_DIR: $SPIKE_SORTING_DIR"

################################################################################
#### OpenEphys
################################################################################

V_SPIKE_SORTING_DIR_0=$SPIKE_SORTING_DIR/v_probe_0
# V_SPIKE_SORTING_DIR_1=$SPIKE_SORTING_DIR/v_probe_1

# copy open ephys spike sorting data to /om2
echo "Copying v_probe_0 spike sorting data to $V_SPIKE_SORTING_DIR_0"
cp -r $OM4_SESSION_DIR/spike_sorting/v_probe_0/ks_3_output_pre_v6_curated \
    $V_SPIKE_SORTING_DIR_0/
# echo "Copying v_probe_1 spike sorting data to $V_SPIKE_SORTING_DIR_1"
# cp -r $OM4_SESSION_DIR/spike_sorting/v_probe_1/ks_3_output_pre_v6_curated \
#     $V_SPIKE_SORTING_DIR_1/

echo 'RUNNING SPIKE PROCESSING FOR V-PROBES'
sbatch spike_processing/spike_processing.sh \
    $SESSION v_probe_0 ks_3_output_pre_v6_curated
# sbatch spike_processing/spike_processing.sh \
#     $SESSION v_probe_1 ks_3_output_pre_v6_curated

echo "DONE with main_spike_processing_after_curating.sh"