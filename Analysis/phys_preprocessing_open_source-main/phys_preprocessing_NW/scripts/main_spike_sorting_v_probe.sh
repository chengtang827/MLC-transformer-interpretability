#!/usr/bin/env bash

# Run Kilosort spike sorting on raw data.

#SBATCH -o ./slurm_logs/%A.out
#SBATCH -t 03:00:00
#SBATCH -n 1
#SBATCH --mem-per-cpu 128G
#SBATCH --mail-type=NONE
#SBATCH --mail-user=nwatters@mit.edu
#SBATCH --partition=jazayeri

################################################################################
#### Preamble
################################################################################

echo -e '\n\n'
echo '######################################'
echo '##  STARTING main_spike_sorting.sh  ##'
echo '######################################'
echo -e '\n\n'

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

OM2_SESSION_DIR=$OM2_BASE_DIR/phys_data/$SESSION
OM2_RAW_DATA_DIR=$OM2_SESSION_DIR/raw_data
mkdir $OM2_SESSION_DIR/spike_sorting

################################################################################
#### Copy raw data to /om2
################################################################################

# bash spike_sorting/copy_data_from_om4_to_om2_v_probe.sh $SESSION

################################################################################
#### Launch open ephys spike sorting
################################################################################

# Process open ephys data
# bash spike_sorting/process_open_ephys_data.sh $SESSION

# Launch open ephys spike sorting
for X in $(find $OM2_RAW_DATA_DIR -maxdepth 1 -name 'v_probe_*'); do
    PROBE_NAME=$(basename ${X})
    echo "PROBE_NAME: $PROBE_NAME"
    sbatch spike_sorting/run_kilosort3_vprobe_64_50.sh $SESSION $PROBE_NAME
done

echo "DONE with main_spike_sorting.sh"