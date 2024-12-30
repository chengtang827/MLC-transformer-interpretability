#!/usr/bin/env bash

# Run KiloSort3 for a 64-channel V-probe with 50 um contact spacing.

#SBATCH -o ./slurm_logs/%A.out
#SBATCH -t 24:00:00
#SBATCH -N 1
#SBATCH --mem 80G
#SBATCH --mail-type=NONE
#SBATCH --mail-user=nwatters@mit.edu
#SBATCH -p jazayeri
#SBATCH --gres=gpu:QUADRORTX6000:1

################################################################################
#### Preamble
################################################################################

echo -e '\n\n'
echo '##############################################'
echo '##  STARTING run_kilosort3_vprobe_64_50.sh  ##'
echo '##############################################'
echo -e '\n\n'

OM2_BASE_DIR=/om2/user/nwatters/multi_prediction

################################################################################
#### Fetch arguments
################################################################################

SESSION=$1  # Argument passed in by user. Should be a date string.
echo "SESSION: $SESSION"

PROBE_NAME=$2  # Argument passed in by user. Should be a string.
echo "PROBE_NAME: $PROBE_NAME"

################################################################################
#### Get directories
################################################################################

OM2_SESSION_DIR=$OM2_BASE_DIR/phys_data/$SESSION
OM2_RAW_DATA_DIR=$OM2_SESSION_DIR/raw_data
OPEN_EPHYS_DATA_DIR=$OM2_RAW_DATA_DIR/$PROBE_NAME
echo "OPEN_EPHYS_DATA_DIR: ${OPEN_EPHYS_DATA_DIR}"

# Make spike sorting directory
WRITE_DIR=$OM2_SESSION_DIR/spike_sorting/$PROBE_NAME
mkdir $WRITE_DIR

# Get sample rate file
SAMPLE_RATE_FILE=$WRITE_DIR/sample_rate
echo "SAMPLE_RATE_FILE: ${SAMPLE_RATE_FILE}"

################################################################################
#### Spike-sort
################################################################################

# Write sample rate of 30000Hz
echo "30000.0" > $SAMPLE_RATE_FILE

# Create temporary matlab file with processing steps
NEW_UUID=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 8 | head -n 1)
tmp_fn=tmp_ks$NEW_UUID
echo "cd ../kilosort" >> $tmp_fn.m
echo "run_kilosort3_vprobe_64_50('$OPEN_EPHYS_DATA_DIR');" >> $tmp_fn.m
echo "exit;" >> $tmp_fn.m

# Run and then remove temporary matlab file
echo 'STARTING MATLAB'
export MW_NVCC_PATH=/cm/shared/openmind/cuda/9.1/bin  # Cuda driver
module add openmind/cuda/9.1
module add openmind/cudnn/9.1-7.0.5
module add openmind/gcc/5.3.0
module add mit/matlab/2018b
matlab -nodisplay -r "$tmp_fn"
rm -f $tmp_fn.m

################################################################################
#### Launch job to get spike times
################################################################################

echo 'RUNNING SPIKE PROCESSING'
sbatch spike_processing/spike_processing.sh \
    $SESSION $PROBE_NAME ks_3_output_pre_v6
