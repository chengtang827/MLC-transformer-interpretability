#!/usr/bin/env bash

# Run spike processing.

#SBATCH -o ./slurm_logs/%A.out
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH --mem 32G
#SBATCH --mail-type=NONE
#SBATCH --mail-user=nwatters@mit.edu
#SBATCH -p jazayeri

################################################################################
#### Preamble
################################################################################

echo -e '\n\n'
echo '####################################'
echo '##  STARTING spike_processing.sh  ##'
echo '####################################'
echo -e '\n\n'

OM2_BASE_DIR=/om2/user/nwatters/multi_prediction
BASE_PWD=`pwd`

################################################################################
#### Fetch arguments
################################################################################

SESSION=$1  # Argument passed in by user.
echo "SESSION: $SESSION"

PROBE_NAME=$2  # Argument passed in by user.
echo "PROBE_NAME: $PROBE_NAME"

KILOSORT_RUN_NAME=$3  # Argument passed in by user.
echo "KILOSORT_RUN_NAME: $KILOSORT_RUN_NAME"


################################################################################
#### Clean spikes directory
################################################################################

SPIKES_DIR=$OM2_BASE_DIR/phys_data/$SESSION/spikes/$PROBE_NAME
echo "Removing SPIKES_DIR $SPIKES_DIR"
# rm -r $SPIKES_DIR

################################################################################
#### Get spike times per cluster
################################################################################

# Activate conda to run python
source /home/nwatters/.bashrc
conda activate phys_analysis

# Get spike times per cluster
echo -e '\n'
echo 'RUNNING get_spike_times_per_cluster.py'
cd $OM2_BASE_DIR/phys_preprocessing/spikes
python get_spike_times_per_cluster.py \
    "$SESSION" "$PROBE_NAME" "$KILOSORT_RUN_NAME"
cd $BASE_PWD
echo -e '\n'

################################################################################
#### Get spike times per trial
################################################################################

# Get spike times per trial
echo -e '\n'
echo 'RUNNING get_spike_times_per_trial.py'
cd $OM2_BASE_DIR/phys_preprocessing/spikes
python get_spike_times_per_trial.py \
    "$SESSION" "$PROBE_NAME/$KILOSORT_RUN_NAME"
cd $BASE_PWD
echo -e '\n'

# ################################################################################
# #### Plot rasters
# ################################################################################

echo -e '\n'
echo 'RUNNING plot_rasters.py'
cd $OM2_BASE_DIR/phys_preprocessing/spikes
python plot_rasters.py "$SESSION" "$PROBE_NAME/$KILOSORT_RUN_NAME"
cd $BASE_PWD
echo -e '\n'


echo 'Done running spike_processing.sh'
