#!/bin/sh

# This script organizes raw data and computes trial-centric data (such as start
# time, end time, and phase times) of all trials for MOOG, MWorks, Open Ephys,
# and SpikeGLX data acquisitions.

# Example usage:  sbatch main_trial_structure.sh Perle/2022-06-01

# The result of this script is a directory `trial_structure` in both om2 and om4
# directories for the given session. In `trial_structure` are written raw trials
# and sync signal events for each data acquisition and an aggregated `trials`
# file with information for all data acquisitions for each trial.

#SBATCH -o ./slurm_logs/%A.out
#SBATCH -t 08:00:00
# # SBATCH -t 01:00:00
#SBATCH -n 1
#SBATCH --mem-per-cpu 64G
#SBATCH --mail-type=NONE
#SBATCH --mail-user=nwatters@mit.edu
#SBATCH --partition=jazayeri

echo -e '\n\n'
echo '########################################'
echo '##  STARTING main_trial_structure.sh  ##'
echo '########################################'
echo -e '\n\n'

SESSION=$1  # Argument passed in by user. Should be a date string.
echo "SESSION: $SESSION"
if [ -z "$SESSION" ]; then
    echo "No session specified, exiting."
    exit
fi

################################################################################
#### Clean up directory structure before starting processing
################################################################################

bash data_reorganization/rename_open_ephys_node_directories.sh $SESSION

################################################################################
#### Find where the raw data lives
################################################################################

bash data_reorganization/get_raw_data_folders.sh $SESSION

################################################################################
#### Get trial structure
################################################################################

bash get_trial_structure/copy_data_from_om4_to_om2.sh $SESSION
bash get_trial_structure/extract_raw_trials.sh $SESSION
bash get_trial_structure/aggregate_trials.sh $SESSION
bash get_trial_structure/copy_data_from_om2_to_om4.sh $SESSION
bash get_trial_structure/remove_raw_data_from_om2.sh $SESSION

echo "DONE with main_trial_structure.sh"