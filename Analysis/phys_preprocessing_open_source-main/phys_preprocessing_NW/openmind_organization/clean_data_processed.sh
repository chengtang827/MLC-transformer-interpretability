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

# This script removes unnecessary files from om4/data_processed.

################################################################################
################################################################################

echo -e '\n\n'
echo '########################################'
echo '##  STARTING clean_data_processed.sh  ##'
echo '########################################'
echo -e '\n\n'

# Base directories
BASE_DIR=/om4/group/jazlab/nwatters/multi_prediction/data_processed

echo "Removing sync_events"
rm -r $BASE_DIR/**/**/trial_structure/sync_events

echo "Removing channel_positions.npy"
rm -r $BASE_DIR/**/**/spike_sorting/**/**/channel_positions.npy
echo "Removing channel_map.npy"
rm -r $BASE_DIR/**/**/spike_sorting/**/**/channel_map.npy
echo "Removing cluster_Amplitude.tsv"
rm -r $BASE_DIR/**/**/spike_sorting/**/**/cluster_Amplitude.tsv
echo "Removing cluster_ContamPct.tsv"
rm -r $BASE_DIR/**/**/spike_sorting/**/**/cluster_ContamPct.tsv
echo "Removing cluster_info.tsv"
rm -r $BASE_DIR/**/**/spike_sorting/**/**/cluster_info.tsv
echo "Removing diary_*"
rm -r $BASE_DIR/**/**/spike_sorting/**/**/diary_*
echo "Removing cluster_group.tsv"
rm -r $BASE_DIR/**/**/spike_sorting/**/**/cluster_group.tsv
echo "Removing cluster_KSLabel.tsv"
rm -r $BASE_DIR/**/**/spike_sorting/**/**/cluster_KSLabel.tsv
echo "Removing mean_amplitudes.tsv"
rm -r $BASE_DIR/**/**/spike_sorting/**/**/mean_amplitudes.tsv
echo "Removing params.py"
rm -r $BASE_DIR/**/**/spike_sorting/**/**/params.py
echo "Removing phy.log"
rm -r $BASE_DIR/**/**/spike_sorting/**/**/phy.log
echo "Removing similar_templates.npy"
rm -r $BASE_DIR/**/**/spike_sorting/**/**/similar_templates.npy
echo "Removing spike_templates.npy"
rm -r $BASE_DIR/**/**/spike_sorting/**/**/spike_templates.npy
echo "Removing templates_ind.npy"
rm -r $BASE_DIR/**/**/spike_sorting/**/**/templates_ind.npy
echo "Removing templates.npy"
rm -r $BASE_DIR/**/**/spike_sorting/**/**/templates.npy
echo "Removing whitening_mat_inv.npy"
rm -r $BASE_DIR/**/**/spike_sorting/**/**/whitening_mat_inv.npy
echo "Removing whitening_mat.npy"
rm -r $BASE_DIR/**/**/spike_sorting/**/**/whitening_mat.npy

SESSIONS=$(find $BASE_DIR -maxdepth 2 -mindepth 2)
echo "SESSIONS: ${SESSIONS}"

# Eliminate probe spike_sorting directories that have no kilosort in them
PROBES=$(find $BASE_DIR/**/**/spike_sorting -mindepth 1 -maxdepth 1)
for x in $PROBES; do
    echo -e '\n'
    echo "Processing probe $x"
    
    for kilosort_dir in $x/ks_*; do
        if ! [ -d "$kilosort_dir" ]; then
            echo 'No kilosort directory, removing probe directory'
            rm -r $x
        fi
    done

    for cluster_group_file in $x/ks_*/curated_cluster_group.tsv; do
        if ! [ -f "$cluster_group_file" ]; then
            echo 'No curated cluster groups, removing probe directory'
            rm -r $x
        fi
    done

    echo "Renaming curated_cluster_group.tsv to cluster_label.tsv"
    for kilosort_dir in $x/ks_*; do
        mv $kilosort_dir/curated_cluster_group.tsv \
            $kilosort_dir/cluster_label.tsv
    done
    
    echo "Removing kilsort directory and moving files out"
    PROCESSED_KILOSORT_DIR=false
    for kilosort_dir in $x/ks_*; do
        if $PROCESSED_KILOSORT_DIR; then
            echo "FOUND MULTIPLE KILOSORT DIRECTORIES. SOMETHING WENT WRONG."
        fi
        mv $kilosort_dir/* $x/
        rm -r $kilosort_dir
        PROCESSED_KILOSORT_DIR=true
    done
done

echo -e '\n\n'
echo 'DONE WITH EVERYTHING'
