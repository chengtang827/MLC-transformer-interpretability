#!/bin/sh

# Prepare total data.

#SBATCH -o ./slurm_logs/%A.out
#SBATCH -t 8:00:00
#SBATCH -n 1
#SBATCH --mem-per-cpu 64G
#SBATCH --mail-type=NONE
#SBATCH --mail-user=nwatters@mit.edu
#SBATCH --partition=jazayeri

echo '#######################################'
echo '##  STARTING run_curate_clusters.sh  ##'
echo '#######################################'

BASE_SOURCE=/om2/user/nwatters/multi_prediction/phys_data/Perle/
# BASE_SOURCE=/om2/user/nwatters/multi_prediction/phys_data/Elgar/

SESSIONS=$(find $BASE_SOURCE -maxdepth 1)
echo "SESSIONS: ${SESSIONS}"

echo 'SOURCING BASHRC'
source /home/nwatters/.bashrc
echo 'ACTIVATING CONDA ENVIRONMENT'
conda activate /om/user/nwatters/venvs/python_basic

for x in $SESSIONS; do
    echo "PROCESSING SESSION $x"
    echo 'STARTING PYTHON'
    python3 run_curate_clusters.py $x 'np_0' 'ks_3_output_v2'
    # python3 run_curate_clusters.py $x 'np_1' 'ks_3_output_v2'
    # python3 curate_clusters.py $x 'v_probe_0' 'ks_3_output_pre_v0_curated'
    # python3 curate_clusters.py $x 'v_probe_1' 'ks_3_output_pre_v6_curated'
    echo 'DONE'
done

echo "DONE"