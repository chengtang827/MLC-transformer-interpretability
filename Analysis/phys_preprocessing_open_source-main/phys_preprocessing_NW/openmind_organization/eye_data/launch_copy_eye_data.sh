#!/bin/sh

# Prepare total data.

#SBATCH -o ./slurm_logs/%A.out
#SBATCH -t 12:00:00
#SBATCH -n 1
#SBATCH --mem-per-cpu 32G
#SBATCH --mail-type=NONE
#SBATCH --mail-user=nwatters@mit.edu
#SBATCH --partition=jazayeri

echo '########################################'
echo '##  STARTING launch_copy_eye_data.sh  ##'
echo '########################################'

echo 'SOURCING BASHRC'
source /home/nwatters/.bashrc
echo 'ACTIVATING CONDA ENVIRONMENT'
# conda activate /om/user/nwatters/venvs/python_basic
conda activate phys_analysis

echo 'STARTING PYTHON'
python3 copy_eye_data_to_data_processed.py

echo "DONE"

# Perle/2022-04-28: Last 3 trials are second session
# Perle/2022-05-09: Half and half
# Perle/2022-05-26: Half and half
# Perle/2022-05-28:  Third and third and third
# Perle/2022-06-07:  Half and half
# Perle/2022-06-14:  Half and half

# Elgar/2022-05-07: Seven sessions
# Elgar/2022-05-08: Two sessions
# Elgar/2022-05-10: Four sessions
# Elgar/2022-05-18: Two sessions
# Elgar/2022-09-04: Two sessions