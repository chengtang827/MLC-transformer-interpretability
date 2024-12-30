#!/usr/bin/env bash



#SBATCH -o ./slurm_output/R-%x.%j.out
#SBATCH -t 24:00:00
#SBATCH -n 8
#SBATCH --mem=16G
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=c_tang@mit.edu
#SBATCH --partition=jazayeri



source /home/$USER/.bashrc
echo '##############################################'
echo '##  STARTING run_parse_sessions.sh  ##'
echo '##############################################'


python run_parse_sessions.py