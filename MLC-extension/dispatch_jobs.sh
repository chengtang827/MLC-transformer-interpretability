#!/usr/bin/env bash


#SBATCH -o ./slurm_output/R-%x.%j.out
#SBATCH -t 24:00:00
#SBATCH -n 8
#SBATCH --mem=256G
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=c_tang@mit.edu
#SBATCH --partition=jazayeri
#SBATCH --gres=gpu:a100:1


source /home/$USER/.bashrc


################################################################################
#### Loading modules
################################################################################

export MW_NVCC_PATH=/cm/shared/openmind/cuda/9.1/bin  # Cuda driver
module add openmind/cuda/9.1
module add openmind/cudnn/9.1-7.0.5
module add openmind/gcc/5.3.0

conda activate MI
cd /om2/user/c_tang/jazlab/MLC-extension
python train.py --episode_type algebraic_noise --fn_out_model net-HookedBIMLNew.pt


