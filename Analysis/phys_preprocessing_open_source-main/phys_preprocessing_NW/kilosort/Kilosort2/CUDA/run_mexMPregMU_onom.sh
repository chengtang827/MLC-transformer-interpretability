#!/bin/sh

#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH -c1
#SBATCH --ntasks=1
#SBATCH --mail-type=ALL
#SBATCH --mem=200000
#SBATCH --time=04:00:00
#SBATCH --partition=jazayeri


module add openmind/cuda/9.1
export MW_NVCC_PATH=/cm/shared/openmind/cuda/9.1/bin
module add mit/matlab/2018b
matlab -nodisplay -r "mexGPUall"
