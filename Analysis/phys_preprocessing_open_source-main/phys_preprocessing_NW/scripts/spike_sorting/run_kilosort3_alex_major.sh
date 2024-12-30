#!/usr/bin/env bash

# Run KiloSort3 for alex major data

#SBATCH -o ./slurm_logs/%A.out
#SBATCH -t 4:00:00
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
echo '##  STARTING run_kilosort3_alex_major.sh  ##'
echo '##############################################'
echo -e '\n\n'

# Allow module commands to work
source /etc/profile.d/modules.sh

################################################################################
#### Get directories
################################################################################

DATA_DIR=/om2/user/nwatters/alex_major/data
echo "DATA_DIR: ${DATA_DIR}"

# Make spike sorting directory
WRITE_DIR=$DATA_DIR/spike_sorting
mkdir $WRITE_DIR

################################################################################
#### Spike-sort
################################################################################

# Create temporary matlab file with processing steps
NEW_UUID=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 8 | head -n 1)
tmp_fn=tmp_ks$NEW_UUID
echo "cd ../../kilosort" >> $tmp_fn.m
echo "run_kilosort3_alex_major('$DATA_DIR');" >> $tmp_fn.m
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

