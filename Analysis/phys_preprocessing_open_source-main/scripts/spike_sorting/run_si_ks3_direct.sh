#!/usr/bin/env bash

# Run ironclust for neuropixel data and perform post processing steps

#SBATCH -o ./slurm_output/R-%x.%j.out
#SBATCH -t 24:00:00
#SBATCH -n 24
#SBATCH --mem=128G
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=c_tang@mit.edu
#SBATCH --partition=jazayeri
#SBATCH --gres=gpu:QUADRORTX6000:1
# TODO: This does not handle multiple probes 
# TODO: This does not write sample rate, which might be necessary for 
# spike postprocessing

################################################################################
#### Preamble
################################################################################

echo -e '\n\n'
echo '####################################'
echo '##  STARTING run_kilosort3_np.sh  ##'
echo '####################################'
echo -e '\n\n'

source /home/$USER/.bashrc
#conda activate phys_analysis

################################################################################
#### Creating spike sorting directory for spikeglx
################################################################################

#mkdir -p $OM2_SPIKE_SORTING_DIR/spikeglx

################################################################################
#### Loading modules
################################################################################

export MW_NVCC_PATH=/cm/shared/openmind/cuda/9.1/bin  # Cuda driver
module add openmind/cuda/9.1
module add openmind/cudnn/9.1-7.0.5
module add openmind/gcc/5.3.0
module add mit/matlab/2018b

################################################################################
#### Spike sort
################################################################################


# $1 = '20230928_F_g0'

mv /om4/group/jazlab/Cheng/sorted/Faure/NP/$1 /om2/user/c_tang/Sorting/Data/Faure/NP/

cd ../../spike_sorting
#python preprocess_spikeglx_data.py '/om2/user/c_tang/Sorting/Data/Faure/NP/'$1'/'$1'_imec0'
#python run_estimate_motion.py '/om2/user/c_tang/Sorting/Data/Faure/NP/'$1'/'$1'_imec0'
python run_kilosort3.py '/om2/user/c_tang/Sorting/Data/Faure/NP/'$1'/'$1'_imec0/'
mv  /om2/user/c_tang/Sorting/Data/Faure/NP/$1 /om4/group/jazlab/Cheng/sorted/Faure/NP/
##########a######################################################################
#### Post-processing 
################################################################################

#python postprocess_spikeglx_data.py $OM2_SESSION_DIR
