#!/usr/bin/env bash

# Run CatGT on raw spikeglx data.

#SBATCH -o ./slurm_logs/%A.out
#SBATCH -t 18:00:00
#SBATCH -n 1
#SBATCH --mem-per-cpu 128G
#SBATCH --mail-type=NONE
#SBATCH --mail-user=nwatters@mit.edu
#SBATCH --partition=jazayeri


YEAR=$1  # Argument passed in by user.
echo "YEAR: $YEAR"
MONTH=$2  # Argument passed in by user.
echo "MONTH: $MONTH"
DAY=$3  # Argument passed in by user.
echo "DAY: $DAY"

SUBJECT=Elgar
echo "SUBJECT: $SUBJECT"
SUBJECT_LOWERCASE=elgar
echo "SUBJECT_LOWERCASE: $SUBJECT_LOWERCASE"

DATE=$YEAR-$MONTH-$DAY
echo "DATE: $DATE"
SESSION=$SUBJECT/$DATE
echo "SESSION: $SESSION"
DATE_UNDERSCORE=${YEAR}_${MONTH}_${DAY}
echo "DATE_UNDERSCORE: $DATE_UNDERSCORE"

echo "COPYING DATA TO OM2"

bash spike_sorting/copy_data_from_om4_to_om_np.sh $SESSION

CATGT_WRITE_DIR=/om/user/nwatters/multi_prediction/phys_data/$SESSION/raw_data/spikeglx_catgt_v6
echo "CATGT_WRITE_DIR: $CATGT_WRITE_DIR"
mkdir $CATGT_WRITE_DIR

echo "STARTING CATGT"

cd ../CatGT/CatGT-linux

# -run=${DATE_UNDERSCORE}_${SUBJECT_LOWERCASE}_task \
# -run=${DATE_UNDERSCORE}_${SUBJECT_LOWERCASE} \
# -run=${DATE_UNDERSCORE}_${SUBJECT_LOWERCASE}_settling \

bash runit.sh \
-dir=/om/user/nwatters/multi_prediction/phys_data/$SESSION/raw_data/spikeglx/ \
-run=${DATE_UNDERSCORE}_${SUBJECT_LOWERCASE}_task \
-dest=$CATGT_WRITE_DIR \
-no_run_fld \
-ap \
-prb=0 \
-g=0 \
-t=0 \
-xa=0,0,2,3.0,4.5,25 \
-xd=2,0,384,6,500 \
-xia=0,0,2,3.0,4.5,2 \
-xid=2,0,384,6,50 \
-gblcar \
-loccar=40,160 \
-apfilter=butter,12,300,10000 \
-gfix=0.40,0.10,0.02

cd ../../scripts

echo "FINISHED CATGT"

echo "LAUNCHING KILOSORT CATGT"

sbatch spike_sorting/run_kilosort3_np_catgt.sh $SESSION