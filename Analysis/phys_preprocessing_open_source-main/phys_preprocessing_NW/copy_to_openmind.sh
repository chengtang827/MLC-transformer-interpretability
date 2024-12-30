#!/bin/bash

# This script copies the codebase to openmind, both to om2 and om4.
# Directories and files that are not needed for running on openmind are excluded
# from the copying.

# If you are forking/using this file, be sure to (i) change the destination
# path(s) and (ii) make sure that you exclude any additional unnecessary
# directories/files that you have introduced into the codebase.

rsync -av -e ssh \
--exclude='copy_to_openmind.sh' \
--exclude='deprecated**' \
--exclude='phys_data**' \
--exclude='behavior_data**' \
--exclude='pykilosort**' \
--exclude='CatGT/CatGT-linux/links**' \
. \
nwatters@openmind7.mit.edu:/om2/user/nwatters/multi_prediction/phys_preprocessing/

rsync -av -e ssh \
--exclude='copy_to_openmind.sh' \
--exclude='deprecated**' \
--exclude='phys_data**' \
--exclude='behavior_data**' \
--exclude='pykilosort**' \
--exclude='CatGT/CatGT-linux/links**' \
. \
nwatters@openmind7.mit.edu:/om4/group/jazlab/nwatters/multi_prediction/phys_preprocessing/