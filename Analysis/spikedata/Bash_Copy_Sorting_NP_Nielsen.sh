#!/bin/sh


SESSION_NAME=$1

TARGET=/Volumes/Elements/Kilosort/${SESSION_NAME}
mkdir ${TARGET}
scp mramadan@openmind7.mit.edu:"/om4/group/jazlab/Mahdi/Neurophys/Sorting/Data_KS/Nielsen/NP/${SESSION_NAME}/*.bin" ${TARGET}/
scp mramadan@openmind7.mit.edu:"/om4/group/jazlab/Mahdi/Neurophys/Sorting/Data_KS/Nielsen/NP/${SESSION_NAME}/*.meta" ${TARGET}/
scp -r mramadan@openmind7.mit.edu:"/om4/group/jazlab/Mahdi/Neurophys/Sorting/Data_KS/Nielsen/NP/${SESSION_NAME}/${SESSION_NAME}_imec0/*.ap.meta" ${TARGET}/
scp -r mramadan@openmind7.mit.edu:"/om4/group/jazlab/Mahdi/Neurophys/Sorting/Data_KS/Nielsen/NP/${SESSION_NAME}/${SESSION_NAME}_imec0/ks_3_output_pre_v7/*" ${TARGET}/
