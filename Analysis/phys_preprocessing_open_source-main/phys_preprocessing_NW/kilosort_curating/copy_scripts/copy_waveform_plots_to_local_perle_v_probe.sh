#!/bin/sh

echo '################################################'
echo '##  STARTING copy_waveform_plots_to_local.sh  ##'
echo '################################################'

SESSIONS=(
    '2022-03-13'
    '2022-03-17'
    '2022-04-08'
    '2022-04-13'
    '2022-04-26'
    '2022-04-29'
    '2022-05-02'
    '2022-03-18'
    '2022-04-09'
    '2022-04-24'
    '2022-04-27'
    '2022-04-30'
    '2022-05-03'
    '2022-05-17'
    '2022-05-27'
    '2022-05-30'
    '2022-06-03'
    '2022-06-06'
    '2022-06-09'
    '2022-06-12'
    '2022-06-15'
    '2022-06-18'
    '2022-03-14'
    '2022-04-11'
    '2022-04-25'
    '2022-04-28'
    '2022-05-01'
    '2022-05-04'
    '2022-05-08'
    '2022-05-12'
    '2022-05-16'
    '2022-05-26'
    '2022-05-29'
    '2022-05-05'
    '2022-05-09'
    '2022-05-06'
    '2022-05-10'
    '2022-05-28'
    '2022-05-31'
    '2022-06-04'
    '2022-06-07'
    '2022-06-10'
    '2022-06-13'
    '2022-06-16'
    '2022-06-19'
    '2022-03-16'
    '2022-06-01'
    '2022-06-05'
    '2022-06-08'
    '2022-06-11'
    '2022-06-14'
    '2022-05-13'
    '2022-05-15'
    '2022-05-18'
    '2022-04-07'
    '2022-06-17'
)

BASE_SOURCE=/om2/user/nwatters/multi_prediction/phys_data/Perle
BASE_TARGET=../../phys_data/Perle

for s in "${SESSIONS[@]}"
do
    SOURCE_0=$BASE_SOURCE/$s/spike_sorting/v_probe_0/ks_3_output_pre_v6_curated/cluster_plots
    if ssh nwatters@openmind7.mit.edu "[ -d ${SOURCE_0} ]"; then
        echo "Session 0:  $s"
        TARGET_0=$BASE_TARGET/$s/spike_sorting/v_probe_0
        rm -r $TARGET_0
        mkdir -p $TARGET_0
        scp -r nwatters@openmind7.mit.edu:$SOURCE_0 $TARGET_0
    fi
    SOURCE_1=$BASE_SOURCE/$s/spike_sorting/v_probe_1/ks_3_output_pre_v6_curated/cluster_plots
    if ssh nwatters@openmind7.mit.edu "[ -d ${SOURCE_1} ]"; then
        echo "Session 1:  $s"
        TARGET_1=$BASE_TARGET/$s/spike_sorting/v_probe_1
        rm -r $TARGET_1
        mkdir -p $TARGET_1
        scp -r nwatters@openmind7.mit.edu:$SOURCE_1 $TARGET_1
    fi
done

echo "DONE"