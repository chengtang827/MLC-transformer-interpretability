# Phys Data Preprocessing

General workflow of a neurophysiology session:
* Record a physiology session in the rig.
* From the rig computers, copy all data to
    `om4/group/jazlab/nwatters/multi_prediction/phys_data/$MONKEY/$DATE_STRING/raw_data`.
* Run the scripts in this directory by following the instructions belof to get
    neuron spike times per trial.


## Data Organization

Data for the multi-prediction project lives in two places: On `om4` in
`om4/group/jazlab/nwatters/multi_prediction/phys_data` and on `om2` in
`om2/user/nwatters/multi_prediction/phys_data`. The `om4` directory contains all
data, including raw data. The `om2` directory only contains processed data such
as spike times and select behavior variables, which requires much less storage
than the raw data.

### Diretory Structure on `om4`

Within `om4/group/jazlab/nwatters/multi_prediction/phys_data`, there are
directories Perle and Elgar for the two monkeys. Within each of these are
session directories of the form `2022-MM-DD`. Each session directory has the
following structure:

```
|-- 2022-MM-DD/
|   |-- raw_data/
|       |-- behavior/
|           |-- moog/
|           |-- mworks/
|       |-- open_ephys/
|           |-- 2022-MM-DD_HH-MM-SS/
|           |-- 2022-MM-DD_settling/
|       |-- spikeglx/
|           |-- 2022_MM_DD_$MONKEY_g0/
|           |-- 2022_MM_DD_$MONKEY_settling_g0/
|       |-- paths_to_task_data/
|           |-- open_ephys
|           |-- open_ephys_photodiode_filename
|           |-- spikeglx
|       |-- v_probe_0/
|           |-- raw_data.dat
|       |-- v_probe_1/
|           |-- raw_data.dat
|   |-- trial_structure/
|       |-- raw_trials/
|           |-- moog_trials
|           |-- mworks_trials
|           |-- mworks_trials_no_eye
|           |-- open_ephys_trials
|           |-- spikeglx_trials
|       |-- sync_events/
|           |-- mworks/
|           |-- open_ephys/
|           |-- spikeglx/
|       |-- trials
|   |-- spike_sorting/
|       |-- np_0/
|       |-- v_probe_0/
|       |-- v_probe_1/
|   |-- spikes/
|       |-- np_0/
|       |-- v_probe_0/
|       |-- v_probe_1/
```

A few points of explanation:
* Directories `v_probe_0` and `v_probe_1` are included in `raw_data` because the
    raw Open Ephys data contains all channels and is not ordered by the channel
    map, so it is convenient to have the channel mapped Open Ephys data by probe
    as well as the raw data.
* `paths_to_task_data` has files that contain a string path to useful data
    files. For example `paths_to_task_data/open_ephys` might contain a string
    like `/om4/group/jazlab/nwatters/multi_prediction/phys_data/Perle/2022-06-01/raw_data/open_ephys/2022-06-01_13-46-03/Record_Node_104`.
* Each file in `trial_structure/raw_trials` is a JSON-serialized list of
    dictionaries, each with fields like `t_start`, `t_end`,
    `relative_phase_times`, and `photodiode_delay`. These are the raw trials
    from each data acquisiton in the time coordinates of that data acquisition.
    `trial_structure/trial` contains an aggregation of all of these.
* Each directory in `trial_structure/sync_events` has a bunch of `.csv` files
    containing sync signal events and photodiode values.
* Each directory in `spike_sorting` contains kilosot output and sample rate for
    each probe.
* Each directory in `spikes` contains data post spike sorting. For example, this
    contains JSON-serialized files of spike times per cluster, spike times per
    cluster per trial, a directory of raster plots, etc.

### Diretory Structure on `om2`

Within `om4/group/jazlab/nwatters/multi_prediction/phys_data`, the structure is
exactly the same as on `om4` except without `raw_data` directories.


## Data Preprocessing Instructions

All of the top-level scripts for preprocessing data lie in the `scripts`
directory. To run the preprocessing pipeline on a session of data
`$MONKEY/2022-MM-DD`, enter the `scripts` directory on openmind and follow these
steps:
1. Run `$ sbatch main_trial_structure.sh $MONKEY/2022-MM-DD`. This creates and
   propagates the `trial_structure` directory in the data folder.
2. Run `$ sbatch main_spike_sorting.sh $MONKEY/2022-MM-DD`. This runs Kilsort on
   all the probes.
3. Manually curate V-Probe kilosort output, and copy the curated spike sorting
   back to `om4`.
4. Run `$ sbatch main_spike_processing_after_curating.sh $MONKEY/2022-MM-DD`.
   This re-runs spike processing on the curated spike sorting.
4. Run `$ sbatch main_copy_spikes_to_om4.sh $MONKEY/2022-MM-DD`. This copies the
   `spikes` directory from `om2` to `om4` and cleans up the session directories
   in both `om2` and `om4`.

## Structure of This Directory