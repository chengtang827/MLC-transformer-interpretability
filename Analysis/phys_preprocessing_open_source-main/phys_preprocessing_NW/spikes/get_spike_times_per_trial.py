"""Compute and write spike times per trial per cluster."""

import copy
import json
from multiprocessing.sharedctypes import Value
import numpy as np
import os
import sys
from matplotlib import pyplot as plt

_OM2_BASE_DIR = '/om2/user/nwatters/multi_prediction'
# _OM2_BASE_DIR = '/Users/nicholaswatters/Desktop/grad_school/research/mehrdad/multi_prediction/phys'


def _get_fr(trial):
    """Get firing rate from trial dictionary."""
    trial_duration = trial['t_end'] - trial['t_start']
    num_spikes = len(trial['relative_spike_times'])
    fr = num_spikes / trial_duration
    return fr


def _get_true_blocks(x):
    x = 2 * np.concatenate([[False], x, [False]]).astype(int) - 1
    turn_on = np.convolve(x, [1, -1], mode='valid')
    turn_off = np.convolve(x, [-1, 1], mode='valid')
    turn_on_inds = np.argwhere(turn_on == 2)[:, 0]
    turn_off_inds = np.argwhere(turn_off == 2)[:, 0] - 1
    blocks = list(zip(turn_on_inds, turn_off_inds))
    return blocks


def _refine_subset(subset, start):
    smooth_subset = np.convolve(subset, np.ones(3), mode='valid')
    derivative = smooth_subset[1:] - smooth_subset[:-1]
    if start:
        max_slope_ind = np.argmax(derivative)
        subset[:max_slope_ind + 1] = False
    else:
        max_slope_ind = np.argmin(derivative)
        subset[max_slope_ind + 2:] = False


def _get_valid_fr_trials(fr_per_trial,
                         block_size_for_max_fr=60,
                         smooth_trial_num=60,
                         smooth_threshold=0.25,
                         min_block_length=200,
                         plot=False):
    # Smooth the firing rate
    smooth_kernel = np.ones(smooth_trial_num) / smooth_trial_num
    smoothed_fr = np.convolve(fr_per_trial, smooth_kernel, mode='valid')

    # Identify block_size_for_max_fr-trial block with max firing rate
    kernel_max = np.ones(block_size_for_max_fr) / block_size_for_max_fr
    smoothed_fr_max = np.convolve(fr_per_trial, kernel_max, mode='valid')
    max_fr_max = np.max(smoothed_fr_max)

    # Threshold away smoothed points with less than one third the max
    low_fr = smoothed_fr < smooth_threshold * max_fr_max
    n_pad_beginning = smooth_trial_num // 2
    n_pad_end = smooth_trial_num - n_pad_beginning - 1
    low_fr = np.concatenate([
        (low_fr[0] * np.ones(n_pad_beginning)).astype(bool),
        low_fr,
        (low_fr[-1] * np.ones(n_pad_end)).astype(bool),
    ])
    valid_fr = np.ones_like(fr_per_trial).astype(bool)
    valid_fr[low_fr] = False

    # Plot if necessary
    if plot:
        plt.figure()
        plt.plot(low_fr)
        plt.figure()
        plt.plot(smoothed_fr)
        plt.show()

    # Remove blocks less than min_block_length trials long
    blocks = _get_true_blocks(valid_fr)
    for (i, j) in blocks:
        if j - i < min_block_length:
            valid_fr[i: j + 1] = False

    # Refine boundaries
    blocks = _get_true_blocks(valid_fr)
    half_smooth_trial_num = smooth_trial_num // 2
    for (i, j) in blocks:
        if i > 0:
            subset = valid_fr[
                max(0, i - half_smooth_trial_num): i + half_smooth_trial_num
            ]
            _refine_subset(subset, start=True)
        if j < len(valid_fr) - 1:
            subset = valid_fr[
                j - half_smooth_trial_num:
                min(len(valid_fr), j + half_smooth_trial_num)
            ]
            _refine_subset(subset, start=False)

    return valid_fr


def _get_valid_fr_blocks(valid_fr_trials):
    # Get indices of blocks with valid firing rate
    if sum(valid_fr_trials) == 0:
        return []
    blocks = _get_true_blocks(valid_fr_trials)
    return blocks


def _curate_spike_times_per_trial(spike_times_per_trial):
    """Compute firing rate and whether valid for each trial."""
    fr_per_trial = np.array(
        [_get_fr(trial) for trial in spike_times_per_trial])
    completed_trials = np.array([
        1. < trial['t_end'] - trial['t_start'] < 5.
        for trial in spike_times_per_trial
    ])
    completed_inds = np.argwhere(completed_trials)[:, 0]
    filtered_fr_per_trial = fr_per_trial[completed_trials]

    filtered_valid_fr_trials = _get_valid_fr_trials(filtered_fr_per_trial)
    filtered_valid_fr_blocks = _get_valid_fr_blocks(filtered_valid_fr_trials)
    valid_fr_blocks = [
        (completed_inds[i], completed_inds[j])
        for i, j in filtered_valid_fr_blocks
    ]

    valid_fr_trials = np.array(len(spike_times_per_trial) * [False])
    for (i, j) in valid_fr_blocks:
        valid_fr_trials[i: j + 1] = True

    for valid, trial in zip(valid_fr_trials, spike_times_per_trial):
        trial['valid'] = valid.astype(float)

    return


def _add_fr_per_phase(trial_dict):
    spike_times = np.array(trial_dict['relative_spike_times'])
    phase_times = trial_dict['relative_phase_times']
    bins = [0.] + phase_times
    fr_per_phase = []
    for (start, end) in zip(bins[:-1], bins[1:]):
        num_spikes = np.sum((spike_times > start) * (spike_times < end))
        fr_per_phase.append(float(num_spikes) / (end - start))
    trial_dict['fr_per_phase'] = fr_per_phase
    trial_dict['fr'] = _get_fr(trial_dict)


def _get_time_variables(trial, daq_name):
    photodiode_delay = trial['mworks_photodiode_delay']
    t_start = photodiode_delay + trial[daq_name + '_t_start']
    t_end = photodiode_delay + trial[daq_name + '_t_end']
    return t_start, t_end


def _get_spike_times_per_trial(trials, spike_times, daq_name):
    """Get spike times per trial given unit's spike times."""
    spike_times_per_trial = []
    
    trial_ind = 0
    while daq_name + '_t_start' not in trials[trial_ind]:
        trial_ind += 1
        if trial_ind > len(trials):
            raise ValueError('Could not find any trials')
    t_start, t_end = _get_time_variables(trials[trial_ind], daq_name)

    current_trial_spike_times = []
    
    if len(spike_times) == 0:
        return None
    
    spike_ind = 0
    while True:
        if trial_ind >= len(trials):
            break
        if daq_name + '_t_start' not in trials[trial_ind]:
            trial_ind += 1
            current_trial_spike_times = []
            continue
        t_start, t_end = _get_time_variables(trials[trial_ind], daq_name)
        if spike_ind >= len(spike_times):
            break
        spike = spike_times[spike_ind]
        if spike < t_start:
            spike_ind += 1
            continue
        if spike < t_end:
            spike_ind += 1
            current_trial_spike_times.append(spike - t_start)
        else:
            trial_dict = {
                'relative_spike_times': copy.deepcopy(
                    current_trial_spike_times),
                'trial_num': trials[trial_ind]['mworks_trial_num'],
                't_start': trials[trial_ind][daq_name + '_t_start'],
                't_end': trials[trial_ind][daq_name + '_t_end'],
                'relative_phase_times': copy.deepcopy(
                    trials[trial_ind][daq_name + '_relative_phase_times']),
            }
            _add_fr_per_phase(trial_dict)
            spike_times_per_trial.append(trial_dict)
            
            trial_ind += 1
            current_trial_spike_times = []

    if len(spike_times_per_trial) >= 100:
        # The conditional has no effect since min_block_size > 100 above, but
        # avoids code hitting error
        _curate_spike_times_per_trial(spike_times_per_trial)
    else:
        for x in spike_times_per_trial:
            x['valid'] = False
            
    return spike_times_per_trial
    
    
def _valid_unit(spike_times_per_trial):
    """Determine whether unit is valid."""
    if spike_times_per_trial is None:
        print('spike_times_per_trial is None')
        return False
    
    fr_when_valid = [
        d['fr'] for d in spike_times_per_trial if d['valid']
    ]
    if len(fr_when_valid) < 100:
        # Neuron not valid for enough trials
        print(f'len(fr_when_valid) = {len(fr_when_valid)}')
        return False
    if np.mean(fr_when_valid) < 0.5:
        # Neuron not high enough firing rate
        print(f'np.mean(fr_when_valid) = {np.mean(fr_when_valid)}')
        return False
    return True


def main(session, probe_name):
    """Compute and write spike times per cluster."""

    ############################################################################
    ####  Find Data Paths
    ############################################################################

    print(f'session: {session}')
    print(f'probe_name: {probe_name}')

    session_dir = os.path.join(_OM2_BASE_DIR, 'phys_data', session)

    # Get spikes spikes directory
    spikes_dir = os.path.join(session_dir, 'spikes', probe_name)
    print(f'spikes_dir: {spikes_dir}')
    if not os.path.exists(spikes_dir):
        os.makedirs(spikes_dir)

    # Get behavior directory
    trials_path = os.path.join(session_dir, 'trial_structure', 'trials')
    print(f'trials_path: {trials_path}')

    # Get daq_name
    if 'np' in probe_name:
        daq_name = 'spikeglx'
    elif 'v_probe' in probe_name:
        daq_name = 'open_ephys'
    else:
        raise ValueError(f'Invalid probe_name {probe_name}')

    ############################################################################
    ####  LOAD DATA
    ############################################################################

    print('LOADING BEHAVIOR TRIALS')
    trials = json.load(open(trials_path, 'r'))

    print('LOADING SPIKE TIMES PER CLUSTER')
    spike_times_per_cluster_path = os.path.join(
        spikes_dir, 'spike_times_per_cluster')
    spike_times_per_cluster = json.load(open(spike_times_per_cluster_path, 'r'))

    ############################################################################
    ####  COMPUTE SPIKE TIMES PER TRIAL PER CLUSTER
    ############################################################################

    print('COMPUTING SPIKE TIMES PER TRIAL PER CLUSTER')

    spike_times_per_trial_per_cluster = {}
    valid_units = {}

    num_units = len(spike_times_per_cluster)
    for i, (k, v) in enumerate(spike_times_per_cluster.items()):
        print(f'Unit {i} / {num_units}')
        spike_times_per_trial = _get_spike_times_per_trial(trials, v, daq_name)
        spike_times_per_trial_per_cluster[k] = spike_times_per_trial
        
        valid_units[k] = _valid_unit(spike_times_per_trial)

    ############################################################################
    ####  WRITE DATA
    ############################################################################

    print('SAVING')

    spike_times_per_trial_path = os.path.join(
        spikes_dir, 'spike_times_per_trial')
    print(f'spike_times_per_trial_path: {spike_times_per_trial_path}')
    json.dump(
        spike_times_per_trial_per_cluster,
        open(spike_times_per_trial_path, 'w')
    )
    valid_units_path = os.path.join(spikes_dir, 'valid_units')
    print(f'valid_units_path: {valid_units_path}')
    json.dump(valid_units, open(valid_units_path, 'w'))

    print('DONE')

    return


if __name__ == "__main__":
    session = sys.argv[1]
    probe_name = sys.argv[2]
    main(session, probe_name)