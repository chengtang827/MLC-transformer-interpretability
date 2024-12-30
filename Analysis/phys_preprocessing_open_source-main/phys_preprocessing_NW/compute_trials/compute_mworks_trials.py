"""Compute and write MWorks trials.

This script generates and saves a list of "trials", based on MOOG logs. Each
trial is a dictionary with all needed data about the trial.
"""

import json
import itertools
import os
import sys
import numpy as np
import re

sys.path.append('../utils')
import digital_to_on_off
import mworks_reader
import photodiode
import serialize


def _dummy_trial():
    trial = {
        'trial_num': None,
        't_start': np.nan,
        't_end': np.nan,
        'relative_phase_times': [],
        'photodiode_delay': np.nan,
    }
    return trial


def _load_mworks_data(mwk_file):
    print(f'loading mworks data from {mwk_file}')
    variable_names = [
        'sync_trial_start',
        'sync_phase',
        'total_trial_num',
        'photodiode',
        'eye_h_calibrated',
        'eye_v_calibrated',
        'pupil_size_r',
        'sync_trial_start_rebound',
    ]
    values = {s: [] for s in variable_names}
    found_codes = False
    reader = mworks_reader.MWK2Reader(mwk_file)
    with reader as event_file:
        for code, time, data in event_file:
            if not found_codes:
                name_to_code = dict((data[c]['tagname'], c) for c in data)
                code_to_name = {name_to_code[s]: s for s in variable_names}
                found_codes = True
                continue
            
            if code in code_to_name:
                name = code_to_name[code]
                values[name].append([time / 1000000., data])

        for name in values:
            values[name] = np.array(values[name])

    trial_start_on_inds, trial_start_off_inds = (
        digital_to_on_off.digital_to_on_off(values['sync_trial_start'][:, 1]))
    trial_start_on_times = values['sync_trial_start'][trial_start_on_inds, 0]
    trial_start_off_times = values['sync_trial_start'][trial_start_off_inds, 0]

    phase_on_inds, _ = digital_to_on_off.digital_to_on_off(
        values['sync_phase'][:, 1])
    phase_on_times = values['sync_phase'][phase_on_inds, 0]

    values['trial_start_on_times'] = trial_start_on_times
    values['trial_start_off_times'] = trial_start_off_times
    values['trial_start_rebound_times'] = (
        values['sync_trial_start_rebound'][:, 0])
    values['trial_start_rebound_values'] = (
        values['sync_trial_start_rebound'][:, 1])
    values['phase_on_times'] = phase_on_times
    values['photodiode_times'] = values['photodiode'][:, 0]
    values['photodiode_values'] = values['photodiode'][:, 1]

    return values


def _check_inferred_expected_trials(inferred_trial_nums,
                                    expected_trial_nums):
    """Make sure inferred and expected trial numbers mostly agree."""
    inferred_trial_nums = np.array(inferred_trial_nums)
    expected_trial_nums = np.array(expected_trial_nums)
    expected_success_rate_last_100_trials = np.mean(
        inferred_trial_nums[-100:] == expected_trial_nums[-100:])

    # Make sure expected trial nums agree with inferred trial nums
    if expected_success_rate_last_100_trials < 0.8:
        raise ValueError(
            'expected_success_rate_last_100_trials is '
            '{expected_success_rate_last_100_trials}, which is too low. '
            'Something went wrong with the trial number inference.'
        )


def _get_trials(data):

    ############################################################################
    ####  GET PHOTODIODE DELAYS
    ############################################################################

    print('GETTING PHOTODIODE DELAYS')

    trial_start_on_times = data['trial_start_on_times']
    trial_start_off_times = data['trial_start_off_times']
    phase_on_times = data['phase_on_times']

    photodiode_times = data['photodiode_times']
    photodiode_vals = data['photodiode_values']

    photodiode_delays = photodiode.get_photodiode_delays(
        photodiode_vals, photodiode_times, trial_start_on_times,
        trial_start_off_times)

    ##########################################################################
    ####  GET TRIALS
    ##########################################################################

    print('GETTING TRIALS')

    total_trial_num_times = data['total_trial_num'][:, 0]
    total_trial_num_vals = data['total_trial_num'][:, 1]
    eye_h_calibrated = data['eye_h_calibrated']
    eye_v_calibrated = data['eye_v_calibrated']
    pupil_size_r = data['pupil_size_r']

    def _times_within_trial(times, t_start, t_end):
        """Get times within a trial."""
        x = np.copy(times[(times > t_start) * (times < t_end)]).tolist()
        return x

    def _variable_in_trial(variable, t_start, t_end):
        """Get relative times and values of variable within trial."""
        inds_in_trial = (variable[:, 0] > t_start) * (variable[:, 0] < t_end)
        x = variable[inds_in_trial, :]
        return x[:, 0] - t_start, x[:, 1]

    trials = []
    trial_timeframes = list(
        zip(trial_start_off_times[:-1], trial_start_off_times[1:])
    )
    if len(photodiode_delays) != len(trial_timeframes) + 1:
        raise ValueError(
            f'photodiode_delays has length {len(photodiode_delays)}, but '
            f'trial_timeframes has length {len(trial_timeframes)}.'
        )
    
    inferred_trial_nums = []
    expected_trial_nums = []
    photodiode_errors = 0
    for i, (t_start, t_end) in enumerate(trial_timeframes):
        print(f'Computing trial {i} / {len(trial_start_off_times)}')

        # Get relative phase times
        phase_on = _times_within_trial(phase_on_times, t_start, t_end)
        relative_phase_times = (phase_on - t_start).tolist()

        # Fetch trial number
        trial_num = total_trial_num_vals[
            (total_trial_num_times > t_start) * (total_trial_num_times < t_end)
        ]
        trial_num = trial_num[0] - 1

        # Update inferred and expected trial nums
        inferred_trial_nums.append(trial_num)
        if i == 0:
            expected_trial_num = trial_num
        else:
            expected_trial_num = expected_trial_nums[-1] + 1
        expected_trial_nums.append(expected_trial_num)

        # Skip trial if unable to associate photodiode
        if np.isnan(photodiode_delays[i]):
            photodiode_errors += 1

        # Append trial data to `trials`
        eye_h_times, eye_h_vals = _variable_in_trial(
            eye_h_calibrated, t_start, t_end)
        eye_v_times, eye_v_vals = _variable_in_trial(
            eye_v_calibrated, t_start, t_end)
        pupil_size_r_times, pupil_size_r_vals = _variable_in_trial(
            pupil_size_r, t_start, t_end)
        trials.append(serialize.serialize({
            'trial_num': expected_trial_num,
            't_start': t_start,
            't_end': t_end,
            'relative_phase_times': relative_phase_times,
            'photodiode_delay': photodiode_delays[i],
            'eye_h_calibrated_times': eye_h_times,
            'eye_h_calibrated_vals': eye_h_vals,
            'eye_v_calibrated_times': eye_v_times,
            'eye_v_calibrated_vals': eye_v_vals,
            'pupil_size_r_times': pupil_size_r_times,
            'pupil_size_r_vals': pupil_size_r_vals,
        }))

    _check_inferred_expected_trials(inferred_trial_nums, expected_trial_nums)

    # Print trial errors, and abort if errors are too high
    print(f'photodiode_errors: {photodiode_errors}')
    if photodiode_errors > 0.1 * len(trial_timeframes):
        raise ValueError(
            f'photodiode_errors is {photodiode_errors} of '
            f'{len(trial_timeframes)} trials.'
        )

    return trials
    

def main():
    """Compute and write MWorks data for each trial.

    This data is a list of dictionaries, one for each valid trial. It is
    JSON-serialized and written to a file $write_dir/mworks_trials.
    
    Args:
        behavior_dir: String. Full path to moog raw data directory.
        write_dir: String. Full path to directory to write raw moog trials.
        sync_events_dir: String. Full path to directory to write sync events.
    """

    print('STARTED compute_mworks_trials.py')

    ############################################################################
    ####  Find Data Paths
    ############################################################################

    behavior_dir = sys.argv[1]
    print(f'behavior_dir: {behavior_dir}')
    write_dir = sys.argv[2]
    print(f'write_dir: {write_dir}')
    sync_events_dir = sys.argv[3]
    print(f'sync_events_dir: {sync_events_dir}')

    # Find all the mworks files
    mworks_re = re.compile('.*.mwk2')
    mworks_log_files = sorted([
        os.path.join(behavior_dir, x) for x in os.listdir(behavior_dir)
        if mworks_re.match(x)
    ])
    print(f'mworks_log_files: {mworks_log_files}')

    # Load mworks data
    print('LOADING MWORKS DATA')
    mworks_data = [_load_mworks_data(s) for s in mworks_log_files]

    # Write sync variables
    print('WRITING SYNC VARIABLES')
    sync_write_names = [
        'trial_start_on_times',
        'trial_start_off_times',
        'phase_on_times',
        'photodiode_times',
        'photodiode_values',
        'total_trial_num',
        'eye_h_calibrated',
        'eye_v_calibrated',
        'pupil_size_r',
        'trial_start_rebound_times',
        'trial_start_rebound_values',
    ]
    for filename, data in zip(mworks_log_files, mworks_data):
        filename = os.path.basename(filename).split('.')[0]
        file_write_dir = os.path.join(sync_events_dir, filename)
        if not os.path.exists(file_write_dir):
            os.makedirs(file_write_dir)
        print(f'Writing sync variables for file {filename} to {file_write_dir}')
        for k in sync_write_names:
            json.dump(
                serialize.serialize(data[k]),
                open(os.path.join(file_write_dir, k), 'w'),
            )


    # sync_events_dir = (
    #     '/Users/nicholaswatters/Desktop/grad_school/research/mehrdad/'
    #     'multi_prediction/phys/phys_data/Elgar/2022-06-09/trial_structure/'
    #     'sync_events/mworks'
    # )
    # mworks_variable_names = [
    #     'phase_on_times', 'total_trial_num', 'trial_start_off_times',
    #     'trial_start_on_times', 'photodiode_times', 'photodiode_values',
    # ]
    # mworks_data = [{
    #     x: np.array(json.load(open(os.path.join(sync_events_dir, x), 'r')))
    #     for x in mworks_variable_names
    # }]


    # Compute trials
    print('COMPUTING TRIALS')
    all_mworks_trials = [_get_trials(data) for data in mworks_data]

    for i, data in enumerate(all_mworks_trials):
        for d in data:
            d['mworks_session_num'] = i
        if i < len(all_mworks_trials) - 1:
            data.append(_dummy_trial())
    mworks_trials = list(itertools.chain(*all_mworks_trials))

    print('SAVING MWORKS TRIALS')

    write_path = os.path.join(write_dir, 'mworks_trials')
    print(f'write_path: {write_path}')
    json.dump(mworks_trials, open(write_path, 'w'))

    print('SAVING MWORKS TRIALS WITHOUT EYE DATA')

    mworks_trials_no_eye = [
        {k: d[k] for k in d.keys() if ('eye' not in k) and ('pupil' not in k)}
        for d in mworks_trials
    ]
    write_path = os.path.join(write_dir, 'mworks_trials_no_eye')
    print(f'write_path: {write_path}')
    json.dump(mworks_trials_no_eye, open(write_path, 'w'))

    print('FINISHED compute_mworks_trials.py')

    return


if __name__ == "__main__":
    main()

