"""Compute trials for phys sync signals."""

import numpy as np
import os
import sys

sys.path.append('../utils')
import serialize
import photodiode


def _times_in_trial(times, trial_timeframe):
    t_start, t_end = trial_timeframe
    times_in_trial = times[(times > t_start) & (times < t_end)]
    return times_in_trial - t_start


def _get_photodiode_delays(events_dir,
                           sync_trial_start_on,
                           sync_trial_start_off):
    """Get photodiode delays."""
    if 'spikeglx' in events_dir:
        photodiode_vals = np.genfromtxt(
            os.path.join(events_dir, 'photodiode_raw.csv'))
        sample_rate = float(np.genfromtxt(
            os.path.join(events_dir, 'sample_rate')))
        photodiode_times = np.arange(len(photodiode_vals)) / sample_rate
    elif 'open_ephys' in events_dir:
        photodiode_vals = np.genfromtxt(
            os.path.join(events_dir, 'photodiode_values.csv'))
        photodiode_times = np.genfromtxt(
            os.path.join(events_dir, 'photodiode_times.csv'))
    else:
        raise ValueError(f'Invalid events_dir {events_dir}')

    photodiode_delays = photodiode.get_photodiode_delays(
        photodiode_vals, photodiode_times, sync_trial_start_on,
        sync_trial_start_off)
    
    return photodiode_delays


def _infer_other_trial_num_intervals(on,
                                     off,
                                     time_unit,
                                     num_variables_on,
                                     num_variables_off):
    """If we only have one trial number interval variables, infer the other."""

    # Get all on/off events, padded by start_time and end_time
    events = np.sort(np.concatenate([on, off]))
    events = np.concatenate([[num_variables_on], events, [num_variables_off]])

    # Get inter-event intervals
    event_diffs = events[1:] - events[:-1]
    if event_diffs[0] > 0.5 * time_unit:
        start_off = True
    else:
        start_off = False
    event_diffs = event_diffs[event_diffs > 0.5 * time_unit]

    # Get event intervals, in units of time_unit
    intervals = np.round(event_diffs / time_unit).astype(int)
    
    if start_off:
        intervals_self = intervals[1::2]
        intervals_other = intervals[::2]
    else:
        intervals_self = intervals[::2]
        intervals_other = intervals[1::2]

    return intervals_self, intervals_other


def _get_trial_num_intervals(zero_on, zero_off, one_on, one_off, time_unit):
    """Get trial num intervals from trial num variable on and off times."""
    if len(zero_on) != len(zero_off) or len(one_on) != len(one_off):
        return None, None
    
    zero_intervals = np.round((zero_off - zero_on) / time_unit).astype(int)
    one_intervals = np.round((one_off - one_on) / time_unit).astype(int)
    if len(one_intervals) == 0:
        one_intervals = np.array([0])
    
    return zero_intervals, one_intervals


def _get_trial_num(zero_intervals, one_intervals):
    """Get trial num from zero and one intervals."""
    if zero_intervals is None or one_intervals is None:
        return np.nan

    if len(one_intervals) == len(zero_intervals) - 1:
        one_intervals = np.concatenate([[0], one_intervals])
    if len(one_intervals) != len(zero_intervals):
        return np.nan

    binary_number = []
    for zero, one in zip(zero_intervals, one_intervals):
        binary_number.extend(one * [1])
        binary_number.extend(zero * [0])
    binary_number = [str(x) for x in binary_number]
    binary_number_str = ''.join(binary_number[::-1])
    trial_num = int(binary_number_str, 2)

    return trial_num


def get_trials_from_events_dir(events_dir):
    """Compute list of trials from sync signal events directory."""

    # Read events
    sync_trial_start_on = np.genfromtxt(
        os.path.join(events_dir, 'sync_trial_start_on.csv'))
    sync_trial_start_off = np.genfromtxt(
        os.path.join(events_dir, 'sync_trial_start_off.csv'))
    sync_trial_num_zero_on = np.genfromtxt(
        os.path.join(events_dir, 'sync_trial_num_zero_on.csv'))
    sync_trial_num_zero_off = np.genfromtxt(
        os.path.join(events_dir, 'sync_trial_num_zero_off.csv'))
    sync_trial_num_one_on = np.genfromtxt(
        os.path.join(events_dir, 'sync_trial_num_one_on.csv'))
    sync_trial_num_one_off = np.genfromtxt(
        os.path.join(events_dir, 'sync_trial_num_one_off.csv'))
    sync_phase_on = np.genfromtxt(
        os.path.join(events_dir, 'sync_phase_on.csv'))
    sync_phase_off = np.genfromtxt(
        os.path.join(events_dir, 'sync_phase_off.csv'))

    # Trial timeframes
    # trial_timeframes_on = list(
    #     zip(sync_trial_start_on[:-1], sync_trial_start_on[1:]))
    trial_timeframes = list(
        zip(sync_trial_start_off[:-1], sync_trial_start_off[1:]))
    zero_on_per_trial = [
        _times_in_trial(sync_trial_num_zero_on, x) for x in trial_timeframes]
    zero_off_per_trial = [
        _times_in_trial(sync_trial_num_zero_off, x) for x in trial_timeframes]
    one_on_per_trial = [
        _times_in_trial(sync_trial_num_one_on, x) for x in trial_timeframes]
    one_off_per_trial = [
        _times_in_trial(sync_trial_num_one_off, x) for x in trial_timeframes]
    sync_phase_on_per_trial = [
        _times_in_trial(sync_phase_on, x) for x in trial_timeframes]
    sync_phase_off_per_trial = [
        _times_in_trial(sync_phase_off, x) for x in trial_timeframes]

    # Compute framerate time_unit, i.e. the duration of one frame
    inter_phase_intervals = np.array([
        np.min(phase_off - phase_on)
        for phase_off, phase_on in zip(
            sync_phase_off_per_trial, sync_phase_on_per_trial)
        if (len(phase_on) == len(phase_off) and len(phase_on) > 0)
    ])
    inter_phase_intervals = inter_phase_intervals[inter_phase_intervals > 0.]
    # Framerate should be the smallest duration between phase-on and phase-off
    # deltas, so we take the 50% quantile just in case there are some
    # anomalously small or large deltas for some reason
    time_unit = np.quantile(inter_phase_intervals, 0.65)

    # Compute interval in which trial num variables turn on and off
    num_variables_on = np.array([
        np.min(zero_on) for zero_on in zero_on_per_trial
        if len(zero_on) > 0
    ])
    num_variables_on = np.quantile(num_variables_on, 0.3)
    num_variables_off = np.array([
        np.max(zero_off) for zero_off in zero_off_per_trial
        if len(zero_off) > 0
    ])
    num_variables_off = np.quantile(num_variables_off, 0.7)

    # Sanity check that the time unit is what we expect
    expected_time_unit = (num_variables_off - num_variables_on) / 13.
    print(f'expected_time_unit = {expected_time_unit}')
    print(f'time_unit = {time_unit}')
    if not np.isclose(time_unit, expected_time_unit, rtol=0.1):
        raise ValueError(
            f'expected_time_unit is {expected_time_unit}, but time_unit is '
            f'{time_unit}'
        )

    # For some sessions, the trial_num_one variables were not recorded (likely
    # because of a loose BNC cable). Here we check to see if that's the case for
    # the current session, and if so raise a flag to infer the trial num ones
    # from trial num zeros.
    if np.mean([len(x) for x in one_on_per_trial]) < 0.5:
        lost_trial_num_ones = True
    else:
        lost_trial_num_ones = False

    # Get photodiode delays
    photodiode_delays = _get_photodiode_delays(
        events_dir, sync_trial_start_on, sync_trial_start_off)
    if len(photodiode_delays) != len(trial_timeframes) + 1:
        raise ValueError(
            f'photodiode_delays has length {len(photodiode_delays)}, but '
            f'trial_timeframes has length {len(trial_timeframes)}.'
        )
    
    # Get recording start
    record_start_time = float(
        np.genfromtxt(os.path.join(events_dir, 'record_start_time')))

    # Get trials
    trials = []
    photodiode_errors = 0
    trial_num_errors = 0
    print(f'Computing {len(trial_timeframes)} trials.')
    for i in range(len(trial_timeframes)):
        t_start = sync_trial_start_off[i]
        if i >= len(sync_trial_start_off) - 1:
            t_end = np.inf
        else:
            t_end = sync_trial_start_off[i + 1]
        timeframe = trial_timeframes[i]
        # must adjust phase times by t_start - timeframe[0] because timeframe is
        # computed based on sync_trial_start_on, but the trial actually starts
        # at sync_trial_start_off
        phase_times = sync_phase_on_per_trial[i] - (t_start - timeframe[0])
        zero_on = zero_on_per_trial[i]
        zero_off = zero_off_per_trial[i]
        one_on = one_on_per_trial[i]
        one_off = one_off_per_trial[i]

        if lost_trial_num_ones:
            zero_intervals, one_intervals = _infer_other_trial_num_intervals(
                zero_on, zero_off, time_unit, num_variables_on,
                num_variables_off,
            )
        else:
            zero_intervals, one_intervals = _get_trial_num_intervals(
                zero_on, zero_off, one_on, one_off, time_unit,
            )

        # Skip trial if unable to associate photodiode
        if np.isnan(photodiode_delays[i]):
            photodiode_errors += 1

        trial_num = _get_trial_num(zero_intervals, one_intervals)
        if np.isnan(trial_num):
            trial_num_errors += 1

        trial = {
            'trial_num': trial_num,
            't_start': t_start - record_start_time,
            't_end': t_end - record_start_time,
            'relative_phase_times': phase_times,
            'photodiode_delay': photodiode_delays[i],
            'noisy_trial_numbers': lost_trial_num_ones,
        }
        trials.append(serialize.serialize(trial))

    # Sanity check trial numbers generally increase
    trial_nums = np.array([t['trial_num'] for t in trials]).astype(int)
    trial_num_diffs = trial_nums[1:] - trial_nums[:-1]
    trial_num_error_rate_threshold = 0.7 if lost_trial_num_ones else 0.4
    trial_num_error_rate = np.mean(trial_num_diffs != 1)
    print(f'trial_num_error_rate = {trial_num_error_rate}')
    if trial_num_error_rate > trial_num_error_rate_threshold:
        raise ValueError(
            'Trial numbers do not increment. Error rate is '
            f'{trial_num_error_rate}. Something is wrong with the sync '
            'variables or trial number processing.'
        )

    # Print trial errors, and abort if errors are too high
    print(f'trial_num_errors: {trial_num_errors}')
    print(f'photodiode_errors: {photodiode_errors}')
    if trial_num_errors > 0.25 * len(trial_timeframes):
        raise ValueError(
            f'trial_num_errors is {trial_num_errors} of '
            f'{len(trial_timeframes)} trials.'
        )
    if photodiode_errors > 0.25 * len(trial_timeframes):
        print('Photodiode did not work. Replacing photodiode_delays with NaNs.')
        for t in trials:
            t['photodiode_delay'] = np.nan

    print(f'Total trials found: {len(trials)}')


    return trials
