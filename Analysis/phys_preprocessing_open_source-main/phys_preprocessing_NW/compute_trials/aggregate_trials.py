"""Aggregate trials across mworks, moog, OpenEphys, and SpikeGLX.

The purpose of this file is to read trials from mworks, moog, OpenEphys, and
SpikeGLX and create an aggregated list of trials with all relevant data from
each source.

Usage:
    $ python3 aggregate_trials.py /om2/.../2022-06-01/trial_structure

This script assumes the following of the input `trial_structure` directory:
    `trial_structure`
        `raw_trials`
            `moog_trials`
            `mworks_trials_no_eye`
            `open_ephys_trial`
            `spikeglx_trials`

This script adds a file `trials` to the `trial_structure` directory.
"""

import json
import numpy as np
import os
import sys


def _find_all_corresponding_phys_trials(mworks_trials, phys_trials):
    # if len(phys_trials) == 2659 and phys_trials[513]['t_start'] == 1805.96:
    #     def _dummy_phys_trial():
    #         return {'trial_num': np.nan}
    #     phys_trials[513] = _dummy_phys_trial()
    #     phys_trials.insert(514, _dummy_phys_trial())
    #     phys_trials.insert(514, _dummy_phys_trial())
    
    if len(mworks_trials) == len(phys_trials):
        # Lengths match, so there's probably a 1-to-1 correspondence, but
        # double-check some trial numbers to be sure
        tail = 300
        last_trial_nums_mw = np.array(
            [t['trial_num'] for t in mworks_trials[-tail:]])
        last_trial_nums_phys = np.array(
            [t['trial_num'] for t in phys_trials[-tail:]])
        alignment = np.mean(last_trial_nums_mw == last_trial_nums_phys)
        if alignment < 0.3:
            raise ValueError(
                'mworks_trials and phys_trials do not align. Alignment is '
                f'{alignment}, last_trial_nums_mw = {last_trial_nums_mw}, '
                f'last_trial_nums_phys = {last_trial_nums_phys}'
            )
        else:
            return phys_trials
    else:
        if len(phys_trials) < len(mworks_trials):
            remainder = len(mworks_trials) - len(phys_trials)
            
            # Check to see if the beginning aligns
            phys_t_start = np.array([x['t_start'] for x in phys_trials])
            mworks_t_start = np.array([x['t_start'] for x in mworks_trials])
            phys_diffs = phys_t_start[1:] - phys_t_start[:-1]
            mworks_diffs = mworks_t_start[1:] - mworks_t_start[:-1]
            close_beginning = np.isclose(
                phys_diffs, mworks_diffs[:len(phys_diffs)], rtol=0.01)
            if np.mean(close_beginning) > 0.98:
                return phys_trials + remainder * [{}]

            close_end = np.isclose(
                phys_diffs, mworks_diffs[-len(phys_diffs):], rtol=0.01)
            if np.mean(close_end) > 0.98:
                return remainder * [{}] + phys_trials
        
        else:
            remainder = len(phys_trials) - len(mworks_trials)
            
            # Check to see if the beginning aligns
            phys_t_start = np.array([x['t_start'] for x in phys_trials])
            mworks_t_start = np.array([x['t_start'] for x in mworks_trials])
            phys_diffs = phys_t_start[1:] - phys_t_start[:-1]
            mworks_diffs = mworks_t_start[1:] - mworks_t_start[:-1]
            close_beginning = np.isclose(
                phys_diffs[:-remainder], mworks_diffs, rtol=0.01)
            if np.mean(close_beginning) > 0.98:
                return phys_trials[:-remainder]

            close_end = np.isclose(
                phys_diffs[remainder:], mworks_diffs, rtol=0.01)
            if np.mean(close_end) > 0.98:
                return phys_trials[remainder:]

        # DELETE THIS
        if len(phys_diffs) == 2028 and len(mworks_diffs) == 2029:
            x1 = phys_diffs[1:]
            x2 = mworks_diffs[:-2]
            if np.allclose(x1, x2, rtol=0.1):
                return phys_trials[1:] + 2 * [{}]

        # np.allclose(phys_diffs[510:520], mworks_diffs[510:520], rtol=0.01)

        # if len(mworks_diffs) == len(phys_diffs) + 2:
        #     if np.allclose(mworks_diffs[:-3], phys_diffs[1:], rtol=0.001):
        #         to_return = phys_trials[1:] + 3 * [{}]
        #         print(len(mworks_trials))
        #         print(len(to_return))
        #         return to_return

        # if len(mworks_diffs) == len(phys_diffs) + 1:
        #     if np.allclose(mworks_diffs[:-2], phys_diffs[1:], rtol=0.001):
        #         to_return = phys_trials[1:] + 2 * [{}]
        #         print(len(mworks_trials))
        #         print(len(to_return))
        #         return to_return


        # print('SPECIAL EXCEPTION HANDLING')

        # phys_starts = np.array([t['t_start'] for t in phys_trials])
        # mworks_starts = np.array([t['t_start'] for t in mworks_trials])
        # transform = (phys_starts[-1] - phys_starts[0]) / (
        #     mworks_starts[-1] - mworks_starts[11])
        # putative_phys_starts = phys_starts[0] + transform * (
        #     mworks_starts - mworks_starts[11])
        # phys_trials_return = []
        # num_errors = 0
        # for x in putative_phys_starts:
        #     errors = np.abs(phys_starts - x)
        #     min_ind = np.argmin(errors)
        #     if errors[min_ind] < 0.001:
        #         phys_trials_return.append(phys_trials[min_ind])
        #     else:
        #         num_errors += 1
        #         phys_trials_return.append({})

        # print(f'num_errors = {num_errors}')

        # return phys_trials_return

        # import pdb; pdb.set_trace()

        raise ValueError(
            f'mworks_trials has length {len(mworks_trials)}, but phys_trials '
            f'has length {len(phys_trials)}, so cannot easily align trials. '
            'Consider using aggregate_trials_last_resort.py in this directory.'
        )


def _find_all_corresponding_moog_trials(mworks_trials, moog_trials):
    moog_index = 0
    corresponding_moog_trials = []
    for t in mworks_trials:
        if moog_trials[moog_index]['total_trial_num'] is None:
            # This can happen if mworks was re-started, so moog has a dummy
            # trial.
            moog_index += 1
            corresponding_moog_trials.append(None)
            continue
        if t['trial_num'] is None:
            corresponding_moog_trials.append(None)
        while moog_trials[moog_index]['total_trial_num'] != t['trial_num']:
            moog_index += 1
        corresponding_moog_trials.append(moog_trials[moog_index])
    
    not_found_num = np.sum([x is None for x in corresponding_moog_trials])
    not_found_message = (
        f'Could not find corresponding moog trial for {not_found_num} mworks '
        'trials.'
    )
    if not_found_num > 15:
        raise ValueError(not_found_message)
    else:
        print(not_found_message)
        
    return corresponding_moog_trials


def _curate_phase_times(trials):
    """Curate relative phase times.
    
    Sometimes, a phase pulse for trial start will be at the beginning of
    relative phase times. In this case, we remove it, so that the first relative
    phase time is the end of the fixation phase.
    """
    for t in trials:
        rel_phase_times = t['relative_phase_times']
        if len(rel_phase_times) > 0 and rel_phase_times[0] < 0.35:
            t['relative_phase_times'] = rel_phase_times[1:]
    
    return trials


def _curate_photodiode_delays(trials):
    for t in trials:
        if 'open_ephys_photodiode_delay' in t:
            del t['open_ephys_photodiode_delay']
        if 'spikeglx_photodiode_delay' in t:
            del t['spikeglx_photodiode_delay']
    
    return trials


def _should_truncate_mworks_trials(mworks_trials, phys_trials):
    # See if should truncate mworks trials by 1.
    tail_length = 20
    for v in phys_trials.values():
        if v is None:
            continue
        if len(v) == len(mworks_trials) - 1:
            last_phys_t_start = np.array(
                [x['t_start'] for x in v[-tail_length:]])
            last_mworks_t_start = np.array(
                [x['t_start'] for x in mworks_trials[-tail_length - 1:-1]])
            phys_diffs = last_phys_t_start[1:] - last_phys_t_start[:-1]
            mworks_diffs = last_mworks_t_start[1:] - last_mworks_t_start[:-1]
            should_truncate = np.allclose(phys_diffs, mworks_diffs, rtol=0.01)
            return should_truncate
    return False


def main(trial_structure_dir):
    """Compute and write aggregated data for each trial.

    This data is a list of dictionaries, one for each valid trial. It is
    JSON-serialized and written to a file $write_dir/trials.
    
    Args:
        raw_trials_dir: String. Full path to raw trials directory.
        write_dir: String. Full path to directory to write raw moog trials.
    """

    print('STARTED aggregate_trials.py')

    ############################################################################
    ####  Find Data Paths
    ############################################################################

    print(f'trial_structure_dir: {trial_structure_dir}')

    raw_trials_dir = os.path.join(trial_structure_dir, 'raw_trials')

    moog_trials = json.load(open(os.path.join(raw_trials_dir, 'moog_trials'), 'r'))
    print(f'len(moog_trials) = {len(moog_trials)}')

    mworks_trials_no_eye = json.load(
        open(os.path.join(raw_trials_dir, 'mworks_trials_no_eye'), 'r'))
    mworks_trials_no_eye = _curate_phase_times(mworks_trials_no_eye)
    print(f'len(mworks_trials_no_eye) = {len(mworks_trials_no_eye)}')

    phys_trials = {}

    if 'open_ephys_trials' in os.listdir(raw_trials_dir):
        open_ephys_trials = json.load(
            open(os.path.join(raw_trials_dir, 'open_ephys_trials'), 'r'))
        print(f'len(open_ephys_trials) = {len(open_ephys_trials)}')
        phys_trials['open_ephys'] = _curate_phase_times(open_ephys_trials)
    else:
        phys_trials['open_ephys'] = None

    if 'spikeglx_trials' in os.listdir(raw_trials_dir):
        spikeglx_trials = json.load(
            open(os.path.join(raw_trials_dir, 'spikeglx_trials'), 'r'))
        print(f'len(spikeglx_trials) = {len(spikeglx_trials)}')
        phys_trials['spikeglx'] = _curate_phase_times(spikeglx_trials)
    else:
        phys_trials['spikeglx'] = None

    # Truncate mworks trials if necessary
    should_truncate_mworks_trials = _should_truncate_mworks_trials(
        mworks_trials_no_eye, phys_trials
    )
    # if should_truncate_mworks_trials:
    #     mworks_trials_no_eye = mworks_trials_no_eye[:-1]
    #     for k, v in phys_trials.items():
    #         phys_trials[k] = v[:len(mworks_trials_no_eye)]

    corresponding_moog_trials = _find_all_corresponding_moog_trials(
        mworks_trials_no_eye, moog_trials)

    corresponding_phys_trials = {}
    for k, v in phys_trials.items():
        if v is not None:
            phys_trials = _find_all_corresponding_phys_trials(
                mworks_trials_no_eye, v)
            phys_trials = [
                {k + '_' + key: value for key, value in d.items()}
                for d in phys_trials
            ]
            corresponding_phys_trials[k] = phys_trials

    trials = []
    for i in range(len(mworks_trials_no_eye)):
        # Start with mworks metadata
        mw_trial = mworks_trials_no_eye[i]
        if mw_trial['trial_num'] is None:
            continue
        trial = dict(
            mworks_trial_num=int(mw_trial['trial_num']),
            mworks_t_start=mw_trial['t_start'],
            mworks_t_end=mw_trial['t_end'],
            mworks_relative_phase_times=mw_trial['relative_phase_times'],
            mworks_photodiode_delay=mw_trial['photodiode_delay'],
            mworks_session_num=mw_trial['mworks_session_num'],
        )

        # Add moog data
        trial['moog_data'] = corresponding_moog_trials[i]

        # Add phys metadata
        for k, corr_phys_trials in corresponding_phys_trials.items():
            phys_trial = corr_phys_trials[i]
            if k + '_t_start' not in phys_trial:
                continue
            trial.update({
                k + '_t_start': phys_trial[k + '_t_start'],
                k + '_t_end': phys_trial[k + '_t_end'],
                k + '_relative_phase_times': phys_trial[
                    k + '_relative_phase_times'],
                k + '_photodiode_delay': phys_trial[k + '_photodiode_delay'],
            })

        trials.append(trial)

    trials = _curate_photodiode_delays(trials)

    print('SAVING AGGREGATED TRIALS')

    write_path = os.path.join(trial_structure_dir, 'trials')
    print(f'write_path: {write_path}')
    json.dump(trials, open(write_path, 'w'))

    print('FINISHED aggregate_trials.py')

    return


if __name__ == "__main__":
    trial_structure_dir = sys.argv[1]
    # trial_structure_dir = (
    #     '/Users/nicholaswatters/Desktop/grad_school/research/mehrdad/'
    #     'multi_prediction/phys/phys_data/Elgar/2022-09-21/trial_structure'
    # )
    main(trial_structure_dir)

