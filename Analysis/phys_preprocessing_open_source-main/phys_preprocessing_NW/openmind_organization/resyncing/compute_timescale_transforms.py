"""Compute timescale transforms in data_processed.

Done, ran
"""

import json
from matplotlib import pyplot as plt
import numpy as np
import os

from sklearn import linear_model as sklearn_linear_model

# _DATA_DIR = '/om4/group/jazlab/nwatters/multi_prediction/data_processed'
_DATA_DIR = (
    '/Users/nicholaswatters/Desktop/grad_school/research/mehrdad/'
    'multi_prediction/phys/data_processed'
)
_MIN_MWORKS_SESSION_LENGTH = 5
_MAX_TRIAL_OFFSET = 12


def _timescale_transform(t_source, t_target, print_stats=False):
    if len(t_source) != len(t_target):
        raise ValueError(
            f'len(t_source) = {len(t_source)} but len(t_target) = '
            f'{len(t_target)}'
        )
    regr = sklearn_linear_model.LinearRegression()
    regr.fit(t_source[:, np.newaxis], t_target[:, np.newaxis])
    pred_t_target = regr.predict(t_source[:, np.newaxis])[:, 0]
    residuals = t_target - pred_t_target
    residual_std = np.round(1000. * np.std(residuals), decimals=1)
    residual_max_error = np.round(1000. * np.max(np.abs(residuals)), decimals=1)

    if print_stats:
        print(f'    residual_std: {residual_std} ms')
        print(f'    residual_max_error: {residual_max_error} ms')

    coef = regr.coef_[0, 0]
    intercept = regr.intercept_[0]

    return coef, intercept, residual_std


def _async_timescale_transform(t_source,
                               t_target,
                               print_stats=False,
                               last_try=False,
                               thresh=None):
    """Find relative offset and timescale transform between two timeseries.
    
    Args:
        t_source: Array of times. Typically trial start times for physiology.
        t_target: Array of times. Typically trial start times for mworks. Need
            not have the same length as t_source.
        print_stats: Bool. Whether to print statistics of the transform.
    """

    candidate_transforms = []
    # for relative_start_index in [0]:
    for relative_start_index in range(-_MAX_TRIAL_OFFSET, _MAX_TRIAL_OFFSET):
        if relative_start_index < 0:
            tmp_t_source = t_source[-1 * relative_start_index:]
            tmp_t_target = np.copy(t_target)
        else:
            tmp_t_target = t_target[relative_start_index:]
            tmp_t_source = np.copy(t_source)
        min_length = min(len(tmp_t_source), len(tmp_t_target))
        if min_length < 5:
            continue
        tmp_t_source = tmp_t_source[:min_length]
        tmp_t_target = tmp_t_target[:min_length]
        
        coef, intercept, residual_std = _timescale_transform(tmp_t_source, tmp_t_target, print_stats=print_stats)
        candidate_transforms.append(
            (relative_start_index, coef, intercept, residual_std)
        )
    
    residual_stds = np.array([x[3] for x in candidate_transforms])
    min_ind = np.argmin(residual_stds)
    min_residual_std = residual_stds[min_ind]
    if thresh is None:
        thresh = 1.
    if min_residual_std > thresh:
        diffs_source = t_source[1:] - t_source[:-1]
        diffs_target = t_target[1:] - t_target[:-1]
        print('diffs_source[:20]')
        print(np.round(diffs_source[:20], decimals=2))
        print('diffs_target[:20]')
        print(np.round(diffs_target[:20], decimals=2))
        import pdb; pdb.set_trace()
        if last_try:
            raise ValueError(
                f'min_residual_std = {residual_stds[min_ind]}, which is too '
                'high'
            )
        else:
            # Pop first element off t_source and t_target and try again
            return _async_timescale_transform(
                t_source[1:], t_target[1:], print_stats=print_stats,
                last_try=True, thresh=thresh)
    relative_start_index, coef, intercept, _ = candidate_transforms[min_ind]

    return relative_start_index, coef, intercept


def _concatenate_mworks_times(mworks_times, phys_times):
    mworks_times = [
        x for x in mworks_times if len(x) > _MIN_MWORKS_SESSION_LENGTH
    ]
    if len(mworks_times) == 0:
        raise ValueError('mworks_times is empty')
    elif len(mworks_times) == 1:
        all_mworks_times = mworks_times[0]
    else:
        coefs = []
        intercepts = []
        for mw_times in mworks_times:
            relative_start_index, coef, intercept = _async_timescale_transform(
                mw_times, phys_times)

            phys_times = phys_times[len(mw_times) + relative_start_index:]
            coefs.append(coef)
            intercepts.append(intercept)
        
        # Check that all coefficients are similar
        similar_coefs = np.isclose(np.max(coefs), np.min(coefs), rtol=0.001)
        if not similar_coefs:
            raise ValueError(
                f'np.max(coefs) = {np.max(coefs)}, but np.min(coefs) = '
                f'{np.min(coefs)}'
            )

        # Chain mworks times with appropriate offsets given coefs and intercepts
        # by first converting them all to phys times, then converting back to
        # the longest mworks session timescale
        all_mworks_times_in_phys_timescale = [
            intercept + coef * mw_times
            for coef, intercept, mw_times in zip(coefs, intercepts, mworks_times)
        ]
        longest_mw_sess_ind = np.argmax([len(x) for x in mworks_times])
        all_mworks_times = np.concatenate([
            (x - intercepts[longest_mw_sess_ind]) / coefs[longest_mw_sess_ind]
            for x in all_mworks_times_in_phys_timescale
        ])

    return all_mworks_times


def _compute_transform(mworks_times, phys_times, phys_sync_dir, thresh=None):
    relative_start_index, coef, intercept = _async_timescale_transform(
        phys_times, mworks_times, thresh=thresh)
    to_write = {
        'phys_start_index_minus_mworks_start_index': relative_start_index,
        'coef': coef,
        'intercept': intercept,
    }
    json.dump(to_write, open(os.path.join(phys_sync_dir, 'transform'), 'w'))

    return


def _process_session(session_dir, thresh=None):
    print(f'\nsession_dir: {session_dir}')

    sync_dir = os.path.join(session_dir, 'sync_pulses')
    mworks_sync_dir = os.path.join(sync_dir, 'mworks')

    # Load mworks sync pulses
    mworks_times = [
        np.array(json.load(open(
            os.path.join(mworks_sync_dir, mw_session, 'trial_start_off_times'),
            'r',
        )))
        for mw_session in sorted(os.listdir(mworks_sync_dir))
        if mw_session[:6] == 'mworks'
    ]

    # Load open_ephys and spikeglx sync pulses
    open_ephys_sync_dir = os.path.join(sync_dir, 'open_ephys')
    if os.path.exists(open_ephys_sync_dir):
        open_ephys_times = np.genfromtxt(
            os.path.join(open_ephys_sync_dir, 'sync_trial_start_off.csv'))
    else:
        open_ephys_times = None
    spikeglx_sync_dir = os.path.join(sync_dir, 'spikeglx')
    if os.path.exists(spikeglx_sync_dir):
        spikeglx_times = np.genfromtxt(
            os.path.join(spikeglx_sync_dir, 'sync_trial_start_off.csv'))
    else:
        spikeglx_times = None
    
    # Print shapes
    mworks_trial_nums = [len(x) for x in mworks_times]
    print(f'Mworks trial numbers: {mworks_trial_nums}')
    if open_ephys_times is not None:
        print(f'Open Ephys trial number: {len(open_ephys_times)}')

        # Debugging
        if session_dir[-16:] == 'Perle/2022-06-07':
            open_ephys_times = np.concatenate([
                open_ephys_times[:514],
                [
                    open_ephys_times[513] + 4.049028,
                    open_ephys_times[513] + 4.049028 + 2.899783,
                ],
                open_ephys_times[514:],
            ])

        if session_dir[-16:] == 'Elgar/2022-05-10':
            open_ephys_times = open_ephys_times[11:]
        
        # Another debugging thing
        open_ephys_times = open_ephys_times.tolist()
        to_pop = []
        for i in range(len(open_ephys_times) - 1):
            if open_ephys_times[i + 1] - open_ephys_times[i] < 0.1:
                to_pop.append(i + 1)
        for i in to_pop[::-1]:
            open_ephys_times.pop(i)
        open_ephys_times = np.array(open_ephys_times)

    if spikeglx_times is not None:
        print(f'SpikeGLX trial number: {len(spikeglx_times)}')

        # Debugging
        if session_dir[-16:] == 'Elgar/2022-05-04':
            spikeglx_times = spikeglx_times[10:]

        if session_dir[-16:] == 'Elgar/2022-09-07':
            spikeglx_times = spikeglx_times[:-3]

    # Concatenate and write mworks times
    if open_ephys_times is not None:
        mworks_times = _concatenate_mworks_times(mworks_times, open_ephys_times)
    else:
        mworks_times = _concatenate_mworks_times(mworks_times, spikeglx_times)
    json.dump(
        mworks_times.tolist(),
        open(os.path.join(mworks_sync_dir, 'trial_start_off_times.json'), 'w'),
    )

    # Now compute transform for open ephys and spikeglx
    if open_ephys_times is not None:
        _compute_transform(
            mworks_times, open_ephys_times, open_ephys_sync_dir, thresh=thresh)
    if spikeglx_times is not None:
        _compute_transform(
            mworks_times, spikeglx_times, spikeglx_sync_dir, thresh=thresh)


def main():
    """Copy sync pulses to data_processed."""

    high_thresh_sessions= {
        ('Perle', '2022-05-03'): 5.,
        ('Elgar', '2022-09-18'): 3.,
    }

    _process_session(os.path.join(_DATA_DIR, 'Elgar', '2022-09-19'))

    # for subject in os.listdir(_DATA_DIR):
    #     subject_dir = os.path.join(_DATA_DIR, subject)
    #     for session_date in os.listdir(subject_dir):
    #         session_dir = os.path.join(
    #             subject_dir, session_date)
            
    #         if (subject, session_date) in high_thresh_sessions:
    #             thresh = high_thresh_sessions[(subject, session_date)]
    #         else:
    #             thresh = None
            
    #         _process_session(session_dir, thresh=thresh)


if __name__ == "__main__":
    main()



# TODO:  Copy this to openmind and run it to fix the problematic sessions.
# Then run mworks_open_source...