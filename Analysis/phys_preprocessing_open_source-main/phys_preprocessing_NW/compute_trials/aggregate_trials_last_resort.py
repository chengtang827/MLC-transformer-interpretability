"""This file has functions that help align trials.

The main function in this file is _find_all_corresponding_phys_trials(). This
function can be used to align physiology trials with mworks trials. It first
attempts to align based on trial number (inferred from zero and one sync signals
in physiology), and if that fails tries to match trials based on relative phase
times.

This does not work very well and is brittle. These functions should only be used
as a last resort if the total number of physiology and mworks trials differs so
there is not a clear one-to-one correspondence.

If you wish to use any of the code in this file, first carefully read and
understand it, then remove the deprecation warnings and use with caution.
"""

import numpy as np
import warnings


def _match_relative_phase_times(mworks_rel_phase_times, phys_rel_phase_times):
    """Check if the mworks and phys relative phase times are similar.
    
    The physiology relative phase times typically include fixation onset time,
    so we only consider the last num_mworks_phases relative phase times in the
    phys trial.
    """

    raise warnings.DeprecationWarning(
        '_match_relative_phase_times is depracated. See source code '
        'documentation.'
    )
    
    # Remove reset phase time, since that often differs between mworks and phys
    mworks_rel_phase_times = mworks_rel_phase_times[:-1]
    
    num_mworks_phases = len(mworks_rel_phase_times)
    num_phys_phases = len(phys_rel_phase_times)
    
    if num_phys_phases < num_mworks_phases:
        return False
    
    # Try to align to first phys phase
    align_to_beginning = np.allclose(
        phys_rel_phase_times[:num_mworks_phases],
        mworks_rel_phase_times,
        atol=1e-3,
    )
    if align_to_beginning:
        return True
    
    # Sometimes the phys trial will include a fixation plse at the beginning,
    # so try to align to the second pulse
    if num_phys_phases > num_mworks_phases:
        align_to_second_pulse = np.allclose(
            phys_rel_phase_times[1:1 + num_mworks_phases],
            mworks_rel_phase_times,
            atol=1e-3,
        )
        if align_to_second_pulse:
            return True
        
    # Cannot align
    return False


def _find_corresponding_phys_trial(mworks_trial, phys_trials):
    """Finds corresponding physiology trial for given mworks trial."""
    
    raise warnings.DeprecationWarning(
        '_find_corresponding_phys_trial is depracated. See source code '
        'documentation.'
    )

    # First check if trial num agrees
    mw_trial_num = mworks_trial['trial_num']
    phys_trial_nums = np.array([t['trial_num'] for t in phys_trials])
    matching_trial_num_ind = np.argwhere(phys_trial_nums == mw_trial_num)
    if len(matching_trial_num_ind) == 1:
        matching_trial_num_ind = matching_trial_num_ind[0, 0]
        return matching_trial_num_ind, phys_trials[matching_trial_num_ind]
    else:
        # Next check relative phase times
        mworks_rel_phase_times = mworks_trial['relative_phase_times']
        phys_rel_phase_times = [
            t['relative_phase_times'] for t in phys_trials
        ]
        matching_rel_phase_times = [
            _match_relative_phase_times(mworks_rel_phase_times, phys_rel)
            for phys_rel in phys_rel_phase_times
        ]
        if len(matching_rel_phase_times) == 0:
            print('Could not find corresponding phys trial.')
            return None, None
        matching_trial_num_ind = np.argwhere(matching_rel_phase_times)
        if len(matching_trial_num_ind) > 0:
            matching_trial_num_ind = matching_trial_num_ind[0, 0]
            return matching_trial_num_ind, phys_trials[matching_trial_num_ind]
        else:
            print('Could not find corresponding phys trial.')
            return None, None


def _find_all_corresponding_phys_trials(mworks_trials, phys_trials):
    """Finds corresponding physiology trials for given mworks trials.
    
    Args:
        mworks_trials: List of mworks trials. Each element is a dictionary with
            fields including 'trial_num' and 'relative_phase_times'.
        phys_trials: List of physiology trials. Each element is a dictionary
            with fields including 'trial_num' and 'relative_phase_times'.

    Returns:
        corresponding_phys_trials: List of physiology trials of the same length
            as mworks_trials. Each element is either a dictionary representing a
            physiology trial or None if no matching phys trial can be found for
            the given mworks trial.
    """

    raise warnings.DeprecationWarning(
        '_find_all_corresponding_phys_trials is depracated. See source code '
        'documentation.'
    )
    
    start_index = 0
    corresponding_phys_trials = []
    for i, t in enumerate(mworks_trials):
        if i % 10 == 0:
            print(i)
        if start_index >= len(phys_trials):
            corresponding_phys_trials.append(None)
        else:
            ind, phys_trial = _find_corresponding_phys_trial(
                t, phys_trials[start_index: start_index + 10])
            corresponding_phys_trials.append(phys_trial)
            if ind is not None:
                start_index = start_index + ind + 1
    
    not_found_num = np.sum([x is None for x in corresponding_phys_trials])
    not_found_message = (
        f'Could not find corresponding phys trial for {not_found_num} mworks '
        'trials.'
    )
    if not_found_num > 10:
        raise ValueError(not_found_message)
    else:
        print(not_found_message)
        
    return corresponding_phys_trials
