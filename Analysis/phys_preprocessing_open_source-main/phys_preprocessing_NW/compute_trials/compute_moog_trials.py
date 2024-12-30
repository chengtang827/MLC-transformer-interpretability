"""Compute and write MOOG trials.

This script generates and saves a list of "trials", based on MOOG logs. Each
trial is a dictionary with all needed data about the trial.
"""

import json
import itertools
import os
import sys
import numpy as np
import re
import copy

sys.path.append('../utils')
import serialize

# Information about indices of useful data in MOOG logs
_TIME_IND = 0
_REWARD_IND = 1
_METASTATE_IND = 4
_STATE_IND = 5
_STATE_PREY_IND = 0
_STATE_FIXATION_IND = 4
_SPRITE_SCALE_IND = 5
_INITIALIZATION_PREY_IND = 1
_INITIALIZATION_METADATA_IND = 14
_BACKGROUND_INDICES_IND = 7


def _dummy_trial():
    trial = {
        'trial_num': None,
        'total_trial_num': None,
    }
    return trial


def _load_moog_trial(trial_path):
    """Compute data from moog trial path"""
    moog_trial = json.load(open(trial_path, 'r'))
    stimulus = copy.deepcopy(moog_trial[0][1]['stim'])
    prey_init = moog_trial[0][0][_INITIALIZATION_PREY_IND][1]
    background_indices = moog_trial[0][0][_BACKGROUND_INDICES_IND][1]
    prey_metadata = [x[_INITIALIZATION_METADATA_IND] for x in prey_init]
    trial_dict = {
        'trial_num': int(trial_path.split('/')[-1]),
        'prey_init': [
            {
                'x': s['x'],
                'y': s['y'],
                'x_vel': s['x_vel'],
                'y_vel': s['y_vel'],
                'id': i,
                'target': m['target'],
            }
            for s, i, m in zip(
                stimulus['state'], stimulus['prey_ids'], prey_metadata,
            )
        ],
        'visible_steps': stimulus['visible_steps'],
        'blank': stimulus['blank'],
        'filename': stimulus['filename'],
        'repeats': stimulus['repeats'],
        'cue_steps': stimulus['cue_steps'],
        'blank': stimulus['blank'],
        'background_indices': background_indices,
    }

    # Add phase list and delay steps
    phase_list = np.array(
        [x[_METASTATE_IND][1]['phase'] for x in moog_trial[1:]]
    )
    trial_dict['phase_list'] = phase_list
    trial_dict['delay_steps'] = np.sum(phase_list == 'delay')

    # Add total_trial_num (which is set from mworks)
    trial_dict['total_trial_num'] = (
        moog_trial[1][_METASTATE_IND][1]['total_trial_num']
    )

    # Add total_trial_num (which is set from mworks)
    trial_dict['total_trial_num'] = (
        moog_trial[1][_METASTATE_IND][1]['total_trial_num']
    )

    # Add prey position during visible and stuff
    prey_states = np.array([
        [s[:4] for s in x[_STATE_IND][_STATE_PREY_IND][1]]
        for x in moog_trial[1:]
    ])
    trial_dict['visible_prey_states'] = prey_states[phase_list == 'visible']
    trial_dict['delay_prey_states'] = prey_states[phase_list == 'delay']
    trial_dict['cue_prey_states'] = prey_states[phase_list == 'cue']
    trial_dict['response_prey_states'] = prey_states[phase_list == 'response']
    trial_dict['reveal_prey_states'] = prey_states[phase_list == 'reveal']

    # Add fixation cross scale
    fixation_scale = [
        t[_STATE_IND][_STATE_FIXATION_IND][1][0][_SPRITE_SCALE_IND]
        for t in moog_trial[1:]
    ]
    trial_dict['fixation_cross_scale'] = fixation_scale

    # Add step times
    step_times = np.array([x[_TIME_IND][1] for x in moog_trial[1:]])
    trial_dict['step_times'] = step_times - step_times[0]

    # Add response
    final_metastate = moog_trial[-1][_METASTATE_IND][1]
    trial_dict['success'] = final_metastate['success']
    trial_dict['response'] = final_metastate['response']
    trial_dict['broke_fixation'] = final_metastate['broke_fixation']
    trial_dict['response_prey'] = final_metastate['response_prey']
    trial_dict['error'] = final_metastate['error']

    # Add reward
    reward = np.array([x[_REWARD_IND][1] for x in moog_trial[1:]])
    trial_dict['reward'] = np.max(reward)

    return serialize.serialize(trial_dict)


def _load_moog_data(moog_dir):
    re_trial_file = re.compile('[0-9]{5}')
    moog_data = [
        _load_moog_trial(os.path.join(moog_dir, x))
        for x in sorted(os.listdir(moog_dir)) if re_trial_file.match(x)
    ]
    return moog_data


def main():
    """Compute and write MOOG data for each trial.

    This data is a list of dictionaries, one for each valid trial. It is
    JSON-serialized and written to a file $write_dir/moog_trials.
    
    Args:
        behavior_dir: String. Full path to moog raw data directory.
        write_dir: String. Full path to directory to write raw moog trials.
    """

    print('STARTED compute_moog_trials.py')

    ############################################################################
    ####  Find Data Paths
    ############################################################################

    behavior_dir = sys.argv[1]
    print(f'behavior_dir: {behavior_dir}')
    write_dir = sys.argv[2]
    print(f'write_dir: {write_dir}')

    # behavior_dir = (
    #     '/Users/nicholaswatters/Desktop/grad_school/research/mehrdad/'
    #     'multi_prediction/phys/phys_data/Perle/2022-05-28/trial_structure/moog'
    # )
    # write_dir = (
    #     '/Users/nicholaswatters/Desktop/grad_school/research/mehrdad/'
    #     'multi_prediction/phys/phys_data/Perle/2022-05-28/trial_structure'
    # )

    # Find all the moog files
    moog_re = re.compile('...._.._.._.._.._..')
    moog_dirs = sorted([
        os.path.join(behavior_dir, x) for x in os.listdir(behavior_dir)
        if moog_re.match(x)
    ])
    print(f'moog_dirs: {moog_dirs}')

    # Load moog data
    print('LOADING MOOG DATA')
    moog_data = [_load_moog_data(moog_dir) for moog_dir in moog_dirs]
    for i, data in enumerate(moog_data):
        for d in data:
            d['moog_session_num'] = i
        if i < len(moog_data) - 1:
            data.append(_dummy_trial())
    moog_trials = list(itertools.chain(*moog_data))
    print(f'Number of moog trials: {len(moog_trials)}')

    print('SAVING MOOG TRIALS METADATA')

    write_path = os.path.join(write_dir, 'moog_trials')
    print(f'write_path: {write_path}')
    json.dump(moog_trials, open(write_path, 'w'))

    print('FINISHED compute_moog_trials.py')

    return


if __name__ == "__main__":
    main()

