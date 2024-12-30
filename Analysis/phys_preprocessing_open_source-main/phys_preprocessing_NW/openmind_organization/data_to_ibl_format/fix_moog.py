"""Fix errors in data.

In the initial IBL conversion there was some flaw, inherited from a flat in the
trial dictionary computation. The result was that the target identities were
mis-assigned. This goes through every trial and validates/corrects all the MOOG
data, using the raw MOOG logs.
"""

import constants
import json
import numpy as np
import os

_METASTATE_IND = 4
_INIT_PREY_IND = 1
_INIT_CUE_IND = 3

_INIT_SPRITE_X_IND = 0
_INIT_SPRITE_Y_IND = 1
_INIT_SPRITE_METADATA_IND = 14

_EPSILON = 1e-6


def _get_true_stim(moog_trial, trial):

    # Sanity check response
    response_moog = moog_trial[-1][_METASTATE_IND][1]['response']
    response_trial = trial['moog_data']['response']
    if response_moog is None or response_trial is None:
        if response_moog is not None or response_trial is not None:
            raise ValueError(
                f'response_moog = {response_moog} but response_trial = '
                f'{response_trial}'
            )
    elif not np.allclose(response_moog, response_trial, atol=_EPSILON):
        raise ValueError(
            f'response_moog = {response_moog} but response_trial = '
            f'{response_trial}'
        )

    # Get true response
    init_full_state = moog_trial[0][0]
    init_prey_state = init_full_state[_INIT_PREY_IND][1]
    init_cue_state = init_full_state[_INIT_CUE_IND][1]
    prey_init = [
        {
            'x': s[_INIT_SPRITE_X_IND],
            'y': s[_INIT_SPRITE_Y_IND],
            'id': s[_INIT_SPRITE_METADATA_IND]['id'],
            'target': False,
        }
        for s in init_prey_state
    ]
    target_prey_ind = (
        init_cue_state[0][_INIT_SPRITE_METADATA_IND]['target_prey_ind']
    )
    prey_init[target_prey_ind]['target'] = True

    # Sanity check prey positions and identities and add velocity
    trial_prey_init = trial['moog_data']['prey_init']
    trial_prey_ids = np.array([x['id'] for x in trial_prey_init])
    for prey in prey_init:
        corresponding_trial_prey_id = np.argwhere(trial_prey_ids == prey['id'])
        if not len(corresponding_trial_prey_id) == 1:
            raise ValueError(
                f'prey = {prey} but trial_prey_ids = {trial_prey_ids}'
            )
        if not len(corresponding_trial_prey_id[0]) == 1:
            raise ValueError(
                f'prey = {prey} but trial_prey_ids = {trial_prey_ids}'
            )
        trial_prey = trial_prey_init[corresponding_trial_prey_id[0][0]]
        if not np.isclose(trial_prey['x'], prey['x'], atol=_EPSILON):
            raise ValueError(f'prey = {prey} but trial_prey = {trial_prey}')
        if not np.isclose(trial_prey['y'], prey['y'], atol=_EPSILON):
            raise ValueError(f'prey = {prey} but trial_prey = {trial_prey}')
        prey['x_vel'] = trial_prey['x_vel']
        prey['y_vel'] = trial_prey['y_vel']

    return prey_init


def _process_session(raw_moog_path, trials_path, task_target_dir):
    
    # Read stimuli_init from task_target_dir
    stimuli_init_path = os.path.join(
        task_target_dir, 'trials.stimuli_init.json')
    stimuli_init = json.load(open(stimuli_init_path, 'r'))

    # Read trials from trials_path
    trials = json.load(open(trials_path, 'r'))

    # Sanity check
    if len(trials) != len(stimuli_init):
        raise ValueError(
            f'len(trials) = {len(trials)} but len(stimuli_init) = '
            f'{len(stimuli_init)}'
        )
    
    moog_sessions = sorted(os.listdir(raw_moog_path))
    print(f'Number of moog sessions: {len(moog_sessions)}')

    # Fix stimuli_init
    moog_session_index = 0
    old_moog_trial_num = -1
    true_stimuli_init = []
    for trial_index, trial in enumerate(trials):
        
        # Read raw moog trial
        moog_trial_num = trial['moog_data']['trial_num']
        moog_session_num = trial['moog_data']['moog_session_num']
        if moog_sessions[0] in ['2022_05_17_14_34_16', '2022_08_21_17_16_48']:
            moog_session_num -= 1
        if moog_session_num >= len(moog_sessions):
            import pdb; pdb.set_trace()
            if trial_index < len(trials) - 1:
                import pdb; pdb.set_trace()
            true_stimuli_init.append(stimuli_init[trial_index])
            continue

        if moog_trial_num < old_moog_trial_num:
            moog_session_index += 1
        moog_trial_path = os.path.join(
            raw_moog_path,
            moog_sessions[moog_session_num],
            str(moog_trial_num).zfill(5)
        )
        moog_trial = json.load(open(moog_trial_path, 'r'))

        # Sanity check raw moog trial and get true stimulus
        true_stim = _get_true_stim(moog_trial, trial)
        true_stimuli_init.append(true_stim)

    # Write new stimuli_init
    # json.dump(true_stimuli_init, open(stimuli_init_path, 'w'))

    return


def main():
    """Convert task data to IBL format."""

    # for monkey in os.listdir(constants.SOURCE_BASE_DIR)[:1]:
    for monkey in os.listdir(constants.SOURCE_BASE_DIR):
        monkey_source_dir = os.path.join(constants.SOURCE_BASE_DIR, monkey)
        monkey_target_dir = os.path.join(
            constants.TARGET_BASE_DIR, constants.MONKEY_TO_ID[monkey])
        if not os.path.exists(monkey_target_dir):
            os.makedirs(monkey_target_dir)
        # for session_date in os.listdir(monkey_source_dir):
        for session_date in os.listdir(monkey_source_dir):
            if session_date == '2022-10-08':
                continue
            raw_moog_path = os.path.join(
                constants.RAW_BASE_DIR,
                monkey,
                session_date,
                'raw_data/behavior/moog',
            )
            trials_path = os.path.join(
                monkey_source_dir,
                session_date,
                'trial_structure',
                'trials_curated',
            )
            print(f'monkey: {monkey}, session_date: {session_date}')
            task_target_dir = os.path.join(
                monkey_target_dir, session_date, '001', 'task')
            if not os.path.exists(task_target_dir):
                raise ValueError(
                    f'Does not exist task_target_dir {task_target_dir}'
                )
            _process_session(raw_moog_path, trials_path, task_target_dir)


if __name__ == "__main__":
    main()

