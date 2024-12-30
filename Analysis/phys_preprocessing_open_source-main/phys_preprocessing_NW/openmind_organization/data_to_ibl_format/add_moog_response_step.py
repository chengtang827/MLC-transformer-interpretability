"""Add moog response step."""

import constants
import json
import numpy as np
import os

_METASTATE_IND = 4


def _get_response_location(moog_trial):
    response_moog = moog_trial[-1][_METASTATE_IND][1]['response']
    return response_moog


def _get_response_moog_step(moog_trial):
    count = -1
    for t in moog_trial[1:]:
        if t[_METASTATE_IND][1]['phase'] == 'response':
            count += 1
        if t[_METASTATE_IND][1]['response'] is not None:
            return count
    return None


def _process_session(raw_moog_path, trials_path, behavior_target_dir):
    
    # Read stimuli_init from behavior_target_dir
    response_location_path = os.path.join(
        behavior_target_dir, 'trials.response.location.json')
    response_location = json.load(open(response_location_path, 'r'))

    # Read trials from trials_path
    trials = json.load(open(trials_path, 'r'))

    # Sanity check
    if len(trials) != len(response_location):
        raise ValueError(
            f'len(trials) = {len(trials)} but len(response_location) = '
            f'{len(response_location)}'
        )
    
    moog_sessions = sorted(os.listdir(raw_moog_path))

    # Extract response moog step
    moog_session_index = 0
    old_moog_trial_num = -1
    response_moog_steps = []
    for trial_index, trial in enumerate(trials):
        
        # Read raw moog trial
        moog_trial_num = trial['moog_data']['trial_num']
        moog_session_num = trial['moog_data']['moog_session_num']
        if moog_sessions[0] in ['2022_05_17_14_34_16', '2022_08_21_17_16_48']:
            moog_session_num -= 1
        if moog_session_num >= len(moog_sessions):
            import pdb; pdb.set_trace()
            raise ValueError(
                f'moog_session_num = {moog_session_num} but len(moog_sessions) '
                f'= {len(moog_sessions)}'
            )

        # Update moog_session_index if necessary
        if moog_trial_num < old_moog_trial_num:
            moog_session_index += 1
        moog_trial_path = os.path.join(
            raw_moog_path,
            moog_sessions[moog_session_num],
            str(moog_trial_num).zfill(5)
        )
        moog_trial = json.load(open(moog_trial_path, 'r'))

        # Sanity check response
        old_response = response_location[trial_index]
        new_response = _get_response_location(moog_trial)
        if old_response != new_response:
            import pdb; pdb.set_trace()
            a = 4

        # Extract response time
        response_moog_step = _get_response_moog_step(moog_trial)
        response_moog_steps.append(response_moog_step)

        old_moog_trial_num = moog_trial_num

    # Write new stimuli_init
    response_moog_steps_path = os.path.join(
        behavior_target_dir, 'trials.response.moog_step_index.json')
    print(f'Writing to {response_moog_steps_path}')
    json.dump(response_moog_steps, open(response_moog_steps_path, 'w'))

    return


def main():
    """Convert task data to IBL format."""

    for monkey in os.listdir(constants.SOURCE_BASE_DIR):
        monkey_source_dir = os.path.join(constants.SOURCE_BASE_DIR, monkey)
        monkey_target_dir = os.path.join(
            constants.TARGET_BASE_DIR, constants.MONKEY_TO_ID[monkey])
        if not os.path.exists(monkey_target_dir):
            os.makedirs(monkey_target_dir)
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
            behavior_target_dir = os.path.join(
                monkey_target_dir, session_date, '001', 'behavior')
            if not os.path.exists(behavior_target_dir):
                raise ValueError(
                    f'Does not exist behavior_target_dir {behavior_target_dir}'
                )
            _process_session(raw_moog_path, trials_path, behavior_target_dir)


if __name__ == "__main__":
    main()

