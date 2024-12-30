"""Get spike times per cluster."""

import json
import numpy as np
import os
from matplotlib import pyplot as plt

from sklearn import linear_model as sklearn_linear_model

_DATA_PROCESSED_BASE_DIR = '/om4/group/jazlab/nwatters/multi_prediction/data_processed'


def _get_reward_times_durations(reward_times, reward_values):
    on_off_inds = np.argwhere(
        np.convolve(2 * reward_values - 1., [-1, 1], mode='valid') == 2.)
    if len(on_off_inds) == 0:
        return np.array([]), np.array([])
    on_off_inds = on_off_inds[:, 0]
    r_times = reward_times[on_off_inds]
    r_durations = reward_times[on_off_inds + 1] - reward_times[on_off_inds]
    return r_times, r_durations


def _process_session(session_dir):
    print(f'Processing {session_dir}')

    ############################################################################
    #### READ TRIALS
    ############################################################################

    trials_path = os.path.join(session_dir, 'trial_structure/trials')
    trials = json.load(open(trials_path, 'r'))
    mworks_session_num = np.array([t['mworks_session_num'] for t in trials])
    unique_mworks_session_nums = np.sort(np.unique(mworks_session_num))
    trial_inds_per_session_num = [
        np.sort(np.squeeze(np.argwhere(mworks_session_num == x)))
        for x in unique_mworks_session_nums
    ]

    ############################################################################
    #### PROCESS EACH MWORKS SESSION
    ############################################################################

    reward_line_base_dir = os.path.join(session_dir, 'reward_line')
    trial_inds_without_moog = []
    moog_offset = 0

    # Session Perle/2022-04-28 had a short second session that's messed up
    if (session_dir.split('/')[-2] == 'Perle' and 
            session_dir.split('/')[-1] == '2022-04-28'):
        unique_mworks_session_nums = unique_mworks_session_nums[:1]
        for i in trial_inds_per_session_num[1]:
            trials[i]['moog_data'] = None
            trial_inds_without_moog.append(i)
        trial_inds_per_session_num = trial_inds_per_session_num[:1]

    for session_num, trial_inds in zip(
            unique_mworks_session_nums, trial_inds_per_session_num):
        print(f'session_num: {session_num}')
        
        # Get times and durations of rewarding events
        reward_line_dir = os.path.join(
            reward_line_base_dir, 'mwork_session_num_' + str(session_num))
        reward_times = np.load(
            os.path.join(reward_line_dir, 'reward_times.npy'))
        reward_values = np.load(
            os.path.join(reward_line_dir, 'reward_values.npy'))
        r_times, r_durations = _get_reward_times_durations(
            reward_times, reward_values)

        # Associate those rewarding events with trials
        gatling_reward_trials = 0
        no_moog_inds = []
        trial_reward_times = []
        trial_reward_durations = []
        for i in trial_inds:
            t = trials[i]
            mworks_t_start = t['mworks_t_start']
            mworks_t_end = t['mworks_t_end']

            moog_data_trial_ind = i + moog_offset
            if moog_data_trial_ind not in trial_inds:
                moog_data = None
            else:
                moog_data = trials[moog_data_trial_ind]['moog_data']
            if moog_data is None or 'success' not in moog_data:
                no_moog_inds.append(i)
                trial_reward_times.append(None)
                trial_reward_durations.append(0.)
                continue
            success = moog_data['success']
            trials[i]['moog_data'] = moog_data

            reward_ind = np.argwhere(
                np.logical_and(r_times > mworks_t_start, r_times < mworks_t_end)
            )
            if len(reward_ind) == 0:
                if success:
                    print(f'Trial {i} had success but no reward found.')
                    raise ValueError
                trial_reward_times.append(None)
                trial_reward_durations.append(0.)
            elif len(reward_ind) == 1:
                reward_ind = reward_ind[0, 0]
                if not success:
                    gatling_reward_trials += 1
                    no_moog_inds.append(i)
                    trial_reward_times.append(None)
                    trial_reward_durations.append(0.)
                    print(f'Trial {i} had no success but reward was found.')
                    continue
                reward_time = np.round(
                    r_times[reward_ind] - mworks_t_start, decimals=4)
                trial_reward_times.append(reward_time)
                trial_reward_durations.append(
                    np.round(r_durations[reward_ind], decimals=4))

                # Verify that reward time happened near reveal phase
                reveal_phase_time = t['mworks_relative_phase_times'][4]
                if np.abs(reveal_phase_time - reward_time) > 0.1:
                    raise ValueError(
                        f'reveal_phase_time is {reveal_phase_time} but '
                        f'reward_time is {reward_time}'
                    )
            else:
                gatling_reward_trials += 1
                no_moog_inds.append(i)
                trial_reward_times.append(None)
                trial_reward_durations.append(0.)
                print(
                    f'For trial {i} found multiple reward indices '
                    f'{reward_ind}'
                )
                continue

        print(f'gatling_reward_trials: {gatling_reward_trials}')
        moog_offset += 1

        if not trial_reward_times:
            raise ValueError('Cannot align rewards')
        
        trial_inds_without_moog.extend(no_moog_inds)
        for i, r_time, r_duration in zip(
                trial_inds,  trial_reward_times, trial_reward_durations):
            trials[i]['reward_time'] = r_time
            trials[i]['reward_duration'] = r_duration

    # Make sure there are not too many trials without moog data
    if len(trial_inds_without_moog) > 10:
        raise ValueError(
            f'Found {len(trial_inds_without_moog)} trials without moog data')
    trial_inds_without_moog = sorted(trial_inds_without_moog)
    for i in trial_inds_without_moog[::-1]:
        trials.pop(i)

    # Write updated trials
    trials_curated_path = os.path.join(
        session_dir, 'trial_structure/trials_curated')
    json.dump(trials, open(trials_curated_path, 'w'))
        
    return


def main():
    """Write MWorks reward data to data_processed for each session."""

    print('STARTED copy_rewards_to_data_processed.py\n')

    # _process_session(
    #     '/Users/nicholaswatters/Desktop/grad_school/research/mehrdad/'
    #     'multi_prediction/phys/phys_data/Elgar/2022-05-07'
    # )

    for monkey in ['Perle', 'Elgar']:
        monkey_data_dir = os.path.join(_DATA_PROCESSED_BASE_DIR, monkey)
        for session_date in os.listdir(monkey_data_dir):
            print(f'\n{session_date}')
            session_data_dir = os.path.join(monkey_data_dir, session_date)
            _process_session(session_data_dir)
            
    print('\nFINISHED copy_rewards_to_data_processed.py')

    return


if __name__ == "__main__":
    main()


