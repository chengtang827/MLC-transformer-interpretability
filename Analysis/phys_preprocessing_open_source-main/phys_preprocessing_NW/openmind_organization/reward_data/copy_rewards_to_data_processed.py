"""Copy reward data to data_processed."""

import json
import os
import numpy as np
import sys

sys.path.append('../../utils')
import mworks_reader

_DATA_PROCESSED_BASE_DIR = '/om4/group/jazlab/nwatters/multi_prediction/data_processed'
_RAW_DATA_BASE_DIR = '/om4/group/jazlab/nwatters/multi_prediction/phys_data'


def _get_reward_line_data(mwk_file):
    print(f'loading mworks data from {mwk_file}')
    values = []
    times = []
    found_code = False
    reader = mworks_reader.MWK2Reader(mwk_file)
    with reader as event_file:
        for code, time, data in event_file:
            if not found_code:
                name_to_code = dict((data[c]['tagname'], c) for c in data)
                reward_line_code = name_to_code['reward_line']
                found_code = True
                continue
            
            if code == reward_line_code:
                times.append(time / 1000000.)
                values.append(data)

    times = np.array(times)
    values = np.array(values)

    return times, values


def _process_session(session_processed_data_dir, session_raw_data_dir):
    # First, read trials in processed data
    trials_path = os.path.join(
        session_processed_data_dir, 'trial_structure', 'trials')
    print(f'session_processed_data_dir: {session_processed_data_dir}')
    print(f'session_raw_data_dir: {session_raw_data_dir}')
    if not os.path.exists(trials_path):
        print(f'trials_path {trials_path} does not exist')
        return
    trials = json.load(open(trials_path, 'r'))

    # Figure out which mworks sessions are used
    mworks_session_nums = [t['mworks_session_num'] for t in trials]
    mworks_session_nums = np.unique(mworks_session_nums)

    # Find all mworks logs in raw_data_dir
    mworks_raw_data_dir = os.path.join(
        session_raw_data_dir, 'raw_data', 'behavior', 'mworks')
    mworks_log_files = sorted(
        [x for x in os.listdir(mworks_raw_data_dir) if x[-5:] == '.mwk2'])
    mworks_log_files = [
        os.path.join(mworks_raw_data_dir, x) for x in mworks_log_files]
    print('mworks_log_files')
    print(mworks_log_files)

    # Extract and write reward data for desired sessions
    for session_num in mworks_session_nums:
        reward_times, reward_values = _get_reward_line_data(
            mworks_log_files[session_num])
        write_dir = os.path.join(
            session_processed_data_dir,
            'reward_line',
            'mwork_session_num_' + str(session_num),
        )
        if not os.path.exists(write_dir):
            os.makedirs(write_dir)
        print(f'Writing reward data to {write_dir}')
        np.save(os.path.join(write_dir, 'reward_times.npy'), reward_times)
        np.save(os.path.join(write_dir, 'reward_values.npy'), reward_values)
    
    return
    

def main():
    """Write MWorks reward data to data_processed for each session."""

    print('STARTED copy_rewards_to_data_processed.py\n')

    for monkey in ['Perle', 'Elgar']:
        monkey_processed_data_dir = os.path.join(
            _DATA_PROCESSED_BASE_DIR, monkey)
        monkey_raw_data_dir = os.path.join(_RAW_DATA_BASE_DIR, monkey)
        for session_date in os.listdir(monkey_processed_data_dir):
            print(f'\n{session_date}')
            session_processed_data_dir = os.path.join(
                monkey_processed_data_dir, session_date)
            session_raw_data_dir = os.path.join(
                monkey_raw_data_dir, session_date)
            _process_session(session_processed_data_dir, session_raw_data_dir)
            
    print('\nFINISHED copy_rewards_to_data_processed.py')

    return


if __name__ == "__main__":
    main()

