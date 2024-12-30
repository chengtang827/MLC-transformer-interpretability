"""Compute timescale offset between mworks in data_processed and open-source.

Run this after computer_timescale_transforms.py, which is run after
copy_sync_pulses.py.

Done, ran
"""

import json
import numpy as np
import os

_DATA_PROCESSED_DIR = (
    '/om4/group/jazlab/nwatters/multi_prediction/data_processed'
)
_DATA_OPEN_SOURCE_DIR = (
    '/om4/group/jazlab/nwatters/multi_prediction/data_open_source/Subjects'
)
# _DATA_PROCESSED_DIR = (
#     '/Users/nicholaswatters/Desktop/grad_school/research/mehrdad/'
#     'multi_prediction/phys/data_processed'
# )
# _DATA_OPEN_SOURCE_DIR = (
#     '/Users/nicholaswatters/Desktop/grad_school/research/mehrdad/'
#     'multi_prediction/phys/data_open_source/Subjects'
# )
_MONKEY_TO_ID = {
    'Perle': 'monkey0',
    'Elgar': 'monkey1',
}
_MAX_OFFSET_TRIALS = 2
_WINDOW = 5


def _process_session(session_processed_dir, session_open_source_dir):
    print(f'\nsession_processed_dir: {session_processed_dir}')
    print(f'session_open_source_dir: {session_open_source_dir}')

    # Read mworks times from processed and open_source directories
    times_processed_path = os.path.join(
        session_processed_dir, 'sync_pulses', 'mworks',
        'trial_start_off_times.json')
    times_processed = np.array(json.load(open(times_processed_path, 'r')))

    times_open_source_path = os.path.join(
        session_open_source_dir, 'task', 'trials.start_times.json')
    times_open_source = np.array(json.load(open(times_open_source_path, 'r')))

    # Look at inter trial intervals to match them
    diffs_processed = times_processed[1:] - times_processed[:-1]
    diffs_open_source = times_open_source[1:] - times_open_source[:-1]

    u_diffs_processed = times_processed[100:] - times_processed[:-100]
    u_diffs_open_source = times_open_source[100:] - times_open_source[:-100]

    # Compute match of diffs_processed and diffs_open_source
    errors = []
    error_inds = []
    for i in range(-_MAX_OFFSET_TRIALS, _MAX_OFFSET_TRIALS):
        if i < 0:
            tmp_diffs_processed = diffs_processed[-1 * i:]
            tmp_diffs_open_source = np.copy(diffs_open_source)
        else:
            tmp_diffs_open_source = diffs_open_source[i:]
            tmp_diffs_processed = np.copy(diffs_processed)
        
        errors.append(np.min([
            np.nanmax(np.abs(
                tmp_diffs_open_source[j: j + _WINDOW] -
                tmp_diffs_processed[j: j + _WINDOW]))
            for j in range(3)
        ]))
        error_inds.append(i)
    
    if np.min(errors) > 0.025:
        import pdb; pdb.set_trace()
        raise ValueError(errors)
    min_error_ind = error_inds[np.argmin(errors)]
    buffer = 3
    if min_error_ind < 0:
        t_processed = times_processed[-1 * min_error_ind + buffer]
        t_open_source = times_open_source[buffer]
    else:
        t_open_source = times_open_source[min_error_ind + buffer]
        t_processed = times_processed[buffer]
    
    open_source_minus_processed = t_open_source - t_processed
    if not np.isfinite(open_source_minus_processed):
        raise ValueError(
            f'open_source_minus_processed = {open_source_minus_processed}')
    print(f'open_source_minus_processed = {open_source_minus_processed}')
    
    # Save open_source_minus_processed
    open_source_minus_processed_path = os.path.join(
        session_processed_dir,
        'sync_pulses',
        'mworks',
        'open_source_minus_processed'
    )
    print(f'Writing to {open_source_minus_processed_path}')
    json.dump(
        open_source_minus_processed,
        open(open_source_minus_processed_path, 'w'),
    )

    return


def main():
    """Copy sync pulses to data_processed."""

    for monkey, monkey_id in _MONKEY_TO_ID.items():
        monkey_processed_dir = os.path.join(_DATA_PROCESSED_DIR, monkey)
        monkey_open_source_dir = os.path.join(_DATA_OPEN_SOURCE_DIR, monkey_id)
        for session_date in os.listdir(monkey_open_source_dir):
            print((monkey, session_date))
            session_processed_dir = os.path.join(
                monkey_processed_dir, session_date)
            session_open_source_dir = os.path.join(
                monkey_open_source_dir, session_date, '001')
            _process_session(session_processed_dir, session_open_source_dir)


if __name__ == "__main__":
    main()


"""
MONKEY='Elgar'
MONKEY_ID='monkey1'
DATE='2022-10-08'
WRITE_P=data_processed/$MONKEY/$DATE/sync_pulses/mworks
mkdir -p $WRITE_P
scp -r nwatters@openmind7.mit.edu:/om4/group/jazlab/nwatters/multi_prediction/$WRITE_P/trial_start_off_times.json $WRITE_P/
WRITE_OS=data_open_source/Subjects/$MONKEY_ID/$DATE/001/task
mkdir -p $WRITE_OS
scp -r nwatters@openmind7.mit.edu:/om4/group/jazlab/nwatters/multi_prediction/$WRITE_OS/trials.start_times.json $WRITE_OS/
"""