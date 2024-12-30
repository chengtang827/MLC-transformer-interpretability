"""Copy sync pulses to data_processed."""

import json
import numpy as np
import os

_RAW_DATA_DIR = '/om4/group/jazlab/nwatters/multi_prediction/phys_data'
_PROCESSED_DATA_DIR = '/om4/group/jazlab/nwatters/multi_prediction/data_processed'


def _process_session(raw_session_dir, processed_session_dir):
    
    read_dir = os.path.join(
        raw_session_dir, 'trial_structure', 'sync_events', 'open_ephys')
    if not os.path.exists(read_dir):
        return
    write_dir = os.path.join(processed_session_dir, 'sync_pulses', 'open_ephys')

    photodiode_times_path = os.path.join(read_dir, 'photodiode_times.csv')
    if not os.path.exists(photodiode_times_path):
        raise ValueError(
            f'photodiode_times_path {photodiode_times_path} does not exist')

    first_photodiode_time = np.genfromtxt(
        photodiode_times_path, max_rows=1).item()

    write_path = os.path.join(write_dir, 'recording_start_time')
    print(f'Writing {first_photodiode_time} to {write_path}')
    json.dump(first_photodiode_time, open(write_path, 'w'))


def main():
    """Copy sync pulses to data_processed."""

    for subject in os.listdir(_PROCESSED_DATA_DIR):
        raw_subject_dir = os.path.join(_RAW_DATA_DIR, subject)
        processed_subject_dir = os.path.join(_PROCESSED_DATA_DIR, subject)
        for session_date in os.listdir(processed_subject_dir):
            raw_session_dir = os.path.join(raw_subject_dir, session_date)
            processed_session_dir = os.path.join(
                processed_subject_dir, session_date)
            _process_session(raw_session_dir, processed_session_dir)


if __name__ == "__main__":
    main()

