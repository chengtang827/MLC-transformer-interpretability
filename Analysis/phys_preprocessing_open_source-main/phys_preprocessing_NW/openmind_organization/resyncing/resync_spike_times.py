"""Compute transformed spike times per probe.

Running now.
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


def _process_session(session_processed_dir):
    print(f'\nsession_processed_dir: {session_processed_dir}')

    # Read timescale transform
    sync_dir = os.path.join(session_processed_dir, 'sync_pulses')
    open_source_minus_processed_path = os.path.join(
        sync_dir, 'mworks', 'open_source_minus_processed')
    open_source_minus_processed = json.load(
        open(open_source_minus_processed_path, 'r'))

    spike_sorting_dir = os.path.join(session_processed_dir, 'spike_sorting')
    
    if 'open_ephys' in os.listdir(sync_dir):
        transform_path = os.path.join(sync_dir, 'open_ephys', 'transform')
        transform = json.load(open(transform_path, 'r'))
        coef = transform['coef']
        intercept = transform['intercept']

        recording_start_time_path = os.path.join(
            sync_dir, 'open_ephys', 'recording_start_time')
        recording_start_time = json.load(open(recording_start_time_path, 'r'))

        for probe_name in os.listdir(spike_sorting_dir):
            if probe_name[:7] != 'v_probe':
                continue
            probe_dir = os.path.join(spike_sorting_dir, probe_name)
            spike_times = np.load(os.path.join(probe_dir, 'spike_times.npy'))
            sample_rate = json.load(
                open(os.path.join(probe_dir, 'sample_rate'), 'r'))
            spike_times = spike_times.astype(float) / sample_rate
            spike_times += recording_start_time
            transformed_spike_times = (
                intercept + open_source_minus_processed + coef * spike_times)
            transformed_spike_times_path = os.path.join(
                probe_dir, 'spike_times_open_source')
            print(f'Writing to {transformed_spike_times_path}')
            np.save(transformed_spike_times_path, transformed_spike_times)
    
    if 'spikeglx' in os.listdir(sync_dir):
        transform_path = os.path.join(sync_dir, 'spikeglx', 'transform')
        transform = json.load(open(transform_path, 'r'))
        coef = transform['coef']
        intercept = transform['intercept']

        for probe_name in os.listdir(spike_sorting_dir):
            if probe_name[:2] != 'np':
                continue
            probe_dir = os.path.join(spike_sorting_dir, probe_name)
            spike_times = np.load(os.path.join(probe_dir, 'spike_times.npy'))
            sample_rate = json.load(
                open(os.path.join(probe_dir, 'sample_rate'), 'r'))
            spike_times = spike_times.astype(float) / sample_rate
            transformed_spike_times = (
                intercept + open_source_minus_processed + coef * spike_times)
            transformed_spike_times_path = os.path.join(
                probe_dir, 'spike_times_open_source')
            print(f'Writing to {transformed_spike_times_path}')
            np.save(transformed_spike_times_path, transformed_spike_times)

    return


def main():
    """Copy sync pulses to data_processed."""

    session_processed_dir = (
        '/om4/group/jazlab/nwatters/multi_prediction/data_processed/Elgar/'
        '2022-10-12'
    )
    _process_session(session_processed_dir)

    # for monkey, monkey_id in _MONKEY_TO_ID.items():
    #     monkey_processed_dir = os.path.join(_DATA_PROCESSED_DIR, monkey)
    #     monkey_open_source_dir = os.path.join(_DATA_OPEN_SOURCE_DIR, monkey_id)
    #     for session_date in os.listdir(monkey_open_source_dir):
    #         print((monkey, session_date))
    #         session_processed_dir = os.path.join(
    #             monkey_processed_dir, session_date)
    #         _process_session(session_processed_dir)


if __name__ == "__main__":
    main()
