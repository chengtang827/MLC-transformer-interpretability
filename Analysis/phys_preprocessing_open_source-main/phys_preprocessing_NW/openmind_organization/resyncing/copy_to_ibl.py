"""Convert physiology data to IBL format."""

import json
import numpy as np
import os

SOURCE_BASE_DIR = '/om4/group/jazlab/nwatters/multi_prediction/data_processed'
TARGET_BASE_DIR = (
    # '/om4/group/jazlab/nwatters/multi_prediction/data_open_source/Subjects'
    '/om2/user/nwatters/multi_prediction/data_open_source/Subjects'
)
MONKEY_TO_ID = {
    'Perle': 'monkey0',
    'Elgar': 'monkey1',
}


def _copy_spike_sorting_data(session_source_dir,
                             probe_write_dir,
                             source_probe_name):
    spike_sorting_probe_dir = os.path.join(
        session_source_dir, 'spike_sorting', source_probe_name)

    # Sanity check spike times shape is consistent
    spike_times_path = os.path.join(probe_write_dir, 'spikes.times.npy')
    if not os.path.exists(spike_times_path):
        return
    spike_times = np.load(spike_times_path)
    spike_times_open_source_path = os.path.join(
        spike_sorting_probe_dir, 'spike_times_open_source.npy')
    spike_times_open_source = np.load(spike_times_open_source_path)
    if len(spike_times) != len(spike_times_open_source):
        raise ValueError(
            f'len(spike_times) = {len(spike_times)} but '
            f'len(spike_times_open_source) = {len(spike_times_open_source)}'
        )

    # Copy spike times
    bash_command_copy_spike_times = 'cp {} {}'.format(
        spike_times_open_source_path,
        os.path.join(probe_write_dir, 'spikes.times_open_source.npy')
    )
    print('Running bash command {}'.format(bash_command_copy_spike_times))
    os.system(bash_command_copy_spike_times)
    
    return


def _process_session(session_source_dir, session_target_dir):

    # Load physiology metadata
    phys_metadata = json.load(
        open(os.path.join(session_source_dir, 'physiology_metadata.json'), 'r'))
    
    # Find which probes were used and write them
    spike_sorting_dir = os.path.join(session_source_dir, 'spike_sorting')
    probe_count = 0

    if (phys_metadata['neuropixel_0_coordinates'] and 
            'np_0' in os.listdir(spike_sorting_dir)):
        probe_label = 'probe' + str(probe_count).zfill(2)
        probe_write_dir = os.path.join(session_target_dir, probe_label)
        _copy_spike_sorting_data(session_source_dir, probe_write_dir, 'np_0')
        probe_count += 1

    if (phys_metadata['neuropixel_1_coordinates'] and 
            'np_1' in os.listdir(spike_sorting_dir)):
        probe_label = 'probe' + str(probe_count).zfill(2)
        probe_write_dir = os.path.join(session_target_dir, probe_label)
        _copy_spike_sorting_data(session_source_dir, probe_write_dir, 'np_1')
        probe_count += 1

    if (phys_metadata['v_probe_0_coordinates'] and 
            'v_probe_0' in os.listdir(spike_sorting_dir)):
        probe_label = 'probe' + str(probe_count).zfill(2)
        probe_write_dir = os.path.join(session_target_dir, probe_label)
        _copy_spike_sorting_data(
            session_source_dir, probe_write_dir, 'v_probe_0')
        probe_count += 1

    if (phys_metadata['v_probe_1_coordinates'] and 
            'v_probe_1' in os.listdir(spike_sorting_dir)):
        probe_label = 'probe' + str(probe_count).zfill(2)
        probe_write_dir = os.path.join(session_target_dir, probe_label)
        _copy_spike_sorting_data(
            session_source_dir, probe_write_dir, 'v_probe_1')
        probe_count += 1

    return


def main():
    """Convert physiology data to IBL format."""

    for monkey in os.listdir(SOURCE_BASE_DIR):
        monkey_source_dir = os.path.join(SOURCE_BASE_DIR, monkey)
        monkey_target_dir = os.path.join(TARGET_BASE_DIR, MONKEY_TO_ID[monkey])
        if not os.path.exists(monkey_target_dir):
            os.makedirs(monkey_target_dir)
        for session_date in os.listdir(monkey_source_dir):
            session_source_dir = os.path.join(monkey_source_dir, session_date)
            session_target_dir = os.path.join(
                monkey_target_dir, session_date, '001')
            
            print(f'session_source_dir: {session_source_dir}')
            print(f'session_target_dir: {session_target_dir}')

            if not os.path.exists(session_target_dir):
                os.makedirs(session_target_dir)

            _process_session(session_source_dir, session_target_dir)


if __name__ == "__main__":
    main()

