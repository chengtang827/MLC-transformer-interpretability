"""Convert physiology data to IBL format."""

import json
import numpy as np
import os
import constants

_NEUROPIXEL_COORDINATE_SYSTEM = (
    'Coordinate system is stereotactic coordinates with origin at zero ear bar '
    'and pitched to the bottom of the eye orbit. Coordinate vectors are '
    '[left-right, posterior-anterior]. Units are in millimeters.'
)
_V_PROBE_COORDINATE_SYSTEM = (
    'Coordinate system is stereotactic coordinates with origin at zero ear bar '
    'and pitched to the bottom of the eye orbit. Coordinate vectors are '
    '[left-right, posterior-anterior, inferior-superior]. Units are in '
    'millimeters.'
)


def _copy_spike_sorting_data(session_source_dir,
                             probe_write_dir,
                             source_probe_name):
    # Create probe_write_dir if necessary
    if not os.path.exists(probe_write_dir):
        os.makedirs(probe_write_dir)
    
    spike_sorting_probe_dir = os.path.join(
        session_source_dir, 'spike_sorting', source_probe_name)

    # Copy spike times
    bash_command_copy_spike_times = 'cp {} {}'.format(
        os.path.join(spike_sorting_probe_dir, 'corrected_spike_times.npy'),
        os.path.join(probe_write_dir, 'spikes.times.npy')
    )
    print('Running bash command {}'.format(bash_command_copy_spike_times))
    os.system(bash_command_copy_spike_times)

    # Copy spike clusters
    bash_command_copy_spike_clusters = 'cp {} {}'.format(
        os.path.join(spike_sorting_probe_dir, 'spike_clusters.npy'),
        os.path.join(probe_write_dir, 'spikes.clusters.npy')
    )
    print('Running bash command {}'.format(bash_command_copy_spike_clusters))
    os.system(bash_command_copy_spike_clusters)

    # Copy cluster labels
    bash_command_copy_cluster_labels = 'cp {} {}'.format(
        os.path.join(spike_sorting_probe_dir, 'cluster_label.tsv'),
        os.path.join(probe_write_dir, 'clusters.labels.tsv')
    )
    print('Running bash command {}'.format(bash_command_copy_cluster_labels))
    os.system(bash_command_copy_cluster_labels)

    # Copy mean waveforms
    bash_command_copy_mean_waveforms = 'cp {} {}'.format(
        os.path.join(spike_sorting_probe_dir, 'mean_waveforms.npy'),
        os.path.join(probe_write_dir, 'clusters.waveforms.npy')
    )
    print('Running bash command {}'.format(bash_command_copy_mean_waveforms))
    os.system(bash_command_copy_mean_waveforms)
    
    return


def _process_session(session_source_dir, session_target_dir):

    # Load physiology metadata
    phys_metadata = json.load(
        open(os.path.join(session_source_dir, 'physiology_metadata.json'), 'r'))

    # Setup probes.description
    probes_description = []
    
    # Find which probes were used and write them
    spike_sorting_dir = os.path.join(session_source_dir, 'spike_sorting')
    probe_count = 0

    if (phys_metadata['neuropixel_0_coordinates'] and 
            'np_0' in os.listdir(spike_sorting_dir)):
        probe_label = 'probe' + str(probe_count).zfill(2)
        probes_description.append({
            'label': probe_label,
            'probe_type': 'Neuropixels',
            'coordinate_system': _NEUROPIXEL_COORDINATE_SYSTEM,
            'depth_from_surface': phys_metadata['neuropixel_depth'],
            'coordinates': np.round(
                phys_metadata['neuropixel_0_coordinates'][:2],
                decimals=2,
            ).tolist(),
        })
        probe_write_dir = os.path.join(session_target_dir, probe_label)
        _copy_spike_sorting_data(session_source_dir, probe_write_dir, 'np_0')
        probe_count += 1

    if (phys_metadata['neuropixel_1_coordinates'] and 
            'np_1' in os.listdir(spike_sorting_dir)):
        probe_label = 'probe' + str(probe_count).zfill(2)
        probes_description.append({
            'label': probe_label,
            'probe_type': 'Neuropixels',
            'coordinate_system': _NEUROPIXEL_COORDINATE_SYSTEM,
            'depth_from_surface': phys_metadata['neuropixel_depth'],
            'coordinates': np.round(
                phys_metadata['neuropixel_1_coordinates'][:2],
                decimals=2,
            ).tolist(),
        })
        probe_write_dir = os.path.join(session_target_dir, probe_label)
        _copy_spike_sorting_data(session_source_dir, probe_write_dir, 'np_1')
        probe_count += 1

    if (phys_metadata['v_probe_0_coordinates'] and 
            'v_probe_0' in os.listdir(spike_sorting_dir)):
        probe_label = 'probe' + str(probe_count).zfill(2)
        probes_description.append({
            'label': probe_label,
            'probe_type': 'V-Probe 64',
            'coordinate_system': _V_PROBE_COORDINATE_SYSTEM,
            'depth_from_surface': phys_metadata['v_probe_depth'],
            'coordinates': {
                k: np.round(v, decimals=2).tolist()
                for k, v in phys_metadata['v_probe_0_coordinates'].items()
            },
        })
        probe_write_dir = os.path.join(session_target_dir, probe_label)
        _copy_spike_sorting_data(
            session_source_dir, probe_write_dir, 'v_probe_0')
        probe_count += 1

    if (phys_metadata['v_probe_1_coordinates'] and 
            'v_probe_1' in os.listdir(spike_sorting_dir)):
        probe_label = 'probe' + str(probe_count).zfill(2)
        probes_description.append({
            'label': probe_label,
            'probe_type': 'V-Probe 64',
            'coordinate_system': _V_PROBE_COORDINATE_SYSTEM,
            'depth_from_surface': phys_metadata['v_probe_depth'],
            'coordinates': {
                k: np.round(v, decimals=2).tolist()
                for k, v in phys_metadata['v_probe_1_coordinates'].items()
            },
        })
        probe_write_dir = os.path.join(session_target_dir, probe_label)
        _copy_spike_sorting_data(
            session_source_dir, probe_write_dir, 'v_probe_1')
        probe_count += 1

    # Write probes_description
    json.dump(
        probes_description,
        open(os.path.join(session_target_dir, 'probes.metadata.json'), 'w'),
    )

    return


def main():
    """Convert physiology data to IBL format."""

    for monkey in os.listdir(constants.SOURCE_BASE_DIR):
        monkey_source_dir = os.path.join(constants.SOURCE_BASE_DIR, monkey)
        monkey_target_dir = os.path.join(
            constants.TARGET_BASE_DIR, constants.MONKEY_TO_ID[monkey])
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

