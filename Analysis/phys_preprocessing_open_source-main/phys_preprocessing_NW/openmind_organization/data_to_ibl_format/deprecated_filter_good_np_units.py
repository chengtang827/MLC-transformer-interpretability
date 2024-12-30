"""Filter good neuropixel units.

NOTE: NEVER RAN THIS FILE BECAUSE THE NEXT TWO LINES ARE FALSE.

Neuropixel units labeled as "good" by kilosort were not filtered for firing rate
by the earlier cluster curation script. So we'll filter them here.
"""

import json
import numpy as np
import os

# _DATA_DIR = '/om2/user/nwatters/multi_prediction/data_open_source'
_DATA_DIR = (
    '/Users/nicholaswatters/Desktop/grad_school/research/mehrdad/'
    'multi_prediction/phys/data_open_source/Subjects'
)

_SAMPLE_RATE = 30000  # Physiology sample rate in Hz
_NUM_SPIKES_FOR_FR_THRESHOLD = 200
_MIN_FR = 0.25


def _process_probe(probe_dir):
    labels_path = os.path.join(probe_dir, 'clusters.labels.tsv')
    labels = np.genfromtxt(
        labels_path,
        delimiter='\t',
        skip_header=True,
        dtype=str,
    )
    waveforms = np.load(os.path.join(probe_dir, 'clusters.waveforms.npy'))
    if waveforms.shape[0] != len(labels):
        raise ValueError(
            f'waveforms.shape[0] = {waveforms.shape[0]} but len(labels) = '
            f'{len(labels)}'
        )
    spikes_clusters = np.load(os.path.join(probe_dir, 'spikes.clusters.npy'))
    spikes_times = np.load(os.path.join(probe_dir, 'spikes.times.npy'))

    # Compute indices of bad clusters
    cluster_ids = labels[:, 0].astype(int)
    bad_cluster_indices = []
    bad_cluster_spike_inds = []
    for i, cluster_id in enumerate(cluster_ids):
        spike_inds = np.argwhere(spikes_clusters == cluster_id)
        if len(spike_inds) < _NUM_SPIKES_FOR_FR_THRESHOLD + 1:
            bad_cluster_indices.append(i)
            bad_cluster_spike_inds.append(spike_inds)
            continue
        times = spikes_times[spike_inds]
        firing_freq = (
            times[_NUM_SPIKES_FOR_FR_THRESHOLD:] -
            times[:-_NUM_SPIKES_FOR_FR_THRESHOLD]
        )
        firing_freq_threshold = (
            _NUM_SPIKES_FOR_FR_THRESHOLD * float(_SAMPLE_RATE) / _MIN_FR
        )
        if np.max(firing_freq) > firing_freq_threshold:
            # Never sustains firing over 0.25Hz for 200 spikes
            bad_cluster_indices.append(i)
            bad_cluster_spike_inds.append(spike_inds)
        
        if cluster_id in[25, '25']:
            import pdb; pdb.set_trace()
    
    # Remove bad clusters if any exist
    if len(bad_cluster_indices) == 0:
        print('Found no bad clusters')
        return
    print(f'Found bad_cluster_indices {bad_cluster_indices}')
    bad_cluster_spike_inds = np.concatenate(bad_cluster_spike_inds)
    labels = np.delete(labels, bad_cluster_indices, axis=0)
    waveforms = np.delete(waveforms, bad_cluster_indices, axis=0)
    spikes_clusters = np.delete(spikes_clusters, bad_cluster_spike_inds)
    spikes_times = np.delete(spikes_times, bad_cluster_spike_inds)

    import pdb; pdb.set_trace()

    # Write curated data
    print('Writing labels')
    with open(labels_path, 'w') as f:
        f.write('cluster_id\tlabel\n')
        for cluster_id, label in labels:
            f.write(
                str(cluster_id) +
                '\t' +
                str(label) +
                '\n'
            )
    print('Writing waveforms')
    np.save(waveforms, os.path.join(probe_dir, 'clusters.waveforms'))
    print('Writing spike clusters')
    np.save(spikes_clusters, os.path.join(probe_dir, 'spikes.clusters'))
    print('Writing spike times')
    np.save(spikes_times, os.path.join(probe_dir, 'spikes.times'))

    return


def _process_session(session_dir):
    """Compute and write spike times per cluster."""
    print(session_dir)

    # Load probe metadata
    probes_metadata_path = os.path.join(session_dir, 'probes.metadata.json')
    probes_metadata = json.load(open(probes_metadata_path, 'r'))

    for probe_metadata in probes_metadata:
        probe_label = probe_metadata['label']
        probe_dir = os.path.join(session_dir, probe_label)
        _process_probe(probe_dir)


def main():
    """Make summary plot for every unit."""

    for subject in os.listdir(_DATA_DIR):
        subject_dir = os.path.join(_DATA_DIR, subject)
        for session_date in os.listdir(subject_dir):
            session_dir = os.path.join(subject_dir, session_date, '001')
            _process_session(session_dir)


if __name__ == "__main__":
    main()

