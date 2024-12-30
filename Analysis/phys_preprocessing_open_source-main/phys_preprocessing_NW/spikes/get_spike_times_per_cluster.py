"""Get spike times per cluster."""

import csv
import json
import numpy as np
import os
import sys

_OM2_BASE_DIR = '/om2/user/nwatters/multi_prediction'
_OM_BASE_DIR = '/om2/user/nwatters/multi_prediction'
# _OM2_BASE_DIR = '/Users/nicholaswatters/Desktop/grad_school/research/mehrdad/multi_prediction/phys'


def main(session, probe_name, kilosort_run_name):
    """Compute and write spike times per cluster."""

    ############################################################################
    ####  Find Data Paths
    ############################################################################

    print(f'session: {session}')
    print(f'probe_name: {probe_name}')
    print(f'kilosort_run_name: {kilosort_run_name}')

    session_dir = os.path.join(_OM2_BASE_DIR, 'phys_data', session)
    session_dir_om = os.path.join(_OM_BASE_DIR, 'phys_data', session)
    spike_sorting_dir = os.path.join(session_dir, 'spike_sorting', probe_name)
    spike_sorting_dir_om = os.path.join(session_dir_om, 'spike_sorting', probe_name)
    print(f'spike_sorting_dir: {spike_sorting_dir}')

    # Get kilosort directory
    kilosort_dir = os.path.join(spike_sorting_dir_om, kilosort_run_name)
    print(f'kilosort_dir: {kilosort_dir}')

    # Get spikes write directory
    base_write_dir = os.path.join(session_dir, 'spikes', probe_name)
    print(f'base_write_dir: {base_write_dir}')
    if not os.path.exists(base_write_dir):
        os.makedirs(base_write_dir)

    # Get spikes write directory
    write_dir = os.path.join(base_write_dir, kilosort_run_name)
    print(f'write_dir: {write_dir}')
    if not os.path.exists(write_dir):
        os.makedirs(write_dir)

    ############################################################################
    ####  Read data
    ############################################################################

    print('READING DATA')

    cluster_label_reader = csv.reader(
        open(os.path.join(kilosort_dir, 'cluster_group.tsv')),
        delimiter="\t")
    cluster_labels_raw = [x for x in cluster_label_reader]
    cluster_labels = {
        int(k): v for k, v in cluster_labels_raw[1:]
    }
    cluster_ids = [k for k in cluster_labels.keys()]

    spike_times = np.squeeze(
        np.load(os.path.join(kilosort_dir, 'spike_times.npy')))
    
    # Read sample_rate and adjust spike times accordingly
    sample_rate_path = os.path.join(spike_sorting_dir, 'sample_rate')
    print(f'Reading sample_rate from {sample_rate_path}')
    sample_rate = json.load(open(sample_rate_path, 'r'))
    print(f'sample_rate = {sample_rate}')
    spike_times = spike_times.astype(float) / float(sample_rate)
    
    spike_clusters = np.squeeze(
        np.load(os.path.join(kilosort_dir, 'spike_clusters.npy')))

    ############################################################################
    ####  Get spike times per cluster
    ############################################################################

    print('GETTING SPIKE TIMES PER CLUSTER')
    
    spike_times_per_cluster = {
        k: spike_times[spike_clusters == k].tolist()
        for k in cluster_ids
    }

    ############################################################################
    ####  Save results
    ############################################################################

    print('SAVING')

    cluster_labels_file = os.path.join(write_dir, 'cluster_labels')
    print(f'Writing cluster labels to {cluster_labels_file}')
    json.dump(cluster_labels, open(cluster_labels_file, 'w'))
    spike_times_per_cluster_file = os.path.join(
        write_dir, 'spike_times_per_cluster')
    print(
        'Writing cluster spike times per cluster to '
        f'{spike_times_per_cluster_file}'
    )
    json.dump(spike_times_per_cluster, open(spike_times_per_cluster_file, 'w'))

    return


if __name__ == "__main__":
    session = sys.argv[1]
    probe_name = sys.argv[2]
    kilosort_run_name = sys.argv[3]
    main(session, probe_name, kilosort_run_name)

