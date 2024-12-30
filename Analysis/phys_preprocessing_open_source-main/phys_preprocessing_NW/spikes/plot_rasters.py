"""Create raster plots."""

import json
import pdb
import numpy as np
import os
import sys
from matplotlib import pyplot as plt

_OM2_BASE_DIR = '/om2/user/nwatters/multi_prediction'


def _plot_raster(cluster_label, cluster_ind, spikes_data):

    if spikes_data is None:
        print('No spikes data')
        return None

    # Get spike and phase data
    spikes_raster = [[], []]
    phases_raster = [[[], []] for _ in range(6)]
    count = 0
    for d in spikes_data:
        trial_num = d['trial_num']
        raw_rel_spike_times = d['relative_spike_times']
        raw_rel_phase_times = d['relative_phase_times']
        num_phases = len(raw_rel_phase_times)
        if num_phases != 6:
            # Incomplete trial
            continue
            
        rel_spike_times = np.array(raw_rel_spike_times) - raw_rel_phase_times[0]
        rel_phase_times = np.array(raw_rel_phase_times) - raw_rel_phase_times[0]
        
        spikes_raster[0].extend(list(rel_spike_times))
        count += 1
        spikes_raster[1].extend(len(rel_spike_times) * [trial_num])
        
        for phase_data, phase_time in zip(phases_raster, rel_phase_times):
            phase_data[0].append(phase_time)
            phase_data[1].append(trial_num)
            
    # Make Plot
    fig, ax = plt.subplots(figsize=(8, 15))
    ax.scatter(spikes_raster[0], spikes_raster[1], c='k', s=1)
    phase_names_colors = [
        ('visible', 'r'),
        ('memory', 'g'),
        ('cue', 'b'),
        ('response', 'm'),
        ('reveal', 'c'),
        ('reset', 'y'),
    ]
    for i in range(6):
        p = phases_raster[i]
        label, color = phase_names_colors[i]
        ax.scatter(p[0], p[1], c=color, s=0.1, label=label)
    
    # Add record of valid trials
    valid = [d['trial_num'] for d in spikes_data if d['valid']]
    ax.scatter(
        len(valid) * [5.5], valid, color=[0.7, 0.7, 0.7], s=5, marker='s')
    
    ax.set_xlim([-1., 6.])
    ax.set_ylabel('Trial Number')
    ax.set_xlabel('Time Within Trial (s)')
    ax.set_title(f'cluster_number: {cluster_ind}, {cluster_label}')
    legend = ax.legend(bbox_to_anchor=[1., 1.])
    # Set marker size in legend
    for handle in legend.legendHandles:
        handle.set_sizes([20])

    return fig


def main(session, probe_name):
    """Compute and write spike times per cluster."""

    ############################################################################
    ####  Find Data Paths
    ############################################################################

    print(f'session: {session}')
    print(f'probe_name: {probe_name}')

    # Get spikes directory
    session_dir = os.path.join(_OM2_BASE_DIR, 'phys_data', session)
    spikes_dir = os.path.join(session_dir, 'spikes', probe_name)
    
    # spikes_dir = (
    #     '/Users/nicholaswatters/Desktop/grad_school/research/mehrdad/'
    #     'multi_prediction/phys/phys_data/Perle/2022-06-03/'
    #     'new_spikes/v_probe_0'
    # )

    cluster_labels_path = os.path.join(spikes_dir, 'cluster_labels')
    valid_units_path = os.path.join(spikes_dir, 'valid_units')
    spikes_per_trial_path = os.path.join(spikes_dir, 'spike_times_per_trial')
    rasters_dir = os.path.join(spikes_dir, 'rasters')
    print(f'rasters_dir: {rasters_dir}')
    if not os.path.exists(rasters_dir):
        os.makedirs(rasters_dir)

    ############################################################################
    ####  LOAD DATA
    ############################################################################

    print('Loading cluster labels')
    cluster_labels = json.load(open(cluster_labels_path, 'r'))

    print('Loading valid units')
    valid_units = json.load(open(valid_units_path, 'r'))

    print('Loading spikes per trial')
    spikes_per_trial = json.load(open(spikes_per_trial_path, 'r'))

    print('Generating rasters')

    print(f'Generating rasters in {rasters_dir}')

    for i, (cluster_ind, valid) in enumerate(valid_units.items()):
        print(f'{i} / {len(valid_units)}')
        # if not valid:
        #     continue

        cluster_label = cluster_labels[cluster_ind]
        spikes_data = spikes_per_trial[cluster_ind]

        fig_raster = _plot_raster(cluster_label, cluster_ind, spikes_data)
        if fig_raster is None:
            continue
        fig_raster.savefig(os.path.join(rasters_dir, str(cluster_ind) + '.png'))
        plt.close(fig_raster)

    print('Done running plot_rasters.py')
    
    return


if __name__ == "__main__":
    session = sys.argv[1]
    probe_name = sys.argv[2]
    main(session, probe_name)

