"""Remove noise units from processed physiology data.

This was run on openmind in srun sessions.

Run this before syncing/establish_common_clock.py.
"""

import numpy as np
import os

_DATA_DIR = '/om4/group/jazlab/nwatters/multi_prediction/data_processed'


def _process_probe_sorting(sorting_dir):
    
    ############################################################################
    #### LOAD DATA
    ############################################################################

    cluster_label_path = os.path.join(sorting_dir, 'cluster_label.tsv')
    cluster_label = np.genfromtxt(
        cluster_label_path,
        delimiter='\t',
        skip_header=True,
        dtype=str,
    )
    depths_path = os.path.join(sorting_dir, 'depths.tsv')
    depths = np.genfromtxt(
        depths_path,
        delimiter='\t',
        skip_header=True,
        dtype=str,
    )
    mean_waveforms = np.load(os.path.join(sorting_dir, 'mean_waveforms.npy'))
    amplitudes = np.squeeze(np.load(os.path.join(sorting_dir, 'amplitudes.npy')))
    spike_clusters = np.squeeze(np.load(os.path.join(sorting_dir, 'spike_clusters.npy')))
    spike_times = np.squeeze(np.load(os.path.join(sorting_dir, 'spike_times.npy')))
    corrected_spike_times = np.squeeze(np.load(os.path.join(sorting_dir, 'corrected_spike_times.npy')))

    ############################################################################
    #### SANITY CHECK SHAPES
    ############################################################################

    num_units = len(cluster_label)
    if len(depths) != num_units:
        raise ValueError(
            f'len(depths) = {len(depths)}, but num_units = {num_units}')
    if mean_waveforms.shape[0] != num_units:
        raise ValueError(
            f'mean_waveforms.shape[0] = {mean_waveforms.shape[0]}, but '
            f'num_units = {num_units}'
        )
    num_spikes = len(amplitudes)
    if len(spike_clusters) != num_spikes:
        raise ValueError(
            f'len(spike_clusters) = {len(spike_clusters)}, but num_spikes = '
            f'{num_spikes}'
        )
    if len(spike_times) != num_spikes:
        raise ValueError(
            f'len(spike_times) = {len(spike_times)}, but num_spikes = '
            f'{num_spikes}'
        )

    ############################################################################
    #### CURATE DATA
    ############################################################################

    print('    curating data')

    # Find indices of good and mua cluster labels
    save_cluster_inds = [x[1] in ['mua', 'good'] for x in cluster_label]

    exists_noise_clusters = np.any(
        [x[1] not in ['mua', 'good'] for x in cluster_label])
    if not exists_noise_clusters:
        print('ALREADY REMOVED NOISE UNITS')
        return

    # Curate cluster_label, depths, and mean_waveforms
    new_cluster_label = [
        x for x, save in zip(cluster_label, save_cluster_inds) if save
    ]
    new_depths = [
        x for x, save in zip(depths, save_cluster_inds) if save
    ]
    new_mean_waveforms = mean_waveforms[np.array(save_cluster_inds)]

    # Curate amplitudes, spike_clusters, and spike_times
    save_cluster_ids = [
        int(x[0]) for x, save in zip(cluster_label, save_cluster_inds) if save
    ]
    save_spike_inds = np.array([x in save_cluster_ids for x in spike_clusters])

    new_spike_times = spike_times[save_spike_inds]
    new_corrected_spike_times = corrected_spike_times[save_spike_inds]
    new_spike_clusters = spike_clusters[save_spike_inds]
    new_amplitudes = amplitudes[save_spike_inds]

    ############################################################################
    #### SAVE DATA
    ############################################################################

    print('    saving data')

    print('    writing cluster_label')
    with open(cluster_label_path, 'w') as f:
        f.write('cluster_id\label\n')
        for x in new_cluster_label:
            f.write(x[0] + '\t' + x[1] + '\n')
    
    print('    writing depths')
    with open(depths_path, 'w') as f:
        f.write('cluster_id\label\n')
        for x in new_depths:
            f.write(x[0] + '\t' + x[1] + '\n')

    print(len(spike_times))
    print(len(new_spike_times))
    print(len(new_corrected_spike_times))

    print('    saving mean_waveforms')
    np.save(os.path.join(sorting_dir, 'mean_waveforms'), new_mean_waveforms)
    print('    saving amplitudes')
    np.save(os.path.join(sorting_dir, 'amplitudes'), new_amplitudes)
    print('    saving spike_clusters')
    np.save(os.path.join(sorting_dir, 'spike_clusters'), new_spike_clusters)
    print('    saving spike_times')
    np.save(os.path.join(sorting_dir, 'spike_times'), new_spike_times)
    print('    saving corrected_spike_times')
    np.save(os.path.join(sorting_dir, 'corrected_spike_times'), new_corrected_spike_times)

    return


def main():
    """Compute and write spike times per cluster."""

    for monkey in ['Perle', 'Elgar']:
        monkey_data_dir = os.path.join(_DATA_DIR, monkey)
        for sorting_dir in os.listdir(monkey_data_dir)[::-1]:
            spike_sorting_dir = os.path.join(
                monkey_data_dir, sorting_dir, 'spike_sorting')
            for probe_name in os.listdir(spike_sorting_dir):
                probe_sorting_dir = os.path.join(spike_sorting_dir, probe_name)
                print(f'\n{probe_sorting_dir}')
                _process_probe_sorting(probe_sorting_dir)


# Elgar 2022-09-07:  Solved, copied trials back
# Elgar 2022-09-19:  Solved. Copied trials, spike_times, spike_clusters, amplitudes. Check rasters.

# Old notes:
#   * Figure out what happened to Perle/2022-04-07 V-Probe syncing
#       * Same for old rasters too, but no problem for neuropixel.
#       * Something with open ephys trial times.
#       * Check open ephys sync pulses.
#       * Conclusion: open Ephys sync pulses seem okay, but the raster still go 
#           unstable after the first ~300 trials. Unresolvable. Discard V-Probe
#           data this session.
#   * Figure out what happened to Elgar/2022-05-12 rasters, both neuropixel and v-probe.
#       * Existed before. Maybe trials missing or spike times missing.
#       * Everything seems fine in test_syncing notebook.
#   * Figure out what happened to Elgar/2022-06-25 rasters
#       * V-probe rasters all stop after a few trials. This did not happen earlier.
#       * Everything seems fine in test_syncing notebook.
#   * Figure out what happened to Elgar/2022-06-30 rasters
#       * Neuropixel rasters all stop after ~500 trials
#       * Looks like neuropixel data collection stopped 40 mins into session. Not an issue.
#   * Figure out what happened to Elgar/2022-09-09 Neuropixel rasters, maybe synchronization is bad
#       * Rasters looked like they had no information, both in new and old rasters.
#       * Everything seems fine in test_syncing notebook. Rasters do have information.
#   * Figure out what happened to Elgar/2022-09-14 rasters for both probes.
#       * Rasters on both probes cut off after ~1000 trials.
#       * Phys computers ran out of memory. Raster do cut off.
#   * Figure out what happened to V-Probe sync signals for Elgar/2022-09-19
#       * Same for old rasters too, but no problem for neuropixel.
#       * Something with open ephys trial times.
#       * Check open ephys sync pulses.
#       * Conclusion: open Ephys sync pulses seem okay, but the raster still go 
#           unstable after the first ~300 trials. Unresolvable. Discard V-Probe
#           data this session.
#   * Fix slanted neuropixel rasters in Elgar/2022-10-13
#       * Happened because np_0/sample_rate on om4 is 30000.00
#       * Actually the sample rate is more like 30000.5
#       * This happened for other sessions nearby, like Elgar/2022-10-12 and
#           Elgar/2022-10-14.
#       * Must go back and fix them all.
#       * Rasters get back starting 2022-10-12. Not sure if 2022-10-11 is bad, rasters look good.

# General to-do:
#   * Area maps
#   * Add shuffled null to mutual information area map metric
#   * PCA
#   * Build models
#   * Copy data_open_source to om4
#   * Write documentation and finalize dataset
#   * Clean up preprocessing code


# URGENT:
#   * Elgar/2022-05-12, no neurons, solved
#   * Elgar/2022-06-25, sync error, solved
#   * Elgar/2022-06-30, sync error, solved
#   * Elgar/2022-10-12, sync error slanted, solved

#   * Elgar/2022-09-19, sync error V-probe, solved
            

if __name__ == "__main__":
    main()

