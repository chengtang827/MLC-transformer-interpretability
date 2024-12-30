"""Curate cluster labels."""

import numpy as np
import os
import sys
from scipy import linalg as scipy_linalg

_OM2_BASE_DIR = '/om2/user/nwatters/multi_prediction'
# _OM2_BASE_DIR = '/Users/nicholaswatters/Desktop/grad_school/research/mehrdad/multi_prediction/phys'

_SAMPLE_RATE = 30000  # Physiology sample rate in Hz
_NUM_SPIKES_FOR_FR_THRESHOLD = 200
_MIN_FR = 0.25

_NEUROPIXEL_INTER_CHANNEL_DISTANCE = 0.01
_V_PROBE_INTER_CHANNEL_DISTANCE = 0.05


def _get_dissimilarity_score(waveform_0, waveform_1):
    """Get dissimilarity score between two 1-channel waveforms.

    This measures the difference between the shapes of the two waveforms. It is
    invariant to the amplitude of the waveforms and temporal offsets of the
    waveforms.
    
    Args:
        waveform_0: Float array of size [timesteps]. Waveform on a single
            channel.
        waveform_1: Float array of size [timesteps]. Waveform on a single
            channel.

    Returns:
        dissimilarity_score: Float in [0, 1]. Lower means waveform shapes are
            more similar.
    """
    # First normalize the waveforms
    waveform_0 = waveform_0 - np.mean(waveform_0)
    waveform_0 = waveform_0 / np.sum(np.abs(waveform_0))
    waveform_1 = waveform_1 - np.mean(waveform_1)
    waveform_1 = waveform_1 / np.sum(np.abs(waveform_1))

    # Now compute the maximum difference between the two waveforms for every
    # temporal offset.
    w_first_deriv_circulant = scipy_linalg.circulant(waveform_0)
    diff_first_second = np.abs(w_first_deriv_circulant - waveform_1[:, np.newaxis])
    diff_fs = np.max(diff_first_second, axis=0)

    # The dissimilarity score is the difference between the min and max of these
    # differences.
    dissimilarity_score = np.min(diff_fs) / np.max(diff_fs)
    
    return dissimilarity_score


def _detect_noise_waveform(w,
                           max_top_2_channels_amplitude_ratio=3.,
                           max_top_2_channels_distance=10,
                           min_amplitude=5.,
                           max_top_2_dissimilarity_score=0.6,
                           max_waveform_spread=30,
                           high_amplitude_fraction=0.6):
    """Evaluate whether waveform is noise based on its shape.
    
    This is needed because Kilosort labels many noise clusters as mua.

    A waveform is considered noise if any of the following are true:
        * It has NaN values. This can happen if there were not enough spikes to
            compute the waveform.
        * If it is too sparse across channels, i.e. only appears on a single
            channel.
        * If it is significantly multimodal across channels.
        * If it's amplitude is too low.
        * If it exhibits significantly different shapes on different channels.
    See arguments for more details about these criteria.

    Args:
        w: Float array of size [timesteps, num_channels]. The waveform.
            Typically timesteps is 82 (that's Kilosort's default for templates),
            and for neuropixe num_channels is 384.
        max_top_2_channels_amplitude_ratio: Float. Maximum ratio of the
            amplitude of the waveform on the highest amplitude channel versus
            the second-highest amplitude channel. If this is exceeded, then the
            waveform is extremely sparse (essentially appearing on only one
            channel) so is noise. A conservative default that is unlikely to
            filter out real units is 5, though you may want a lower value to be
            less conservative.
        max_top_2_channels_distance: Maximum difference of the index of the
            highest amplitude channel versus the index of the second-highest
            amplitude channel. If this is exceeded, then the two highest
            amplitude channels are very far apart on the probe, so the waveform
            is multimodal in space and hence is noise. A conservative default
            that is unlikely to filter out real units is 10, though you may want
            a lower value to be less conservative.
        min_amplitude: Minimum amplitude of the waveform. If the amplitude on
            the highest-amplitude channel is less than this, then the unit is
            very low-amplitude and likely noise. A reasonable default for a
            typical neuropixel with ordinary impedance is 5.
        max_top_2_dissimilarity_score: Maximum dissimilarity between the
            waveform shapes on the highest amplitude channel and the
            second-highest amplitude channel. See _get_dissimilarity_score()
            above for the dissimilarity score. A conservative default that is
            unlikely to filter out real units is 0.5, though you may want a
            lower value to be less conservative and filter out more units.
        max_waveform_spread: Int. Maximum number of channels allowed between two
            high-amplitude channels. Units of number of channels. Waveforms
            exceeding this are too spread out along the probe.
        high_amplitude_fraction: Fraction of max amplitude to consider as high
            amplitude channel for maximum waveform spread criterion.

    Returns:
        Noise: Bool. Whether the waveform is noise.
        Label: String. Label of the waveform if it is noise. If it is not noise,
            empty string.
    """
    if np.any(np.isnan(w)):
        return True, 'waveform_nan'
    
    # Sort the channel indices by amplitude of the waveform
    y_diff = np.abs(np.max(w, axis=0) - np.min(w, axis=0))
    top_channel_inds = np.argsort(y_diff)[::-1]
    
    # If amplitude radio for the highest amplitude channel vs second-highest
    # amplitude channel ratio is too high, waveform is too sparse across
    # channels, so label as noise
    first_max = y_diff[top_channel_inds[0]]
    second_max = y_diff[top_channel_inds[1]]
    if first_max > max_top_2_channels_amplitude_ratio * second_max:
        return True, 'waveform_sparsity'
    
    # If highest amplitude channel and second-highest amplitude channel are too
    # far apart on the probe, label as noise
    top_2_channel_sep = np.abs(top_channel_inds[0] - top_channel_inds[1])
    if top_2_channel_sep > max_top_2_channels_distance:
        return True, 'waveform_multimodal'
    
    # If amplitude is low, label as noise
    if first_max < min_amplitude:
        return True, 'waveform_amplitude'
    
    # If amplitude is too high across too many channels, label as noise because
    # the waveform is too spread out
    # Consider the mean of the top 3 channels to be the max amplitude
    max_amplitude = np.mean(y_diff[top_channel_inds[:3]])
    # Now find the spread of channels that have at least 60% that max
    high_amplitude_channels = np.argwhere(
        y_diff > high_amplitude_fraction * max_amplitude)
    waveform_spread = (
        np.max(high_amplitude_channels) - np.min(high_amplitude_channels)
    )
    if waveform_spread > max_waveform_spread:
        return True, 'waveform_spread'
    
    # If dissimilarity score between highest amplitude channel and
    # second-highest amplitude channel is too high, label as noise.
    dissimilarity_score = _get_dissimilarity_score(
        w[:, top_channel_inds[0]], w[:, top_channel_inds[1]])
    if dissimilarity_score > max_top_2_dissimilarity_score:
        return True, 'waveform_dissimilarity'
    
    return False, ''


def _get_max_amplitude_channel_inds(mean_waveforms):
    """Get indices of maximum amplitude channels for each waveform.
    
    Args:
        mean_waveforms: Float array of size
            [num_clusters, timesteps_per_waveform, num_channels].
    
    Returns:
        max_amplitude_channel_inds: Int array of size [num_clusters].
    """
    amplitudes_per_channel = (
        np.max(mean_waveforms, axis=1) - np.min(mean_waveforms, axis=1)
    )
    max_amplitudes = np.max(amplitudes_per_channel, axis=1)
    max_amplitude_channel_inds = np.argmax(
        amplitudes_per_channel, axis=1).astype(float)
    max_amplitude_channel_inds[np.isnan(max_amplitudes)] = np.NaN
    
    return max_amplitude_channel_inds


def main(session, probe_name, kilosort_run_name):
    """Compute and write spike times per cluster."""

    ############################################################################
    ####  Find Data Paths And Load Data
    ############################################################################
    
    print('\n\n')
    print(f'session: {session}')
    print(f'probe_name: {probe_name}')
    print(f'kilosort_run_name: {kilosort_run_name}')

    session_dir = os.path.join(_OM2_BASE_DIR, 'phys_data', session)
    spike_sorting_dir = os.path.join(session_dir, 'spike_sorting', probe_name)
    print(f'spike_sorting_dir: {spike_sorting_dir}')
    kilosort_dir = os.path.join(spike_sorting_dir, kilosort_run_name)
    print(f'kilosort_dir: {kilosort_dir}')

    if not os.path.exists(kilosort_dir):
        print('\nNO KILOSORT DIRECTORY\n')
        return

    mean_waveforms = np.load(os.path.join(kilosort_dir, 'mean_waveforms.npy'))
    templates = np.load(os.path.join(kilosort_dir, 'templates.npy'))
    cluster_group = np.genfromtxt(
        os.path.join(kilosort_dir, 'cluster_group.tsv'),
        delimiter='\t',
        skip_header=True,
        dtype=str,
    )

    num_clusters = templates.shape[0]
    print(f'num_clusters: {num_clusters}')

    amplitudes = np.squeeze(
        np.load(os.path.join(kilosort_dir, 'amplitudes.npy')))
    spike_clusters = np.squeeze(
        np.load(os.path.join(kilosort_dir, 'spike_clusters.npy')))
    spike_times = np.squeeze(
        np.load(os.path.join(kilosort_dir, 'spike_times.npy')))

    # Sanity check shape of mean_waveforms
    if mean_waveforms.shape[0] != len(cluster_group):
        raise ValueError(
            f'Found {mean_waveforms.shape[0]} mean_waveforms, but '
            '{len(cluster_group)} cluster_groups'
        )

    # Compute indices of clusters
    cluster_ids = cluster_group[:, 0].astype(int)
    spike_indices = {}
    for cluster_id in cluster_ids:
        spike_indices[cluster_id] = np.argwhere(spike_clusters == cluster_id)

    # Create curated_cluster_group, which will contain curated cluster labels
    curated_cluster_group = {
        int(cluster_id): group for (cluster_id, group) in cluster_group
    }
    for k, v in curated_cluster_group.items():
        if v == 'noise':
            curated_cluster_group[k] = 'noise: kilosort'

    ############################################################################
    ####  Compute and save empirical mean amplitudes
    ############################################################################

    mean_amplitudes_write_file = os.path.join(
        kilosort_dir, 'mean_amplitudes.tsv')
    print(f'Writing mean amplitudes to {mean_amplitudes_write_file}')
    with open(mean_amplitudes_write_file, 'w') as f:
        f.write('cluster_id\tmean_amplitude\n')
        for cluster_id in cluster_ids:
            f.write(
                str(cluster_id) +
                '\t' +
                str(float(np.mean(amplitudes[spike_indices[cluster_id]]))) +
                '\n'
            )

    ############################################################################
    ####  Compute and save depth from tip of probe
    ############################################################################

    max_amplitude_channel_inds = _get_max_amplitude_channel_inds(mean_waveforms)

    if 'np' in probe_name:
        depths = (
            -1. *
            _NEUROPIXEL_INTER_CHANNEL_DISTANCE *
            max_amplitude_channel_inds
        )
    elif 'v_probe' in probe_name:
        depths = (
            -1. * _V_PROBE_INTER_CHANNEL_DISTANCE * max_amplitude_channel_inds
        )
    
    depths_write_file = os.path.join(
        kilosort_dir, 'depths.tsv')
    print(f'Writing depths to {depths_write_file}')
    with open(depths_write_file, 'w') as f:
        f.write('cluster_id\tdepth_from_probe_tip\n')
        for cluster_id, depth in zip(cluster_ids, depths):
            f.write(str(cluster_id) + '\t' + str(float(depth)) + '\n')

    ############################################################################
    ####  Relabel clusters with too low maximum firing as noise
    ############################################################################

    for cluster_id in cluster_ids:
        spike_inds = spike_indices[cluster_id]
        if len(spike_inds) < _NUM_SPIKES_FOR_FR_THRESHOLD + 1:
            curated_cluster_group[cluster_id] = 'noise: too_few_spikes'
            continue
        times = spike_times[spike_inds]
        firing_freq = (
            times[_NUM_SPIKES_FOR_FR_THRESHOLD:] -
            times[:-_NUM_SPIKES_FOR_FR_THRESHOLD]
        )
        firing_freq_threshold = (
            _NUM_SPIKES_FOR_FR_THRESHOLD * float(_SAMPLE_RATE) / _MIN_FR
        )
        if np.max(firing_freq) > firing_freq_threshold:
            # Never sustains firing over 0.25Hz for 200 spikes
            curated_cluster_group[cluster_id] = 'noise: low_max_firing_rate'

    ############################################################################
    ####  Evaluate out-of-brain based on mean waveform shape
    ############################################################################

    if 'np' in probe_name:  # V-probe is already curated
        for i, cluster_id in enumerate(cluster_ids):
            noise, reason = _detect_noise_waveform(mean_waveforms[i])
            if noise:
                curated_cluster_group[cluster_id] = 'noise: ' + reason

    ############################################################################
    ####  Save curated_cluster_group
    ############################################################################

    curated_cluster_group_write_file = os.path.join(
        kilosort_dir, 'curated_cluster_group.tsv')
    print(
        f'Writing curated cluster groups to {curated_cluster_group_write_file}'
    )
    with open(curated_cluster_group_write_file, 'w') as f:
        f.write('cluster_id\tcurated_cluster_group\n')
        for cluster_id in cluster_ids:
            f.write(
                str(cluster_id) +
                '\t' +
                str(curated_cluster_group[cluster_id]) +
                '\n'
            )


if __name__ == "__main__":
    session = sys.argv[1]
    probe_name = sys.argv[2]
    kilosort_run_name = sys.argv[3]
    main(session, probe_name, kilosort_run_name)

