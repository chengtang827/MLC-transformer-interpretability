"""Utils for computing mean waveforms.."""

from matplotlib import pyplot as plt
import numpy as np
import os
import shutil
import pathlib

OM2_BASE_DIR = '/om2/user/nwatters/multi_prediction'
OM4_BASE_DIR = '/om4/group/jazlab/nwatters/multi_prediction'


def _plot_random_clusters(mean_waveforms,
                          kilosort_dir,
                          spread=15,
                          y_sep=100.,
                          num_to_plot=20):
    """Generate and save plots of mean waveforms for random clusters."""
    
    cluster_plot_dir = os.path.join(kilosort_dir, 'cluster_plots')
    if os.path.exists(cluster_plot_dir):
        shutil.rmtree(cluster_plot_dir)
    os.makedirs(cluster_plot_dir)
    print('Plotting clusters in {}'.format(cluster_plot_dir))
    
    num_clusters = mean_waveforms.shape[0]
    num_channels = mean_waveforms.shape[2]
    cluster_inds_to_plot = np.random.randint(num_clusters, size=num_to_plot)
    for cluster_ind in cluster_inds_to_plot:
        waveform = mean_waveforms[cluster_ind]
        peak_channel = np.argmax(
            np.max(waveform, axis=0) - np.min(waveform, axis=0)
        )
        start_channel = max(0, peak_channel - spread)
        end_channel = min(num_channels - 1, peak_channel + spread)
        _, ax = plt.subplots(figsize=(8, 1 + 2 * spread))
        channels_to_plot = np.arange(start_channel, end_channel)
        for i, chan in enumerate(channels_to_plot):
            y_offset = i * y_sep
            ax.plot(y_offset + waveform[:, chan])
        _ = ax.set_yticks(y_sep * np.arange(len(channels_to_plot)))
        _ = ax.set_yticklabels(channels_to_plot)
        write_file = os.path.join(cluster_plot_dir, str(cluster_ind) + '.pdf')
        plt.savefig(write_file, format='pdf', bbox_inches='tight', dpi=60)
        plt.close()
        

def _memmap_neuropixel(raw_data_path):
    """Memory map raw neuropixel data."""
    path = pathlib.Path(raw_data_path)
    n_channels = 385
    dtype = 'int16'
    mode = 'r+'

    fsize = path.stat().st_size
    item_size = np.dtype(dtype).itemsize
    if fsize % (item_size * n_channels) != 0:
        print(
            'Inconsistent number of channels between the params file and the '
            'binary dat file')

    num_timesteps = fsize // (item_size * n_channels)
    shape = (num_timesteps, n_channels)

    raw_data = np.memmap(path, dtype=dtype, offset=0, shape=shape, mode=mode)
    raw_data = raw_data[:, :-1]  # Remove sync channel

    return raw_data, n_channels - 1


def _memmap_v_probe(raw_data_path):
    """Memory map raw v_probe data."""

    # Setup raw data memory map
    path = pathlib.Path(raw_data_path)
    num_channels = 64
    dtype = 'int16'
    mode = 'r+'

    fsize = path.stat().st_size
    item_size = np.dtype(dtype).itemsize
    if fsize % (item_size * num_channels) != 0:
        print(
            'Inconsistent number of channels between the params file and the '
            'binary dat file')

    num_timesteps = fsize // (item_size * num_channels)
    shape = (num_timesteps, num_channels)

    raw_data = np.memmap(path, dtype=dtype, offset=0, shape=shape, mode=mode)

    return raw_data, num_channels


def compute_mean_waveforms(kilosort_dir,
                           raw_data_path,
                           max_spike_to_average=1000):
    """Compute and save mean waveforms."""

    if 'spikeglx' in raw_data_path:
        raw_data, num_channels = _memmap_neuropixel(raw_data_path)
    elif 'v_probe' in raw_data_path:
        raw_data, num_channels = _memmap_v_probe(raw_data_path)
    else:
        raise ValueError(
            f'raw_data_path {raw_data_path} does not contain spikeglx or '
            'v_probe'
        )

    spike_times = np.squeeze(np.load(os.path.join(kilosort_dir, 'spike_times.npy')))
    spike_clusters = np.squeeze(np.load(os.path.join(kilosort_dir, 'spike_clusters.npy')))
    cluster_group = np.genfromtxt(
        fname=os.path.join(kilosort_dir, 'cluster_group.tsv'), delimiter="\t",
        skip_header=True)
    cluster_nums = cluster_group[:, 0].astype(int)

    mean_waveforms = []

    for cluster_num in cluster_nums:
        cluster_spike_indices = spike_clusters == cluster_num
        cluster_spike_times = spike_times[cluster_spike_indices].astype(int)
        
        if len(cluster_spike_times) > max_spike_to_average:
            cluster_spike_times = np.random.choice(
                cluster_spike_times, size=max_spike_to_average)

        sum_waveform = np.zeros((82, num_channels), dtype=float)
        count_waveforms = 0
        for t in cluster_spike_times:
            new_waveform = raw_data[t - 41: t + 41, :].astype(float)
            if new_waveform.shape[0] == 82:
                count_waveforms += 1
                sum_waveform += new_waveform

        # print(f'count_waveforms: {count_waveforms}')
        mean_waveform = sum_waveform / count_waveforms
        mean_waveforms.append(mean_waveform)

    mean_waveforms = np.array(mean_waveforms)

    # Save mean waveforms
    mean_waveforms_write_path = os.path.join(kilosort_dir, 'mean_waveforms')
    print('Saving waveforms to {}'.format(mean_waveforms_write_path))
    np.save(mean_waveforms_write_path, mean_waveforms)

    _plot_random_clusters(mean_waveforms, kilosort_dir)
    
    return
