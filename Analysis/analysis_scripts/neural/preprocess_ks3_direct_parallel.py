import copy
import pickle
import matplotlib

# matplotlib.use('TkAgg')

# import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import scipy.io as io
import scipy.stats as stats
import scipy.ndimage as ndimage
import os
from os.path import exists
from ibllib.io import spikeglx
import sys
import shutil
from run_parse_trials import *
from pathlib import Path
import concurrent.futures
import multiprocessing

matplotlib.use('agg')  # Use the Agg backend

sys.path.append('/Users/laptopd/Documents/Compositionality/Analysis')


def _get_session(date_str):
    sessions_path = '/Users/laptopd/Documents/Compositionality/Analysis/sessions_F.obj'
    session_file = open(sessions_path, 'rb')

    sessions = pickle.load(session_file)
    session_file.close()
    session_id = 0
    for idx in range(len(sessions)):
        if sessions[idx].date == date_str:
            session_id = idx
            break

    session = sessions[session_id]
    return session


def load_raster_to_trials(path_internal, date_str, rewrite=0):
    """
    :param path_internal:
    :return: load the spiketimes of neurons into the trial structure
    """
    path_source = path_internal.parents[2]
    path_save = path_internal.parents[0]

    if exists(path_save / 'raster_phy.obj') and rewrite == 0:
        return

    session = _get_session(date_str)

    def _get_nidaq_meta(spikeglx_nidaq_dir):
        """Read nidaq metadata to get sample rate."""
        nidaq_meta_file = [
            x for x in os.listdir(spikeglx_nidaq_dir) if 'nidq.meta' in x
        ]
        if len(nidaq_meta_file) != 1:
            raise ValueError(
                f'Did not find a single nidaq meta file, instead found '
                f'{nidaq_meta_file}'
            )
        nidaq_meta_file = os.path.join(spikeglx_nidaq_dir, nidaq_meta_file[0])
        with open(nidaq_meta_file) as f:
            nidaq_meta_lines = f.readlines()
        nidaq_meta = {
            x.split('=', 1)[0]: x.split('=', 1)[1][:-1]
            for x in nidaq_meta_lines
        }
        # sample_rate = float(nidaq_meta['niSampRate'])

        return nidaq_meta

    def _get_imec_meta(dir):
        """Read nidaq metadata to get sample rate."""
        imec_meta_file = [
            x for x in os.listdir(dir) if 'ap.meta' in x
        ]
        if len(imec_meta_file) != 1:
            raise ValueError(
                f'Did not find a single imec meta file, instead found '
                f'{imec_meta_file}'
            )
        imec_meta_file = os.path.join(dir, imec_meta_file[0])
        with open(imec_meta_file) as f:
            imec_meta_lines = f.readlines()
        imec_meta = {
            x.split('=', 1)[0]: x.split('=', 1)[1][:-1]
            for x in imec_meta_lines
        }

        return imec_meta

    def _get_nidaq_reader(spikeglx_nidaq_dir):
        """Get nidaq reader."""
        nidaq_bin_file = [
            x for x in os.listdir(spikeglx_nidaq_dir) if 'nidq.bin' in x
        ]
        if len(nidaq_bin_file) != 1:
            raise ValueError(
                f'Did not find a single nidaq bin file, instead found '
                f'{nidaq_bin_file}'
            )
        nidaq_bin_file = nidaq_bin_file[0]
        reader = spikeglx.Reader(os.path.join(spikeglx_nidaq_dir, nidaq_bin_file))

        return reader

    nidaq_meta = _get_nidaq_meta(path_source)
    sample_rate_nd = float(nidaq_meta['niSampRate'])
    start_nd = float(nidaq_meta['firstSample']) / sample_rate_nd
    len_nd = float(nidaq_meta['fileTimeSecs'])

    imec_meta = _get_imec_meta(path_source)
    sample_rate_im = float(imec_meta['imSampRate'])
    start_im = float(imec_meta['firstSample']) / sample_rate_im
    len_im = float(imec_meta['fileTimeSecs'])

    reader = _get_nidaq_reader(path_source)
    stride = int(np.round(sample_rate_nd / 1000))
    analog_raw, digital_raw = reader.read(nsel=slice(0, reader.ns, stride))
    read_sample_rate = sample_rate_nd / float(stride)

    sync_raw = digital_raw[:, 7]
    sync = sync_raw == 1
    photodiode_raw = analog_raw[:, 1]
    photodiode = photodiode_raw > 3
    square_nidq = analog_raw[:, 0] < 3
    square_nidq_ = np.roll(square_nidq, -1)
    idx_flip_square = np.where(square_nidq != square_nidq_)[0]
    idx_flip_square = idx_flip_square[np.arange(1, len(idx_flip_square) + 1, 2)]
    # square_nidq_times = idx_flip_square / read_sample_rate
    # square_imec_times = io.loadmat(path + 'sync_times.mat')['sync_times']

    # checking the square wave, there is a linear scaling drift between im and nd
    # nd timestamps should first multiply the ratio, then subtract the start offset
    scaling_ratio = len_im / len_nd
    start_offset = start_im - start_nd

    sync_ = np.roll(sync, -1)
    idx_flip_sync = np.where(sync != sync_)[0]
    idx_flip_sync = idx_flip_sync[np.arange(1, len(idx_flip_sync) + 1, 2)]
    sync_times_probe = idx_flip_sync / read_sample_rate

    photodiode_ = np.roll(photodiode, -1)
    idx_flip_pho = np.where(photodiode != photodiode_)[0]
    idx_flip_pho = idx_flip_pho[np.arange(1, len(idx_flip_pho) + 1, 2)]
    photo_times_probe = idx_flip_pho / read_sample_rate

    file_spiketimes = path_internal / 'spike_times.npy'
    spiketimes = np.load(file_spiketimes)
    sample_rate_probe = 3e4
    spiketimes = spiketimes / sample_rate_probe

    # file_cluster_label = path_internal / 'cluster_group.tsv'
    # df = pd.read_csv(file_cluster_label, sep='\t')
    # # idx_good = np.array(df['cluster_id'][df['KSLabel'] == 'good'].array)
    # group = np.array(df['group'].array)
    # cluster_id = np.array(df['cluster_id'].array)
    # print('good: ' + str(sum(group == 'good')) + ' mua: ' + str(sum(group == 'mua')))
    # cluster_id = cluster_id[group!='noise']
    # group = group[group!='noise']
    # file_clusters = path_internal / 'spike_clusters.npy'
    # spike_clusters = np.load(file_clusters)
    # n_units = len(cluster_id)
    # spiketimes_all_units = np.ndarray(n_units, dtype=object)
    file_cluster_label = data_dir_internal / 'cluster_KSLabel.tsv'
    df = pd.read_csv(file_cluster_label, sep='\t')
    # idx_good = np.array(df['cluster_id'][df['KSLabel'] == 'good'].array)
    group = np.array(df['KSLabel'].array)
    cluster_id = np.array(df['cluster_id'].array)
    print('good: ' + str(sum(group == 'good')) + ' mua: ' + str(sum(group == 'mua')))
    file_clusters = data_dir_internal / 'spike_templates.npy'
    spike_clusters = np.load(file_clusters)
    n_units = len(group)
    spiketimes_all_units = np.ndarray(n_units, dtype=object)

    for i in range(n_units):
        spiketimes_all_units[i] = spiketimes[spike_clusters == cluster_id[i]]

    sync_time_start_idx = []
    for i in range(session.n_total):
        time_i = session.cue1_start_time[i]
        idx = np.where(session.trialsync_times > time_i)[0][0]
        sync_time_start_idx.append(idx)

    sync_time_start_idx = np.array(sync_time_start_idx)
    # plotted sync times against photo times as a sanity check
    sync_times_probe = sync_times_probe[sync_time_start_idx]
    sync_times_probe = sync_times_probe * scaling_ratio - start_offset

    photo_times_probe = photo_times_probe[:len(sync_time_start_idx)]
    photo_times_probe = photo_times_probe * scaling_ratio - start_offset

    raster = np.ndarray((n_units, session.n_total), dtype=object)
    before = -3
    after = 6

    def _spiketime_to_raster(idx_unit):
        spiketimes_i = spiketimes_all_units[idx_unit]
        idx_s = 0
        idx_start = 0
        for t in range(session.n_total):
            raster_it = []
            cue1_t = session.cue1_start_time[t]
            end_t = cue1_t + after  # session.trial_len[t]
            start_t = cue1_t + before
            spiketimes_it = spiketimes_i + cue1_t - photo_times_probe[t]
            idx_s = idx_start
            while True:

                try:
                    if spiketimes_it[idx_s] < start_t:
                        idx_s += 1

                    if start_t <= spiketimes_it[idx_s] <= end_t:
                        if spiketimes_it[idx_s - 1] < start_t:
                            idx_start = idx_s
                        raster_it.append(spiketimes_it[idx_s] - cue1_t)
                        idx_s += 1

                    if end_t < spiketimes_it[idx_s]:
                        break
                except:
                    # idx_s exceeding total spike numbers
                    break

            raster[idx_unit][t] = np.array(raster_it)
        print(idx_unit)

    if __name__ == '__main__':
        num_workers = multiprocessing.cpu_count()  # Number of available CPU cores
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Define the items to process (e.g., a list of 1000 items)
            items = np.arange(n_units)

            # Submit the tasks to the executor
            results = executor.map(_spiketime_to_raster, items)

    raster_file = {
        'raster': raster,
        'group': group,
        'window': [before, after]
    }
    file_to_save = open(path_save / 'raster_phy.obj', 'wb+')
    pickle.dump(raster_file, file_to_save)
    file_to_save.close()


def filter_trials(path_internal, date_str, rewrite=0):
    # remove low firing trials
    path_save = path_internal.parents[0]

    if exists(path_save / 'raster_phy_filtered.obj') and rewrite == 0:
        return

    def _moving_average(data, window):
        data_ma = np.zeros_like(data).astype(float)
        for i in range(len(data)):
            xmin = int(max(0, np.floor(i - window / 2)))
            xmax = int(min(len(data), np.ceil((i + window / 2))))
            data_ma[i] = np.sum(data[xmin:xmax]) / len(data[xmin:xmax])
        return data_ma

    raster_handle = open(path_save / 'raster_phy.obj', 'rb')
    raster_file = pickle.load(raster_handle)
    raster_handle.close()

    session = _get_session(date_str)

    raster = raster_file['raster']
    window = raster_file['window']
    group = raster_file['group']
    n_units = raster.shape[0]
    n_trials = raster.shape[1]

    # idx_high_units = np.ones(n_units, dtype=int)
    idx_good_trials = np.ndarray(n_units, dtype=object)
    win_size = 50
    for n in np.arange(n_units):
        spikes_raw = np.array([len(raster[n, t]) for t in range(n_trials)])
        spikes_smooth = _moving_average(spikes_raw, win_size)
        spikes_asc = np.sort(spikes_smooth)
        idx_asc = np.argsort(spikes_smooth)
        cumsum = np.cumsum(spikes_asc) / sum(spikes_asc)
        try:
            idx_cut = np.where(cumsum >= 0.05)[0][0]
        except:
            continue
        idx_good_trials[n] = idx_asc[idx_cut:]
        print(n)

    # raster_good_units = raster[idx_high_units == 1, :]
    # idx_good_unit_trials = idx_good_trials[idx_high_units == 1]
    # group = group[idx_high_units == 1]
    raster_filtered_file = {
        'raster': raster,
        'idx_good_trials': idx_good_trials,
        'group': group,
        'window': window
    }
    file_to_save = open(path_save / 'raster_phy_filtered.obj', 'wb+')
    pickle.dump(raster_filtered_file, file_to_save)
    file_to_save.close()

    return


def __plot_raster(params):
    idx_unit = params[0]
    raster1_i = params[1]
    raster3_i = params[2]
    session = params[3]
    raster_path = params[4]
    group = params[5]
    idx_good_trials = params[6]
    # for idx_unit in range(n_units):
    raster_name_i = raster_path / (str(idx_unit) + '_' + group[idx_unit] + '_raster_p' + '.pdf')
    if exists(raster_name_i):
        return
        # continue

    idx_good_trials_n = idx_good_trials[idx_unit]

    # raster1_i = raster[idx_unit, :]
    # raster3_i = raster_sac_on[idx_unit, :]

    n_subplots = 2
    fig, ax = plt.subplots(1, n_subplots, figsize=(20, 5))

    for t in range(session.n_total):
        x1 = raster1_i[t]
        x3 = raster3_i[t]
        y = np.ones(len(x1)) * t
        ax[0].scatter(x1, y, s=0.25, c='k')
        ax[1].scatter(x3, y, s=0.25, c='k')

    ax[0].plot([0, 0], [0, session.n_total], linestyle='--', color=[1, 0, 0], linewidth=1)
    ax[1].plot([0, 0], [0, session.n_total], linestyle='--', color=[1, 0, 0], linewidth=1)
    ax[0].set_title('Cue1 on')
    ax[1].set_title('Saccade on')
    try:
        ax[0].scatter(np.zeros(len(idx_good_trials_n)) + 6.1, idx_good_trials_n, s=1, color='r')
    except:
        pass

    # plt.show()
    plt.ioff()
    fig.savefig(raster_name_i)
    plt.close()
    print('raster' + str(idx_unit))


def gen_rasters(path_internal, path_external, date_str, rewrite=0):
    path_save = path_internal.parents[0]

    raster_handle = open(path_save / 'raster_phy_filtered.obj', 'rb')
    raster_file = pickle.load(raster_handle)
    raster_handle.close()
    raster = raster_file['raster']
    group = raster_file['group']
    idx_good_trials = raster_file['idx_good_trials']
    session = _get_session(date_str)

    n_units = raster.shape[0]
    # n_units = 3
    n_trials = raster.shape[1]

    raster_cue1_on = raster
    raster_sac_on = np.ndarray(raster.shape, dtype=object)

    for t in range(n_trials):
        target_xy = session.target_xy[t, :]
        saccade_on = session.saccade_onset[t] - session.cue1_start_time[t]
        for n in range(n_units):
            raster_sac_on[n, t] = raster_cue1_on[n, t] - saccade_on

    raster_path = path_external / 'raster'
    if not os.path.exists(raster_path):
        os.mkdir(raster_path)

    if __name__ == '__main__':
        num_workers = multiprocessing.cpu_count()  # Number of available CPU cores
        # os.environ['DISPLAY'] = ':0'
        pool = multiprocessing.Pool(num_workers)

        # Define the items to process (e.g., a list of 1000 items)
        items = []  # Replace 30 with your desired number of items
        [items.append((i, raster[i, :], raster_sac_on[i, :], session, raster_path, group, idx_good_trials)) for i in
         range(n_units)]

        # Use the map method to parallelize the tasks
        pool.map(__plot_raster, items)

        # Close the pool to release resources
        pool.close()
        pool.join()
    return


def raster_to_bins(path_internal, path_external, date_str, rewrite=0):
    '''
    Load raster into bins, and perform Gaussian smoothing
    :param path_internal:
    :param path_external:
    :param date_str:
    :param rewrite:
    :return:
    '''
    save_dir = path_external / 'fr_1ms'

    raster_handle = open(path_internal.parents[0] / 'raster_phy_filtered.obj', 'rb')
    raster_file = pickle.load(raster_handle)
    raster_handle.close()
    raster = raster_file['raster']
    idx_high_trials = raster_file['idx_good_trials']
    group = raster_file['group']

    session = _get_session(date_str)

    n_units = raster.shape[0]
    n_trials = raster.shape[1]

    raster_cue1_on = raster
    raster_sac_on = np.ndarray(raster.shape, dtype=object)
    speed = 3  # 3 units/s

    for t in range(n_trials):
        target_xy = session.target_xy[t, :]
        saccade_on = session.saccade_onset[t] - session.cue1_start_time[t]
        for n in range(n_units):
            raster_sac_on[n, t] = raster_cue1_on[n, t] - saccade_on

    width = 1 / 1000  # 1ms
    overlap = 0  # 33ms
    lim_cue1 = [-1, 2.2]
    lim_sac = [-0.7, 1.2]
    bins_cue1 = np.arange(lim_cue1[0], lim_cue1[1], width)
    bins_cue1 = np.stack((bins_cue1, bins_cue1 + width), axis=0)
    bins_sac = np.arange(lim_sac[0], lim_sac[1], width)
    bins_sac = np.stack((bins_sac, bins_sac + width), axis=0)

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # load spike times into PSTH bins
    def _raster_to_bin(idx_unit):
        save_name = save_dir / (str(idx_unit) + '_p.obj')
        if exists(save_name) and rewrite == 0:
            return
        print(idx_unit)
        psth_cue1 = np.zeros((n_trials, bins_cue1.shape[1]))
        psth_sac = np.zeros((n_trials, bins_sac.shape[1]))

        for t in range(n_trials):
            raster_cue1_t = raster_cue1_on[idx_unit, t]
            raster_sac_t = raster_sac_on[idx_unit, t]
            idx_cue1_start = 0
            idx_sac_start = 0

            for b in range(bins_cue1.shape[1]):
                idx_cue1 = idx_cue1_start
                while True:
                    try:
                        if raster_cue1_t[idx_cue1] < bins_cue1[0, b]:
                            idx_cue1 += 1

                        if bins_cue1[0, b] < raster_cue1_t[idx_cue1] < bins_cue1[1, b]:
                            try:
                                if raster_cue1_t[idx_cue1 - 1] < bins_cue1[0, b]:
                                    idx_cue1_start = idx_cue1
                            except:
                                pass
                            psth_cue1[t, b] += 1
                            idx_cue1 += 1

                        if bins_cue1[1, b] < raster_cue1_t[idx_cue1]:
                            break
                    except:
                        break
            # psth_cue1[t, :] = ndimage.gaussian_filter(input=psth_cue1[t, :], sigma=0.1/width, truncate=2)

            for b in range(bins_sac.shape[1]):
                idx_sac = idx_sac_start
                while True:
                    try:
                        if raster_sac_t[idx_sac] < bins_sac[0, b]:
                            idx_sac += 1

                        if bins_sac[0, b] < raster_sac_t[idx_sac] < bins_sac[1, b]:
                            try:
                                if raster_sac_t[idx_sac - 1] < bins_sac[0, b]:
                                    idx_sac_start = idx_sac
                            except:
                                pass
                            psth_sac[t, b] += 1
                            idx_sac += 1

                        if bins_sac[1, b] < raster_sac_t[idx_sac]:
                            break
                    except:
                        break
            # psth_sac[t, :] = ndimage.gaussian_filter(input=psth_sac[t, :], sigma=0.1/width, truncate=2)

        psth_cue1 /= width
        psth_sac /= width

        fr_bin_file = {
            'psth_cue1': psth_cue1,
            'bins_cue1': bins_cue1,
            'psth_sac': psth_sac,
            'bins_sac': bins_sac,
            'group': group[idx_unit],
            'idx_high_trials': idx_high_trials[idx_unit]
        }
        file_to_save = open(save_name, 'wb+')
        pickle.dump(fr_bin_file, file_to_save)
        file_to_save.close()
        print(idx_unit)

    if __name__ == '__main__':
        num_workers = multiprocessing.cpu_count()  # Number of available CPU cores
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Define the items to process (e.g., a list of 1000 items)
            items = np.arange(n_units)
            # items = np.arange(100)
            # Submit the tasks to the executor
            results = executor.map(_raster_to_bin, items)
    return


def __plot_psth(params):
    fr = params[0]
    fr_bin_path = params[1]
    session = params[2]
    target_set = params[3]
    list_idx_tar = params[4]
    psth_path = params[5]
    idx_cor_trials = session.true_correct
    idx_completed_trials = np.where(session.behavior == 'completed')[0]

    n_targets = len(target_set)
    n = fr.split('_')[0]
    file = fr_bin_path / fr
    # file = fr_bin_path /'0_p.obj'
    fr_bin_file = open(file, 'rb')
    fr_bin = pickle.load(fr_bin_file)
    fr_bin_file.close()
    psth_cue1_raw = fr_bin['psth_cue1']
    psth_sac_raw = fr_bin['psth_sac']
    psth_cue1_raw = ndimage.gaussian_filter(input=psth_cue1_raw, sigma=[0, 50], truncate=1)
    psth_sac_raw = ndimage.gaussian_filter(input=psth_sac_raw, sigma=[0, 50], truncate=1)



    bins_cue1_raw = fr_bin['bins_cue1']
    bins_sac_raw = fr_bin['bins_sac']

    group = fr_bin['group']

    idx_high_trials = fr_bin['idx_high_trials']
    n_trials = psth_cue1_raw.shape[0]

    speed = 3  # 3 units/sec
    width = 3 / 30  # 66ms
    step = 1 / 30  # 33ms

    bins_cue1 = np.arange(bins_cue1_raw[0, 0], bins_cue1_raw[0, -1], step)
    bins_cue1 = np.stack((bins_cue1, bins_cue1 + width), axis=0)
    bins_sac = np.arange(bins_sac_raw[0, 0], bins_sac_raw[0, -1], step)
    bins_sac = np.stack((bins_sac, bins_sac + width), axis=0)

    psth_cue1 = np.zeros((n_trials, bins_cue1.shape[1]))
    psth_sac = np.zeros((n_trials, bins_sac.shape[1]))
    dataset = [psth_cue1, psth_sac]
    dataset_raw = [psth_cue1_raw, psth_sac_raw]
    bins = [bins_cue1, bins_sac]
    bins_raw = [bins_cue1_raw, bins_sac_raw]

    for d in range(len(dataset)):

        for t in range(dataset[d].shape[1]):
            ind = np.logical_and(bins[d][0, t] <= bins_raw[d][0, :], bins_raw[d][0, :] < bins[d][1, t])
            for tri in range(n_trials):
                # dataset[d][n, tri, t] = np.mean(dataset_raw[d][n, tri, ind], axis=0)
                dataset[d][tri, t] = np.mean(dataset_raw[d][tri, ind])

    xmin = [-1, -0.7]
    xmax = abs(np.array(target_set).sum(axis=2)[:, 1])
    list_xmax = [xmax, np.ones(n_targets) * 2]
    titles = ['Cue1', 'Saccade']
    cmap = matplotlib.cm.get_cmap("Set1", n_targets)

    # n = 1
    # trials wanted for each target
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 5), sharex='col', sharey='col')
    ymax = 0
    for tar in range(n_targets):
        # for trained targets only correct trials
        try:
            # look at correct trials
            idx_high_good_trials = \
                np.intersect1d(np.intersect1d(idx_high_trials, idx_completed_trials), list_idx_tar[tar])
            # idx_high_good_trials = \
            #     np.intersect1d(np.intersect1d(idx_high_trials, idx_cor_trials), list_idx_tar[tar])
        except:
            idx_high_good_trials = []

        for f in range(2):

            x = np.arange(xmin[f], (list_xmax[f][tar]) / speed, step)
            try:
                y_all_trials = dataset[f][idx_high_good_trials, :len(x)]
            except:
                y_all_trials = np.zeros((1, len(x)))
            y = np.mean(y_all_trials, 0)
            y_se = np.std(y_all_trials, 0) / np.sqrt(y_all_trials.shape[0])

            # ax[f].plot(x, y, label=target_label[tar], color=cmap(tar))
            ax[f].plot(x, y, color=cmap(tar))

            ax[f].fill(np.concatenate((x, np.flip(x))), np.concatenate((y + y_se, np.flip(y - y_se))),
                       facecolor=[cmap(tar)[i] * (1 - i // 3) + i // 3 * 0.1 for i in range(4)])
            ymax = max(ymax, max(y + y_se))
            ax[f].set_title(titles[f])
            ax[f].set_xlabel('Time (S)')
            ax[f].set_ylabel('Firing rate (Hz)')
            # if f == 0:
            #     ax[f].legend()

    for f in range(2):

        ax[f].set_ylim([0, ymax])
        if f < 2:
            ax[f].plot([1 / 3, 1 / 3], [0, ymax], color=(0, 0, 0, 0.15), linestyle='--')
            ax[f].plot([2 / 3, 2 / 3], [0, ymax], color=(0, 0, 0, 0.15), linestyle='--')
            ax[f].plot([1, 1], [0, ymax], color=(0, 0, 0, 0.15), linestyle='--')
            if f == 0:
                ax[f].plot([4 / 3, 4 / 3], [0, ymax], color=(0, 0, 0, 0.15), linestyle='--')
                ax[f].plot([-0.5, -0.5], [0, ymax], color=(0, 0, 0, 0.15), linestyle='--')

        ax[f].plot([0, 0], [0, ymax], color=(0, 0, 0, 0.3))

    plt.tight_layout()
    # plt.show()

    psth_name_n = psth_path / (str(n) + '_' + group + '_psth_all_par' + '.pdf')
    fig.savefig(psth_name_n)
    plt.ioff()
    plt.close()

    print(n)


def gen_psth(save_dir_external, date_str, target_params, rewrite=0):
    psth_path = save_dir_external / 'psth'
    if not os.path.exists(psth_path):
        os.mkdir(psth_path)

    fr_bin_path = save_dir_external / 'fr_1ms'
    fr_files = os.listdir(fr_bin_path)

    session = _get_session(date_str)




    # if len(session.target_set) == 72:  # set1
    #     target_set = target_params['set1']['target_set']
    #     target_label = target_params['set1']['target_label']
    #
    # elif len(session.target_set) == 73:  # set2
    #     target_set = target_params['set2']['target_set']
    #     target_label = target_params['set2']['target_label']

    test_target_id = np.where((np.array(session.alpha_group_id_list) == 1))[0]
    target_set = []
    [target_set.append(session.target_set[i]) for i in test_target_id]

    n_targets = len(target_set)
    target_id = []
    for i in range(n_targets):
        target_i = target_set[i]
        [target_id.append(i0) for i0, t0 in enumerate(session.target_set) if t0 == target_i]
    list_idx_tar = np.ndarray(n_targets, dtype=object)
    for t in range(n_targets):
        list_idx_tar[t] = np.where(session.target_id == target_id[t])[0]


    if __name__ == '__main__':
        num_workers = multiprocessing.cpu_count()  # Number of available CPU cores
        # os.environ['DISPLAY'] = ':0'
        pool = multiprocessing.Pool(num_workers)

        # # Define the items to process (e.g., a list of 1000 items)
        items = []  # Replace 30 with your desired number of items

        [items.append((fr_files[i], fr_bin_path, session, target_set, list_idx_tar, psth_path)) for i in range(len(fr_files))]

        # Use the map method to parallelize the tasks
        pool.map(__plot_psth, items)

        # Close the pool to release resources
        pool.close()
        pool.join()

    return




def filter_units(path_internal, path_external, date_str, target_params, rewrite=0):
    min_num_trials = 10
    min_fr = 1
    psth_good_path = path_internal + '/psth_good'
    psth_bad_path = path_internal + '/psth_bad'
    if not os.path.exists(psth_good_path):
        os.mkdir(psth_good_path)
    if not os.path.exists(psth_bad_path):
        os.mkdir(psth_bad_path)

    psth_path = path_internal + '/psth/'

    fr_bin_path = path_external + '/fr_1ms/'
    fr_files = os.listdir(fr_bin_path)
    n_units = len(fr_files)

    session = _get_session(date_str)
    has_cue2 = True
    targets = session.target_xy
    if (targets == 0).any():
        has_cue2 = False

    speed = 3  # 3 units/sec
    width = 2 / 30  # 66ms
    step = 1 / 30  # 33ms

    idx_cor_trials = session.true_correct
    idx_completed_trials = np.where(session.behavior == 'completed')[0]
    target_set = np.array(target_params['target_set'])
    target_order = target_params['target_set_order']
    target_label = target_params['target_label']
    n_trained = target_params['n_trained']
    n_targets = len(target_set)

    idx_filter_units = np.ones(n_units)

    list_idx_tar = np.ndarray(n_targets, dtype=object)
    for t in range(n_targets):
        idx_tar = np.where(np.sum(session.target_xy == target_set[t, :], axis=1) == 2)[0]
        idx_order = np.where(session.order_xy == target_order[t])[0]
        list_idx_tar[t] = np.intersect1d(idx_tar, idx_order)

    for n in range(len(fr_files)):
        # n=12
        file = fr_bin_path + fr_files[n]
        fr_bin_file = open(file, 'rb')
        fr_bin = pickle.load(fr_bin_file)
        fr_bin_file.close()
        psth_cue1_raw = fr_bin['psth_cue1']
        psth_cue2_raw = fr_bin['psth_cue2']
        psth_sac_raw = fr_bin['psth_sac']
        bins_cue1_raw = fr_bin['bins_cue1']
        bins_cue2_raw = fr_bin['bins_cue2']
        bins_sac_raw = fr_bin['bins_sac']
        group = fr_bin['group']

        idx_high_trials = fr_bin['idx_high_trials']
        n_trials = psth_cue1_raw.shape[0]

        bins_cue1 = np.arange(bins_cue1_raw[0, 0], bins_cue1_raw[0, -1], step)
        bins_cue1 = np.stack((bins_cue1, bins_cue1 + width), axis=0)
        bins_cue2 = np.arange(bins_cue2_raw[0, 0], bins_cue2_raw[0, -1], step)
        bins_cue2 = np.stack((bins_cue2, bins_cue2 + width), axis=0)
        bins_sac = np.arange(bins_sac_raw[0, 0], bins_sac_raw[0, -1], step)
        bins_sac = np.stack((bins_sac, bins_sac + width), axis=0)

        psth_cue1 = np.zeros((n_trials, bins_cue1.shape[1]))
        psth_cue2 = np.zeros((n_trials, bins_cue2.shape[1]))
        psth_sac = np.zeros((n_trials, bins_sac.shape[1]))
        dataset = [psth_cue1, psth_cue2, psth_sac]
        dataset_raw = [psth_cue1_raw, psth_cue2_raw, psth_sac_raw]
        bins = [bins_cue1, bins_cue2, bins_sac]
        bins_raw = [bins_cue1_raw, bins_cue2_raw, bins_sac_raw]
        # n = 1
        #     for d in range(len(dataset)):
        #         if not has_cue2 and d == 1:
        #             continue
        #
        #         for t in range(dataset[d].shape[1]):
        #             ind = np.logical_and(bins[d][0, t] <= bins_raw[d][0, :], bins_raw[d][0, :] < bins[d][1, t])
        #             for tri in range(n_trials):
        #                 # dataset[d][n, tri, t] = np.mean(dataset_raw[d][n, tri, ind], axis=0)
        #                 dataset[d][tri, t] = np.mean(dataset_raw[d][tri, ind])

        xmin = [-1, -0.1, -0.7]
        xmax = np.max(target_set, 1)
        list_xmax = [xmax, np.ones(n_targets), np.ones(n_targets) * 3]
        titles = ['Cue1', 'Cue2', 'Saccade']
        cmap = matplotlib.cm.get_cmap("Set1")

        # n = 1
        # trials wanted for each target
        is_high = 0
        baseline_list = []
        baseline = np.where(bins_cue1_raw[0, :] < -0.05)[0]
        for tar in range(n_trained):
            # for trained targets only correct trials
            try:
                if tar < n_trained:
                    idx_high_good_trials = \
                        np.intersect1d(np.intersect1d(idx_high_trials, idx_cor_trials), list_idx_tar[tar])

                    if len(idx_high_good_trials) < min_num_trials:
                        idx_filter_units[n] = 0
                        break
                    trials_tmp = psth_cue1_raw[idx_high_good_trials, :]
                    baseline_list.append(np.mean(trials_tmp[:, baseline], axis=1))


                # testing targets look at completed trials
                else:
                    idx_high_good_trials = \
                        np.intersect1d(np.intersect1d(idx_high_trials, idx_completed_trials), list_idx_tar[tar])
            except:
                idx_high_good_trials = []

            # if len(idx_high_good_trials) < min_num_trials:
            #     idx_good_units[n] = 0
            #     break

            for d in range(len(dataset)):
                if not has_cue2 and d == 1:
                    continue
                if np.mean(dataset_raw[d][idx_high_good_trials, :]) > min_fr:
                    is_high = 1

        if is_high == 0:
            idx_filter_units[n] = 0

        if len(baseline_list) == n_trained:
            bl = baseline_list
            stat, p = stats.f_oneway(bl[0], bl[1], bl[2], bl[3], axis=0)
            if p < 0.01:
                idx_filter_units[n] = 0

            # for f in range(3):
            #     if not has_cue2 and f == 1:
            #         continue
            #
            #     x = np.arange(xmin[f], (list_xmax[f][tar] + 0.01) / speed, step)
            #     try:
            #         y_all_trials = dataset[f][idx_high_good_trials, :len(x)]
            #     except:
            #         y_all_trials = np.zeros((1, len(x)))
            #     y = np.mean(y_all_trials, 0)
            #     y_se = np.std(y_all_trials, 0) / np.sqrt(y_all_trials.shape[0])

        psth_name_n = '/' + str(n) + '_' + group + '_psth' + '.pdf'

        if idx_filter_units[n]:
            shutil.copy(psth_path + psth_name_n, psth_good_path + psth_name_n)
        else:
            shutil.copy(psth_path + psth_name_n, psth_bad_path + psth_name_n)

        print(n)

    path = path_internal + '/ks_3_output_pre_v6/'
    file_cluster_label = path + 'cluster_KSLabel.tsv'
    df = pd.read_csv(file_cluster_label, sep='\t')
    group = np.array(df['KSLabel'].array)

    # 假设是计算出来后，要存的数据
    idx_flter_file = {
        'idx_filter_units': idx_filter_units,
        'group': group
    }
    # 把这个数据存到电脑里，叫这个文件名字
    save_name = path_internal + '/idx_filter_units.obj'

    # 打开这个文件名，在程序里的handle叫这个
    file_to_save = open(save_name, 'wb+')

    # 把要存的数据dump到这个handle里
    pickle.dump(idx_flter_file, file_to_save)
    # 关掉这个handle
    file_to_save.close()
    return


if __name__ == '__main__':
    NP_dates = ['20231106_F_g0']
    session_dates = ['20231106']
    n_sessions = len(session_dates)
    for s in range(n_sessions):
        date_str = session_dates[s]
        session_dir_internal = Path('/Users/laptopd/Documents/Compositionality/Analysis/spikedata/F/NP/') / NP_dates[s]
        session_dir_external = Path('/Volumes/Expansion/Cheng/processed data/') / NP_dates[s]

        # list all the iterations
        dir_list = os.listdir(session_dir_internal / 'kilosort3_corrected_output')
        dir_list.sort()
        for f in dir_list:
            if not 'iteration' in f:
                continue

        data_dir_internal = session_dir_internal / 'kilosort3_corrected_output' / f / 'sorter_output'
        save_dir_external = session_dir_external / 'kilosort3_corrected_output' / f

        if not os.path.exists(save_dir_external):
            os.makedirs(save_dir_external)

        load_raster_to_trials(data_dir_internal, date_str, rewrite=0)
        # #
        # # # filter out the low firing trials
        filter_trials(data_dir_internal, date_str, rewrite=0)

        gen_rasters(data_dir_internal, save_dir_external, date_str, rewrite=0)

        # load raster into bins
        raster_to_bins(data_dir_internal, save_dir_external, date_str, rewrite=0)

        target_params = {
            'set1': {
                'target_label': ['[-1,-1]->[-1,1]', '[1,1]->[1,-1]'],
                'target_set': [[[-1, -1], [0, 2]], [[-1, 1], [0, -2]]]
            },
            'set2': {
                'target_label': ['[3,-2]->[-3,-2]', '[2,3]->[2,-3]'],
                'target_set': [[[3, -2], [-6, 0]], [[2, 3], [0, -6]]]
            }
        }
        # plot and save the psth for all units
        gen_psth(save_dir_external, date_str, target_params, rewrite=0)
        # #
        # # # filter out low-trial units
        # filter_units(path_internal, path_external, date_str, target_params, rewrite=1)
