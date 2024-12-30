import copy
import pickle
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as io
import os
from os.path import exists
from ibllib.io import spikeglx
import sys
import shutil
from run_parse_trials import *


sys.path.append('/Users/laptopd/Documents/Compositionality/Analysis')


def get_session(date_str):
    sessions_path = '/Users/laptopd/Documents/Compositionality/Analysis/sessions.obj'
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
    if exists(path_internal + '/raster.obj') and rewrite == 0:
        return

    path = path_internal + '/ks_3_output_pre_v6/'

    session = get_session(date_str)

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

    nidaq_meta = _get_nidaq_meta(path)
    sample_rate_nd = float(nidaq_meta['niSampRate'])
    start_nd = float(nidaq_meta['firstSample']) / sample_rate_nd
    len_nd = float(nidaq_meta['fileTimeSecs'])

    imec_meta = _get_imec_meta(path)
    sample_rate_im = float(imec_meta['imSampRate'])
    start_im = float(imec_meta['firstSample']) / sample_rate_im
    len_im = float(imec_meta['fileTimeSecs'])

    reader = _get_nidaq_reader(path)
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

    file_spiketimes = path + 'spike_times.npy'
    spiketimes = np.load(file_spiketimes)
    sample_rate_probe = 3e4
    spiketimes = spiketimes / sample_rate_probe

    file_cluster_label = path + 'cluster_KSLabel.tsv'
    df = pd.read_csv(file_cluster_label, sep='\t')
    # idx_good = np.array(df['cluster_id'][df['KSLabel'] == 'good'].array)
    group = np.array(df['KSLabel'].array)
    cluster_id = np.array(df['cluster_id'].array)
    print('good: ' + str(sum(group == 'good')) + ' mua: ' + str(sum(group == 'mua')))
    file_clusters = path + 'spike_templates.npy'
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
    for i in range(n_units):
        spiketimes_i = spiketimes_all_units[i]
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

            raster[i][t] = np.array(raster_it)
        print(i)

    raster_file = {
        'raster': raster,
        'group': group,
        'window': [before, after]
    }
    file_to_save = open(path_internal + '/raster.obj', 'wb+')
    pickle.dump(raster_file, file_to_save)
    file_to_save.close()


def gen_rasters(path_internal, path_external, date_str, rewrite=0):
    raster_handle = open(path_internal + '/raster.obj', 'rb')
    raster_file = pickle.load(raster_handle)
    raster_handle.close()
    raster = raster_file['raster']
    group = raster_file['group']

    session = get_session(date_str)

    n_units = raster.shape[0]
    # n_units = 3
    n_trials = raster.shape[1]

    raster_cue1_on = raster
    raster_cue2_on = np.ndarray(raster.shape, dtype=object)
    raster_sac_on = np.ndarray(raster.shape, dtype=object)
    speed = 3  # 3 units/s
    intv = 0.01  # sec
    for t in range(n_trials):
        target_xy = session.target_xy[t, :]
        cue2_on = target_xy[0] / speed + intv
        saccade_on = session.saccade_onset[t] - session.cue1_start_time[t]
        for n in range(n_units):
            raster_cue2_on[n, t] = raster_cue1_on[n, t] - cue2_on
            raster_sac_on[n, t] = raster_cue1_on[n, t] - saccade_on

    raster_path = path_external + '/raster'
    if not os.path.exists(raster_path):
        os.mkdir(raster_path)

    for i in range(n_units):
        raster1_i = raster[i, :]
        raster3_i = raster_sac_on[i, :]
        fig, ax = plt.subplots(1, 2, figsize=(20, 5))
        for t in range(session.n_total):
            x1 = raster1_i[t]
            x3 = raster3_i[t]
            y = np.ones(len(x1)) * t
            ax[0].scatter(x1, y, s=0.5, c='k')
            ax[1].scatter(x3, y, s=0.5, c='k')
        ax[0].plot([0, 0], [0, session.n_total], linestyle='--', color=[1, 0, 0], linewidth=1)
        ax[1].plot([0, 0], [0, session.n_total], linestyle='--', color=[1, 0, 0], linewidth=1)
        ax[0].set_title('Cue1 on')
        ax[1].set_title('Saccade on')
        raster_name_i = raster_path + '/' + str(i) + '_' + group[i] + '_raster' + '.pdf'
        # plt.show()
        fig.savefig(raster_name_i)
        plt.close()
        print('raster'+str(i))
    return


def filter_trials(path_internal, date_str, rewrite=0):
    # remove low firing trials

    if exists(path_internal + '/raster_filtered.obj') and rewrite == 0:
        return

    def _moving_average(data, window):
        data_ma = np.zeros_like(data).astype(float)
        for i in range(len(data)):
            xmin = int(max(0, np.floor(i - window / 2)))
            xmax = int(min(len(data), np.ceil((i + window / 2))))
            data_ma[i] = np.sum(data[xmin:xmax]) / len(data[xmin:xmax])
        return data_ma

    raster_handle = open(path_internal + '/raster.obj', 'rb')
    raster_file = pickle.load(raster_handle)
    raster_handle.close()

    session = get_session(date_str)

    raster = raster_file['raster']
    window = raster_file['window']
    group = raster_file['group']
    n_units = raster.shape[0]
    n_trials = raster.shape[1]

    # idx_high_units = np.ones(n_units, dtype=int)
    idx_good_trials = np.ndarray(n_units, dtype=object)
    win_size = 50
    for n in np.arange(n_units):
        spikes_per_trial = np.array([len(raster[n, t]) for t in range(n_trials)])

        # # filter out low firing units below 1 Hz
        # if np.max(spikes_per_trial) / (window[1] - window[0]) < 1:
        #     idx_high_units[n] = 0

        spikes_ma = _moving_average(spikes_per_trial, win_size)
        idx_good_trials[n] = np.where(spikes_ma > np.max(spikes_ma) / 3)[0]

    # raster_good_units = raster[idx_high_units == 1, :]
    # idx_good_unit_trials = idx_good_trials[idx_high_units == 1]
    # group = group[idx_high_units == 1]
    ratio_good_trials = np.array([len(idx_good_trials[n]) / n_trials for n in range(n_units)])
    raster_filtered_file = {
        'raster': raster,
        'idx_trials': idx_good_trials,
        'group': group,
        'window': window
    }
    file_to_save = open(path_internal + '/raster_filtered.obj', 'wb+')
    pickle.dump(raster_filtered_file, file_to_save)
    file_to_save.close()

    return


def raster_to_bins(path_internal, path_external, date_str, rewrite=0):
    if exists(path_external + '/fr_bin_1ms.obj') and rewrite == 0:
        return


    raster_handle = open(path_internal + '/raster_filtered.obj', 'rb')
    raster_file = pickle.load(raster_handle)
    raster_handle.close()
    raster = raster_file['raster']
    idx_high_trials = raster_file['idx_trials']
    group = raster_file['group']

    session = get_session(date_str)

    n_units = raster.shape[0]
    # n_units = 3
    n_trials = raster.shape[1]

    raster_cue1_on = raster
    raster_cue2_on = np.ndarray(raster.shape, dtype=object)
    raster_sac_on = np.ndarray(raster.shape, dtype=object)
    speed = 3  # 3 units/s
    intv = 0.01  # sec
    for t in range(n_trials):
        target_xy = session.target_xy[t, :]
        cue2_on = target_xy[0] / speed + intv
        saccade_on = session.saccade_onset[t] - session.cue1_start_time[t]
        for n in range(n_units):
            raster_cue2_on[n, t] = raster_cue1_on[n, t] - cue2_on
            raster_sac_on[n, t] = raster_cue1_on[n, t] - saccade_on

    width = 1 / 1000  # 1ms
    overlap = 0  # 33ms
    lim_cue1 = [-1, 1.8]
    lim_cue2 = [-0.1, 1.8]
    lim_sac = [-0.7, 1.2]
    bins_cue1 = np.arange(lim_cue1[0], lim_cue1[1], width)
    bins_cue1 = np.stack((bins_cue1, bins_cue1 + width), axis=0)
    bins_cue2 = np.arange(lim_cue2[0], lim_cue2[1], width)
    bins_cue2 = np.stack((bins_cue2, bins_cue2 + width), axis=0)
    bins_sac = np.arange(lim_sac[0], lim_sac[1], width)
    bins_sac = np.stack((bins_sac, bins_sac + width), axis=0)

    psth_cue1 = np.zeros((n_units, n_trials, bins_cue1.shape[1]))
    psth_cue2 = np.zeros((n_units, n_trials, bins_cue2.shape[1]))
    psth_sac = np.zeros((n_units, n_trials, bins_sac.shape[1]))

    # load spike times into PSTH bins
    for n in range(n_units):
        print(n)
        for t in range(n_trials):
            raster_cue1_t = raster_cue1_on[n, t]
            raster_cue2_t = raster_cue2_on[n, t]
            raster_sac_t = raster_sac_on[n, t]
            idx_cue1_start = 0
            idx_cue2_start = 0
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
                            psth_cue1[n, t, b] += 1
                            idx_cue1 += 1

                        if bins_cue1[1, b] < raster_cue1_t[idx_cue1]:
                            break
                    except:
                        break

            for b in range(bins_cue2.shape[1]):
                idx_cue2 = idx_cue2_start
                while True:
                    try:
                        if raster_cue2_t[idx_cue2] < bins_cue2[0, b]:
                            idx_cue2 += 1

                        if bins_cue2[0, b] < raster_cue2_t[idx_cue2] < bins_cue2[1, b]:
                            try:
                                if raster_cue2_t[idx_cue2 - 1] < bins_cue2[0, b]:
                                    idx_cue2_start = idx_cue2
                            except:
                                pass
                            psth_cue2[n, t, b] += 1
                            idx_cue2 += 1

                        if bins_cue2[1, b] < raster_cue2_t[idx_cue2]:
                            break
                    except:
                        break

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
                            psth_sac[n, t, b] += 1
                            idx_sac += 1

                        if bins_sac[1, b] < raster_sac_t[idx_sac]:
                            break
                    except:
                        break

    psth_cue1 /= width
    psth_cue2 /= width
    psth_sac /= width

    fr_bin_file = {
        'psth_cue1': psth_cue1,
        'bins_cue1': bins_cue1,
        'psth_cue2': psth_cue2,
        'bins_cue2': bins_cue2,
        'psth_sac': psth_sac,
        'bins_sac': bins_sac,
        'group': group,
        'idx_high_trials': idx_high_trials
    }
    file_to_save = open(path_external + '/fr_bin_1ms.obj', 'wb+')
    pickle.dump(fr_bin_file, file_to_save)
    file_to_save.close()

    return


def gen_psth(path_internal, path_external, date_str, target_params, rewrite=0):
    psth_path = path_internal + '/psth'
    if not os.path.exists(psth_path):
        os.mkdir(psth_path)

    fr_bin_path = path_external + '/fr_bin_1ms.obj'
    fr_bin_file = open(fr_bin_path, 'rb')
    fr_bin = pickle.load(fr_bin_file)
    fr_bin_file.close()
    psth_cue1_raw = fr_bin['psth_cue1']
    psth_cue2_raw = fr_bin['psth_cue2']
    psth_sac_raw = fr_bin['psth_sac']
    bins_cue1_raw = fr_bin['bins_cue1']
    bins_cue2_raw = fr_bin['bins_cue2']
    bins_sac_raw = fr_bin['bins_sac']
    group = fr_bin['group']
    n_units = len(group)
    idx_high_trials = fr_bin['idx_high_trials']
    n_trials = psth_cue1_raw.shape[1]
    session = get_session(date_str)

    speed = 3  # 3 units/sec
    width = 2 / 30  # 66ms
    step = 1 / 30  # 33ms

    bins_cue1 = np.arange(bins_cue1_raw[0, 0], bins_cue1_raw[0, -1], step)
    bins_cue1 = np.stack((bins_cue1, bins_cue1 + width), axis=0)
    bins_cue2 = np.arange(bins_cue2_raw[0, 0], bins_cue2_raw[0, -1], step)
    bins_cue2 = np.stack((bins_cue2, bins_cue2 + width), axis=0)
    bins_sac = np.arange(bins_sac_raw[0, 0], bins_sac_raw[0, -1], step)
    bins_sac = np.stack((bins_sac, bins_sac + width), axis=0)

    psth_cue1 = np.zeros((n_units, n_trials, bins_cue1.shape[1]))
    psth_cue2 = np.zeros((n_units, n_trials, bins_cue2.shape[1]))
    psth_sac = np.zeros((n_units, n_trials, bins_sac.shape[1]))
    dataset = [psth_cue1, psth_cue2, psth_sac]
    dataset_raw = [psth_cue1_raw, psth_cue2_raw, psth_sac_raw]
    bins = [bins_cue1, bins_cue2, bins_sac]
    bins_raw = [bins_cue1_raw, bins_cue2_raw, bins_sac_raw]
    # n = 148
    for d in range(len(dataset)):
        # for n in range(n_units):
        # for tr in range(n_trials):
        for t in range(dataset[d].shape[2]):
            ind = np.logical_and(bins[d][0, t] <= bins_raw[d][0, :], bins_raw[d][0, :] < bins[d][1, t])
            for tri in range(n_trials):
                # dataset[d][n, tri, t] = np.mean(dataset_raw[d][n, tri, ind], axis=0)
                dataset[d][:, tri, t] = np.mean(dataset_raw[d][:, tri, ind], axis=1)

                # print(tri)
            print(t)
    idx_cor_trials = session.true_correct
    idx_completed_trials = np.where(session.behavior == 'completed')[0]
    target_set = np.array(target_params['target_set'])
    target_order = target_params['target_set_order']
    # list_xmax = np.array([[1, 1, 3], [1, 3, 3], [3, 1, 3], [3, 1, 3], [1, 3, 3],
    #                       [2, 2, 3], [1, 2, 3], [2, 1, 3], [4, 1, 3]])
    target_label = target_params['target_label']
    n_trained = target_params['n_trained']
    n_targets = len(target_set)
    list_idx_tar = np.ndarray(n_targets, dtype=object)
    for t in range(n_targets):
        idx_tar = np.where(np.sum(session.target_xy == target_set[t, :], axis=1) == 2)[0]
        idx_order = np.where(session.order_xy == target_order[t])[0]
        list_idx_tar[t] = np.intersect1d(idx_tar, idx_order)

    xmin = [-1, -0.1, -0.7]
    xmax = np.max(target_set, 1)
    list_xmax = [xmax, np.ones(n_targets), np.ones(n_targets)*3]
    titles = ['Cue1', 'Cue2', 'Saccade']
    cmap = matplotlib.cm.get_cmap("Set1")
    for n in range(n_units):
        # trials wanted for each target
        list_idx_high_good_trials = np.ndarray(n_targets, dtype=object)
        idx_high_trials_n = idx_high_trials[n]
        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(30, 5), sharex='col', sharey='col')
        ymax = 0
        for tar in range(n_targets):
            # first 3 trained targets only correct trials
            if tar < n_trained:
                list_idx_high_good_trials[tar] = \
                    np.intersect1d(np.intersect1d(idx_high_trials_n, idx_cor_trials), list_idx_tar[tar])
            # testing targets look at completed trials
            else:
                list_idx_high_good_trials[tar] = \
                    np.intersect1d(np.intersect1d(idx_high_trials_n, idx_completed_trials), list_idx_tar[tar])

            for f in range(3):
                x = np.arange(xmin[f], (list_xmax[f][tar] + 0.01) / speed, step)
                try:
                    y_all_trials = dataset[f][n, list_idx_high_good_trials[tar], :len(x)]
                except:
                    y_all_trials = np.zeros((1, len(x)))
                y = np.mean(y_all_trials, 0)
                y_se = np.std(y_all_trials, 0) / np.sqrt(y_all_trials.shape[0])

                ax[f].plot(x, y, label=target_label[tar], color=cmap(tar))
                ax[f].fill(np.concatenate((x, np.flip(x))), np.concatenate((y + y_se, np.flip(y - y_se))),
                           facecolor=[cmap(tar)[i] * (1 - i // 3) + i // 3 * 0.1 for i in range(4)])
                ymax = max(ymax, max(y + y_se))
                ax[f].set_title(titles[f])
                ax[f].set_xlabel('Time (S)')
                ax[f].set_ylabel('Firing rate (Hz)')
                if f == 2:
                    ax[f].legend()
        for f in range(3):
            ax[f].set_ylim([0, ymax])
            if f < 2:
                ax[f].plot([1 / 3, 1 / 3], [0, ymax], color=(0, 0, 0, 0.15), linestyle='--')
                ax[f].plot([2 / 3, 2 / 3], [0, ymax], color=(0, 0, 0, 0.15), linestyle='--')
                ax[f].plot([1, 1], [0, ymax], color=(0, 0, 0, 0.15), linestyle='--')
                if f == 0:
                    ax[f].plot([4 / 3, 4 / 3], [0, ymax], color=(0, 0, 0, 0.15), linestyle='--')

            ax[f].plot([0, 0], [0, ymax], color=(0, 0, 0, 0.3))

        plt.tight_layout()
        # plt.show()

        psth_name_n = psth_path + '/' + str(n) + '_' + group[n] + '_psth' + '.pdf'
        fig.savefig(psth_name_n)
        plt.close()
        print(n)

    return


def filter_units(NP_path, date_str, rewrite=0):
    fr_bin_path = NP_path + '/fr_bin.obj'
    fr_bin_file = open(fr_bin_path, 'rb')
    fr_bin = pickle.load(fr_bin_file)
    fr_bin_file.close()
    psth_cue1 = fr_bin['psth_cue1']
    psth_cue2 = fr_bin['psth_cue2']
    psth_sac = fr_bin['psth_sac']

    group = fr_bin['group']
    n_units = len(group)
    idx_high_trials = fr_bin['idx_high_trials']

    session = get_session(date_str)

    speed = 3  # 3 units/sec
    width = 2 / 30  # 66ms
    overlap = 1 / 30  # 33ms

    idx_cor_trials = session.true_correct
    idx_completed_trials = np.where(session.behavior == 'completed')[0]
    # list_target = np.array([[1, 1], [1, 3], [3, 1], [1, 3], [3, 1], [2, 2], [1, 2], [2, 1], [4, 1]])
    # list_order = np.array([1, 1, 1, 0, 0, 1, 1, 1, 1])
    # list_xmax = np.array([[1, 1, 3], [1, 3, 3], [3, 1, 3], [3, 1, 3], [1, 3, 3],
    #                       [2, 2, 3], [1, 2, 3], [2, 1, 3], [4, 1, 3]])
    # list_label = ['1,1', '1,3', '3,1', '1,3 r', '3,1 r', '2,2', '1,2', '2,1', '4,1']
    list_target = np.array([[1, 0], [3, 0], [0, 2], [2, 0], [4, 0], [0, 1], [0, 3], [0, 4]])
    list_order = np.array([1, 1, 1, 1, 1, 1, 1, 1])
    list_xmax = np.array([[1, 1, 3], [3, 1, 3], [2, 1, 3], [2, 1, 3], [4, 1, 3],
                          [1, 1, 3], [3, 1, 3], [4, 1, 3]])
    list_label = ['1,0', '3,0', '0,2', '2,0', '4,0', '0,1', '0,3', '0,4']
    n_targets = len(list_target)
    list_idx_tar = np.ndarray(n_targets, dtype=object)
    for t in range(n_targets):
        idx_tar = np.where(np.sum(session.target_xy == list_target[t, :], axis=1) == 2)[0]
        idx_order = np.where(session.order_xy == list_order[t])[0]
        list_idx_tar[t] = np.intersect1d(idx_tar, idx_order)

    dataset = [psth_cue1, psth_cue2, psth_sac]

    xmin = [-1, -0.1, -0.7]
    titles = ['Cue1', 'Cue2', 'Saccade']
    cmap = matplotlib.cm.get_cmap("Set1")
    idx_modulated_units_bi = np.zeros(n_units)
    for n in range(n_units):
        # trials wanted for each target

        list_idx_high_good_trials = np.ndarray(n_targets, dtype=object)
        idx_high_trials_n = idx_high_trials[n]

        # for tar in range(n_targets):
        #     # first 3 trained targets only correct trials
        #     if tar < 3:
        #         list_idx_high_good_trials[tar] = \
        #             np.intersect1d(np.intersect1d(idx_high_trials_n, idx_cor_trials), list_idx_tar[tar])
        #     # testing targets look at completed trials
        #     else:
        #         list_idx_high_good_trials[tar] = \
        #             np.intersect1d(np.intersect1d(idx_high_trials_n, idx_completed_trials), list_idx_tar[tar])
        #
        #     if len(list_idx_high_good_trials[tar]) <= 1:
        #         continue

        for tar in range(n_targets):
            # first 3 trained targets only correct trials
            if tar < 3:
                list_idx_high_good_trials[tar] = \
                    np.intersect1d(np.intersect1d(idx_high_trials_n, idx_cor_trials), list_idx_tar[tar])
            # testing targets look at completed trials
            else:
                list_idx_high_good_trials[tar] = \
                    np.intersect1d(np.intersect1d(idx_high_trials_n, idx_completed_trials), list_idx_tar[tar])

            for f in range(3):
                x = np.arange(xmin[f], (list_xmax[tar, f] + 0.01) / speed, overlap)
                try:
                    y_all_trials = dataset[f][n, list_idx_high_good_trials[tar], :len(x)].reshape(
                        len(list_idx_high_good_trials[tar]), len(x)
                    )
                except:
                    y_all_trials = np.zeros((1, len(x)))

                y_se = np.std(y_all_trials, 0) / np.sqrt(y_all_trials.shape[0])
                if (y_se == 0).all():
                    continue
                y_mean = np.mean(y_all_trials, 0)
                if ((y_mean - y_se) > 3).any():
                    idx_modulated_units_bi[n] = 1
                    continue

        print(n)

    idx_modulated_units = np.where(idx_modulated_units_bi == 1)[0]
    psths = os.listdir(NP_path + '/psth')
    psth_good_path = NP_path + '/psth_good'
    psth_bad_path = NP_path + '/psth_bad'

    if not os.path.exists(psth_good_path):
        os.mkdir(psth_good_path)
    if not os.path.exists(psth_bad_path):
        os.mkdir(psth_bad_path)
    for i in range(len(psths)):
        num = int(psths[i].split('_')[0])
        if num in idx_modulated_units:
            shutil.copy(NP_path + '/psth/' + psths[i], NP_path + '/psth_good/' + psths[i])
        else:
            shutil.copy(NP_path + '/psth/' + psths[i], NP_path + '/psth_bad/' + psths[i])

    fr_bin_filtered = copy.deepcopy(fr_bin)
    fr_bin_filtered['idx_modulated_units'] = idx_modulated_units

    file_to_save = open(NP_path + '/fr_bin_filtered.obj', 'wb+')
    pickle.dump(fr_bin_filtered, file_to_save)
    file_to_save.close()
    return


if __name__ == '__main__':
    # NP_dates = ['Jan_16_F_g0','Jan_17_F_g0','Jan_18_F_g0','Jan_20_F_g0','Jan_22_F_g0']
    # session_dates = ['20230116', '20230117', '20230118', '20230120', '20230122']
    NP_dates = ['20230213_F_g0']
    session_dates = ['20230213']
    n_sessions = len(NP_dates)
    for s in range(n_sessions):
        path_internal = '/Users/laptopd/Documents/Compositionality/Analysis/spikedata/F/NP/' + NP_dates[s]
        path_external = '/Volumes/Expansion/Cheng/processed data/' + NP_dates[s]
        if not os.path.exists(path_external):
            os.mkdir(path_external)

        date_str = session_dates[s]
        load_raster_to_trials(path_internal, date_str, rewrite=0)
        # #
        #
        gen_rasters(path_internal, path_external, date_str, rewrite=0)
        #
        # # # filter out the low firing trials
        filter_trials(path_internal, date_str, rewrite=0)
        #
        # # load raster into bins
        raster_to_bins(path_internal, path_external, date_str, rewrite=0)

        target_params = {
            'target_label': ['1, 0', '3, 0', '0, 2', '0, 4', '2, 0', '4, 0', '0, 1', '0, 3'],
            'target_set': [[1, 0], [3, 0], [0, 2], [0, 4], [2, 0], [4, 0], [0, 1], [0, 3]],
            'target_set_order': [1, 1, 1, 1, 1, 1, 1, 1],
            'n_trained': 4
        }

        # # plot and save the psth for all units
        gen_psth(path_internal, path_external, date_str, target_params, rewrite=1)
        #
        # # filter out the non-modulated units
        # filter_units(NP_path, date_str, rewrite=1)
