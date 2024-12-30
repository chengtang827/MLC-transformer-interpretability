import numpy as np
import pandas as pd
import pickle
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import scipy.io as io
import os
from ibllib.io import spikeglx


date_str = 'Oct_12_g0'
session_date = '20221012'
path = date_str + '/ks_3_output_pre_v6/'

raster_path = date_str + '/raster'
if not os.path.exists(raster_path):
    os.mkdir(raster_path)

session_file = open('/Users/laptopd/Documents/Compositionality/Analysis/sessions.obj', 'rb')
sessions = pickle.load(session_file)
session_file.close()


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
start_nd = float(nidaq_meta['firstSample'])/sample_rate_nd
len_nd = float(nidaq_meta['fileTimeSecs'])

imec_meta = _get_imec_meta(path)
sample_rate_im = float(imec_meta['imSampRate'])
start_im = float(imec_meta['firstSample'])/sample_rate_im
len_im = float(imec_meta['fileTimeSecs'])

reader = _get_nidaq_reader(path)
stride = int(np.round(sample_rate_nd / 1000))
analog_raw, digital_raw = reader.read(nsel=slice(0, reader.ns, stride))
read_sample_rate = sample_rate_nd / float(stride)

sync_raw = digital_raw[:, 7]
sync = sync_raw == 1
photodiode_raw = analog_raw[:, 1]
photodiode = photodiode_raw > 3
square_nidq = analog_raw[:, 0]<3
square_nidq_ = np.roll(square_nidq, -1)
idx_flip_square = np.where(square_nidq != square_nidq_)[0]
idx_flip_square = idx_flip_square[np.arange(1, len(idx_flip_square) + 1, 2)]
square_nidq_times = idx_flip_square/read_sample_rate
square_imec_times = io.loadmat(path + 'sync_times.mat')['sync_times']

# checking the square wave, there is a linear scaling drift between im and nd
# nd timestamps should first multiply the ratio, then subtract the start offset
scaling_ratio = len_im/len_nd
start_offset = start_im - start_nd

sync_ = np.roll(sync, -1)
idx_flip_sync = np.where(sync != sync_)[0]
idx_flip_sync = idx_flip_sync[np.arange(1, len(idx_flip_sync) + 1, 2)]
sync_times_probe = idx_flip_sync/read_sample_rate

photodiode_ = np.roll(photodiode, -1)
idx_flip_pho = np.where(photodiode != photodiode_)[0]
idx_flip_pho = idx_flip_pho[np.arange(1, len(idx_flip_pho) + 1, 2)]
photo_times_probe = idx_flip_pho/read_sample_rate
# photodiode_times = np.array([photodiode[idx].time for idx in idx_flip])

# find session
# date_num = date_str.split('_')[0].split('-')
# session_date = ''
# for i in date_num:
#     session_date += i

session_id = 0
for idx in range(len(sessions)):
    if sessions[idx].date == session_date:
        session_id = idx
        break

session = sessions[session_id]
# session.trialsync_times = session.trialsync_times / 1e6

file_spiketimes = path + 'spike_times.npy'
spiketimes = np.load(file_spiketimes)
sample_rate_probe = 3e4
spiketimes = spiketimes / sample_rate_probe



file_cluster_label = path + 'cluster_KSLabel.tsv'
df = pd.read_csv(file_cluster_label, sep='\t')
# idx_good = np.array(df['cluster_id'][df['KSLabel'] == 'good'].array)
group = np.array(df['KSLabel'].array)
cluster_id = np.array(df['cluster_id'].array)
print('good: '+str(sum(group=='good'))+' mua: '+str(sum(group=='mua')))
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
# sync_times_probe = sync_times_probe[sync_time_start_idx]
sync_times_probe = sync_times_probe*scaling_ratio-start_offset


photo_times_probe = photo_times_probe[:len(sync_time_start_idx)]
photo_times_probe = photo_times_probe*scaling_ratio-start_offset

# a = photo_times_probe - sync_times_probe
# print('Sync delay max: ' + str(max(abs(a))))


raster = np.ndarray((n_units, session.n_total), dtype=object)
before = -2
after = 6
for i in range(n_units):
    spiketimes_i = spiketimes_all_units[i]
    idx_s = 0
    idx_start = 0
    # idx_start_pre = 0
    for t in range(session.n_total):

        raster_it = []
        cue1_t = session.cue1_start_time[t]
        end_t = cue1_t + after  # session.trial_len[t]
        start_t = cue1_t + before
        spiketimes_it = spiketimes_i + cue1_t - photo_times_probe[t]
        idx_s = idx_start
        while True:
            if idx_s == len(spiketimes_i):
                break

            if spiketimes_it[idx_s] < start_t:
                idx_s += 1
            if idx_s == len(spiketimes_i):
                break
            if start_t <= spiketimes_it[idx_s] <= end_t:
                if spiketimes_it[idx_s-1] < start_t:
                    idx_start = idx_s
                raster_it.append(spiketimes_it[idx_s] - cue1_t)
                idx_s += 1
            if idx_s == len(spiketimes_i):
                break
            if end_t < spiketimes_it[idx_s]:
                break

        raster[i][t] = np.array(raster_it)



bin_step = 0.05
bin_size = 0.1
bins = np.arange(before, after, bin_step)
bins = np.stack((bins, bins + bin_size), axis=0)
psth_cue = np.zeros((n_units, session.n_total, bins.shape[1]))
psth_sac = np.zeros((n_units, session.n_total, bins.shape[1]))
for i in range(len(group)):
    psth_cue_i = np.zeros((session.n_total, bins.shape[1]))
    psth_sac_i = np.zeros((session.n_total, bins.shape[1]))

    raster_i = raster[i, :]
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    for t in range(session.n_total):
        x = raster_i[t]
        x_saccade = raster_i[t] + session.cue1_start_time[t] - session.saccade_onset[t]
        y = np.ones(len(x)) * t
        ax[0].scatter(x, y, s=0.3, c='k')
        ax[1].scatter(x_saccade, y, s=0.3, c='k')

    ax[0].plot([0, 0], [0, session.n_total], linestyle='--', color=[1, 0, 0], linewidth=1)
    ax[0].set_title('Cue onset')
    ax[1].plot([0, 0], [0, session.n_total], linestyle='--', color=[1, 0, 0], linewidth=1)
    ax[1].set_title('Saccade onset')

    raster_name_i = raster_path + '/' + str(cluster_id[i]) + '_' + group[i] + '_raster' + '.pdf'
    fig.savefig(raster_name_i)
    plt.close()
    # print(i)
    # plt.show()

    ind_cor = np.where(session.has_reward == 1)[0]
    ind_order = np.where(session.order_xy == 1)[0]
    ind_11 = np.intersect1d(np.where(np.sum(session.target_xy == [1, 1], axis=1) == 2)[0], ind_cor)
    ind_11 = np.intersect1d(ind_11, ind_order)
    ind_13 = np.intersect1d(np.where(np.sum(session.target_xy == [1, 3], axis=1) == 2)[0], ind_cor)
    ind_13 = np.intersect1d(ind_13, ind_order)

    ind_31 = np.intersect1d(np.where(np.sum(session.target_xy == [3, 1], axis=1) == 2)[0], ind_cor)
    ind_31 = np.intersect1d(ind_31, ind_order)


    for t in range(session.n_total):
        raster_cue_t = raster_i[t]
        raster_sac_t = raster_i[t] + session.cue1_start_time[t] - session.saccade_onset[t]
        idx_cue = 0
        idx_cue_start = 0
        idx_sac = 0
        idx_sac_start = 0
        for b in range(bins.shape[1]):
            idx_cue = idx_cue_start
            while True:
                if idx_cue == len(raster_cue_t):
                    break
                if raster_cue_t[idx_cue] < bins[0, b]:
                    idx_cue += 1
                if idx_cue == len(raster_cue_t):
                    break
                if bins[0, b] < raster_cue_t[idx_cue] < bins[1, b]:
                    try:
                        if raster_cue_t[idx_cue - 1] < bins[0, b]:
                            idx_cue_start = idx_cue
                    except:
                        pass
                    psth_cue_i[t, b] += 1
                    idx_cue += 1
                if idx_cue == len(raster_cue_t):
                    break
                if bins[1, b] < raster_cue_t[idx_cue]:
                    break

            idx_sac = idx_sac_start
            while True:
                if idx_sac == len(raster_sac_t):
                    break
                if raster_sac_t[idx_sac] < bins[0, b]:
                    idx_sac += 1
                if idx_sac == len(raster_sac_t):
                    break
                if bins[0, b] < raster_sac_t[idx_sac] < bins[1, b]:
                    try:
                        if raster_sac_t[idx_sac - 1] < bins[0, b]:
                            idx_sac_start = idx_sac
                    except:
                        pass
                    psth_sac_i[t, b] += 1
                    idx_sac += 1
                if idx_sac == len(raster_sac_t):
                    break
                if bins[1, b] < raster_sac_t[idx_sac]:
                    break


    psth_cue_i = psth_cue_i / bin_size
    psth_sac_i = psth_sac_i / bin_size
    psth_cue[i, :, :] = psth_cue_i
    psth_sac[i, :, :] = psth_sac_i


    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    x = bins[0, :-1]
    ind_list = [ind_11, ind_13, ind_31]
    color_list = ['r', 'g', 'b']
    lgd = []
    ymin = 0
    ymax = -1
    for k in range(len(ind_list)):
        y = np.mean(psth_cue_i[ind_list[k], :-1], 0)
        ste = np.std(psth_cue_i[ind_list[k], :-1], 0) / np.sqrt(len(ind_list[k]))
        h, = ax[0].plot(x, y, color=color_list[k], linewidth=2)
        ax[0].plot(x, y + ste, color=color_list[k], linewidth=0.5)
        ax[0].plot(x, y - ste, color=color_list[k], linewidth=0.5)
        ymax = max(ymax, max(y + ste))
        lgd.append(h)

    ax[0].legend(lgd, ['1,1', '1,3', '3,1'])
    ymin = 0
    ax[0].set_ylim([ymin, ymax * 1.2])
    ax[0].plot([0, 0], [ymin, ymax], linestyle='--', color='k', linewidth=1)

    lgd = []
    for k in range(len(ind_list)):
        y = np.mean(psth_sac_i[ind_list[k], :-1], 0)
        ste = np.std(psth_sac_i[ind_list[k], :-1], 0) / np.sqrt(len(ind_list[k]))
        h, = ax[1].plot(x, y, color=color_list[k], linewidth=2)
        ax[1].plot(x, y + ste, color=color_list[k], linewidth=0.5)
        ax[1].plot(x, y - ste, color=color_list[k], linewidth=0.5)
        ymax = max(ymax, max(y + ste))
        lgd.append(h)

    ax[1].legend(lgd, ['1,1', '1,3', '3,1'])
    ymin = 0
    ax[1].set_ylim([ymin, ymax * 1.2])
    ax[1].plot([0, 0], [ymin, ymax], linestyle='--', color='k', linewidth=1)

    ax[0].set_title('Cue onset')
    ax[1].set_title('Saccade onset')
    psth_name_i = raster_path + '/' + str(cluster_id[i]) + '_' + group[i] + '_psth' + '.pdf'

    fig.savefig(psth_name_i)
    plt.close()
    print(i)
    # plt.show()

psth = {
    'psth_cue_on': psth_cue,
    'psth_sac_on': psth_sac,
    'session': session,
    'bins': bins,
    'group': group
}

file_to_save = open(date_str + '/psth.obj', 'wb+')
pickle.dump(psth, file_to_save)
file_to_save.close()
a = 1
