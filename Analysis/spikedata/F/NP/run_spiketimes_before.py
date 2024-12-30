import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import scipy.io as io
import os

date_str = '2022-09-09_12-47-12'
path = './spikedata/F/' + date_str + '/ks_3_output_pre_v6/'

raster_path = './spikedata/F/' + date_str + '/raster'
if not os.path.exists(raster_path):
    os.mkdir(raster_path)

session_file = open('./sessions.obj', 'rb')
sessions = pickle.load(session_file)
session_file.close()

# find session
date_num = date_str.split('_')[0].split('-')
holder = ''
for i in date_num:
    holder += i

session_id = 0
for idx in range(len(sessions)):
    if sessions[idx].date == holder:
        session_id = idx
        break

session = sessions[session_id]
session.trialsync_times = session.trialsync_times / 1e6

file_spiketimes = path + 'spike_times.npy'
spiketimes = np.load(file_spiketimes)
sample_rate = 3e4
spiketimes = spiketimes / sample_rate

file_cluster_label = path + 'cluster_KSLabel.tsv'
df = pd.read_csv(file_cluster_label, sep='\t')
# idx_good = np.array(df['cluster_id'][df['KSLabel'] == 'good'].array)
label = np.array(df['KSLabel'].array)
cluster_id = np.array(df['cluster_id'].array)
file_clusters = path + 'spike_templates.npy'
spike_clusters = np.load(file_clusters)
n_units = len(label)
spiketimes_all_units = np.ndarray(n_units, dtype=object)

for i in range(n_units):
    spiketimes_all_units[i] = spiketimes[spike_clusters == cluster_id[i]]

sync_time_start = np.zeros(session.n_total)
sync_time_start_idx = []
for i in range(session.n_total):
    time_i = session.cue1_start_time[i]
    idx = np.where(session.trialsync_times > time_i)[0][0]
    sync_time_start[i] = session.trialsync_times[idx]
    sync_time_start_idx.append(idx)

sync_time_start_idx = np.array(sync_time_start_idx)
sync_time_vprobe = io.loadmat(path + 'sync_times.mat')['sync_times']
sync_time_vprobe = sync_time_vprobe[sync_time_start_idx]

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
        cue1_t = session.photodiode_times[t]
        end_t = cue1_t + after  # session.trial_len[t]
        start_t = cue1_t + before
        spiketimes_it = spiketimes_i + sync_time_start[t] - sync_time_vprobe[t]
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



for i in range(len(label)):
    raster_i = raster[i, :]
    fig, ax = plt.subplots()
    for t in range(session.n_total):
        x = raster_i[t]
        y = np.ones(len(x)) * t
        ax.scatter(x, y, s=0.5, c='k')
    ax.plot([0, 0], [0, session.n_total], linestyle='--', color=[1, 0, 0], linewidth=1)
    raster_name_i = raster_path + '/' + str(cluster_id[i]) + '_' + label[i] + '_raster' + '.pdf'
    fig.savefig(raster_name_i)
    plt.close()
    # print(i)
    # plt.show()

    ind_cor = np.where(session.has_reward == 1)[0]
    ind_11 = np.intersect1d(np.where(np.sum(session.target_xy == [1, 1], axis=1) == 2)[0], ind_cor)
    ind_13 = np.intersect1d(np.where(np.sum(session.target_xy == [1, 3], axis=1) == 2)[0], ind_cor)
    ind_31 = np.intersect1d(np.where(np.sum(session.target_xy == [3, 1], axis=1) == 2)[0], ind_cor)
    # #
    # ind_cat = np.concatenate((np.concatenate((ind_11, ind_13)), ind_31))
    # fig, ax = plt.subplots()
    # for t in range(len(ind_cat)):
    #     x = raster_i[ind_cat[t]]
    #     y = np.ones(len(x)) * t
    #     ax.scatter(x, y, s=0.5, c='k')
    # plt.show()
    bin_step = 0.05
    bin_size = 0.1
    bins = np.arange(before, after, bin_step)
    bins = np.stack((bins, bins + bin_size), axis=0)
    psth = np.zeros((session.n_total, bins.shape[1]))
    for t in range(session.n_total):
        raster_t = raster_i[t]
        for b in range(bins.shape[1]):
            for j in range(len(raster_t)):
                if bins[0, b] < raster_t[j] < bins[1, b]:
                    psth[t, b] += 1
                if bins[1, b] < raster_t[j]:
                    break
    psth = psth/bin_size
    fig, ax = plt.subplots()
    x = bins[0, :-1]
    ind_list = [ind_11, ind_13, ind_31]
    color_list = ['r', 'g', 'b']
    lgd = []
    ymin = 0
    ymax = -1
    for k in range(len(ind_list)):
        y = np.mean(psth[ind_list[k], :-1], 0)
        ste = np.std(psth[ind_list[k], :-1], 0) / np.sqrt(len(ind_list[k]))
        h, = ax.plot(x, y, color=color_list[k], linewidth=2)
        ax.plot(x, y + ste, color=color_list[k], linewidth=0.5)
        ax.plot(x, y - ste, color=color_list[k], linewidth=0.5)
        ymax = max(ymax, max(y + ste))
        lgd.append(h)

    ax.legend(lgd, ['1,1', '1,3', '3,1'])
    ymin = 0
    ax.set_ylim([ymin, ymax * 1.2])
    ax.plot([0, 0], [ymin, ymax], linestyle='--', color='k', linewidth=1)
    psth_name_i = raster_path + '/' + str(cluster_id[i]) + '_' + label[i] + '_psth' + '.pdf'

    fig.savefig(psth_name_i)
    plt.close()
    print(i)
    # plt.show()

a = 1
