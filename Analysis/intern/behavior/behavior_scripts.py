import sys
import matplotlib
import numpy as np

matplotlib.use('TkAgg')
from matplotlib.patches import *

sys.path.insert(1, '//')

from run_parse_trials import *
from scipy.ndimage import gaussian_filter
from scipy.optimize import fmin

import analysis_scripts.behavior.dates_FN as dates


def _get_session(sessions, date):
    """
    :param sessions:
    :param date:
    :return:
    """
    for idx in range(len(sessions)):
        if sessions[idx].date == date:
            return sessions[idx]
    return


def _get_invis_trials(session):
    probe_trials_by_toss = (session.toss_probe <= session.ratio_probe).astype(int)
    probe_trials_by_alpha = (session.ball_alpha <= 0).astype(int)
    probe_trials = np.logical_or(probe_trials_by_toss, probe_trials_by_alpha).astype(int)
    return probe_trials



def _prior_distribution(session_dates):
    n_session = len(session_dates)
    x_range = 13
    y_range = 13  # 8+1+8
    speed = 2
    confusion = np.zeros((n_session, 2, x_range, y_range))  # 0 visible 1 probe

    target_dist = np.zeros((n_session, 2, x_range, y_range))  # 0 visible 1 probe
    saccade_dist = np.zeros((n_session, 2, x_range, y_range))  # 0 visible 1 probe
    reward_dist = np.zeros((n_session, 2, x_range, y_range))  # 0 visible 1 probe
    bias_dist = np.zeros((n_session, 2, x_range, y_range))  # 0 visible 1 probe

    for s in range(n_session):
        session = _get_session(sessions, session_dates[s])
        flag_probe = _get_invis_trials(session)
        # find out trials not starting from origin
        target_set = np.array(session.target_set)
        # trials where there is response
        idx_val_trials = np.where(~np.isnan(session.reported_xy[:, 0].astype(float)))[0].astype(int)
        target_ids_val = session.target_id[idx_val_trials]
        saccade_xy_val = session.reported_xy[idx_val_trials]
        # target_xy_val = session.target_xy[idx_val_trials]
        reward_dur_val = session.reward_dur[idx_val_trials]

        flag_probe = flag_probe[idx_val_trials]

        start_xy = target_set[target_ids_val, 0, :]
        vector_xy = target_set[target_ids_val, 1, :]
        target_xy_val = start_xy + vector_xy

        # for reaction time
        sac_onset = session.saccade_onset[idx_val_trials]
        cue1_start = session.cue1_start_time[idx_val_trials]
        pre_target_delay = session.pre_target_delay[idx_val_trials]

        for i0 in range(len(idx_val_trials)):  # visible/probe
            x = target_xy_val[i0, 0]
            y = target_xy_val[i0, 1]
            x_ = saccade_xy_val[i0, 0]
            y_ = saccade_xy_val[i0, 1]
            x0 = start_xy[i0, 0]
            y0 = start_xy[i0, 1]

            # if x==1 and y==1:
            #     continue
            #
            # if x == 1 and y == -1:
            #     continue

            if flag_probe[i0] == 0:  # visible
                target_dist[s, 0, int(x + x_range / 2), int(y + y_range / 2)] += 1
                saccade_dist[s, 0, int(x_ + x_range / 2), int(y_ + y_range / 2)] += 1
                reward_dist[s, 0, int(x_ + x_range / 2), int(y_ + y_range / 2)] += reward_dur_val[i0]
            else:  # probe
                target_dist[s, 1, int(x + x_range / 2), int(y + y_range / 2)] += 1
                saccade_dist[s, 1, int(x_ + x_range / 2), int(y_ + y_range / 2)] += 1
                reward_dist[s, 1, int(x_ + x_range / 2), int(y_ + y_range / 2)] += reward_dur_val[i0]

    confusion = np.flip(np.swapaxes(confusion, 2, 3), 2)
    target_dist = np.flip(np.swapaxes(target_dist, 2, 3), 2)
    saccade_dist = np.flip(np.swapaxes(saccade_dist, 2, 3), 2)
    reward_dist = np.flip(np.swapaxes(reward_dist, 2, 3), 2)

    # target_dist_filter = gaussian_filter(target_dist, sigma=[0, 0, 0.8, 0.8], mode='mirror')
    # saccade_dist_filter = gaussian_filter(saccade_dist, sigma=[0, 0, 0.8, 0.8], mode='mirror')
    # reward_dist_filter = gaussian_filter(reward_dist, sigma=[0, 0, 0.8, 0.8], mode='mirror')
    #
    target_dist_filter = gaussian_filter(target_dist, sigma=[0, 0, 0, 0], mode='mirror')
    saccade_dist_filter = gaussian_filter(saccade_dist, sigma=[0, 0, 0, 0], mode='mirror')
    reward_dist_filter = gaussian_filter(reward_dist, sigma=[0, 0, 0, 0], mode='mirror')

    # heatmap of target distribution
    n_col = 6
    n_row = int(np.ceil(n_session / n_col))
    fig, ax = plt.subplots(n_row, n_col)  # , figsize=(20, n_row * 2))
    if len(ax.shape) == 1:
        ax = ax[np.newaxis, :]
    cnt = 0
    for s in range(n_col * n_row):

        ax_s = ax[cnt // n_col, cnt % n_col]
        if s >= n_session:
            ax_s.set_visible(False)
            cnt += 1
            continue

        ax_s.imshow(target_dist_filter[s, 1], cmap='jet')
        ax_s.set_xlim((x_range - 1) / 4 - 0.5, x_range - (x_range - 1) / 4 - 0.5)
        ax_s.set_ylim(y_range - (y_range - 1) / 4 - 0.5, (y_range - 1) / 4 - 0.5)
        ax_s.set_title('target')
        cnt += 1

    # heatmap of saccade distribution
    n_col = 6
    n_row = int(np.ceil(n_session / n_col))
    fig, ax = plt.subplots(n_row, n_col)  # , figsize=(20, n_row * 2))
    if len(ax.shape) == 1:
        ax = ax[np.newaxis, :]
    cnt = 0
    for s in range(n_col * n_row):

        ax_s = ax[cnt // n_col, cnt % n_col]
        if s >= n_session:
            ax_s.set_visible(False)
            cnt += 1
            continue

        ax_s.imshow(saccade_dist_filter[s, 1], cmap='jet')
        ax_s.set_xlim((x_range - 1) / 4 - 0.5, x_range - (x_range - 1) / 4 - 0.5)
        ax_s.set_ylim(y_range - (y_range - 1) / 4 - 0.5, (y_range - 1) / 4 - 0.5)
        ax_s.set_title('saccade')
        cnt += 1

    # heatmap of reward distribution
    n_col = 6
    n_row = int(np.ceil(n_session / n_col))
    fig, ax = plt.subplots(n_row, n_col)  # , figsize=(20, n_row * 2))
    if len(ax.shape) == 1:
        ax = ax[np.newaxis, :]
    cnt = 0
    for s in range(n_col * n_row):

        ax_s = ax[cnt // n_col, cnt % n_col]
        if s >= n_session:
            ax_s.set_visible(False)
            cnt += 1
            continue

        ax_s.imshow(reward_dist_filter[s, 1], cmap='jet')
        ax_s.set_xlim((x_range - 1) / 4 - 0.5, x_range - (x_range - 1) / 4 - 0.5)
        ax_s.set_ylim(y_range - (y_range - 1) / 4 - 0.5, (y_range - 1) / 4 - 0.5)
        ax_s.set_title('reward')
        cnt += 1
    plt.show()
    pass



def confusion_each(session_dates, sessions, params):
    is_trained = params['is_trained']
    is_invisible = params['is_invisible']
    n_session = len(session_dates)
    start_all = np.zeros((0, 2))
    target_all = np.zeros((0, 2))
    saccade_all = np.zeros((0, 2))
    coef_sep_all = np.zeros((0, 2))
    n_lines = 9
    for s in range(n_session):
        session = _get_session(sessions, session_dates[s])
        flag_probe = _get_invis_trials(session)
        # find out trials not starting from origin
        target_set = np.array(session.target_set)

        alpha_list = np.array(session.alpha_list)
        alpha_group_id_list = np.array(session.alpha_group_id_list)
        test_targets_id = []
        # if (alpha_list <= 0).any():
        test_group_idx = np.where(alpha_list == 0)[0]
        # locate in alpha_group_id_list

        for i0, j0 in enumerate(alpha_group_id_list):
            if j0 in test_group_idx:
                test_targets_id.append(i0)

        # flag_trained = np.zeros_like(session.target_id)
        # for i0, j0 in enumerate(session.target_id):
        #     if j0 not in test_targets_id:
        #         flag_trained[i0] = 1

        # test_targets_id = list(range(47, 68))

        # trials where there is response
        idx_val_trials = np.where(~np.isnan(session.reported_xy[:, 0].astype(float)))[0].astype(int)
        target_ids_val = session.target_id[idx_val_trials]
        saccade_xy_val = session.reported_xy[idx_val_trials]

        flag_probe = flag_probe[idx_val_trials]

        start_xy = target_set[target_ids_val, 0, :]
        vector_xy = target_set[target_ids_val, 1, :]
        target_xy_val = start_xy + vector_xy
        # target_xy_val = session.target_xy[idx_val_trials]

        start_probe = start_xy[flag_probe == 1, :]
        target_probe = target_xy_val[flag_probe == 1, :]
        saccade_probe = saccade_xy_val[flag_probe == 1, :]
        # start_probe = start_xy[flag_probe == 0, :]
        # target_probe = target_xy_val[flag_probe == 0, :]
        # saccade_probe = saccade_xy_val[flag_probe == 0, :]
        start_all = np.concatenate((start_all, start_probe), axis=0)
        target_all = np.concatenate((target_all, target_probe), axis=0)
        saccade_all = np.concatenate((saccade_all, saccade_probe), axis=0)

    pairs = np.concatenate((start_all, target_all), axis=1).astype(float)
    pairs_unique = np.unique(pairs, axis=0)
    pairs_repeated = []
    saccades = []
    saccades_mean = np.zeros((pairs_unique.shape[0], 2))
    for i, p in enumerate(pairs_unique):
        idx_tmp = np.where((pairs == p).sum(axis=1) == 4)[0]
        if len(idx_tmp) >= 0:
            pairs_repeated.append(p)
            saccades.append(saccade_all[idx_tmp, :])
            saccades_mean[i, :] = np.mean(saccade_all[idx_tmp, :], axis=0)

    n_targets = len(pairs_repeated)

    reported_each = np.array(saccades, dtype=object)

    confusion_each = np.zeros((n_targets, n_lines, n_lines))
    for t in range(n_targets):
        reported_t = reported_each[t]
        for i in range(len(reported_t)):
            if np.isnan(reported_t[i, 0]):
                confusion_each[t, int((n_lines - 1) / 2), int((n_lines - 1) / 2)] += 1
            else:
                tmp = reported_t[i, :]
                confusion_each[t, int(tmp[0] + (n_lines - 1) / 2), int(tmp[1] + (n_lines - 1) / 2)] += 1
        confusion_each[t, :, :] = np.flip(confusion_each[t, :, :].T, 0)

    n_col = 20
    n_row = int(np.ceil(len(test_targets_id) / n_col))
    fig, ax = plt.subplots(n_row, n_col, figsize=(20, n_row * 2))
    cnt = 0
    # n_targets = 21
    for t in range(n_targets):
        x0 = pairs_repeated[t][0]
        y0 = pairs_repeated[t][1]
        x = pairs_repeated[t][2]
        y = pairs_repeated[t][3]

        target_id = np.where(np.sum(
            target_set.reshape(target_set.shape[0], target_set.shape[1] * target_set.shape[2]) == np.array(
                [x0, y0, x - x0, y - y0]), 1) == 4)[0]

        if target_id not in test_targets_id:
            continue
        else:
            ax_t = ax[cnt // n_col, cnt % n_col]
            ax_t.set_xticks(np.arange(0, n_lines), labels=[])
            ax_t.set_yticks(np.arange(0, n_lines), labels=[])
            ax_t.grid()

            confusion_t = np.sqrt(confusion_each[t] / np.sum(confusion_each[t].flatten()))
            for row in range(n_lines):
                for col in range(n_lines):
                    radius = confusion_t[row, col]
                    ax_t.add_patch(
                        Circle((col, n_lines - 1 - row), radius / 2, edgecolor=(0, 0, 0, 0), facecolor=(0, 0, 0, 1),
                               linewidth=1))
            # target
            ax_t.add_patch(
                Circle((x + (n_lines - 1) / 2, y + (n_lines - 1) / 2), 0.5, edgecolor=(0, 1, 0), facecolor=(0, 0, 0, 0),
                       linewidth=2))
            # start
            ax_t.add_patch(
                Circle((x0 + (n_lines - 1) / 2, y0 + (n_lines - 1) / 2), 0.5, edgecolor=(.9, .9, 0),
                       facecolor=(0, 0, 0, 0), linewidth=2))
            # fixation
            ax_t.add_patch(
                Circle(((n_lines - 1) / 2, (n_lines - 1) / 2), 0.5, edgecolor=(1, 0, 0), facecolor=(0, 0, 0, 0),
                       linewidth=2))

            ax_t.plot(np.array([0, saccades_mean[t, 0]])+(n_lines - 1) / 2, np.array([0, saccades_mean[t, 1]])+(n_lines - 1) / 2)

            ax_t.set_title(len(saccades[t]))

            cnt += 1

    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)

    plt.show()


def _choose_focus_target(session, subject):
    '''
    select focused targets for different kinds of sessions
    :param session:
    :return:
    '''

    if subject=='F':
        if session.date in dates.train_dates_F:
            target_ids = list(range(len(session.target_set)))
        elif (session.date in dates.set1_dates_F) or (session.date in dates.set2_dates_F):
            target_ids = list(np.where(np.array(session.alpha_group_id_list)==1)[0])
        elif session.date in dates.generalize_dates_F:
            target_ids = list(np.where(np.array(session.alpha_group_id_list)==0)[0])
    elif subject=='N':
        if session.date in dates.train_on_axis_dates_N:
            target_ids = list(range(len(session.target_set)))

    return target_ids


def mse_n_confusion(session_dates, sessions, params):
    subject = params['subject']
    do_plot_mse_confusion = params['do_plot_mse_n_confusion']
    only_last_session = params['only_last_session']
    do_plot_each_confusion = params['do_plot_each_confusion']
    is_invisible = params['is_invisible']

    n_session = len(session_dates)
    x_range = 13
    y_range = 13  # 8+1+8
    error = np.zeros((n_session, 3))  # vis/invis/invis_train

    target_dist = np.zeros((n_session, 2, x_range, y_range))  # 0 visible 1 probe
    saccade_dist = np.zeros((n_session, 2, x_range, y_range))  # 0 visible 1 probe

    start_all = np.zeros((0, 2))
    target_all = np.zeros((0, 2))
    saccade_all = np.zeros((0, 2))

    n_lines = 13
    for s in range(n_session):
        session = _get_session(sessions, session_dates[s])
        flag_invis = _get_invis_trials(session)
        #
        target_set = np.array(session.target_set)
        focus_target_ids = _choose_focus_target(session, subject)


        # alpha_group_id_list = np.array(session.alpha_group_id_list)
        # test_targets_id = []
        # test_group_idx = [1]
        #
        # for i0, j0 in enumerate(alpha_group_id_list):
        #     if j0 in test_group_idx:
        #         test_targets_id.append(i0)

        # trials where there is response
        idx_val_trials = np.where(~np.isnan(session.reported_xy[:, 0].astype(float)))[0].astype(int)
        target_ids_val = session.target_id[idx_val_trials]
        saccade_xy_val = session.reported_xy[idx_val_trials]

        flag_focus = np.array([target_ids_val[i] in focus_target_ids for i in range(len(target_ids_val))]).astype(int)
        flag_invis = flag_invis[idx_val_trials]
        if is_invisible==1:
            flag_select = np.logical_and(flag_invis, flag_focus)
        else:
            flag_select = np.logical_and(flag_invis==0, flag_focus)

        start_xy = target_set[target_ids_val, 0, :]
        vector_xy = target_set[target_ids_val, 1, :]
        target_xy_val = start_xy + vector_xy

        start_select = start_xy[flag_select == 1, :]
        target_select = target_xy_val[flag_select == 1, :]
        saccade_select = saccade_xy_val[flag_select == 1, :]

        try:
            error[s, 1] = np.sum(np.sum((target_select - saccade_select) ** 2, 1) ** 0.5) / target_select.shape[0]
        except:
            error[s, 1] = np.nan

        start_all = np.concatenate((start_all, start_select), axis=0)
        target_all = np.concatenate((target_all, target_select), axis=0)
        saccade_all = np.concatenate((saccade_all, saccade_select), axis=0)

        if s==len(session_dates)-1:  # the last session
            start_last = start_select
            target_last = target_select
            saccade_last = saccade_select

    if do_plot_mse_confusion:
        confusion_one = np.zeros((n_lines, n_lines))
        if only_last_session==1:
            target_tmp = target_last
            saccade_tmp = saccade_last
        else:
            target_tmp = target_all
            saccade_tmp = saccade_all
        for i in range(target_tmp.shape[0]):
            err_tmp = saccade_tmp[i, :] - target_tmp[i, :]

            confusion_one[int(err_tmp[0] + (n_lines - 1) / 2), int(err_tmp[1] + (n_lines - 1) / 2)] += 1
        confusion_one[:, :] = np.flip(confusion_one[:, :].T, 0)

        confusion_one = np.sqrt(confusion_one / np.sum(confusion_one.flatten()))

        fig, ax = plt.subplots(1, 2)

        ax_t = ax[0]
        x = np.arange(n_session)
        # ax_t.plot(x, error[:, 0], color=(0, 0, 0), linestyle='-', label='vis')

        ax_t.plot(x, error[:, 1], color=(0, 0, 0), linestyle='--')

        ax_t.scatter(x, error[:, 1], color=(0, 0, 0), linestyle='--')

        ax_t.set_xticks(x)
        ax_t.set_xticklabels(session_dates, rotation=90)
        ax_t.set_xlabel('Sessions')
        ax_t.set_title('Mean error distance')
        ax_t.set_ylim([0, max(error[:, 1])*1.1])
        # ax_t.legend()

        ax_t = ax[1]

        ax_t.set_xticks(np.arange(0, n_lines), labels=[])
        ax_t.set_yticks(np.arange(0, n_lines), labels=[])
        ax_t.grid()

        for row in range(n_lines):
            for col in range(n_lines):
                radius = confusion_one[row, col]
                ax_t.add_patch(
                    Circle((col, n_lines - 1 - row), radius / 2, edgecolor=(0, 0, 0, 0), facecolor=(0, 0, 0, 1),
                           linewidth=1))

        ax_t.add_patch(
            Circle(((n_lines - 1) / 2, (n_lines - 1) / 2), 0.5, edgecolor=(1, 0, 0), facecolor=(0, 0, 0, 0),
                   linewidth=1))
        ax_t.set_title('Error pattern')

        plt.show()

    if do_plot_each_confusion:
        n_lines = 9
        if only_last_session==1:
            start_tmp = start_last
            target_tmp = target_last
            saccade_tmp = saccade_last
        else:
            start_tmp = start_all
            target_tmp = target_all
            saccade_tmp = saccade_all
        pairs = np.concatenate((start_tmp, target_tmp), axis=1).astype(float)
        pairs_unique = np.unique(pairs, axis=0)
        pairs_repeated = []
        saccades = []
        saccades_mean = np.zeros((pairs_unique.shape[0], 2))
        for i, p in enumerate(pairs_unique):
            idx_tmp = np.where((pairs == p).sum(axis=1) == 4)[0]
            if len(idx_tmp) >= 0:
                pairs_repeated.append(p)
                saccades.append(saccade_tmp[idx_tmp, :])
                saccades_mean[i, :] = np.mean(saccade_tmp[idx_tmp, :], axis=0)

        n_targets = len(pairs_repeated)

        reported_each = np.array(saccades, dtype=object)

        confusion_each = np.zeros((n_targets, n_lines, n_lines))
        for t in range(n_targets):
            reported_t = reported_each[t]
            for i in range(len(reported_t)):
                if np.isnan(reported_t[i, 0]):
                    confusion_each[t, int((n_lines - 1) / 2), int((n_lines - 1) / 2)] += 1
                else:
                    err_tmp = reported_t[i, :]
                    confusion_each[t, int(err_tmp[0] + (n_lines - 1) / 2), int(err_tmp[1] + (n_lines - 1) / 2)] += 1
            confusion_each[t, :, :] = np.flip(confusion_each[t, :, :].T, 0)

        n_col = 20
        n_row = int(np.ceil(len(focus_target_ids) / n_col))
        fig, ax = plt.subplots(n_row, n_col, figsize=(20, n_row * 2))
        # ax = ax.reshape(-1, 1)
        cnt = 0
        # n_targets = 21
        for t in range(n_targets):
            x0 = pairs_repeated[t][0]
            y0 = pairs_repeated[t][1]
            x = pairs_repeated[t][2]
            y = pairs_repeated[t][3]

            target_id = np.where(np.sum(
                target_set.reshape(target_set.shape[0], target_set.shape[1] * target_set.shape[2]) == np.array(
                    [x0, y0, x - x0, y - y0]), 1) == 4)[0]

            if target_id not in focus_target_ids:
                continue
            else:
                if n_row>1:
                    ax_t = ax[cnt // n_col, cnt % n_col]
                else:
                    ax_t = ax[cnt % n_col]
                ax_t.set_xticks(np.arange(0, n_lines), labels=[])
                ax_t.set_yticks(np.arange(0, n_lines), labels=[])
                ax_t.grid()

                confusion_t = np.sqrt(confusion_each[t] / np.sum(confusion_each[t].flatten()))
                for row in range(n_lines):
                    for col in range(n_lines):
                        radius = confusion_t[row, col]
                        ax_t.add_patch(
                            Circle((col, n_lines - 1 - row), radius / 2, edgecolor=(0, 0, 0, 0), facecolor=(0, 0, 0, 1),
                                   linewidth=1))
                # target
                ax_t.add_patch(
                    Circle((x + (n_lines - 1) / 2, y + (n_lines - 1) / 2), 0.5, edgecolor=(0, 1, 0),
                           facecolor=(0, 0, 0, 0),
                           linewidth=2))
                # start
                ax_t.add_patch(
                    Circle((x0 + (n_lines - 1) / 2, y0 + (n_lines - 1) / 2), 0.5, edgecolor=(.9, .9, 0),
                           facecolor=(0, 0, 0, 0), linewidth=2))
                # fixation
                ax_t.add_patch(
                    Circle(((n_lines - 1) / 2, (n_lines - 1) / 2), 0.5, edgecolor=(1, 0, 0), facecolor=(0, 0, 0, 0),
                           linewidth=2))

                ax_t.plot(np.array([0, saccades_mean[t, 0]]) + (n_lines - 1) / 2,
                          np.array([0, saccades_mean[t, 1]]) + (n_lines - 1) / 2)

                ax_t.set_title(len(saccades[t]))

                cnt += 1

        plt.subplots_adjust(left=0, right=1, bottom=0, top=0.96)

        plt.show()

    return


def _model_fit(session_dates):
    n_session = len(session_dates)
    x_range = 13
    y_range = 13  # 8+1+8
    speed = 2
    confusion = np.zeros((n_session, 2, x_range, y_range))  # 0 visible 1 probe
    error = np.zeros((n_session, 3))  # vis/invis/invis_train

    target_dist = np.zeros((n_session, 2, x_range, y_range))  # 0 visible 1 probe
    saccade_dist = np.zeros((n_session, 2, x_range, y_range))  # 0 visible 1 probe

    start_all = np.zeros((0, 2))
    target_all = np.zeros((0, 2))
    saccade_all = np.zeros((0, 2))

    n_lines = 13
    for s in range(n_session):
        session = _get_session(sessions, session_dates[s])
        alpha_list = np.array(session.alpha_list)

        flag_probe = _get_invis_trials(session)
        # find out trials not starting from origin
        target_set = np.array(session.target_set)

        alpha_group_id_list = np.array(session.alpha_group_id_list)
        test_targets_id = []

        test_group_idx = np.where(alpha_list == 0)[0]
        for i0, j0 in enumerate(alpha_group_id_list):
            if j0 in test_group_idx:
                test_targets_id.append(i0)

        # trials where there is response
        idx_val_trials = np.where(~np.isnan(session.reported_xy[:, 0].astype(float)))[0].astype(int)
        target_ids_val = session.target_id[idx_val_trials]
        saccade_xy_val = session.reported_xy[idx_val_trials]

        flag_probe = flag_probe[idx_val_trials]
        flag_gen = np.array([target_ids_val[i] in test_targets_id for i in range(len(target_ids_val))]).astype(int)
        flag_gen_probe = np.logical_and(flag_gen, flag_probe)

        start_xy = target_set[target_ids_val, 0, :]
        vector_xy = target_set[target_ids_val, 1, :]
        target_xy_val = start_xy + vector_xy

        start_probe = start_xy[flag_gen_probe == 1, :]
        target_probe = target_xy_val[flag_gen_probe == 1, :]
        saccade_probe = saccade_xy_val[flag_gen_probe == 1, :]

        error[s, 1] = np.sum(np.sum((target_probe - saccade_probe) ** 2, 1) ** 0.5) / target_probe.shape[0]

        start_all = np.concatenate((start_all, start_probe), axis=0)
        target_all = np.concatenate((target_all, target_probe), axis=0)
        saccade_all = np.concatenate((saccade_all, saccade_probe), axis=0)

    # model 1, offset + start
    def _objective1(params, start, saccade):
        """
        saccade = [b1, b2]+a0*start
        :param params:
        a0, b1, b2
        """

        y_hat = start_all*params[0]+params[1:]

        return np.mean(np.sum((saccade-y_hat)**2, 1)**0.5)

    # model 2, offset + start + direction
    def _objective2(params, start, direction, saccade):
        """
        saccade = a0*start+a1*direction+[b1, b2]
        :param params:
        a0, a1, b1, b2
        """

        y_hat = start*params[0]+direction*params[1]+params[2:]

        return np.mean(np.sum((saccade-y_hat)**2, 1)**0.5)

    # model 3, offset + start + direction*duration
    def _objective3(params, start, vector, saccade):
        """
        saccade = a0*start+a1*vector+[b1, b2]
        :param params:
        a0, a1, b1, b2
        """

        y_hat = start*params[0]+vector*params[1]+params[2:]

        return np.mean(np.sum((saccade-y_hat)**2, 1)**0.5)

    n_boot = 100
    r1 = np.zeros(n_boot)
    r2 = np.zeros(n_boot)
    r3 = np.zeros(n_boot)
    for n in range(n_boot):
        id_rand = np.random.choice(start_all.shape[0], start_all.shape[0], replace=True)
        start = start_all[id_rand, :]
        target = target_all[id_rand, :]
        saccade = saccade_all[id_rand, :]
        vector = target-start
        direction = vector/abs(vector.sum(axis=1)).reshape(-1, 1)

        params1 = fmin(_objective1, x0=np.ones(3), args=(start, saccade), disp=0)
        r1[n] = _objective1(params1, start, saccade)

        params2 = fmin(_objective2, x0=np.ones(4), args=(start, direction, saccade), disp=0)
        r2[n] = _objective2(params2, start, direction, saccade)

        params3 = fmin(_objective3, x0=np.ones(4), args=(start, vector, saccade), disp=0)
        r3[n] = _objective3(params3, start, vector, saccade)
        print(n)

    residual_target = np.mean(np.sum((saccade_all-target_all)**2, 1)**0.5)

    fig, ax = plt.subplots()
    ax.bar(['Only start', 'start + direction', 'start + vector'], [r1.mean(), r2.mean(), r3.mean()],
           yerr=[r1.std(), r2.std(), r3.std()])
    plt.show()


def _draw_targets_overlay(session_date):
    """

    :param session: session
    :return: None
    """
    session = _get_session(sessions, session_date)
    target_set = session.target_set
    test_id = np.where(np.array(session.alpha_group_id_list) == 1)[0]
    test_set = np.array(target_set)[test_id, :, :]
    n_lines = 9
    fig, ax_t = plt.subplots(figsize=(10, 10))
    cnt = 0

    ax_t.set_xticks(np.arange(0, n_lines), labels=[])
    ax_t.set_yticks(np.arange(0, n_lines), labels=[])
    ax_t.grid()

    ax_t.add_patch(
        Circle(((n_lines - 1) / 2, (n_lines - 1) / 2), 0.5, edgecolor=(1, 0, 0), facecolor=(0, 0, 0, 0),
               linewidth=2))
    for i in range(test_set.shape[0]):
        tar = test_set[i, :, :]
        x0 = (n_lines - 1) / 2
        y0 = (n_lines - 1) / 2
        c = np.random.rand(3)
        c = c.tolist()
        c.append(0.4)
        ax_t.scatter(x0 + tar[0, 0], y0 + tar[0, 1], s=100, color=c)
        ax_t.arrow(x0 + tar[0, 0], y0 + tar[0, 1], tar[1, 0], tar[1, 1], head_width=0.2, head_length=0.2, linewidth=5,
                   color=c)

    ax_t.set_xlim([0, n_lines - 1])
    ax_t.set_ylim([0, n_lines - 1])

    plt.show()
    a = 1


if __name__ == "__main_":
    session_file = open('/Users/laptopd/Documents/Compositionality/Analysis/sessions_F.obj', 'rb')
    sessions = pickle.load(session_file)
    session_file.close()

    session_sample_all_dates = ['20231128', '20231129', '20231130', '20231201', '20231204',  # all samples
                                '20231205', '20231206', '20231207', '20231210', '20231211', '20231213']  # subsamples
    session_dates = session_sample_all_dates

    # _confusion_each_generalization_only(session_dates)
    # _confusion_each_generalization_only([session_dates[-1]])
    # _training_progress_confusion_gen_only(session_dates)
    #
    _model_fit([session_dates[-1]])
    _model_fit(session_dates)


    _prior_distribution(session_dates)

