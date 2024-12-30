import sys
import numpy as np
import os
import pickle
import mwk2reader
from pathlib import Path
import copy

class Session:
    def __init__(self):
        self.cue1_start_time = []
        self.fixation_start_time = []
        self.end_time = []
        self.behavior = []
        self.trial_len = []
        self.has_reward = []
        self.reward_dur = []
        self.target_xy = []
        self.reported_xy = []
        self.eye_xy = []
        self.order_xy = []
        self.training_set = []
        self.training_set_order = []
        self.testing_set = []
        self.testing_set_order = []

        self.target_set = []
        self.target_id = []
        self.alpha_list = []
        self.alpha_group_id_list = []

        self.photodiode_times = []
        self.trialsync_times = []
        self.pre_target_delay = []
        self.saccade_onset = []
        # correct for precise radius
        self.true_correct = []
        self.ball_alpha = []

        self.ratio_probe = []
        self.toss_probe = []

        self.n_no_response = 0
        self.n_aborted = 0
        self.n_completed = 0
        self.n_correct = 0
        self.n_error = 0
        self.checksum = 0
        self.n_total = 0
        self.date = ''

    def merge(self, new_s):
        self.cue1_start_time = np.append(self.cue1_start_time, new_s.cue1_start_time)
        self.fixation_start_time = np.append(self.fixation_start_time, new_s.fixation_start_time)
        self.end_time = np.append(self.end_time, new_s.end_time)
        self.behavior = np.append(self.behavior, new_s.behavior)
        self.trial_len = np.append(self.trial_len, new_s.trial_len)
        self.has_reward = np.append(self.has_reward, new_s.has_reward)
        self.target_xy = np.append(self.target_xy, new_s.target_xy, axis=0)
        self.reported_xy = np.append(self.reported_xy, new_s.reported_xy, axis=0)
        self.eye_xy = np.append(self.eye_xy, new_s.eye_xy)
        self.order_xy = np.append(self.order_xy, new_s.order_xy)
        self.photodiode_times = np.append(self.photodiode_times, new_s.photodiode_times)
        self.trialsync_times = np.append(self.trialsync_times, new_s.trialsync_times)
        self.pre_target_delay = np.append(self.pre_target_delay, new_s.pre_target_delay)
        self.saccade_onset = np.append(self.saccade_onset, new_s.saccade_onset)
        self.true_correct = np.where(np.sum(self.reported_xy == self.target_xy, axis=1) == 2)[0]
        self.ball_alpha = np.append(self.ball_alpha, new_s.ball_alpha)

        try:  # can remove after re-compile all sessions
            self.target_id = np.append(self.target_id, new_s.target_id)
            self.ratio_probe = np.append(self.ratio_probe, new_s.ratio_probe)
            self.toss_probe = np.append(self.toss_probe, new_s.toss_probe)
            self.reward_dur = np.append(self.reward_dur, new_s.reward_dur)
        except:
            pass

        self.n_no_response += new_s.n_no_response
        self.n_aborted += new_s.n_aborted
        self.n_completed += new_s.n_completed
        self.n_correct += new_s.n_correct
        self.n_error += new_s.n_error
        self.checksum = np.logical_and(self.checksum, new_s.checksum)
        self.n_total += new_s.n_total


def _exception_handle(session):
    if session.date == '20231107':
        session.ratio_probe[568] = '0.6'
        session.ratio_probe = (session.ratio_probe).astype(float)
    return session


def _cleanEntry(data, idx_begin):
    data = data[idx_begin:]
    tmp = np.array([i.data for i in data])
    idx_clean = np.where(tmp != np.roll(tmp, 1))[0]  # to remove redundancy
    new_data = []
    [new_data.append(data[i]) for i in idx_clean]
    return new_data


def _getEventInIntv(f, eventNames, start, end, flag='max'):
    events = f.get_events(codes=eventNames, time_range=[start, end])
    timestamps = []
    [timestamps.append(i.time) for i in events]
    if flag == 'min':
        idx = np.argmin(np.array(timestamps))
        return events[idx]
    elif flag == 'max':
        idx = np.argmax(np.array(timestamps))
        return events[idx]

    if flag == 'average':
        data = []
        [data.append(i.data) for i in events]
        return np.mean(np.array(data))

    if flag == 'raw':
        data = []
        [data.append(i.data) for i in events]
        return data


def _convertXY(eye_x, eye_y):
    eye_x /= 4
    eye_y /= 4
    x_range = np.ceil(eye_x + np.arange(-1, 2))
    y_range = np.ceil(eye_y + np.arange(-1, 2))
    dist = np.zeros((3, 3))
    for i in range(len(x_range)):
        for j in range(len(y_range)):
            dist[i, j] = np.sqrt(pow(eye_x - x_range[i], 2) + pow(eye_y - y_range[j], 2))
    idx_min = np.unravel_index(dist.argmin(), dist.shape)

    return [x_range[idx_min[0]], y_range[idx_min[1]]]


def _findLastTimestep(event_list, idx_cur, time):
    while event_list[idx_cur].time < time:
        idx_cur += 1
        try:
            if event_list[idx_cur].time > time:
                idx_cur -= 1
                break
        except:
            idx_cur -= 1
            break
    return idx_cur


def _parse_session_from_mwk2(filename, dir_raw, save_dir):
    session = Session()
    session.date = filename.split('-')[2]
    save_path = save_dir / filename


    f = mwk2reader.MWKFile(str(dir_raw / filename))

    try:
        f.open()
        total = f.get_events(codes=['total'])
    except:
        return

    if len(total) < 10:
        return

    try:
        trialsync = f.get_events(codes=['trialsync'])
        trialsync_times = []
    except:
        trialsync = []
        trialsync_times = []

    try:
        training_set = f.get_events(codes=['trained_targets'])
        training_set_order = f.get_events(codes=['trained_targets_order_xy'])
        testing_set = f.get_events(codes=['test_targets'])
        testing_set_order = f.get_events(codes=['test_targets_order_xy'])
        session.training_set = training_set[-1].data
        session.training_set_order = training_set_order[-1].data
        session.testing_set = testing_set[-1].data
        session.testing_set_order = testing_set_order[-1].data
    except:
        pass

    codec = f.codec

    # discard invalid var_i in the beginning
    tmp = np.array([i.data for i in total])
    idx_begin = np.where(tmp == 1)[0][0]
    total = _cleanEntry(total, idx_begin)  # cue1_start_time

    try:
        photodiode = f.get_events(codes=['photodiode'])
        voltage = []
        [voltage.append(i.data) for i in photodiode]
        voltage = np.array(voltage)

        voltage = voltage < 2

        voltage_ = np.roll(voltage, -1)
        idx_flip = np.where(voltage != voltage_)[0]

        flip_gap = idx_flip[1::2] - idx_flip[::2]

        idx_flip = idx_flip[np.arange(1, len(idx_flip) + 1, 2)]  # count only the rising edge

        idx_real_flip = idx_flip[np.where(flip_gap < 30)[0]]
        photodiode_times = np.array([photodiode[idx].time for idx in idx_real_flip])


    except:
        photodiode_times = []

    try:
        session.target_set = f.get_events(codes=['trained_targets'])[-1].data
        session.alpha_list = f.get_events(codes=['alpha_list'])[0].data
        session.alpha_group_id_list = f.get_events(codes=['alpha_group_id_list'])[0].data

        target_id_all = f.get_events(codes=['target_idx'])

    except:
        pass

    if len(trialsync) > 0:
        trialsync = _cleanEntry(trialsync, idx_begin)
        [trialsync_times.append(i.time) for i in trialsync]

        trialsync_times = np.array(trialsync_times)
        trialsync_times = trialsync_times[np.arange(1, len(trialsync_times), 2)] / 1e6

    try:
        alpha_list_all = f.get_events(codes=['alpha_list'])
        alpha_group_id_all = f.get_events(codes=['alpha_group_id'])
        target_x_all = f.get_events(codes=['target_x'])
        target_y_all = f.get_events(codes=['target_y'])
        order_xy_all = f.get_events(codes=['order_xy'])
        ratio_probe_all = f.get_events(codes=['ratio_probe_trials'])
        toss_probe_all = f.get_events(codes=['toss_probe'])
    except:
        print('Invalid Session ' + session.date)
        return

    idx_alpha_list = 0
    idx_alpha_group_id = 0
    idx_target_pos = 0
    idx_target_id = 0
    idx_order = 0
    idx_ratio_probe = 0
    idx_toss_probe = 0
    reward_total = []

    for i in range(len(total) - 1):  # discard the last trial
        # try catch here
        cue1_start = total[i].time
        try:
            end_i = _getEventInIntv(f, ['aborted', 'no_response', 'completed'], cue1_start, cue1_start + int(10e6),
                                    flag='min')
            is_fixating = _getEventInIntv(f, ['is_fixating'], cue1_start, cue1_start + int(50e6), flag='min')
        except:
            continue

        try:
            reward_total_i = _getEventInIntv(f, ['reward_total'], end_i.time, end_i.time + int(1e6), flag='min').data
        except:
            reward_total_i = -1

        idx_alpha_list = _findLastTimestep(alpha_list_all, idx_alpha_list, cue1_start)
        idx_alpha_group_id = _findLastTimestep(alpha_group_id_all, idx_alpha_group_id, cue1_start)
        idx_target_pos = _findLastTimestep(target_x_all, idx_target_pos, cue1_start)
        idx_target_id = _findLastTimestep(target_id_all, idx_target_id, cue1_start)
        idx_order = _findLastTimestep(order_xy_all, idx_order, cue1_start)
        idx_ratio_probe = _findLastTimestep(ratio_probe_all, idx_ratio_probe, cue1_start)
        idx_toss_probe = _findLastTimestep(toss_probe_all, idx_toss_probe, cue1_start)

        try:
            pre_target_delay = _getEventInIntv(f, ['pre_target_delay'], cue1_start, end_i.time, flag='max')
            pre_target_delay = pre_target_delay.data / 1e6
        except:
            pre_target_delay = -1

        alpha_list_i = alpha_list_all[idx_alpha_list].data
        alpha_group_id_i = alpha_group_id_all[idx_alpha_group_id].data
        try:
            session.ball_alpha.append(alpha_list_i[alpha_group_id_i])
        except:
            session.ball_alpha.append(0)
        session.cue1_start_time.append(cue1_start / 1e6)
        session.end_time.append(end_i.time / 1e6)
        behavior = codec[end_i.code]
        session.behavior.append(behavior)
        if (np.unique(session.training_set_order) == 1).all():
            session.order_xy.append(1)
        else:
            session.order_xy.append(order_xy_all[idx_order].data)
        session.pre_target_delay.append(pre_target_delay)
        session.saccade_onset.append(is_fixating.time / 1e6)

        eye_x = np.array(_getEventInIntv(f, ['eye_x'], cue1_start, end_i.time, flag='raw'))
        eye_y = np.array(_getEventInIntv(f, ['eye_y'], cue1_start, end_i.time, flag='raw'))
        session.eye_xy.append(np.stack((eye_x, eye_y), axis=1))
        if behavior == 'completed':
            is_reward_i = _getEventInIntv(f, ['correct', 'error'], cue1_start, cue1_start + int(5e7), flag='min')
            if codec[is_reward_i.code] == 'correct':
                session.has_reward.append(1)
            elif codec[is_reward_i.code] == 'error':
                session.has_reward.append(0)
            # check the reported target location
            # reward_time_i = getEventInIntv(f, ['rewardOut'], cue1_start, cue1_start + int(1e7), flag='min').time
            # only see the
            eye_x_i = _getEventInIntv(f, ['eye_x'], end_i.time - int(1e5), end_i.time, flag='average')
            eye_y_i = _getEventInIntv(f, ['eye_y'], end_i.time - int(1e5), end_i.time, flag='average')

            reported_xy_i = _convertXY(eye_x_i, eye_y_i)

            session.reported_xy.append(reported_xy_i)
        else:
            session.has_reward.append(-1)
            session.reported_xy.append([np.nan, np.nan])

        session.target_xy.append([target_x_all[idx_target_pos].data / 4, target_y_all[idx_target_pos].data / 4])
        session.target_id.append(target_id_all[idx_target_id].data)
        session.ratio_probe.append(ratio_probe_all[idx_ratio_probe].data)
        session.toss_probe.append(toss_probe_all[idx_toss_probe].data)

        reward_total.append(reward_total_i)

    session.cue1_start_time = np.array(session.cue1_start_time)
    session.end_time = np.array(session.end_time)
    session.trial_len = session.end_time - session.cue1_start_time
    session.behavior = np.asarray(session.behavior, dtype=object)
    session.has_reward = np.array(session.has_reward)
    session.reported_xy = np.asarray(session.reported_xy, dtype=object)
    session.target_xy = np.asarray(session.target_xy, dtype=object)
    session.eye_xy = np.array(session.eye_xy, dtype=object)
    session.order_xy = np.array(session.order_xy)
    session.target_id = np.array(session.target_id)

    session.pre_target_delay = np.array(session.pre_target_delay)
    session.ball_alpha = np.array(session.ball_alpha)

    session.ratio_probe = np.array(session.ratio_probe)
    session.toss_probe = np.array(session.toss_probe)

    reward_total = np.array(reward_total)
    for i0 in range(len(reward_total)):
        if reward_total[i0] == -1:
            try:
                reward_total[i0] = reward_total[i0 - 1]
            except:
                reward_total[i0] = 0
    reward_total_ = np.roll(reward_total, 1)
    reward_total_[0] = 0
    session.reward_dur = reward_total - reward_total_

    if len(photodiode_times) > 0:
        session.photodiode_times = photodiode_times[:len(session.behavior)] / 1e6
    session.trialsync_times = trialsync_times
    session.saccade_onset = np.array(session.saccade_onset)

    session.n_total = len(session.behavior)
    session.n_aborted = sum(session.behavior == 'aborted')
    session.n_completed = sum(session.behavior == 'completed')
    session.n_no_response = sum(session.behavior == 'no_response')
    session.n_correct = sum(session.has_reward == 1)
    session.n_error = sum(session.has_reward == 0)

    # reported_target matches target
    session.true_correct = np.where(np.sum(session.reported_xy == session.target_xy, axis=1) == 2)[0]

    session.checksum = (len(session.behavior) == session.n_completed + session.n_aborted + session.n_no_response) & \
                       (session.n_completed == session.n_correct + session.n_error)
    f.close()

    session = _exception_handle(session)


    return session


if __name__ == "__main__":
    dir_raw = Path('../data_raw')  #./trials/F/'
    dir_parsed = Path('../data_processed') #/F/'
    if not os.path.isdir(dir_parsed):
        os.makedirs(dir_parsed)

    subjects = ['F']
    session_start_date = {
        'F': '20230917',
        'N': ''
    }
    for sub in subjects:
        session_names_raw = os.listdir(dir_raw / sub)
        [session_names_raw.remove(f) for f in session_names_raw if not f.endswith('mwk2')]
        dates = np.array([session_name_i.split('-')[2] for session_name_i in session_names_raw])
        dates_unique = np.unique(dates)

        # a copy of dates, to compare with the original dates after merge
        dates_copy = copy.deepcopy(dates)

        # list of parsed sessions, if exists, skip
        session_names_parsed = os.listdir(dir_parsed / sub)
        [session_names_parsed.remove(f) for f in session_names_parsed if not f.endswith('obj')]
        dates_parsed = [name.split('-')[2] for name in session_names_parsed]
        for date_i in dates_unique:
            if date_i<session_start_date[sub]:
                continue

            session_pieces_idx, = np.where(dates == date_i)

            # if already parsed, skip
            if date_i in dates_parsed:
                continue

            valid_session_pieces_idx = []
            # check if the date is duplicated
            if len(session_pieces_idx) > 1:
                valid_session_pieces = []
                for piece_idx in session_pieces_idx:
                    session_piece = _parse_session_from_mwk2(session_names_raw[piece_idx],
                                                             dir_raw / sub,
                                                             dir_parsed / sub)
                    # keep the valid sessions
                    if session_piece is not None:
                        valid_session_pieces_idx.append(piece_idx)
                        valid_session_pieces.append(session_piece)

                # sort the duplicated dates by cue1_start timing
                valid_session_pieces_idx = np.array(valid_session_pieces_idx)
                valid_session_pieces = np.array(valid_session_pieces)

                cue1_start_list = [session_pieces_i.cue1_start_time[0] for session_pieces_i in valid_session_pieces]
                valid_session_pieces_idx = valid_session_pieces_idx[np.argsort(cue1_start_list).astype(int)]
                valid_session_pieces = valid_session_pieces[np.argsort(cue1_start_list)]

                # join the sessions
                for s in valid_session_pieces[1:]:
                    valid_session_pieces[0].merge(s)

                session_i = valid_session_pieces[0]
            else:
                valid_session_pieces_idx = session_pieces_idx
                session_i = _parse_session_from_mwk2(session_names_raw[session_pieces_idx[0]],
                                                     dir_raw / sub,
                                                     dir_parsed / sub)

            # save the parsed session.obj

            savename = dir_parsed / sub / f'{session_names_raw[valid_session_pieces_idx[0]]}.obj'

            file_to_save = open(savename, 'wb+')
            pickle.dump(session_i, file_to_save)
            file_to_save.close()

            print(savename)



