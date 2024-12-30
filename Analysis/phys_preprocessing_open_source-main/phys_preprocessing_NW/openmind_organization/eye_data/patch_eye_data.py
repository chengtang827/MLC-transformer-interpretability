"""Patch eye data from different sessions together.

Run this before ../syncing/establish_common_clock.py.
"""

import json
import os
import numpy as np
from sklearn import linear_model as sklearn_linear_model

_DATA_BASE_DIR = '/om4/group/jazlab/nwatters/multi_prediction/data_processed'


def _process_session(session_data_dir):
    
    ############################################################################
    #### READ TRIALS
    ############################################################################

    trials_path = os.path.join(session_data_dir, 'trial_structure/trials')
    trials = json.load(open(trials_path, 'r'))
    mworks_session_num = np.array([t['mworks_session_num'] for t in trials])
    unique_mworks_session_nums = np.sort(np.unique(mworks_session_num))
    trial_inds_per_session_num = [
        np.sort(np.squeeze(np.argwhere(mworks_session_num == x)))
        for x in unique_mworks_session_nums
    ]

    ############################################################################
    #### COMPUTE PHYS TO MWORKS TIME SCALING
    ############################################################################

    # For each mworks session, find open_ephys, spikeglx, and mworks trial start
    # times

    mworks_from_open_ephys_slopes = []
    mworks_from_open_ephys_trial_nums = []
    mworks_from_spikeglx_slopes = []
    mworks_from_spikeglx_trial_nums = []
    for trial_inds in trial_inds_per_session_num:
        open_ephys_mworks_t_start = np.array([
            [trials[i]['open_ephys_t_start'], trials[i]['mworks_t_start']]
            for i in trial_inds
            if 'open_ephys_t_start' in trials[i]
        ])
        spikeglx_mworks_t_start = np.array([
            [trials[i]['spikeglx_t_start'], trials[i]['mworks_t_start']]
            for i in trial_inds
            if 'spikeglx_t_start' in trials[i]
        ])
        if open_ephys_mworks_t_start.shape[0] > 1:
            regr_open_ephys = sklearn_linear_model.LinearRegression()
            regr_open_ephys.fit(
                open_ephys_mworks_t_start[:, 0:1],
                open_ephys_mworks_t_start[:, 1:2],
            )
            mworks_from_open_ephys_slopes.append(regr_open_ephys.coef_[0, 0])
            mworks_from_open_ephys_trial_nums.append(
                open_ephys_mworks_t_start.shape[0])
        
        if spikeglx_mworks_t_start.shape[0] > 1:
            regr_spikeglx = sklearn_linear_model.LinearRegression()
            regr_spikeglx.fit(
                spikeglx_mworks_t_start[:, 0:1],
                spikeglx_mworks_t_start[:, 1:2],
            )
            mworks_from_spikeglx_slopes.append(regr_spikeglx.coef_[0, 0])
            mworks_from_spikeglx_trial_nums.append(
                spikeglx_mworks_t_start.shape[0])
    
    mworks_from_open_ephys_slopes = np.array(mworks_from_open_ephys_slopes)
    mworks_from_spikeglx_slopes = np.array(mworks_from_spikeglx_slopes)

    # Compute weighted average open_ephys and spikeglx slope
    
    if len(mworks_from_open_ephys_slopes) > 0:
        mworks_from_open_ephys_slope = np.average(
            mworks_from_open_ephys_slopes,
            weights=mworks_from_open_ephys_trial_nums,
        )
    else:
        mworks_from_open_ephys_slope = None
        
    if len(mworks_from_spikeglx_slopes) > 0:
        mworks_from_spikeglx_slope = np.average(
            mworks_from_spikeglx_slopes,
            weights=mworks_from_spikeglx_trial_nums,
        )
    else:
        mworks_from_spikeglx_slope = None

    ############################################################################
    #### COMPUTE TIME OFFSET FOR EACH MWORKS SESSION
    ############################################################################

    # compute time between successive mworks sessions

    time_data_per_session_num = []
    for i, (session_num, trial_inds) in enumerate(zip(
            unique_mworks_session_nums, trial_inds_per_session_num)):
        start_time = trials[trial_inds[0]]['mworks_t_start']
        end_time = trials[trial_inds[-1]]['mworks_t_end']
        if i == 0:
            # Align so trial 0 starts at time 0.
            offset = -1. * start_time
        else:
            # Compute mworks offset from phys times
            last_trial_old_session = trials[
                time_data_per_session_num[-1]['trial_inds'][-1]
            ]
            first_trial_new_session = trials[trial_inds[0]]
            mworks_old_t = last_trial_old_session['mworks_t_start']
            mworks_new_t = first_trial_new_session['mworks_t_start']
            if ('open_ephys_t_start' in last_trial_old_session and 
                    'open_ephys_t_start' in first_trial_new_session):
                old_t = last_trial_old_session['open_ephys_t_start']
                new_t = first_trial_new_session['open_ephys_t_start']
                true_mworks_new_t = (
                    mworks_old_t +
                    mworks_from_open_ephys_slope * (new_t - old_t)
                )
                offset = true_mworks_new_t - mworks_new_t
            elif ('spikeglx_t_start' in last_trial_old_session and 
                    'spikeglx_t_start' in first_trial_new_session):
                old_t = last_trial_old_session['spikeglx_t_start']
                new_t = first_trial_new_session['spikeglx_t_start']
                true_mworks_new_t = (
                    mworks_old_t +
                    mworks_from_spikeglx_slope * (new_t - old_t)
                )
                offset = true_mworks_new_t - mworks_new_t
            else:
                raise ValueError(
                    f'Neither open_ephys nor spikeglx works for session {i}')

        data = {
            'session_num': session_num,
            'trial_inds': trial_inds,
            'start_time': start_time,
            'end_time': end_time,
            'offset_to_accumulate': offset,
        }
        time_data_per_session_num.append(data)
        
    ############################################################################
    #### CORRECT AND CONCATENATE EYE VARIABLE TIMES
    ############################################################################

    eye_data_path = os.path.join(session_data_dir, 'eye_data')
    for var_name in ['eye_h_calibrated', 'eye_v_calibrated', 'pupil_size_r']:
        var_times_final = []
        var_values_final = []
        agg_offset = 0.
        for data in time_data_per_session_num:
            session_num = data['session_num']
            start_time = data['start_time']
            end_time = data['end_time']
            agg_offset += data['offset_to_accumulate']

            var_path = os.path.join(
                eye_data_path,
                'mwork_session_num_' + str(session_num),
                var_name,
            )
            var_times_path = var_path + '_times.npy'
            var_times = np.load(var_times_path)
            var_values_path = var_path + '_values.npy'
            var_values = np.load(var_values_path)
            save_inds = (var_times > start_time) * (var_times < end_time)
            var_times_curated = agg_offset + var_times[save_inds]
            var_values_curated = var_values[save_inds]
            var_times_final.append(var_times_curated)
            var_values_final.append(var_values_curated)
        
        var_times_final = np.concatenate(var_times_final)
        var_values_final = np.concatenate(var_values_final)
        write_path_times = os.path.join(eye_data_path, var_name + '_times')
        write_path_values = os.path.join(eye_data_path, var_name + '_values')
        print(f'Writing {write_path_times}')
        np.save(write_path_times, var_times_final)
        print(f'Writing {write_path_values}')
        np.save(write_path_values, var_values_final)

    ############################################################################
    #### ADD CORRECTED TIMES TO TRIALS
    ############################################################################

    # Add corrected mworks time to trials
    agg_offset = 0.
    for data in time_data_per_session_num:
        session_num = data['session_num']
        trial_inds = data['trial_inds']
        start_time = data['start_time']
        end_time = data['end_time']
        agg_offset += data['offset_to_accumulate']
        for i in trial_inds:
            trials[i]['t_start_common'] = (
                agg_offset + trials[i]['mworks_t_start'])
            trials[i]['t_end_common'] = agg_offset + trials[i]['mworks_t_end']

    # Write trials
    print(f'Writing {trials_path}')
    json.dump(trials, open(trials_path, 'w'))

    ############################################################################
    #### CHECK THAT WE HAVE EYE DATA FOR ALL TRIALS
    ############################################################################

    for i, t in enumerate(trials):
        t_start = t['t_start_common']
        t_end = t['t_end_common']
        if not np.sum((var_times_final > t_start) * (var_times_final < t_end)):
            raise ValueError('MISSING EYE DATA FOR SOME TRIALS')

    return
    

def main():
    """Compute and write MWorks eye data to data_processed for each session."""

    print('STARTED patch_eye_data.py\n')

    # _process_session(
    #     '/Users/nicholaswatters/Desktop/grad_school/research/mehrdad/'
    #     'multi_prediction/phys/phys_data/sync_plots/data/Elgar/2022-05-07'
    # )

    for monkey in ['Perle', 'Elgar']:
        monkey_data_dir = os.path.join(_DATA_BASE_DIR, monkey)
        for session_date in os.listdir(monkey_data_dir):
            print(f'\n{session_date}')
            session_data_dir = os.path.join(
                monkey_data_dir, session_date)
            _process_session(session_data_dir)
            
    print('\nFINISHED patch_eye_data.py')

    return


if __name__ == "__main__":
    main()

