"""Plot eye speed per trial phase per session.

This was run on openmind in srun sessions.
"""

import json
import numpy as np
import os
from matplotlib import pyplot as plt
import seaborn as sns

_DATA_DIR = '/om4/group/jazlab/nwatters/multi_prediction/data_processed'
_OM2_DATA_DIR = '/om2/user/nwatters/multi_prediction/phys_data'


def _get_padded_eye_vel_per_phase(eye_vel_per_phase):
    phase_lengths = [len(x) for x in eye_vel_per_phase]
    cutoff_phase_length = int(np.median(phase_lengths))
    def _pad_eye_vel(x):
        if len(x) < cutoff_phase_length:
            nan_padded_x = np.concatenate(
                [np.nan * np.ones(cutoff_phase_length - len(x)), x])
            return nan_padded_x
        else:
            return x[-cutoff_phase_length:]
    eye_vel_per_phase = np.array([_pad_eye_vel(x) for x in eye_vel_per_phase])
    return eye_vel_per_phase


def _plot_eye_vel_per_phase(ax,
                            eye_vel_per_phase,
                            phase_name='',
                            bin_size_sec=0.01):
    eye_vel_per_phase = _get_padded_eye_vel_per_phase(eye_vel_per_phase)
    x_axis = np.arange(-eye_vel_per_phase.shape[1], 0) * bin_size_sec
    x_axis = np.tile(x_axis, eye_vel_per_phase.shape[0])
    eye_vel_per_phase = np.ravel(eye_vel_per_phase)
    sns.lineplot(ax=ax, x=x_axis, y=eye_vel_per_phase, errorbar=('ci', 30))
    ax.set_xlabel('sec before phase end')
    ax.set_title(phase_name)
    ax.set_ylabel('eye speed')
    ax.set_ylim([0., 2.])
    return


def _convert_to_velocity(phase_start,
                         phase_end,
                         eye_h_values_interval,
                         eye_h_times_interval,
                         eye_v_values_interval,
                         eye_v_times_interval,
                         bin_size_sec=0.01):

    # Create bins for histograms
    num_bin_edges = int(np.round((phase_end - phase_start) / bin_size_sec))

    # Compute h and v eye velocities
    eye_h_vel = np.abs(eye_h_values_interval[1:] - eye_h_values_interval[:-1])
    eye_v_vel = np.abs(eye_v_values_interval[1:] - eye_v_values_interval[:-1])

    eye_h_bins = np.floor(
        (eye_h_times_interval - phase_start) / bin_size_sec)[:-1]
    eye_v_bins = np.floor(
        (eye_v_times_interval - phase_start) / bin_size_sec)[:-1]

    eye_velocity = np.zeros(num_bin_edges)
    for i in range(num_bin_edges - 1):
        eye_h_mean = np.mean(eye_h_vel[eye_h_bins == i])
        eye_v_mean = np.mean(eye_v_vel[eye_v_bins == i])
        eye_vel = np.sqrt(np.square(eye_h_mean) + np.square(eye_v_mean))
        eye_velocity[i] = eye_vel
    
    return eye_velocity


def _create_plots(session_dir):

    # Get path to trials and eye data
    trials_path = os.path.join(session_dir, 'trial_structure', 'trials')
    eye_data_path = os.path.join(session_dir, 'eye_data')
    
    # Load trials and extract trial start times and relative phase times
    trials = json.load(open(trials_path, 'r'))
    t_start_common = [t['t_start_common'] for t in trials]
    relative_phase_times = [t['mworks_relative_phase_times'] for t in trials]

    # Load eye_h and eye_v values and times
    eye_h_values = np.load(
        os.path.join(eye_data_path, 'eye_h_calibrated_values.npy'))
    eye_h_times = np.load(
        os.path.join(eye_data_path, 'eye_h_calibrated_times.npy'))
    eye_v_values = np.load(
        os.path.join(eye_data_path, 'eye_v_calibrated_values.npy'))
    eye_v_times = np.load(
        os.path.join(eye_data_path, 'eye_v_calibrated_times.npy'))

    # Compute eye velocity in ms bins for each phase of each trial
    eye_vel_per_phase = [[] for _ in range(6)]
    eye_h_ind = 0
    eye_v_ind = 0
    trial_count = 0
    for t in trials:
        trial_count += 1
        if trial_count % 100 == 0:
            print(f'trial {trial_count}/{len(trials)}')
        t_start_common = t['t_start_common']
        relative_phase_times = [0.] + t['mworks_relative_phase_times']
        
        if len(relative_phase_times) > 7:
            # Occassionally there's a mistake in t['mworks_relative_phase_times']
            relative_phase_times = relative_phase_times[:6]
        
        if len(relative_phase_times) < 6:
            # Incomplete trial
            continue
        if relative_phase_times[-1] > 10:
            # Trial took too long
            continue

        for i in range(len(relative_phase_times) - 1):
            phase_start = t_start_common + relative_phase_times[i]
            phase_end = t_start_common + relative_phase_times[i + 1]

            # Scan to start
            skip_trial = False
            while eye_h_times[eye_h_ind] < phase_start:
                eye_h_ind += 1
                if (eye_h_ind >= len(eye_h_times) or
                        eye_h_times[eye_h_ind] > phase_end):
                    skip_trial = True
                    break
            eye_h_start = eye_h_ind
            while eye_v_times[eye_v_ind] < phase_start:
                eye_v_ind += 1
                if (eye_v_ind >= len(eye_v_times) or
                        eye_v_times[eye_v_ind] > phase_end):
                    skip_trial = True
                    break
            eye_v_start = eye_v_ind
            
            # Scan to end
            skip_trial = False
            while eye_h_times[eye_h_ind] < phase_end:
                eye_h_ind += 1
                if eye_h_ind >= len(eye_h_times):
                    skip_trial = True
                    break
            eye_h_end = eye_h_ind
            while eye_v_times[eye_v_ind] < phase_end:
                eye_v_ind += 1
                if eye_v_ind >= len(eye_v_times):
                    skip_trial = True
                    break
            eye_v_end = eye_v_ind

            if skip_trial:
                continue

            # Get the eye values and times in the phase
            eye_h_values_interval = eye_h_values[eye_h_start: eye_h_end]
            eye_h_times_interval = eye_h_times[eye_h_start: eye_h_end]
            eye_v_values_interval = eye_v_values[eye_v_start: eye_v_end]
            eye_v_times_interval = eye_v_times[eye_v_start: eye_v_end]

            eye_velocity = _convert_to_velocity(
                phase_start,
                phase_end,
                eye_h_values_interval,
                eye_h_times_interval,
                eye_v_values_interval,
                eye_v_times_interval,
            )
            eye_vel_per_phase[i].append(eye_velocity)

    # Create figure and axes
    fig, axes = plt.subplots(6, 1, figsize=(3, 12))
    phase_names = ['fixation', 'stimulus', 'delay', 'cue', 'response', 'reward']
    for i in range(6):
        _plot_eye_vel_per_phase(
            axes[i], eye_vel_per_phase[i], phase_name=phase_names[i])

    plt.tight_layout()

    return fig


def main():
    """Compute and write spike times per cluster."""

    base_write_dir = os.path.join(_OM2_DATA_DIR, 'eye_plots')
    if not os.path.exists(base_write_dir):
        os.makedirs(base_write_dir)

    for monkey in ['Perle', 'Elgar']:
        monkey_data_dir = os.path.join(_DATA_DIR, monkey)
        monkey_write_dir = os.path.join(base_write_dir, monkey)
        if not os.path.exists(monkey_write_dir):
            os.makedirs(monkey_write_dir)
        for session_dir in os.listdir(monkey_data_dir):
            session_dir = os.path.join(monkey_data_dir, session_dir)
            print(f'session_dir: {session_dir}')
            fig = _create_plots(session_dir)
            date = session_dir.split('/')[-1]
            fig_path = f'{monkey_write_dir}/{date}.png'
            print(f'fig_path: {fig_path}')
            print(f'{fig_path}')
            fig.savefig(fig_path)
            plt.close(fig)
            

if __name__ == "__main__":
    main()

# To DO:
#     Elgar:  '2022-08-21',  no eye data
#     Perle:  '2022-05-17',  Limited eye data


# Perle total: [
#     '2022-04-07', '2022-04-08', '2022-04-09', '2022-04-11', '2022-04-13',
#     '2022-04-24', '2022-04-25', '2022-04-26', '2022-04-27', '2022-04-28',
#     '2022-04-29', '2022-04-30', '2022-05-01', '2022-05-02', '2022-05-03',
#     '2022-05-04', '2022-05-05', '2022-05-06', '2022-05-08', '2022-05-10',
#     '2022-05-09', '2022-05-12', '2022-05-13', '2022-05-15', '2022-05-16',
#     '2022-05-17', '2022-05-18', '2022-05-26', '2022-05-27', '2022-05-28',
#     '2022-05-29', '2022-05-30', '2022-05-31', '2022-06-01', '2022-06-03',
#     '2022-06-04', '2022-06-05', '2022-06-06', '2022-06-07', '2022-06-08',
#     '2022-06-09', '2022-06-10', '2022-06-11', '2022-06-12', '2022-06-13',
#     '2022-06-14', '2022-06-15', '2022-06-16', '2022-06-17', '2022-06-18',
#     '2022-06-19',
# ]

# Elgar total: [
#     '2022-05-02', '2022-05-06', '2022-05-04', '2022-05-05', '2022-05-07',
#     '2022-05-08', '2022-05-10', '2022-05-12', '2022-05-14', '2022-05-16',
#     '2022-05-17', '2022-05-18', '2022-05-19', '2022-06-02', '2022-06-03',
#     '2022-06-04', '2022-06-05', '2022-06-06', '2022-06-07', '2022-06-13',
#     '2022-06-15', '2022-06-16', '2022-06-17', '2022-06-18', '2022-06-19',
#     '2022-06-20', '2022-06-21', '2022-06-22', '2022-06-23', '2022-06-24',
#     '2022-06-25', '2022-06-29', '2022-06-30', '2022-07-01', '2022-07-02',
#     '2022-08-19', '2022-08-20', '2022-08-22', '2022-08-21', '2022-08-23',
#     '2022-08-24', '2022-08-26', '2022-08-25', '2022-08-31', '2022-09-01',
#     '2022-09-10', '2022-09-11', '2022-09-02', '2022-09-03', '2022-09-05',
#     '2022-09-04', '2022-09-06', '2022-09-07', '2022-09-09', '2022-09-12',
#     '2022-09-13', '2022-09-14', '2022-09-15', '2022-09-18', '2022-09-19',
#     '2022-09-20', '2022-09-21', '2022-09-22', '2022-09-23', '2022-10-02',
#     '2022-10-03', '2022-10-04', '2022-10-05', '2022-10-06', '2022-10-07',
#     '2022-10-11', '2022-10-12', '2022-10-13', '2022-10-14',
# ]
