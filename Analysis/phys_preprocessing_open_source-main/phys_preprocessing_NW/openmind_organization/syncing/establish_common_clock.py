"""Get spike times per cluster.

Run this before data_to_ibl_format stuff.
"""

import json
import numpy as np
import os
from matplotlib import pyplot as plt

from sklearn import linear_model as sklearn_linear_model

_DATA_DIR = '/om4/group/jazlab/nwatters/multi_prediction/data_processed'
_OM2_DATA_DIR = '/om2/user/nwatters/multi_prediction/phys_data'


def _get_timescale_mapping(common_t, phys_t, phys_name):
    times = np.array([
        [x, y] for x, y in zip(phys_t, common_t)
        if x is not None and y is not None
    ])
    if len(times) == 0:
        return None, None
    regr = sklearn_linear_model.LinearRegression()
    regr.fit(times[:, 0:1], times[:, 1:2])
    slope = regr.coef_[0, 0]
    intercept = regr.intercept_[0]
    
    pred_common_t = regr.predict(times[:, 0:1])[:, 0]
    residuals = times[:, 1] - pred_common_t
    residual_std = np.round(1000. * np.std(residuals), decimals=1)
    residual_max_error = np.round(1000. * np.max(np.abs(residuals)), decimals=1)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(np.arange(len(residuals)), residuals, s=5)
    ax.set_xlabel('trial index')
    ax.set_ylabel(f'{phys_name} prediction commont time residuals', fontsize=8)
    ax.set_title(
        f'{phys_name} residual std: {residual_std}, max: {residual_max_error}',
        fontsize=8,
    )

    return fig, (slope, intercept)


def _correct_spike_times(probe_spike_sorting_path, time_mapping):
    # Unpack time_mapping
    slope, intercept = time_mapping

    # Load spike times
    print('        Loading spike times')
    spike_times = np.load(
        os.path.join(probe_spike_sorting_path, 'spike_times.npy'))

    # Read sample_rate and adjust spike times to be in phys timescale
    sample_rate_path = os.path.join(probe_spike_sorting_path, 'sample_rate')
    print(f'        Reading sample_rate from {sample_rate_path}')
    sample_rate = json.load(open(sample_rate_path, 'r'))
    print(f'        sample_rate = {sample_rate}')
    spike_times = spike_times.astype(float) / float(sample_rate)

    # Correct spike times to be in common timescale
    print('        Correcting spike times')
    corrected_spike_times = slope * spike_times + intercept

    # Save spike times
    write_path = os.path.join(probe_spike_sorting_path, 'corrected_spike_times')
    print('        Saving corrected spike times')
    np.save(write_path, corrected_spike_times)
    return


def _process_session(session_dir):
    print(f'Processing {session_dir}')

    # Load trials
    trials_path = os.path.join(session_dir, 'trial_structure', 'trials')
    trials = json.load(open(trials_path, 'r'))
    
    # Get common to phys timescale mapping for open_ephys and spikeglx
    common_t = [t['t_start_common'] for t in trials]
    open_ephys_t = [
        t['open_ephys_t_start'] if 'open_ephys_t_start' in t else None
        for t in trials
    ]
    spikeglx_t = [
        t['spikeglx_t_start'] if 'spikeglx_t_start' in t else None
        for t in trials
    ]
    print('    Computing timescale mappings')
    open_ephys_fig, open_ephys_mapping = _get_timescale_mapping(
        common_t, open_ephys_t, 'open_ephys')
    spikeglx_fig, spikeglx_mapping = _get_timescale_mapping(
        common_t, spikeglx_t, 'spikeglx')

    # Correct, write, and plot correlation of phys spike times
    spike_sorting_path = os.path.join(session_dir, 'spike_sorting')
    probe_names = os.listdir(spike_sorting_path)
    for probe_name in probe_names:
        probe_spike_sorting_path = os.path.join(spike_sorting_path, probe_name)
        if probe_name[:7] == 'v_probe':
            if open_ephys_mapping is None:
                raise ValueError(
                    'Found v_probe spike sorting but no open_ephys_mapping')
            print('    Correcting open_ephys spike times')
            _correct_spike_times(probe_spike_sorting_path, open_ephys_mapping)
        if probe_name[:2] == 'np':
            if spikeglx_mapping is None:
                raise ValueError(
                    'Found np spike sorting but no spikeglx_mapping')
            print('    Correcting spikeglx spike times')
            _correct_spike_times(probe_spike_sorting_path, spikeglx_mapping)

    # Return figures to save
    figs = {
        'open_ephys': open_ephys_fig,
        'spikeglx': spikeglx_fig,
    }

    return figs


def main():
    """Compute and write spike times per cluster."""

    # figs = _process_session(
    #     '/Users/nicholaswatters/Desktop/grad_school/research/mehrdad/'
    #     'multi_prediction/phys/phys_data/Perle/2022-06-01'
    # )

    base_figure_dir = os.path.join(_OM2_DATA_DIR, 'sync_plots_common')
    if not os.path.exists(base_figure_dir):
        os.makedirs(base_figure_dir)

    for monkey in ['Perle', 'Elgar']:
        monkey_data_dir = os.path.join(_DATA_DIR, monkey)
        monkey_figure_dir = os.path.join(base_figure_dir, monkey)
        if not os.path.exists(monkey_figure_dir):
            os.makedirs(monkey_figure_dir)
        for session_dir in os.listdir(monkey_data_dir):
            session_dir = os.path.join(monkey_data_dir, session_dir)
            figs = _process_session(session_dir)
            for phys_name, fig in figs.items():
                if fig is None:
                    continue
                date = session_dir.split('/')[-1]
                fig_path = f'{monkey_figure_dir}/{date}_{phys_name}.png'
                print(f'{fig_path}')
                fig.savefig(fig_path)
                plt.close(fig)


if __name__ == "__main__":
    main()

