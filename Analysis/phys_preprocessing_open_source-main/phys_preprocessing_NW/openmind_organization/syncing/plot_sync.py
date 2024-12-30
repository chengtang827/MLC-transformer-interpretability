"""Get spike times per cluster."""

import json
import numpy as np
import os
from matplotlib import pyplot as plt

from sklearn import linear_model as sklearn_linear_model

_OM2_DATA_DIR = '/om2/user/nwatters/multi_prediction/phys_data'


def _scatterplot_sync(ax, mworks_t, phys_t, phys_name):
    """Scatterplot ignoring None values."""
    to_plot = np.array([
        [x, y] for x, y in zip(mworks_t, phys_t)
        if x is not None and y is not None
    ])
    if len(to_plot) == 0:
        return
    ax.scatter(to_plot[:, 0], to_plot[:, 1], s=2)
    ax.set_xlabel('mworks times')
    ax.set_ylabel(phys_name + ' times')
    ax.set_title('Mworks times vs ' + phys_name + ' times', fontsize=8)


def _plot_residuals(ax, mworks_t, phys_t, phys_name):
    times = np.array([
        [x, y] for x, y in zip(mworks_t, phys_t)
        if x is not None and y is not None
    ])
    if len(times) == 0:
        return
    regr = sklearn_linear_model.LinearRegression()
    regr.fit(times[:, 0:1], times[:, 1:2])
    pred_phys_t = regr.predict(times[:, 0:1])[:, 0]
    residuals = times[:, 1] - pred_phys_t
    residual_std = np.round(1000. * np.std(residuals), decimals=1)
    residual_max_error = np.round(1000. * np.max(np.abs(residuals)), decimals=1)

    ax.scatter(np.arange(len(residuals)), residuals, s=5)
    ax.set_xlabel('trial index')
    ax.set_ylabel(f'{phys_name} prediction residuals')
    ax.set_title(
        f'{phys_name} residual std: {residual_std}, max: {residual_max_error}',
        fontsize=8,
    )


def _create_sync_plots(session_dir):

    # Get path to trials
    trials_path = os.path.join(session_dir, 'trial_structure', 'trials')
    if not os.path.exists(trials_path):
        print(f'trials_path does not exist: {trials_path}')
        return None
    
    # Load trials and extract trial start times
    trials = json.load(open(trials_path, 'r'))
    t_start = {
        'mworks': [
            t['mworks_t_start'] if 'mworks_t_start' in t else None
            for t in trials
        ],
        'open_ephys': [
            t['open_ephys_t_start'] if 'open_ephys_t_start' in t else None
            for t in trials
        ],
        'spikeglx': [
            t['spikeglx_t_start'] if 'spikeglx_t_start' in t else None
            for t in trials
        ],
    }

    # Create figure and axes
    fig, axes = plt.subplots(2, 2, figsize=(6, 6))
    for ax_row, phys_name in zip(axes, ['open_ephys', 'spikeglx']):
        _scatterplot_sync(
            ax_row[0], t_start['mworks'], t_start[phys_name], phys_name)
        _plot_residuals(
            ax_row[1], t_start['mworks'], t_start[phys_name], phys_name)

    plt.tight_layout()

    return fig


def main():
    """Compute and write spike times per cluster."""

    # _create_sync_plots(
    #     '/Users/nicholaswatters/Desktop/grad_school/research/mehrdad/multi_prediction/phys/phys_data/Perle/2022-06-18'
    # )

    base_write_dir = os.path.join(_OM2_DATA_DIR, 'sync_plots')
    if not os.path.exists(base_write_dir):
        os.makedirs(base_write_dir)

    for monkey in ['Perle', 'Elgar']:
        monkey_data_dir = os.path.join(_OM2_DATA_DIR, monkey)
        monkey_write_dir = os.path.join(base_write_dir, monkey)
        if not os.path.exists(monkey_write_dir):
            os.makedirs(monkey_write_dir)
        for session_dir in os.listdir(monkey_data_dir):
            session_dir = os.path.join(monkey_data_dir, session_dir)
            fig = _create_sync_plots(session_dir)
            if fig is None:
                continue
            date = session_dir.split('/')[-1]
            fig_path = f'{monkey_write_dir}/{date}.png'
            print(f'{fig_path}')
            fig.savefig(fig_path)
            plt.close(fig)


# Elgar 2022-09-07:  Solved, copy trials back
# Elgar 2022-09-19:  Solved. Copy trials, spike_times, spike_clusters, amplitudes. Check rasters
# Perle 2022-05-09:  Nothing wrong. Just MWorks had two sessions.

# General to-do:
#   * Copy problem sessions back to om4
#   * Deal with eyelink and behavior data
#   * Pack all data into session-long arrays instead of trials, with one common clock (adjusting for phys sampling rate, etc.).
#   * Plot rasters to check
            

if __name__ == "__main__":
    main()

