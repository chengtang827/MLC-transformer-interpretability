"""Script for estimating motion from preprocessed NP dataset"""
import os
import sys
import numpy as np
import shutil
from pathlib import Path
import spikeinterface.extractors as se
import spikeinterface.core as sc
from spikeinterface.sortingcomponents.peak_detection import detect_peaks
from spikeinterface.sortingcomponents.peak_localization import localize_peaks
from spikeinterface.sortingcomponents.peak_selection import select_peaks
from spikeinterface.sortingcomponents.motion_estimation import estimate_motion
from spikeinterface.sortingcomponents.motion_interpolation import interpolate_motion
from probeinterface.plotting import plot_probe
import matplotlib.pyplot as plt
from spikeinterface.widgets import plot_peak_activity_map, plot_motion, plot_timeseries
import torch
import time

_JOB_KWARGS = dict(
    n_jobs=40,
    chunk_duration='2s',
    chunk_memory='10M',
    progress_bar=True,
)


def _plot_peak_activity_n_save(peak_folder, peaks, peak_locations, preprocessed, namestr='peak_locations'):
    fig, ax = plt.subplots(figsize=(10, 10))
    # plot_peak_activity_map(recording=preprocessed, peaks=peaks, figure=fig)
    # fig.savefig(os.path.join(peak_folder, namestr))

    # ax = axs[2]
    x = peaks['sample_index'] / preprocessed.get_sampling_frequency()
    y = peak_locations['y']
    ax.scatter(x, y, s=1, color='k', alpha=0.05)
    ax.set_ylim(0, 4000)
    ax.set_xlabel('time (s)')
    plt.tight_layout()
    fig.savefig(os.path.join(peak_folder, namestr))


def _correct_motion_n_save(preprocessed, motion, temporal_bins, spatial_bins,
                           corrected_folder):
    """
    """
    rec_corrected = interpolate_motion(recording=preprocessed, motion=motion,
                                       temporal_bins=temporal_bins,
                                       spatial_bins=spatial_bins,
                                       border_mode="remove_channels",
                                       spatial_interpolation_method="idw",  # 'idw'
                                       sigma_um=30.)

    rec_corrected.save(folder=corrected_folder, format='binary',
                       **_JOB_KWARGS)


def main(session_dir, iterations):
    """Estimates motion for a NP dataset

    Args:
        session_dir (str): Path to session directory
    """
    base_folder = Path(session_dir)
    print(base_folder)
    corrected_base_folder = base_folder / 'corrected'
    peak_base_folder = base_folder / 'peaks'

    if corrected_base_folder.exists():
        shutil.rmtree(corrected_base_folder)
    if peak_base_folder.exists():
        shutil.rmtree(peak_base_folder)

    peak_base_folder.mkdir(exist_ok=True)
    corrected_base_folder.mkdir(exist_ok=True)

    for i0 in range(iterations):

        print('Detecting peaks: iteration ' + str(i0))

        if i0 == 0:  # first iteration, go from pre-processed raw data
            preprocess_folder = base_folder / 'preprocess'
        else:  # improve based on the previous correction
            preprocess_folder = corrected_base_folder / ('iteration' + str(int(i0 - 1)))

        if not preprocess_folder.exists():
            raise ValueError('Preprocessed data folder does not exist')

        # noise_levels[noise_levels == 0] = 1

        peak_folder_iteration = peak_base_folder / ('iteration' + str(int(i0)))
        peak_folder_before = peak_base_folder / ('iteration' + str(int(i0))) / 'before'
        peak_folder_after = peak_base_folder / ('iteration' + str(int(i0))) / 'after'

        peak_folder_iteration.mkdir(exist_ok=True)
        peak_folder_before.mkdir(exist_ok=True)
        peak_folder_after.mkdir(exist_ok=True)

        # corrected_folder.mkdir(exist_ok=True)
        corrected_folder = corrected_base_folder / ('iteration' + str(int(i0)))
        preprocessed = sc.load_extractor(preprocess_folder)
        noise_levels = sc.get_noise_levels(preprocessed)

        job_kwargs = dict(chunk_duration="1s", n_jobs=20, progress_bar=True)
        t1 = time.time()

        peaks = detect_peaks(recording=preprocessed, method="by_channel", gather_mode='npy',
                             folder=peak_base_folder / ('iteration' + str(int(i0))) / 'tmp_before', names=['peaks.npy'],
                             detect_threshold=9, peak_sign='neg',
                             noise_levels=noise_levels, **job_kwargs)
        t2 = time.time()
        print('t2: ' + str(t2 - t1))
        some_peaks = select_peaks(peaks, method='uniform', n_peaks=900000)
        # some_peaks = peaks
        t3 = time.time()
        print('t3: ' + str(t3 - t2))

        print('Localizing peaks: iteration ' + str(i0))

        peak_locations = localize_peaks(recording=preprocessed, peaks=some_peaks, method="monopolar_triangulation",
                                        local_radius_um=100, max_distance_um=1000, ms_before=0.3, ms_after=0.6,
                                        optimizer='minimize_with_log_penality', **job_kwargs)

        _plot_peak_activity_n_save(peak_folder=peak_folder_before,
                                   peaks=some_peaks, peak_locations=peak_locations,
                                   preprocessed=preprocessed, namestr='peak_locations_before')
        t5 = time.time()
        print('t5: ' + str(t5 - t3))

        print('Estimating motion: iteration ' + str(i0))

        motion_kwargs = {'torch_device': torch.device('cuda'), 'conv_engine': 'torch', 'corr_threshold': 0.,
                         'time_horizon_s': 360, 'convergence_method': 'lsqr_robust'}
        motion, temporal_bins, spatial_bins = estimate_motion(recording=preprocessed,
                                                              peaks=some_peaks,
                                                              peak_locations=peak_locations,
                                                              method="decentralized",
                                                              direction="y",
                                                              bin_duration_s=2.0,
                                                              bin_um=5.0,
                                                              win_step_um=50.0,
                                                              win_sigma_um=600.0, progress_bar=True, ** motion_kwargs)

        t6 = time.time()
        print('t6: ' + str(t6 - t5))

        t7 = time.time()
        print('t7: ' + str(t7 - t6))
        _correct_motion_n_save(preprocessed=preprocessed, motion=motion,
                               temporal_bins=temporal_bins, spatial_bins=spatial_bins,
                               corrected_folder=corrected_folder)

        t8 = time.time()

        print('t8: ' + str(t8 - t7))

        preprocessed = sc.load_extractor(corrected_folder)
        noise_levels = sc.get_noise_levels(preprocessed)

        peaks = detect_peaks(recording=preprocessed, method="by_channel", gather_mode='npy',
                             folder=peak_base_folder / ('iteration' + str(int(i0))) / 'tmp_after', names=['peaks.npy'],
                             detect_threshold=9, peak_sign='neg',
                             noise_levels=noise_levels, **job_kwargs)

        some_peaks = select_peaks(peaks, method='uniform', n_peaks=900000)
        # some_peaks = peaks
        print('Localizing peaks: iteration ' + str(i0))
        peak_locations = localize_peaks(recording=preprocessed, peaks=some_peaks, method="monopolar_triangulation",
                                        local_radius_um=100,
                                        max_distance_um=1000, **job_kwargs)
        _plot_peak_activity_n_save(peak_folder=peak_folder_after,
                                   peaks=some_peaks, peak_locations=peak_locations,
                                   preprocessed=preprocessed, namestr='peak_locations_after')


if __name__ == "__main__":
    session_dir = sys.argv[1]

    # receive optional parameter on the number of iterations of motion correction, default 1
    try:
        iterations = sys.argv[2]
    except:
        iterations = 1

    # session_dir = '/Users/laptopd/Documents/Compositionality/Analysis/phys_preprocessing_open_source-main'
    main(session_dir, iterations)
