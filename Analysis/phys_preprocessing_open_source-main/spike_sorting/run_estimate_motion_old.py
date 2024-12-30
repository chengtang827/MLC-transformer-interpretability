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
from spikeinterface.sortingcomponents.motion_correction import CorrectMotionRecording
from probeinterface.plotting import plot_probe
import matplotlib.pyplot as plt
from spikeinterface.widgets._legacy_mpl_widgets import plot_pairwise_displacement, plot_displacement
from spikeinterface.widgets import plot
import torch
import time

_JOB_KWARGS = dict(
    n_jobs=40,
    chunk_duration='2s',
    chunk_memory='10M',
    progress_bar=True,
)

def _plot_motion(peak_folder, motion, temporal_bins, spatial_bins, extra_check, namestr='motion'):
    fig, axs = plt.subplots(ncols=2, figsize=(15, 10))
    plot_pairwise_displacement(motion, temporal_bins, spatial_bins, extra_check,
                                ncols=8, ax=axs[0])
    plot_displacement(motion, temporal_bins, spatial_bins, extra_check,
                       with_histogram=True, ax=axs[1])
    # somehow this line gives me error
    # plt.tight_layout()
    fig.savefig(os.path.join(peak_folder, namestr))


def _plot_peak_locations(peak_folder, peak_locations, peaks, rec_preprocessed, namestr='peak_locations'):
    probe = rec_preprocessed.get_probe()

    fig, axs = plt.subplots(ncols=3, sharey=True, figsize=(20, 10))
    ax = axs[0]
    plot_probe(probe, ax=ax)
    ax.scatter(peak_locations['x'], peak_locations['y'], color='k', s=1,
               alpha=0.002)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    if 'z' in peak_locations.dtype.fields:
        ax = axs[1]
        ax.scatter(peak_locations['z'], peak_locations['y'], color='k', s=1,
                   alpha=0.002)
        ax.set_xlabel('z')
    ax.set_ylim(0, 4000)

    ax = axs[2]
    x = peaks['sample_ind'] / rec_preprocessed.get_sampling_frequency()
    y = peak_locations['y']
    ax.scatter(x, y, s=1, color='k', alpha=0.05)
    ax.set_ylim(0, 4000)
    ax.set_xlabel('time (s)')
    plt.tight_layout()
    fig.savefig(os.path.join(peak_folder, namestr))


def _correct_motion(rec_preprocessed, motion, temporal_bins, spatial_bins,
                    corrected_folder):
    """
    """
    rec_corrected = CorrectMotionRecording(rec_preprocessed, motion, temporal_bins,
                                           spatial_bins,
                                           spatial_interpolation_method='idw',
                                           border_mode='remove_channels')

    rec_corrected.save(folder=corrected_folder, format='binary',
                       **_JOB_KWARGS)


def _detect_peaks(peak_folder, rec_preprocessed, noise_levels):
    """Detects peaks from preprocessed recording.
    If peak_folder exists, returns saved peaks

    Args:
        peak_folder (str or Path):
        rec_preprocessed (object):

    Returns:
        np.ndarray: Peaks description
    """

    if not (peak_folder / 'peaks.npy').exists():
        peaks = detect_peaks(
            rec_preprocessed,
            method='locally_exclusive',
            local_radius_um=100,
            peak_sign='neg',
            detect_threshold=9,
            noise_levels=noise_levels,
            **_JOB_KWARGS,
        )
        np.save(peak_folder / 'peaks.npy', peaks)
    else:
        print('Detected peaks exist')
    peaks = np.load(peak_folder / 'peaks.npy')

    return peaks


def _localize_peaks(peak_folder, rec_preprocessed, peaks):
    """Localizes peaks using monopolar triangulation
    Args:
        peak_folder (str or Path):
        rec_preprocessed (object):

    Returns:
        np.ndarray: Peaks description
    """
    if not (peak_folder / 'peak_locations_monopolar_triangulation_log_limit.npy').exists():
        peak_locations = localize_peaks(
            rec_preprocessed,
            peaks,
            ms_before=0.3,
            ms_after=0.6,
            optimizer='minimize_with_log_penality',
            method='monopolar_triangulation',
            local_radius_um=100.,
            max_distance_um=1000.,
            **_JOB_KWARGS,
        )
        np.save(peak_folder / 'peak_locations_monopolar_triangulation_log_limit.npy', peak_locations)
        print(peak_locations.shape)
    else:
        print('Localized peaks exist')
    peak_locations = np.load(peak_folder / 'peak_locations_monopolar_triangulation_log_limit.npy')
    return peak_locations


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

        corrected_folder = corrected_base_folder / ('iteration' + str(int(i0)))
        rec_preprocessed = sc.load_extractor(preprocess_folder)
        noise_levels = sc.get_noise_levels(rec_preprocessed)
        noise_levels[noise_levels == 0] = 1

        peak_folder_iteration = peak_base_folder / ('iteration' + str(int(i0)))
        peak_folder_before = peak_base_folder / ('iteration' + str(int(i0))) / 'before'
        peak_folder_after = peak_base_folder / ('iteration' + str(int(i0))) / 'after'

        peak_folder_iteration.mkdir(exist_ok=True)
        peak_folder_before.mkdir(exist_ok=True)
        peak_folder_after.mkdir(exist_ok=True)

        # corrected_folder.mkdir(exist_ok=True)
        t1 = time.time()
        peaks = _detect_peaks(peak_folder=peak_folder_before,
                              rec_preprocessed=rec_preprocessed,
                              noise_levels=noise_levels)
        t2 = time.time()
        print('t2: '+ str(t2-t1))
        some_peaks = select_peaks(peaks,
                                  method='smart_sampling_amplitudes',
                                  noise_levels=noise_levels,
                                  n_peaks=900000)
        t3 = time.time()
        print('t3: ' + str(t3 - t2))

        print('Localizing peaks: iteration ' + str(i0))
        peak_locations = _localize_peaks(peak_folder=peak_folder_before,
                                         rec_preprocessed=rec_preprocessed,
                                         peaks=some_peaks)
        t4 = time.time()
        print('t4: '+ str(t4-t3))

        _plot_peak_locations(peak_folder=peak_folder_before,
                             peak_locations=peak_locations,
                             peaks=some_peaks,
                             rec_preprocessed=rec_preprocessed, namestr='peak_locations_before')
        t5 = time.time()
        print('t5: '+ str(t5-t4))

        print('Estimating motion: iteration ' + str(i0))

        motion_kwargs = {'torch_device': torch.device('cuda'), 'conv_engine': 'torch', 'corr_threshold': 0.,
                         'time_horizon_s': 360, 'convergence_method': 'lsqr_robust'}
        motion, temporal_bins, spatial_bins, extra_check = estimate_motion(
            rec_preprocessed, some_peaks, peak_locations,
            direction='y', win_step_um=50, win_sigma_um=600,
            output_extra_check=True, progress_bar=True, **motion_kwargs)
        t6 = time.time()
        print('t6: '+ str(t6-t5))

        # print(extra_check.keys())
        # _plot_motion(peak_folder=peak_folder, motion=motion, temporal_bins=temporal_bins,
        #              spatial_bins=spatial_bins, extra_check=extra_check)
        t7 = time.time()
        print('t7: '+ str(t7-t6))

        _correct_motion(rec_preprocessed=rec_preprocessed, motion=motion,
                        temporal_bins=temporal_bins, spatial_bins=spatial_bins,
                        corrected_folder=corrected_folder)
        t8 = time.time()
        print('t8: '+ str(t8-t7))

        # post plot
        rec_preprocessed = sc.load_extractor(corrected_folder)
        noise_levels = sc.get_noise_levels(rec_preprocessed)
        peaks = _detect_peaks(peak_folder=peak_folder_after,
                              rec_preprocessed=rec_preprocessed,
                              noise_levels=noise_levels)
        some_peaks = select_peaks(peaks,
                                  method='smart_sampling_amplitudes',
                                  noise_levels=noise_levels,
                                  n_peaks=900000)
        peak_locations = _localize_peaks(peak_folder=peak_folder_after,
                                         rec_preprocessed=rec_preprocessed,
                                         peaks=some_peaks)
        _plot_peak_locations(peak_folder=peak_folder_after,
                             peak_locations=peak_locations,
                             peaks=some_peaks,
                             rec_preprocessed=rec_preprocessed, namestr='peak_locations_after')



    # bin_um = 5
    # bin_duration_s=5.

    # motion_histogram, temporal_bins, spatial_bins = make_motion_histogram(
    #     rec_preprocessed,
    #     peaks,
    #     peak_locations=peak_locations,
    #     bin_um=bin_um,
    #     bin_duration_s=bin_duration_s,
    #     direction='y',
    #     weight_with_amplitude=False,
    # )
    # print(motion_histogram.shape, temporal_bins.size, spatial_bins.size)


if __name__ == "__main__":
    session_dir = sys.argv[1]

    # receive optional parameter on the number of iterations of motion correction, default 1
    try:
        iterations = sys.argv[2]
    except:
        iterations = 1

    # session_dir = '/Users/laptopd/Documents/Compositionality/Analysis/phys_preprocessing_open_source-main'
    main(session_dir, iterations)
