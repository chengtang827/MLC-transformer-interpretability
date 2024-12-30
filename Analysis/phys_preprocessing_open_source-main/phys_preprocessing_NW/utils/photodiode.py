"""Functions for processing photodiode data.

In the multi-prediction task, there is an MWorks variable sync_trial_start that
is usually 0 put steps to 1 for 50ms at the beginning of each trial. The
photodiode follows this variable. We use the offset of this step as the
beginning of the trial, and detect the corresponding photodiode offset to
evaluate the delay between MWorks and the monitor.

This file has two functions:
    * get_photodiode_flashes(): Detects photodiode flashes
    * get_photodiode_delays(): Computes per-trial photodiode delays relative to
        trial start sync signal.
"""

from matplotlib import pyplot as plt
import numpy as np
import warnings


def get_photodiode_flashes(photodiode_vals,
                           photodiode_times,
                           session_start_time=0.,
                           smooth_window=10,
                           plot=False):
    """Compute times of photodiode flashes.

    This function works as follows:
        * Normalize and smooth the photodiode data.
        * Compute the mean square error with a sliding step function.
        * Find local minima of the mean squared error. These are the photodiode
            change times.

    Args:
        photodiode_vals: Numpy array of photodiode values.
        photodiode_times: Numpy array of photodiode times in seconds for the
            values in photodiode_vals. Same length as photodiode_vals.
        session_start_time: Time at which the session starts. All photodiode
            flashes before this time will be ignored.
        smooth_window: Window (in timesteps, which should be 1ms) over which to
            smooth photodiode values to help accurately detect the change.
            This may seem like a large window size, but empirically, smoothing
            like this helps accuracy of photodiode detection for monkey P data,
            which has a noisy photodiode.

    Returns:
        photodiode_change_times: Numpy array of times at which photodiode turns
            from on (white on screen) to off (black on screen).
    """

    sample_rate = np.mean(photodiode_times[1:] - photodiode_times[:-1])
    target_sample_rate = 1. / 1000  # 1 msec

    # This function works best with a sample rate of 1ms, so if the sample rate
    # is not close to the target of 1ms, raise an error.
    if not np.isclose(sample_rate, target_sample_rate, rtol=0.5):
        raise ValueError(
            f'Sample_rate is {sample_rate}, but should be close to '
            f'{target_sample_rate}.'
        )

    # Normalize photodiode_vals, excluding extreme photodiode values
    photodiode_vals -= np.mean(photodiode_vals[
        (photodiode_vals < np.quantile(photodiode_vals, 0.999)) &
        (photodiode_vals > np.quantile(photodiode_vals, 0.001))
    ])
    photodiode_vals /= np.std(photodiode_vals[
        (photodiode_vals < np.quantile(photodiode_vals, 0.999)) &
        (photodiode_vals > np.quantile(photodiode_vals, 0.001))
    ])

    # Smooth photodiode_vals
    kernel = np.ones(smooth_window) / smooth_window
    photodiode_vals_smoothed = np.convolve(photodiode_vals, kernel, mode='same')

    # Figure out if photodiode goes up or down. For Monkey E data, the
    # photodiode values are negated (black is high value, white is low value),
    # so we handle both signs by inferring whether the white value should be
    # positive or nevative based on the skew of the photodiode data
    # distribution.
    up_quantile = np.quantile(photodiode_vals_smoothed, 0.998)
    down_quantile = np.quantile(photodiode_vals_smoothed, 0.002)
    if up_quantile > -1 * down_quantile:
        white_val = up_quantile
    else:
        white_val = down_quantile

    # Get mse between smoothed photodiode and kernel
    # Since the photodiode is black most of the time, because of normalization
    # the black value should be near zero.
    black_val = 0.
    white_duration = 60  # Photodiode flashes are ~60ms long white steps.
    black_duration = 50
    template = np.concatenate([
        black_val * np.ones(int(black_duration)),
        white_val * np.ones(int(white_duration)),
        black_val * np.ones(int(black_duration)),
    ])
    sliding = np.lib.stride_tricks.sliding_window_view(
        photodiode_vals_smoothed, len(template))
    mse = np.mean(np.square(sliding - template), axis=1)
    mse = np.concatenate([np.inf * np.ones(black_duration + white_duration), mse])

    # Compute times where mse has a local minimum, i.e. where photodiode is
    # similar to the kernel.
    mean_mse = np.mean(mse[np.isfinite(mse)])
    mse_threshold = 0.5 * mean_mse  # Empirically, this threshold works well.
    spikes = 2 * (mse < mse_threshold) - 1
    start_spikes = np.argwhere(np.convolve(spikes, [1, -1]) == 2.)[:, 0]
    spike_peaks = [i + np.argmin(mse[i: i + 100]) for i in start_spikes]
    photodiode_change_inds = np.array(spike_peaks)
    photodiode_change_times = np.array(
        [photodiode_times[i] for i in photodiode_change_inds])

    # Only take photodiode flashes that happen after the first trial start, in
    # case rig door opening or something caused a flash before the task started.
    good_inds = photodiode_change_times > session_start_time
    photodiode_change_times = photodiode_change_times[good_inds]

    # Plot photodiode values around change points if necessary
    if plot:
        _, axes = plt.subplots(1, 3, figsize=(18, 6))
        for i in photodiode_change_inds[:10]:
            axes[0].plot(photodiode_vals[i - 100: i + 100])
            axes[1].plot(photodiode_vals_smoothed[i - 100: i + 100])
            axes[2].plot(mse[i - 100: i + 100])
        plt.subplots()
        plt.plot(template)
        _, axes = plt.subplots(1, 2, figsize=(14, 7))
        axes[0].plot(photodiode_vals_smoothed[-30000:])
        axes[1].plot(mse[-30000:])
        plt.show()

    return photodiode_change_times


def get_photodiode_delays(photodiode_vals,
                          photodiode_times,
                          trial_start_on_times,
                          trial_start_off_times):
    """Compute photodiode delays per trial.

    The photodiode in the task follows the trial_start sync variable, which is
    usually 0 but steps to 1 for 50ms at the beginning of every trial.
    
    Args:
        photodiode_vals: Numpy array of photodiode values.
        photodiode_times: Numpy array of photodiode times in seconds for the
            values in photodiode_vals. Same length as photodiode_vals.
        trial_start_on_times: Numpy array of times when trial_start sync
            variable turns on form 0 to 1.
        trial_start_on_times: Numpy array of times when trial_start sync
            variable turns off from 1 to 0.

    Returns:
        photodiode_delays: Numpy array of photodiode delays in seconds, per
            trial. NaN values for trials where a corresponding photodiode change
            time was not found.
    """

    photodiode_change_times = get_photodiode_flashes(
        photodiode_vals, photodiode_times, trial_start_on_times[0], plot=False)
    
    # Sometimes the photodiode signal was lost, e.g. for Open Ephys for monkey
    # E for many sessions. In this case, we return NaNs. The photodiode delays
    # are computer from recordings on other devices (e.g. spikeglx and/or
    # MWorks) in ../compute_trials/aggregate_trials.py.
    if len(photodiode_change_times) == 0:
        return np.nan * np.ones(len(trial_start_off_times))

    trial_p_change_inds = np.array([
        np.argmin(np.square(photodiode_change_times - t))
        for t in trial_start_off_times
    ])

    trial_p_change_times = photodiode_change_times[trial_p_change_inds]
    sync_photodiode_error = np.abs(trial_start_off_times - trial_p_change_times)
    
    # If photodiode change time is off by more than 0.1 seconds, something went
    # wrong and we use NaN for the photodiode delay.
    error_too_high = sync_photodiode_error > 0.1
    error_inds = np.argwhere(error_too_high)[:, 0]
    trial_p_change_times[error_inds] = np.nan

    # If more than half of trials had an error, return all NaNs and raise
    # warning.
    if np.mean(error_too_high) > 0.5:
        warnings.warn(
            'Photodiode did not work. Using NaNs for photodiode_delays.'
        )
        photodiode_delays = np.nan * np.ones_like(trial_p_change_inds)
    
    photodiode_delays = trial_p_change_times - trial_start_off_times

    return photodiode_delays
