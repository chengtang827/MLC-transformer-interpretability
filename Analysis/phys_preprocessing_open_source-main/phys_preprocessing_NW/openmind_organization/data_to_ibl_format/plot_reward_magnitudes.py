"""Sanity check to make sure reward durations agree with response errors."""

import constants
import json
from matplotlib import pyplot as plt
import os

_WRITE_BASE_DIR = (
    '/om2/user/nwatters/multi_prediction/phys_data/reward_magnitude_plots'
)


def _process_session(session_dir):

    # Extract subject and session date
    subject = session_dir.split('/')[-3]
    session_date = session_dir.split('/')[-2]

    # Load response error and reward duration
    response_error_path = os.path.join(
        session_dir, 'behavior', 'trials.response.error.json')
    response_errors = json.load(open(response_error_path, 'r'))
    reward_duration_path = os.path.join(
        session_dir, 'task', 'trials.reward.duration.json')
    reward_durations = json.load(open(reward_duration_path, 'r'))

    # Sanity check
    if not len(response_errors) == len(reward_durations):
        raise ValueError(
            f'len(response_errors) = {len(response_errors)} but '
            f'len(reward_durations) = {len(reward_durations)}'
        )

    # Plot reward duration vs response error
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(response_errors, reward_durations)
    ax.set_xlim(0., 0.5)
    ax.set_title(f'{subject}/{session_date}')
    ax.set_xlabel('response error')
    ax.set_ylabel('reward duration')

    # Save figure
    if not os.path.exists(_WRITE_BASE_DIR):
        os.makedirs(_WRITE_BASE_DIR)
    subject_path = os.path.join(_WRITE_BASE_DIR, subject)
    if not os.path.exists(subject_path):
        os.makedirs(subject_path)
    fig_path = os.path.join(subject_path, session_date)
    print(f'fig_path: {fig_path}')
    fig.savefig(fig_path)


def main():
    """Sanity check to make sure reward durations agree with response errors."""

    for subject in os.listdir(constants.TARGET_BASE_DIR):
        subject_dir = os.path.join(constants.TARGET_BASE_DIR, subject)
        for session_date in os.listdir(subject_dir):
            session_dir = os.path.join(subject_dir, session_date, '001')
            _process_session(session_dir)


if __name__ == "__main__":
    main()

