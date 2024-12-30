"""Compute and write OpenEphys trials."""

import json
import os
import sys
import phys_trials_calculator


def main():
    """Compute and write timing metadata for each trial.

    This timing metadata is a list of dictionaries, one for each valid trial. It
    is written to a file $write_dir/open_ephys_trials.
    
    Args:
        open_ephys_events_dir: String. Full path to open ephys sync signal
            events directory.
        write_dir: String. Full path to directory to write trials.
    """

    open_ephys_events_dir = sys.argv[1]
    print(f'open_ephys_events_dir: {open_ephys_events_dir}')
    write_dir = sys.argv[2]
    print(f'write_dir: {write_dir}')

    # open_ephys_events_dir = (
    #     '/Users/nicholaswatters/Desktop/grad_school/research/mehrdad/'
    #     'multi_prediction/phys/phys_data/Perle/2022-05-03/trial_structure/'
    #     'sync_events/open_ephys'
    # )

    # open_ephys_events_dir = (
    #     '/Users/nicholaswatters/Desktop/grad_school/research/mehrdad/'
    #     'multi_prediction/phys/phys_data/Elgar/2022-09-05/trial_structure/'
    #     'sync_events/open_ephys'
    # )

    trials = phys_trials_calculator.get_trials_from_events_dir(
        open_ephys_events_dir)

    # import pdb; pdb.set_trace()

    print('SAVING OPEN EPHYS TRIALS')

    write_path = os.path.join(write_dir, 'open_ephys_trials')
    print(f'write_path: {write_path}')
    json.dump(trials, open(write_path, 'w'))

    print('FINISHED compute_open_ephys_trials.py')

    return


if __name__ == "__main__":
    main()
