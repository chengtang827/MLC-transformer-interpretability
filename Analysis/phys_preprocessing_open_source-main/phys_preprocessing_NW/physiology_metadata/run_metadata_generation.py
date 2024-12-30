"""Generate and save physiology metadata.

This was run in python on openmind.
"""

import os
import sys
import phys_metadata_perle
import phys_metadata_elgar

_OM2_BASE_DIR = '/om2/user/nwatters/multi_prediction'
# _OM2_BASE_DIR = '/Users/nicholaswatters/Desktop/grad_school/research/mehrdad/multi_prediction/phys'


def main(monkey):
    """Compute and write physiology metadata."""

    # Get session metadata for monkey
    if monkey == 'Perle':
        sessions = phys_metadata_perle.PERLE
    elif monkey == 'Elgar':
        sessions = phys_metadata_elgar.ELGAR
    else:
        raise ValueError(f'Invalid monkey {monkey}')

    # Iterate through sessions and write metadata for each
    for session_date, session in sessions.items():
        write_dir = os.path.join(
            _OM2_BASE_DIR, 'phys_data', monkey, session_date)
        if not os.path.exists(write_dir):
            os.makedirs(write_dir)
        write_path = os.path.join(write_dir, 'physiology_metadata.json')
        print(f'Writing session_date metadata to {write_path}')
        metadata_string = str(session)
        with open(write_path, 'w') as f:
            f.write(metadata_string)
    
    print('DONE')


if __name__ == "__main__":
    monkey = sys.argv[1]
    main(monkey)

