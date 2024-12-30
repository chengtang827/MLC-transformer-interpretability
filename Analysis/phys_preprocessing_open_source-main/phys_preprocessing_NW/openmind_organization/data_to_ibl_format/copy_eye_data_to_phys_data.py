"""Convert eye data to IBL format."""

import constants
import os

_BASE_DIR = '/om4/group/jazlab/nwatters/multi_prediction/data_open_source/Subjects'
_SUBJECT_DICT = {
    '00': 'monkey0',
    '01': 'monkey1',
}


def main():
    """Convert eye data to IBL format."""

    for subject_source, subject_target in _SUBJECT_DICT.items():
        subject_source_dir = os.path.join(_BASE_DIR, subject_source)
        subject_target_dir = os.path.join(_BASE_DIR, subject_target)
        for session_date in os.listdir(subject_source_dir):
            session_source_dir = os.path.join(
                subject_source_dir, session_date, '001')
            session_target_dir = os.path.join(
                subject_target_dir, session_date, '001')
            
            # Remove target behavior dir
            bash_remove_target_behavior_dir = (
                f'rm -r {session_target_dir}/behavior'
            )
            print(f'Running bash command {bash_remove_target_behavior_dir}')
            os.system(bash_remove_target_behavior_dir)
            # Make target behavior dir
            bash_make_target_behavior_dir = (
                f'mkdir {session_target_dir}/behavior'
            )
            print(f'Running bash command {bash_make_target_behavior_dir}')
            os.system(bash_make_target_behavior_dir)
            # Copy eye data to target behavior dir
            bash_copy_eye_data = (
                f'cp {session_source_dir}/behavior/eye.* '
                f'{session_target_dir}/behavior/'
            )
            print(f'Running bash command {bash_copy_eye_data}')
            os.system(bash_copy_eye_data)


if __name__ == "__main__":
    main()

