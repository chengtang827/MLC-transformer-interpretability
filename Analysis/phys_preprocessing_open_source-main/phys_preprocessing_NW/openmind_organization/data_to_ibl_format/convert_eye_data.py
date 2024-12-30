"""Convert eye data to IBL format."""

import os
import constants


def _process_session(eye_data_source_dir, behavior_target_dir):

    # Copy eye horizontal component
    bash_command_copy_eye_h_values = 'cp {} {}'.format(
        os.path.join(eye_data_source_dir, 'eye_h_calibrated_values.npy'),
        os.path.join(behavior_target_dir, 'eye.h.values.npy')
    )
    print('Running bash command {}'.format(bash_command_copy_eye_h_values))
    os.system(bash_command_copy_eye_h_values)
    bash_command_copy_eye_h_times = 'cp {} {}'.format(
        os.path.join(eye_data_source_dir, 'eye_h_calibrated_times.npy'),
        os.path.join(behavior_target_dir, 'eye.h.times.npy')
    )
    print('Running bash command {}'.format(bash_command_copy_eye_h_times))
    os.system(bash_command_copy_eye_h_times)

    # Copy eye vertical component
    bash_command_copy_eye_v_values = 'cp {} {}'.format(
        os.path.join(eye_data_source_dir, 'eye_v_calibrated_values.npy'),
        os.path.join(behavior_target_dir, 'eye.v.values.npy')
    )
    print('Running bash command {}'.format(bash_command_copy_eye_v_values))
    os.system(bash_command_copy_eye_v_values)
    bash_command_copy_eye_v_times = 'cp {} {}'.format(
        os.path.join(eye_data_source_dir, 'eye_v_calibrated_times.npy'),
        os.path.join(behavior_target_dir, 'eye.v.times.npy')
    )
    print('Running bash command {}'.format(bash_command_copy_eye_v_times))
    os.system(bash_command_copy_eye_v_times)

    # Copy pupil size
    bash_command_copy_eye_pupil_values = 'cp {} {}'.format(
        os.path.join(eye_data_source_dir, 'pupil_size_r_values.npy'),
        os.path.join(behavior_target_dir, 'eye.pupil.values.npy')
    )
    print('Running bash command {}'.format(bash_command_copy_eye_pupil_values))
    os.system(bash_command_copy_eye_pupil_values)
    bash_command_copy_eye_pupil_times = 'cp {} {}'.format(
        os.path.join(eye_data_source_dir, 'pupil_size_r_times.npy'),
        os.path.join(behavior_target_dir, 'eye.pupil.times.npy')
    )
    print('Running bash command {}'.format(bash_command_copy_eye_pupil_times))
    os.system(bash_command_copy_eye_pupil_times)
    
    return


def main():
    """Convert eye data to IBL format."""

    for monkey in os.listdir(constants.SOURCE_BASE_DIR):
        monkey_source_dir = os.path.join(constants.SOURCE_BASE_DIR, monkey)
        monkey_target_dir = os.path.join(
            constants.TARGET_BASE_DIR, constants.MONKEY_TO_ID[monkey])
        if not os.path.exists(monkey_target_dir):
            os.makedirs(monkey_target_dir)
        for session_date in os.listdir(monkey_source_dir):
            eye_data_source_dir = os.path.join(
                monkey_source_dir, session_date, 'eye_data')
            behavior_target_dir = os.path.join(
                monkey_target_dir, session_date, '001', 'behavior')

            if not os.path.exists(behavior_target_dir):
                os.makedirs(behavior_target_dir)

            _process_session(eye_data_source_dir, behavior_target_dir)


if __name__ == "__main__":
    main()

