"""Copy sync pulses to data_processed."""

import os

_RAW_DATA_DIR = '/om4/group/jazlab/nwatters/multi_prediction/phys_data'
_PROCESSED_DATA_DIR = '/om4/group/jazlab/nwatters/multi_prediction/data_processed'


def _copy_mworks_sync_pulses(raw_session_dir, write_dir):
    print(f'Copying mworks sync pulses to {write_dir}')

    # Create write directory
    write_dir = os.path.join(write_dir, 'mworks')
    if not os.path.exists(write_dir):
        os.makedirs(write_dir)
    
    # Iterate through raw mworks sessions, copying sync pulses
    mworks_sync_dir = os.path.join(
        raw_session_dir, 'trial_structure', 'sync_events', 'mworks')
    mworks_sessions = sorted(os.listdir(mworks_sync_dir))
    for i, sess in enumerate(mworks_sessions):
        source = os.path.join(mworks_sync_dir, sess, 'trial_start_off_times')
        target_dir = os.path.join(
            write_dir, f'mworks_session_{i}')
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        target = os.path.join(target_dir, 'trial_start_off_times')
        bash_command = 'cp {} {}'.format(source, target)
        print('Running bash command {}'.format(bash_command))
        os.system(bash_command)


def _copy_open_ephys_sync_pulses(raw_session_dir, write_dir):
    print(f'Copying open_ephys sync pulses to {write_dir}')

    open_ephys_sync_dir = os.path.join(
        raw_session_dir, 'trial_structure', 'sync_events', 'open_ephys')
    if not os.path.exists(open_ephys_sync_dir):
        return

    # Create write directory
    write_dir = os.path.join(write_dir, 'open_ephys')
    if not os.path.exists(write_dir):
        os.makedirs(write_dir)
    
    # Copy sync pulses
    source = os.path.join(open_ephys_sync_dir, 'sync_trial_start_off.csv')
    target = os.path.join(write_dir, 'sync_trial_start_off.csv')
    bash_command = 'cp {} {}'.format(source, target)
    print('Running bash command {}'.format(bash_command))
    os.system(bash_command)


def _copy_spikeglx_sync_pulses(raw_session_dir, write_dir):
    print(f'Copying spikeglx sync pulses to {write_dir}')

    spikeglx_sync_dir = os.path.join(
        raw_session_dir, 'trial_structure', 'sync_events', 'spikeglx')
    if not os.path.exists(spikeglx_sync_dir):
        return

    # Create write directory
    write_dir = os.path.join(write_dir, 'spikeglx')
    if not os.path.exists(write_dir):
        os.makedirs(write_dir)
    
    # Copy sync pulses
    source = os.path.join(spikeglx_sync_dir, 'sync_trial_start_off.csv')
    target = os.path.join(write_dir, 'sync_trial_start_off.csv')
    bash_command = 'cp {} {}'.format(source, target)
    print('Running bash command {}'.format(bash_command))
    os.system(bash_command)


def _process_session(raw_session_dir, processed_session_dir):

    # Establish write directory
    write_dir = os.path.join(processed_session_dir, 'sync_pulses')
    if not os.path.exists(write_dir):
        os.makedirs(write_dir)

    # Copy mworks sync pulses
    _copy_mworks_sync_pulses(raw_session_dir, write_dir)

    # Copy open_ephys sync pulses
    _copy_open_ephys_sync_pulses(raw_session_dir, write_dir)

    # Copy spikeglx sync pulses
    _copy_spikeglx_sync_pulses(raw_session_dir, write_dir)
    


def main():
    """Copy sync pulses to data_processed."""

    for subject in os.listdir(_PROCESSED_DATA_DIR):
        raw_subject_dir = os.path.join(_RAW_DATA_DIR, subject)
        processed_subject_dir = os.path.join(_PROCESSED_DATA_DIR, subject)
        for session_date in os.listdir(processed_subject_dir):
            raw_session_dir = os.path.join(raw_subject_dir, session_date)
            processed_session_dir = os.path.join(
                processed_subject_dir, session_date)
            _process_session(raw_session_dir, processed_session_dir)


if __name__ == "__main__":
    main()

