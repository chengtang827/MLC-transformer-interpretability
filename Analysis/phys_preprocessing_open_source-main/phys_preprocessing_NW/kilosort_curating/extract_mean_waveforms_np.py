"""Extract mean waveforms for neuropixel.

This script was run on openmind in an srun interactive session.
"""

import glob
import os
import utils
import sys


def _run_session(session, probe_name, kilosort_run_name):
    """Compute and write spike times per cluster."""

    ############################################################################
    ####  Find Data Paths
    ############################################################################

    print('\n\n')
    print(f'session: {session}')
    print('probe_name: {}'.format(probe_name))
    print('kilosort_run_name: {}'.format(kilosort_run_name))

    om2_session_dir = os.path.join(utils.OM2_BASE_DIR, 'phys_data', session)
    print('om2_session_dir: {}'.format(om2_session_dir))
    kilosort_dir = os.path.join(
        om2_session_dir, 'spike_sorting', probe_name, kilosort_run_name
    )
    if not os.path.exists(kilosort_dir):
        print('No kilosort directory, exiting')
        return
    print('kilosort_dir: {}'.format(kilosort_dir))
    om4_session_dir = os.path.join(utils.OM4_BASE_DIR, 'phys_data', session)

    om2_raw_data_path = os.path.join(om2_session_dir, 'raw_data')
    om2_raw_data_path_spikeglx = os.path.join(om2_raw_data_path, 'spikeglx')
    print('om2_raw_data_path_spikeglx: {}'.format(om2_raw_data_path_spikeglx))
    if not os.path.exists(om2_raw_data_path_spikeglx):
        os.makedirs(om2_raw_data_path_spikeglx)

    # Look for raw data file in om2
    raw_data_path = os.listdir(om2_raw_data_path_spikeglx)
    if len(raw_data_path) == 0:
        print('Copying raw data from om4')
        # Copy data from om4
        om4_session_dir = os.path.join(utils.OM4_BASE_DIR, 'phys_data', session)
        print('om4_session_dir: {}'.format(om4_session_dir))
        if 'Perle' in session:
            spikeglx_task_dir = os.path.join(
                om4_session_dir, 'raw_data/spikeglx/**perle_g**')
        elif 'Elgar' in session:
            spikeglx_task_dir = os.path.join(
                om4_session_dir, 'raw_data/spikeglx/*task*')
        else:
            raise ValueError(
                f'session {session} has neither Perle nor Elgar in the '
                'name'
            )
        raw_data_path = (
            glob.glob(spikeglx_task_dir + '/*.ap.bin') + 
            glob.glob(spikeglx_task_dir + '/*/*.ap.bin')
        )
        if not raw_data_path:
            print('Found no raw data, exiting')
            return
        raw_data_path = raw_data_path[0]
        print('raw_data_path: {}'.format(raw_data_path))
        bash_command = (
            'cp {} {}/'.format(raw_data_path, om2_raw_data_path_spikeglx)
        )
        print('Running bash command {}'.format(bash_command))
        os.system(bash_command)

    raw_data_file = os.path.join(
        om2_raw_data_path_spikeglx, os.listdir(om2_raw_data_path_spikeglx)[0])
    print('raw_data_file: {}'.format(raw_data_file))

    # Compute and save mean waveforms
    utils.compute_mean_waveforms(kilosort_dir, raw_data_file)

    bash_command_remove_raw_data = 'rm -r {}'.format(om2_raw_data_path)
    print('Running bash command {}'.format(bash_command_remove_raw_data))
    os.system(bash_command_remove_raw_data)
    print('done with session')
    
    return


def main(monkey):
    sessions = [
        # POPULATE THIS WITH A LIST OF SESSION STRINGS, e.g. 2022-mm-dd
        '2022-09-21',
    ]
    sessions = [monkey + '/' + s for s in sessions]
    kilosort_run_name = 'ks_3_output_v2'
    probe_name = 'np_0'

    for session in sessions:
        _run_session(session, probe_name, kilosort_run_name)

    print('DONE')


if __name__ == "__main__":
    monkey = sys.argv[1]
    if monkey not in ['Perle', 'Elgar']:
        raise ValueError(f'Invalid monkey {monkey}')
    main(monkey=monkey)

