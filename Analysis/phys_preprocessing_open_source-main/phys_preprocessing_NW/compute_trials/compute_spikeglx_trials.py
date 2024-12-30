"""Compute and write SpikeGLX trials."""

from ibllib.io import spikeglx
import json
import os
import numpy as np
import sys
import phys_trials_calculator

sys.path.append('../utils')
import digital_to_on_off


def _get_metadata(spikeglx_raw_dir, stream_name):
    """Read nidaq metadata to get sample rate."""
    meta_file = [
        x for x in os.listdir(spikeglx_raw_dir) if stream_name + '.meta' in x
    ]
    if len(meta_file) == 0:
        raise ValueError(f'Did not find a .meta file in {spikeglx_raw_dir}')
    else:
        meta_file = os.path.join(spikeglx_raw_dir, meta_file[0])
    with open(meta_file) as f:
        meta_lines = f.readlines()
    meta = {
        x.split('=', 1)[0]: x.split('=', 1)[1][:-1]
        for x in meta_lines
    }
    return meta


def _get_nidaq_sample_rate(spikeglx_raw_dir):
    """Read nidaq metadata to get sample rate."""
    nidaq_meta = _get_metadata(spikeglx_raw_dir, 'nidq')
    sample_rate = float(nidaq_meta['niSampRate'])
    
    return sample_rate


def _get_imec_relative_delay_to_nidaq(spikeglx_raw_dir):
    """Read nidaq metadata to get sample rate."""
    nidaq_meta = _get_metadata(spikeglx_raw_dir, 'nidq')
    imec_meta = _get_metadata(spikeglx_raw_dir, 'ap')
    nidaq_sample_rate = float(nidaq_meta['niSampRate'])
    imec_sample_rate = float(imec_meta['imSampRate'])
    nidaq_first_sample = float(nidaq_meta['firstSample'])
    imec_first_sample = float(imec_meta['firstSample'])

    # # For session Elgar/2022-10-05, computer from lf.meta
    # imec_sample_rate = 30000.348603
    # imec_first_sample = 387650328

    nidaq_start = float(nidaq_first_sample) / nidaq_sample_rate
    imec_start = float(imec_first_sample) / imec_sample_rate
    relative_start = imec_start - nidaq_start
    
    return relative_start


def _get_nidaq_reader(spikeglx_raw_dir):
    """Get nidaq reader."""
    nidaq_bin_file = [
        x for x in os.listdir(spikeglx_raw_dir) if 'nidq.bin' in x
    ]
    if len(nidaq_bin_file) != 1:
        raise ValueError(
            f'Did not find a single nidaq bin file, instead found '
            f'{nidaq_bin_file}'
        )
    nidaq_bin_file = nidaq_bin_file[0]
    reader = spikeglx.Reader(os.path.join(spikeglx_raw_dir, nidaq_bin_file))
    
    return reader


def _write_sync_events(spikeglx_raw_dir, sync_events_dir):
    """Read sync events from spikeglx logs and write to sync_events_dir."""

    # Read nidaq metadata to get sample rate and relative imec delay
    print('Reading sample rate')
    sample_rate = _get_nidaq_sample_rate(spikeglx_raw_dir)
    imec_relative_delay_to_nidaq = _get_imec_relative_delay_to_nidaq(
        spikeglx_raw_dir)
    
    # Write relative imec delay
    relative_delay_path = os.path.join(
        sync_events_dir, 'imec_relative_delay_to_nidaq')
    print(
        f'Writing relative_delay {imec_relative_delay_to_nidaq} to '
        f'{relative_delay_path}'
    )
    json.dump(imec_relative_delay_to_nidaq, open(relative_delay_path, 'w'))

    ############################################################################
    ####  Read nidaq data analog and digital data
    ############################################################################

    print('Getting nidaq reader')
    reader = _get_nidaq_reader(spikeglx_raw_dir)
    # Extract data with stride to read at ~1ms
    stride = int(np.round(sample_rate / 1000))
    print('Reading analog and digital nidaq data')
    analog_raw, digital_raw = reader.read(nsel=slice(0, reader.ns, stride))
    read_sample_rate = sample_rate / float(stride)

    # Save raw analog and digital data
    print('Extracting analog and digital raw data')
    write_path_analog_raw = os.path.join(sync_events_dir, 'analog_raw.csv')
    print(f'Writing analog raw data to {write_path_analog_raw}')
    np.savetxt(write_path_analog_raw, analog_raw, delimiter=',')
    write_path_digital_raw = os.path.join(sync_events_dir, 'digital_raw.csv')
    print(f'Writing digital raw data to {write_path_digital_raw}')
    np.savetxt(write_path_digital_raw, digital_raw, delimiter=',')

    # Write read_sample_rate
    read_sample_rate_path = os.path.join(sync_events_dir, 'sample_rate')
    print(
        f'Writing read_sample_rate {read_sample_rate} to '
        f'{read_sample_rate_path}'
    )
    json.dump(read_sample_rate, open(read_sample_rate_path, 'w'))

    ############################################################################
    ####  Extract and write nidaq data channels of interest
    ############################################################################

    # Neuropixel internally generated sync variables
    print('Extracting analog and digital sync pulse')
    analog_np_sync = analog_raw[:, 0]
    digital_np_sync = digital_raw[:, 16]
    write_path_analog_np_sync = os.path.join(
        sync_events_dir, 'analog_np_sync_raw.csv')
    print(f'Writing analog sync pulse to {write_path_analog_np_sync}')
    np.savetxt(write_path_analog_np_sync, analog_np_sync, delimiter=',')
    write_path_digital_np_sync = os.path.join(
        sync_events_dir, 'digital_np_sync_raw.csv')
    print(f'Writing digital sync pulse to {write_path_digital_np_sync}')
    np.savetxt(write_path_digital_np_sync, digital_np_sync, delimiter=',')

    # Task sync variables
    print('Extracting task sync variables')
    if 'Perle' in sync_events_dir:
        photodiode = analog_raw[:, 1]
    elif 'Elgar' in sync_events_dir:
        photodiode = analog_raw[:, 8]
    else:
        raise ValueError(f'Invalid sync_events_dir {sync_events_dir}')
        
    digital_sync_vars = {
        'sync_trial_start': digital_raw[:, 0],
        'sync_phase': digital_raw[:, 1],
        'sync_trial_num_zero': digital_raw[:, 2],
        'sync_trial_num_one': digital_raw[:, 3],
    }

    write_path_photodiode = os.path.join(sync_events_dir, 'photodiode_raw.csv')
    print(f'Writing photodiode to {write_path_photodiode}')
    np.savetxt(write_path_photodiode, photodiode, delimiter=',')

    for name, x in digital_sync_vars.items():
        x_on, x_off = digital_to_on_off.digital_to_on_off(x)
        x_on = x_on / read_sample_rate
        x_off = x_off / read_sample_rate
        write_path_on = os.path.join(sync_events_dir, name + '_on.csv')
        print(f'Writing {name}_on to {write_path_on}')
        np.savetxt(write_path_on, x_on, delimiter=',')
        write_path_off = os.path.join(sync_events_dir, name + '_off.csv')
        print(f'Writing {name}_off to {write_path_off}')
        np.savetxt(write_path_off, x_off, delimiter=',')

    print('Done extracting and saving sync variables')

    return imec_relative_delay_to_nidaq


def _fix_delay(trial, imec_relative_delay_to_nidaq):
    trial['t_start'] = trial['t_start'] + imec_relative_delay_to_nidaq
    trial['t_end'] = trial['t_end'] + imec_relative_delay_to_nidaq
    return trial


def main():
    """Compute and write timing metadata for each trial.

    This timing metadata is a list of dictionaries, one for each valid trial. It
    is written to a file $write_dir/spikeglx_trials.
    
    Args:
        spikeglx_raw_dir: String. Full path to spikeglx raw data directory.
        write_dir: String. Full path to directory to write trials.
        sync_events_dir: String. Full path to directory to write sync events.
    """

    spikeglx_raw_dir = sys.argv[1]
    print(f'spikeglx_raw_dir: {spikeglx_raw_dir}')
    write_dir = sys.argv[2]
    print(f'write_dir: {write_dir}')
    sync_events_dir = sys.argv[3]
    print(f'sync_events_dir: {sync_events_dir}')

    # sync_events_dir = (
    #     '/Users/nicholaswatters/Desktop/grad_school/research/mehrdad/'
    #     'multi_prediction/phys/phys_data/Perle/2022-06-11/'
    #     'trial_structure/sync_events/spikeglx'
    # )

    # sync_events_dir = (
    #     '/Users/nicholaswatters/Desktop/grad_school/research/mehrdad/'
    #     'multi_prediction/phys/phys_data/Elgar/2022-09-05/'
    #     'trial_structure/sync_events/spikeglx'
    # )

    imec_relative_delay_to_nidaq = _write_sync_events(
        spikeglx_raw_dir, sync_events_dir)

    print('GENERATING TRIALS')

    trials = phys_trials_calculator.get_trials_from_events_dir(
        sync_events_dir)

    trials = [_fix_delay(t, imec_relative_delay_to_nidaq) for t in trials]

    print('SAVING SPIKEGLX TRIALS')

    write_path = os.path.join(write_dir, 'spikeglx_trials')
    print(f'write_path: {write_path}')
    json.dump(trials, open(write_path, 'w'))

    print('FINISHED compute_spikeglx_trials.py')

    return


if __name__ == "__main__":
    main()
