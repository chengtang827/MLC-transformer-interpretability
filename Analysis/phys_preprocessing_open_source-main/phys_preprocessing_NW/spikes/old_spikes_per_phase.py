"""Create raster plots."""

import copy
import json
import numpy as np
import os
import shutil
import sys

_OM2_BASE_DIR = '/om2/user/nwatters/multi_prediction'

_PHASES = ('fixation', 'visible', 'delay', 'cue', 'response', 'reveal')


def _generate_spikes_per_phase(behavior_trials_dict,
                               spikes_per_trial,
                               phase_ind,
                               write_path):

    print(f'Generating spikes per phase in {write_path}')

    write_data = []
    for i, spike_data in enumerate(spikes_per_trial):
        metadata, trials_spikes = spike_data
        write_metadata = copy.deepcopy(metadata)
        phase_spikes = []
        failed_trials = 0
        for (trial_num, spikes) in trials_spikes:
            spikes = np.array(spikes)
            if trial_num not in behavior_trials_dict:
                print(trial_num)
                failed_trials += 1
                continue
            rel_phase_times = (
                [0.] + behavior_trials_dict[trial_num]['relative_phase_times'])
            if len(rel_phase_times) <= phase_ind + 1:
                continue
                
            phase_start = rel_phase_times[phase_ind]
            phase_end = rel_phase_times[phase_ind + 1]
            trial_phase_spikes = spikes[
                (spikes < phase_end) * (spikes > phase_start)]
            trial_phase_spikes -= phase_start
            trial_phase_data = {
                'trial_num': trial_num,
                'phase_duration': phase_end - phase_start,
                'spikes': trial_phase_spikes.tolist(),
            }
            phase_spikes.append(trial_phase_data)
        if failed_trials > 5:
            import pdb; pdb.set_trace()
            raise ValueError('Too many failed trials. Something is wrong.')

        write_data.append([write_metadata, phase_spikes])
    
    json.dump(write_data, open(write_path, 'w'))

    return


def main(total_data_path):
    """Compute and write spike times per cluster."""

    ############################################################################
    ####  CREATE FIRING RATE DIRECTORIES
    ############################################################################

    spikes_per_phase_dir = os.path.join(total_data_path, 'spikes_per_phase')
    print(f'spikes_per_phase_dir: {spikes_per_phase_dir}')
    if os.path.exists(spikes_per_phase_dir):
        shutil.rmtree(spikes_per_phase_dir)
    os.makedirs(spikes_per_phase_dir)

    ############################################################################
    ####  LOAD DATA
    ############################################################################

    print('Loading behavior_trials')
    behavior_trials_path = os.path.join(
        total_data_path, 'total_behavior_trials')
    behavior_trials = json.load(open(behavior_trials_path, 'r'))
    behavior_trials_dict = {int(d['trial_num']): d for d in behavior_trials}

    print('Loading spikes per trial')
    spikes_per_trial_path = os.path.join(total_data_path, 'spikes_per_trial')
    for probe_name in os.listdir(spikes_per_trial_path):
        print(f'Probe name: {probe_name}')
        probe_path = os.path.join(spikes_per_trial_path, probe_name)
        spikes_per_trial = json.load(open(probe_path, 'r'))
        spikes_per_phase_dir_probe = os.path.join(
            spikes_per_phase_dir, probe_name)
        os.makedirs(spikes_per_phase_dir_probe)
        for phase_ind, phase in enumerate(_PHASES):
            write_path = os.path.join(spikes_per_phase_dir_probe, phase)
            _generate_spikes_per_phase(
                behavior_trials_dict, spikes_per_trial, phase_ind, write_path)

    print('DONE')
    
    return


if __name__ == "__main__":
    session = sys.argv[1]
    probe_name = sys.argv[2]
    main(session, probe_name)
