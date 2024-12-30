"""Convert eye data to IBL format."""

import constants
import json
import numpy as np
import os


def _process_session(trials_path, task_target_dir, behavior_target_dir):

    trials = json.load(open(trials_path, 'r'))

    ############################################################################
    #### TIME START
    ############################################################################

    t_start = [
        t['t_start_common'] + t['mworks_photodiode_delay'] for t in trials
    ]
    t_start_path = os.path.join(
        task_target_dir, 'trials.start_times.json')
    json.dump(t_start, open(t_start_path, 'w'))

    ############################################################################
    #### PHASE TIMES
    ############################################################################
    
    # Phase times are [
    #   visible onset,
    #   delay onset,
    #   cue onset,
    #   response onset,
    #   reveal onset,
    #   ITI onset,
    # ]
    # If fixation was broken, this list is truncated after the last event before
    # fixation was broken. The broken fixation ITI is 1 second.
    def _get_rel_phase_times(t):
        rel_phase_times = t['mworks_relative_phase_times']
        if len(rel_phase_times) > 6:
            rel_phase_times = rel_phase_times[:6]
        return rel_phase_times
    rel_phase_times = [_get_rel_phase_times(t) for t in trials]
    rel_phase_times_path = os.path.join(
        task_target_dir, 'trials.relative_phase_times.json')
    json.dump(rel_phase_times, open(rel_phase_times_path, 'w'))

    ############################################################################
    #### STIMULUS INITIALIZATION
    ############################################################################
    
    # Write stimulus initialization for each trial. In moving object conditions,
    # objects begin moving as soon as they become visible in the visible phase.
    stimuli_init = [t['moog_data']['prey_init'] for t in trials]
    stimuli_init_path = os.path.join(
        task_target_dir, 'trials.stimuli_init.json')
    json.dump(stimuli_init, open(stimuli_init_path, 'w'))
    
    ############################################################################
    #### MOOG STEPS PER PHASE
    ############################################################################
    
    def _get_steps_per_phase(moog_data):
        steps_per_phase = {
            'visible_phase': moog_data['visible_steps'],
            'delay_phase': moog_data['delay_steps'],
            'cue_phase': moog_data['delay_steps'],
        }
        return steps_per_phase
    steps_per_phase = [_get_steps_per_phase(t['moog_data']) for t in trials]
    steps_per_phase_path = os.path.join(
        task_target_dir, 'trials.steps_per_phase.json')
    json.dump(steps_per_phase, open(steps_per_phase_path, 'w'))

    ############################################################################
    #### WHETHER OBJECTS ARE BLANKS
    ############################################################################
    
    object_blanks = [t['moog_data']['blank'] for t in trials]
    object_blanks_path = os.path.join(
        task_target_dir, 'trials.object_blanks.json')
    json.dump(object_blanks, open(object_blanks_path, 'w'))

    ############################################################################
    #### PHASE AT EVERY MOOG STEP
    ############################################################################
    
    moog_phases = [t['moog_data']['phase_list'] for t in trials]
    moog_phases_path = os.path.join(
        task_target_dir, 'trials.phase_per_moog_step.json')
    json.dump(moog_phases, open(moog_phases_path, 'w'))

    ############################################################################
    #### OBJECT STATE AT EVERY STEP IN EACH MOOG PHASE
    ############################################################################
    
    def _get_object_states(moog_data):
        states_per_step = {
            'visible': moog_data['visible_prey_states'],
            'delay': moog_data['delay_prey_states'],
            'cue': moog_data['cue_prey_states'],
            'response': moog_data['response_prey_states'],
            'reveal': moog_data['reveal_prey_states'],
        }
        return states_per_step
    object_states = [_get_object_states(t['moog_data']) for t in trials]
    object_states_path = os.path.join(
        task_target_dir, 'trials.object_states_per_moog_step.json')
    json.dump(object_states, open(object_states_path, 'w'))

    ############################################################################
    #### OBJECT STATE AT EVERY STEP IN EACH MOOG PHASE
    ############################################################################
    
    fixation_cross_scale = [
        t['moog_data']['fixation_cross_scale'] for t in trials
    ]
    fixation_cross_scale_path = os.path.join(
        task_target_dir, 'trials.fixation_cross_scale_per_moog_step.json')
    json.dump(fixation_cross_scale, open(fixation_cross_scale_path, 'w'))

    ############################################################################
    #### RESPONSE
    ############################################################################
    
    response = [t['moog_data']['response'] for t in trials]
    response_path = os.path.join(
        behavior_target_dir, 'trials.response.location.json')
    json.dump(response, open(response_path, 'w'))

    ############################################################################
    #### RESPONSE ERROR
    ############################################################################
    
    response_error = [t['moog_data']['error'] for t in trials]
    response_error_path = os.path.join(
        behavior_target_dir, 'trials.response.error.json')
    json.dump(response_error, open(response_error_path, 'w'))

    ############################################################################
    #### RESPONSE OBJECT
    ############################################################################
    
    response_object = [t['moog_data']['response_prey'] for t in trials]
    response_object_path = os.path.join(
        behavior_target_dir, 'trials.response.object.json')
    json.dump(response_object, open(response_object_path, 'w'))

    ############################################################################
    #### BROKE FIXATION
    ############################################################################
    
    broke_fixation = [t['moog_data']['broke_fixation'] for t in trials]
    broke_fixation_path = os.path.join(
        behavior_target_dir, 'trials.broke_fixation.json')
    json.dump(broke_fixation, open(broke_fixation_path, 'w'))

    ############################################################################
    #### REWARD
    ############################################################################
    
    reward_time = [t['reward_time'] for t in trials]
    reward_time_path = os.path.join(task_target_dir, 'trials.reward.time.json')
    json.dump(reward_time, open(reward_time_path, 'w'))
    reward_duration = [t['reward_duration'] for t in trials]
    reward_duration_path = os.path.join(
        task_target_dir, 'trials.reward.duration.json')
    json.dump(reward_duration, open(reward_duration_path, 'w'))
    
    return


def main():
    """Convert task data to IBL format."""

    for monkey in os.listdir(constants.SOURCE_BASE_DIR):
        monkey_source_dir = os.path.join(constants.SOURCE_BASE_DIR, monkey)
        monkey_target_dir = os.path.join(
            constants.TARGET_BASE_DIR, constants.MONKEY_TO_ID[monkey])
        if not os.path.exists(monkey_target_dir):
            os.makedirs(monkey_target_dir)
        for session_date in os.listdir(monkey_source_dir):
            trials_path = os.path.join(
                monkey_source_dir,
                session_date,
                'trial_structure',
                'trials_curated',
            )
            print(f'monkey: {monkey}, session_date: {session_date}')
            behavior_target_dir = os.path.join(
                monkey_target_dir, session_date, '001', 'behavior')
            if not os.path.exists(behavior_target_dir):
                os.makedirs(behavior_target_dir)
            task_target_dir = os.path.join(
                monkey_target_dir, session_date, '001', 'task')
            if not os.path.exists(task_target_dir):
                os.makedirs(task_target_dir)
            _process_session(trials_path, task_target_dir, behavior_target_dir)


if __name__ == "__main__":
    main()

