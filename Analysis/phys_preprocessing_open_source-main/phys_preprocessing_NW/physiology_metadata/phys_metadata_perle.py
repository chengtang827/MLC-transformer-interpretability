"""Physiology notes."""

import phys_metadata_utils
import phys_params_perle


PERLE = {
    '2022-03-13': phys_metadata_utils.Session(
        phys_params=phys_params_perle.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=6.5,
            from_right=4.,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            through_dura=0,
            first_spikes=0.700,
            end=3.465,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_posterior=4.,
            top_from_right=4.,
            bottom_from_left=7.,
            bottom_from_anterior=7.),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            surface=8.,
            first_spikes=13.,
            end=20.747,
        ),
        notes=(
            '2BAS, LR, 10% random pos. 2-object, old fruits. No mworks rebound '
            'variable.'
        ),
    ),
    '2022-03-14': phys_metadata_utils.Session(
        phys_params=phys_params_perle.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=7.5,
            from_right=6.,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=0.468,
            end=2.240,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_posterior=3.,
            top_from_right=1.5,
            bottom_from_left=8.,
            bottom_from_anterior=8.),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            surface=9.,
            first_spikes=13.,
            end=23.380,
        ),
        notes=(
            '2BAS, LR, 20% random pos. V-probe not many neurons, probably too '
            'medial, missed principal sulcus. Neuropixel not many neurons. No '
            'mworks rebound variable.'
        ),
    ),
    '2022-03-16': phys_metadata_utils.Session(
        phys_params=phys_params_perle.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=7.,
            from_right=5.5,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            through_dura=3.391,
            end=4.850,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_posterior=3.5,
            top_from_right= 2.5,
            bottom_from_left=6.,
            bottom_from_anterior=8.),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            surface=9.,
            first_spikes=12.,
            end=20.020,
        ),
        notes=(
            '2BAS, LR, 20% random pos. Lots of neurons on both probes. Lots of '
            'neurons 2mm before end when lowering V-probe. NIDAQ was not '
            'working.'
        ),
    ),
    '2022-03-17': phys_metadata_utils.Session(
        phys_params=phys_params_perle.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=5.5,
            from_right=3.5,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=0.,
            end=4.500,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_posterior=3.5,
            top_from_right= 2.,
            bottom_from_left=8.,
            bottom_from_anterior=7.),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            surface=14.,
            first_spikes=18.,
            end=27.500,
        ),
        notes=(
            '2BAS, LR, 20% random + 2ABC LR generalization for the first time. '
            'Lots of neuropixel neurons, not many in V-probe. Maybe V-probe '
            'too far medial and missed principal sulcus. NIDAQ was not working.'
        )
    ),
    '2022-03-18': phys_metadata_utils.Session(
        phys_params=phys_params_perle.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=5.,
            from_right=4.5,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            through_dura=0.,
            end=3.204,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_posterior=5.,
            top_from_right=1.,
            bottom_from_left=7.5,
            bottom_from_anterior=7.5),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            surface=11.,
            first_spikes=16.,
            end=24.500,
        ),
        notes=(
            '2BAS, LR, 20% random + 2ABC LR generalization second session. '
            'Lots of V-probe neurons, and pretty clear white matter. Not a lot '
            'of neuropixel neurons. Changed V-probe angle today (more steep '
            'left/right). Seems like the right spot in principal sulcus now. '
            'Some noise on V-probe though. NIDAQ was not working.'
        )
    ),
    '2022-04-07': phys_metadata_utils.Session(
        phys_params=phys_params_perle.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=5.,
            from_right=5.,
        ),
        np_1_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=7.,
            from_right=5.3,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=0.,
            end=0.900,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_posterior=5.,
            top_from_right=1.,
            bottom_from_left=6.5,
            bottom_from_anterior=8.),
        v_probe_1_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_posterior=3.5,
            top_from_right= 1.5,
            bottom_from_left=5.7,
            bottom_from_anterior=9.3),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            surface=20.,
            first_spikes=25.,
            end=28.,
        ),
        notes=(
            'Double neuropixel session. Only used one v-probe, because of '
            'weird data format.'
        ),
    ),
    '2022-04-08': phys_metadata_utils.Session(
        phys_params=phys_params_perle.Params,
        np_1_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=8.5,
            from_right=4.,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=0.,
            end=3.,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_posterior=5.,
            top_from_right=1.,
            bottom_from_left=7.5,
            bottom_from_anterior=8.),
        v_probe_1_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_posterior=3.5,
            top_from_right= 1.5,
            bottom_from_left=6.5,
            bottom_from_anterior=9.5),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            surface=-4.,
            first_spikes=0.,
            end=9.3,
        ),
        notes=(
            '1/2 moving, ABC (same as all recording so far after '
            'generalization to ABC). Broke one neuropixel on insertion, '
            'imec_0. Only used one v-probe, because of weird data format.'
        ),
    ),
    '2022-04-09': phys_metadata_utils.Session(
        phys_params=phys_params_perle.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=5.,
            from_right=3.5,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=0.970,
            end=3.460,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_posterior=3.,
            top_from_right=2.,
            bottom_from_left=6.5,
            bottom_from_anterior=8.),
        v_probe_1_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_posterior=1.5,
            top_from_right= 2.5,
            bottom_from_left=5.5,
            bottom_from_anterior=9.5),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            surface=-4.,
            first_spikes=0.,
            end=10.,
        ),
        notes=(
            'Added discrete stimuli, some conditions (6 single, 3 double) '
            'repeated. 50% 2-object moving, 50% 1-object moving, ABC. This '
            'session, max_delay_steps = 30, which is bad. This may also have '
            'been the case for previous sessions. V-probe might be too deep '
            'and not anterior enough. Only used one v-probe, because of weird '
            'data format.'
        ),
    ),
    '2022-04-11': phys_metadata_utils.Session(
        phys_params=phys_params_perle.Params,
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_posterior=7.,
            top_from_right=1.,
            bottom_from_left=7.5,
            bottom_from_anterior=5.),
        v_probe_1_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_posterior=5.5,
            top_from_right= 2.,
            bottom_from_left=6.5,
            bottom_from_anterior=7.),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            surface=-2.5,
            first_spikes=1.,
            end=9.,
        ),
        notes=(
            'Started final version of 2-moving task today. Fixed OpenEphys ADC '
            'channels (previous sessions had a bad old version and weird '
            'channel map with double V-probe). Also, today reduced 1-object '
            'conditions and increased delay. Broke neuropixel (went too deep '
            'and flex hit top of guide tube). V-probe is pretty far anterior. '
            'No Neuropixel data.'
        ),
    ),
    '2022-04-13': phys_metadata_utils.Session(
        phys_params=phys_params_perle.Params,
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_posterior=4.,
            top_from_right=2.5,
            bottom_from_left=6.5,
            bottom_from_posterior=8.),
        v_probe_1_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_posterior=2.5,
            top_from_right=3.5,
            bottom_from_posterior=6.5,
            bottom_from_left=5.5,
        ),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            surface=-5,
            first_spikes=-1,
            end=3.,
        ),
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=5.5,
            from_right=5.5,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=0.,
            end=1.500,
        ),
        notes=(
            'Good neurons on all probes. V-probe probably in medial wall of '
            'superior arcuate. V-probe posterior guide tube slipped and went '
            'about 2mm into brain.'
        ),
    ),
    '2022-04-24': phys_metadata_utils.Session(
        phys_params=phys_params_perle.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=6.5,
            from_right=4.,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            through_dura=0.,
            end=3.,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_posterior=4.,
            top_from_right=2.5,
            bottom_from_posterior=8.,
            bottom_from_left=6.5,
        ),
        v_probe_1_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_posterior=2.5,
            top_from_right=1.5,
            bottom_from_posterior=6.5,
            bottom_from_left=5.5,
        ),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            surface=-5.,
            first_spikes=-1.,
            end=4.,
        ),
        notes=(
            '2-moving record_moving_1_2_abc. Both guide tubes went ~1mm too '
            'deep. Stopped V-probes in SEF (medial superior arcuate), not yet '
            'in FEF. On probe raising, probe tips were 3.5mm beyond guide tube '
            'ends. Neuropixel lowered with coarse z because did not realize '
            'entered brain because of noise from V-probe.'
        ),
    ),
    '2022-04-25': phys_metadata_utils.Session(
        phys_params=phys_params_perle.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=6.5,
            from_right=3.5,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=0.957,
            end=2.937,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_posterior=4.5,
            top_from_right=1.5,
            bottom_from_posterior=8.,
            bottom_from_left=7.,
        ),
        v_probe_1_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_posterior=2.5,
            top_from_right=2.,
            bottom_from_posterior=6.5,
            bottom_from_left=6.,
        ),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=1.5,
            end=10.,
        ),
        notes=(
            '2-moving record_moving_1_2_abc. Posterior V-probe (v_probe_1) was '
            '0.5mm shallower than anterior (v_probe_0). I think the V-probes '
            'went to lateral side of superior arcuate sulcus. Grid hole may be '
            'too medial and anterior. New V-probe guide tube holder today. '
            'Neuropixel: Great neurons.'
        ),
    ),
    '2022-04-26': phys_metadata_utils.Session(
        phys_params=phys_params_perle.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=8.,
            from_right=3.5,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=2.019,
            end=3.916,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_posterior=3.5,
            top_from_right=2.5,
            bottom_from_posterior=6.5,
            bottom_from_left=6.,
        ),
        v_probe_1_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_posterior=1.5,
            top_from_right=3.,
            bottom_from_posterior=5.,
            bottom_from_left=5.,
        ),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            surface=-1.5,
            first_spikes=2.,
            end=8.6,
        ),
        notes=(
            '2-moving record_moving_1_2_abc. V-probes in medial superior '
            'arcuate (SEF) I think. Neuropixel not as good as yesterday.'
        ),
    ),
    '2022-04-27': phys_metadata_utils.Session(
        phys_params=phys_params_perle.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=7.5,
            from_right=3.5,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            through_dura=2.000,
            end=4.283,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_posterior=3.5,
            top_from_right=2.5,
            bottom_from_posterior=6.5,
            bottom_from_left=6.,
        ),
        v_probe_1_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_posterior=1.5,
            top_from_right=3.,
            bottom_from_posterior=5.,
            bottom_from_left=5.,
        ),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            surface=-0.5,
            through_dura=2.,
            end=8.,
        ),
        notes=(
            'record_moving_1_2_abc. Pretty good neuropixel. V-probe maybe not '
            'lowered deep enough. V-prob upon raising was 8.5mm beyond guide '
            'tube tip, but guide tube was raised 1mm by hand after probe was '
            'through dura. V-probe spikes disappeared after 6mm of raising. '
            'V-probe grid holes same as yesterday.'
        )
    ),
    '2022-04-28': phys_metadata_utils.Session(
        phys_params=phys_params_perle.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=6.5,
            from_right=3.,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=2.050,
            end=4.091,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_posterior=3.5,
            top_from_right=2.5,
            bottom_from_posterior=6.5,
            bottom_from_left=6.,
        ),
        v_probe_1_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_posterior=1.5,
            top_from_right=3.,
            bottom_from_posterior=5.,
            bottom_from_left=5.,
        ),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            through_dura=0.,
            first_spikes=2.,
            end=10.5,
        ),
        notes=(
            'record_moving_1_2_abc. Before today, record node in OpenEphys was '
            'after LFP viewer. Starting today, recorded settling with one sync '
            'pulse from MWorks at the start of settling. V-probe raising: '
            'Probe tip 11mm from guide tube tip. Probably V-probe in lateral '
            'wall of arcuate, tips of probes definitely in cortex with great '
            'neurons. V-probe grid holes same as yesterday.'
        ),
    ),
    '2022-04-29': phys_metadata_utils.Session(
        phys_params=phys_params_perle.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=8.5,
            from_right=2.5,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=0.,
            end=1.550,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_posterior=3.5,
            top_from_right=2.5,
            bottom_from_posterior=6.5,
            bottom_from_left=6.,
        ),
        v_probe_1_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_posterior=1.5,
            top_from_right=3.,
            bottom_from_posterior=5.,
            bottom_from_left=5.,
        ),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            through_dura=0.,
            first_spikes=2.,
            end=11.,
        ),
        notes=(
            'record_moving_1_2_abc. Raising V_probe: probe tips 14mm below '
            'guide tube tips, but guide tubes raised 2mm after dura puncture. '
            'V-probes went deep, definitely in lateral wall of arcuate. '
            'V-probe grid holes same as yesterday. Neuropixel guide tube was '
            'too deep, probe kept bending.'
        ),
    ),
    '2022-04-30': phys_metadata_utils.Session(
        phys_params=phys_params_perle.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=7.3,
            from_right=2.5,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=0.500,
            end=1.869,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_posterior=1.5,
            top_from_right=1.5,
            bottom_from_posterior=6.5,
            bottom_from_left=7.,
        ),
        v_probe_1_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_posterior=-0.5,
            top_from_right=2.,
            bottom_from_posterior=5.,
            bottom_from_left=6.,
        ),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            surface=-3.,
            first_spikes=0.,
            end=6.75,
        ),
        notes=(
            'record_moving_1_2_abc. V-probes probably in medial superior '
            'arcuate sulcus.'
        )
    ),
    '2022-05-01': phys_metadata_utils.Session(
        phys_params=phys_params_perle.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=8.75,
            from_right=5.25,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=0.,
            end=2.175,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_posterior=1.5,
            top_from_right=1.5,
            bottom_from_posterior=6.5,
            bottom_from_left=7.,
        ),
        v_probe_1_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_posterior=-0.5,
            top_from_right=2.,
            bottom_from_posterior=5.,
            bottom_from_left=6.,
        ),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            through_dura=0.,
            first_spikes=0.5,
            end=8.5,
        ),
        notes=(
            'Posterior V-probe guide tube may have been dragged into brain. '
            'Same V-probe holes as yesterday.'
        ),
    ),
    '2022-05-02': phys_metadata_utils.Session(
        phys_params=phys_params_perle.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=7.,
            from_right=3.25,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=2.000,
            end=3.629,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_posterior=2.5,
            top_from_right=2.5,
            bottom_from_posterior=8.,
            bottom_from_left=5.5,
        ),
        v_probe_1_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_posterior=1.,
            top_from_right=3.,
            bottom_from_posterior=6.5,
            bottom_from_left=4.5,
        ),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            surface=-3.5,
            first_spikes=0.,
            end=6.,
        ),
        notes=(
            'record_moving_1_2_abc. Neuropixel not many neurons. V-probe '
            'probably in medial wall of arcuate. Open Ephys dropped signal '
            'mid-session. Ignore V-probes.'
        ),
    ),
    '2022-05-03': phys_metadata_utils.Session(
        phys_params=phys_params_perle.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=4.,
            from_right=4.,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=1.000,
            end=3.206,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_posterior=2.5,
            top_from_right=2.5,
            bottom_from_posterior=8.,
            bottom_from_left=5.5,
        ),
        v_probe_1_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_posterior=1.,
            top_from_right=3.,
            bottom_from_posterior=6.5,
            bottom_from_left=4.5,
        ),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            surface=-4.,
            through_dura=-1.,
            first_spikes=-0.5,
            end=8.,
        ),
        notes=(
            'record_moving_1_2_abc. Accidentally recorded as .bin instead of '
            'OpenEphys format. Neuropixel lots of neurons, pretty far '
            'anterior. V-probe probably in lateral arcuate. Same V-probe holes '
            'as yesterday.'
        ),
    ),
    '2022-05-04': phys_metadata_utils.Session(
        phys_params=phys_params_perle.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=4.5,
            from_right=4.25,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=1.500,
            end=3.440,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_posterior=2.5,
            top_from_right=2.5,
            bottom_from_posterior=8.,
            bottom_from_left=5.5,
        ),
        v_probe_1_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_posterior=1.,
            top_from_right=3.,
            bottom_from_posterior=6.5,
            bottom_from_left=4.5,
        ),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            through_dura=0.,
            first_spikes=1.,
            end=9.5,
        ),
        notes=(
            'V-probe probably in lateral arcuate sulcus. Same V-probe holes as '
            'yesterday.'
        ),
    ),
    '2022-05-05': phys_metadata_utils.Session(
        phys_params=phys_params_perle.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=4.5,
            from_right=3.5,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=2.000,
            end=3.507,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_posterior=2.5,
            top_from_right=2.5,
            bottom_from_posterior=8.,
            bottom_from_left=5.5,
        ),
        v_probe_1_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_posterior=1.,
            top_from_right=3.,
            bottom_from_posterior=6.5,
            bottom_from_left=4.5,
        ),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            through_dura=0.,
            first_spikes=2.5,
            end=12.5,
        ),
        notes=(
            'Neuropixel might be very near bone, since guide tube was hard to '
            'press. Same V-probe holes as yesterday.'
        )
    ),
    '2022-05-06': phys_metadata_utils.Session(
        phys_params=phys_params_perle.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=7.75,
            from_right=3.5,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=1.200,
            end=3.250,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_posterior=6.5,
            top_from_right=1.,
            bottom_from_posterior=7.5,
            bottom_from_left=6.5,
        ),
        v_probe_1_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_posterior=5.5,
            top_from_right=1.,
            bottom_from_posterior=6.,
            bottom_from_left=6.5,
        ),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            through_dura=-1.,
            first_spikes=0.,
            end=7.5,
        ),
        notes=(
            'All V-probe guide tube length is 23mm, top of guide tubes flush '
            'with top of grid. V-probes in lateral arcuate sulcus probably. '
            'New V-probe guide tubes today, with less anterior/posterior angle.'
        ),
    ),
    '2022-05-08': phys_metadata_utils.Session(
        phys_params=phys_params_perle.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=6.5,
            from_right=4.25,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=3.900,
            end=5.677,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_posterior=1.75,
            top_from_right=2.25,
            bottom_from_posterior=7.,
            bottom_from_left=6.,
        ),
        v_probe_1_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_posterior=0.5,
            top_from_right=2.75,
            bottom_from_posterior=5.5,
            bottom_from_left=6.,
        ),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            through_dura=0.,
            first_spikes=2.,
            end=9.25,
        ),
        notes='Open Ephys dropped signal mid-session. Ignore V-probes.',
    ),
    '2022-05-09': phys_metadata_utils.Session(
        phys_params=phys_params_perle.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=3.75,
            from_right=3.5,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            through_dura=0.,
            first_spikes=2.,
            end=3.372,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_posterior=4.5,
            top_from_right=3.,
            bottom_from_posterior=8.,
            bottom_from_left=4.5,
        ),
        v_probe_1_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_posterior=3.,
            top_from_right=3.5,
            bottom_from_posterior=6.5,
            bottom_from_left=5.,
        ),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            through_dura=0.,
            end=8.,
        ),
        notes=(
            '3 moving objects generalization. Note: Ran task, then stopped '
            'after about 40 trials, then re-started MWorks and resumed. '
            'Raising V-probe: probe tip 11mm below tip of guide tube (did not '
            'raise guide tube when lowering).'
        ),
    ),
    '2022-05-10': phys_metadata_utils.Session(
        phys_params=phys_params_perle.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=4.,
            from_right=3.25,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            through_dura=-1.,
            first_spikes=0.550,
            end=2.523,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_posterior=4.25,
            top_from_right=3.,
            bottom_from_posterior=8.5,
            bottom_from_left=5.25,
        ),
        v_probe_1_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_posterior=2.5,
            top_from_right=3.5,
            bottom_from_posterior=7.,
            bottom_from_left=5.5,
        ),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            through_dura=0.,
            first_spikes=2.,
            end=11.,
        ),
        notes=(
            '3-moving object generalization. Raising V-probe: probe tip 13mm '
            'below tip of guide tube (did not raise guide tube when lowering). '
            'Like yesterday, went 1.5mm below dura before seeing spikes on '
            'neuropixel. Neuropixel neurons are reasonable but not great.'
        ),
    ),
    '2022-05-11': phys_metadata_utils.Session(
        phys_params=phys_params_perle.Params,
        notes=(
            'NOTE: Changed Elgar broken fixation ITI (1.5 -> 1) and success '
            'ITI (1 -> 0.5). Perle broke neuropixel, so failed recording day. '
            'Trained 1_2_moving.'
        ),
    ),
    '2022-05-12': phys_metadata_utils.Session(
        phys_params=phys_params_perle.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=5.5,
            from_right=3.,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=1.720,
            end=3.232,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_posterior=6.,
            top_from_right=3.25,
            bottom_from_posterior=9.25,
            bottom_from_left=6.,
        ),
        v_probe_1_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_posterior=4.75,
            top_from_right=3.5,
            bottom_from_posterior=7.5,
            bottom_from_left=6.5,
        ),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            through_dura=0.,
            first_spikes=2.,
            end=11.,
        ),
        notes=(
            '3-moving generalization. Pretty good neurons on neuropixel. '
            'Neuropixel first spikes soon after through dura (~0.5mm). V-probe '
            'raising: probe tip 11mm below guide tube tip (did not raise guide '
            'tubes when lowering). Open Ephys dropped signal mid-session. '
            'Ignore V-probes.'
        ),
    ),
    '2022-05-13': phys_metadata_utils.Session(
        phys_params=phys_params_perle.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=8.,
            from_right=4.,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=-0.800,
            end=0.900,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_posterior=1.5,
            top_from_right=2.,
            bottom_from_posterior=5.,
            bottom_from_left=7.,
        ),
        v_probe_1_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_posterior=0.,
            top_from_right=2.5,
            bottom_from_posterior=3.25,
            bottom_from_left=7.25,
        ),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            through_dura=0.,
            first_spikes=1.,
            end=7.,
        ),
        notes=(
            'NOTE: Until now, -1, 0.__ holes were zero-indexed, so add 1 to '
            'all previous holes that are less than 1. 3-moving generalization. '
            'Raising V-probe: probe tip 7mm below guide tube tip (did not '
            'raise guide tube). Great V-probe neurons. Not great neuropixel '
            'neurons but okay. NOTE: Elgar until 05-12 had V-probe settling '
            'with physiology in the same directory.'
        ),
    ),
    '2022-05-15': phys_metadata_utils.Session(
        phys_params=phys_params_perle.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=4.5,
            from_right=3.5,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=-1.000,
            end=1.284,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_posterior=1.75,
            top_from_right=2.5,
            bottom_from_posterior=5.5,
            bottom_from_left=5.3,
        ),
        v_probe_1_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_posterior=0.,
            top_from_right=2.75,
            bottom_from_posterior=3.75,
            bottom_from_left=5.6,
        ),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            through_dura=0.,
            first_spikes=2.,
            end=9.,
        ),
        notes=(
            '1-moving rotated arena generalization. V-probe raising: probe tip '
            '10mm below guide tube tip (did not lift guide tube). V-probe not '
            'great neurons. Neuropixel very good neurons.'
        ),
    ),
    '2022-05-16': phys_metadata_utils.Session(
        phys_params=phys_params_perle.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=3.5,
            from_right=4.5,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=-1.000,
            end=0.500,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_posterior=3.5,
            top_from_right=1.5,
            bottom_from_posterior=6.25,
            bottom_from_left=6.5,
        ),
        v_probe_1_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_posterior=2.,
            top_from_right=2.,
            bottom_from_posterior=4.5,
            bottom_from_left=7.,
        ),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            through_dura=0.,
            first_spikes=2.5,
            end=9.25,
        ),
        notes=(
            'Okay neurons on V-probe 1. Great neurons on neuropixel. Raising '
            'V-probe: probe tip 11mm below guide tube tip (did not raise guide '
            'tube).'
        ),
    ),
    '2022-05-17': phys_metadata_utils.Session(
        phys_params=phys_params_perle.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=7.5,
            from_right=3.75,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=-1.000,
            end=1.620,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_posterior=4.5,
            top_from_right=1.,
            bottom_from_posterior=7.5,
            bottom_from_left=5.,
        ),
        v_probe_1_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_posterior=3.,
            top_from_right=0.5,
            bottom_from_posterior=5.75,
            bottom_from_left=5.5,
        ),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            through_dura=0.,
            first_spikes=1.,
            end=8.5,
        ),
        notes=(
            'V-probe straddling arcuate sulcus I think. Neuropixel in damaged '
            'cortex, few neurons.'
        ),
    ),
    '2022-05-18': phys_metadata_utils.Session(
        phys_params=phys_params_perle.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=5.,
            from_right=2.,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=-1.000,
            end=1.500,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_posterior=4.5,
            top_from_right=1.,
            bottom_from_posterior=7.5,
            bottom_from_left=5.,
        ),
        v_probe_1_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_posterior=3.,
            top_from_right=0.5,
            bottom_from_posterior=5.75,
            bottom_from_left=5.5,
        ),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            through_dura=0,
            first_spikes=2.,
            end=4.,
        ),
        notes=(
            '1-moving rotated arena. On V-probe 0, no neurons in top 20 '
            'channels, unsure why. Got worried so stopped lowering early. '
            'Neuropixel not many neurons but more than yesterday. Same V-probe '
            'grid holes as yesterday.'
        ),
    ),
    '2022-05-26': phys_metadata_utils.Session(
        phys_params=phys_params_perle.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=7.5,
            from_right=3.25,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=1.000,
            end=2.000,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_posterior=4.,
            top_from_right=0.25,
            bottom_from_posterior=7.5,
            bottom_from_left=6.5,
        ),
        v_probe_1_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_posterior=2.,
            top_from_right=0.,
            bottom_from_posterior=5.5,
            bottom_from_left=6.5,
        ),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            through_dura=0.,
            first_spikes=1.5,
            end=9.75,
        ),
        notes=(
            '3_static_abc. Note: Eyelink was not working well for first ~150 '
            'completed trials. Neuropixel good neurons. V-probes okay neurons. '
            'Raising V-probe: probe tip 10mm below guide tube tip. Kilosort '
            'failed for V-Probe 1.'
        ),
    ),
    '2022-05-27': phys_metadata_utils.Session(
        phys_params=phys_params_perle.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=7.75,
            from_right=2.25,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=1.000,
            end=2.700,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_posterior=4.5,
            top_from_right=0.5,
            bottom_from_posterior=7.75,
            bottom_from_left=5.75,
        ),
        v_probe_1_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_posterior=2.5,
            top_from_right=0.,
            bottom_from_posterior=5.75,
            bottom_from_left=6.,
        ),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            surface=-6.,
            first_spikes=0.,
            end=6.25,
        ),
        notes=(
            '3_static_abc. V-probes very close to bone (guide tube insertion '
            'difficult). Some sporadic noise on neuropixel. Raising V-probe: '
            'probe tip 6mm below guide tube tip.'
        ),
    ),
    '2022-05-28': phys_metadata_utils.Session(
        phys_params=phys_params_perle.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=6.75,
            from_right=1.5,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=0.500,
            end=3.500,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_posterior=4.5,
            top_from_right=0.5,
            bottom_from_posterior=7.75,
            bottom_from_left=5.75,
        ),
        v_probe_1_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_posterior=2.5,
            top_from_right=0.,
            bottom_from_posterior=5.75,
            bottom_from_left=6.,
        ),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            through_dura=0.,
            first_spikes=1.,
            end=8.3,
        ),
        notes=(
            'Switched Eyelink from 50/180 to 50/170 near end of session '
            '(around 800 completed trials) to improve eye detection. V-probe '
            'not great neurons. Neuropixel pretty good neurons. V-probe holes '
            'same as yesterday. Raising V-probe: probe tip 8.5mm below guide '
            'tube tip.'
        ),
    ),
    '2022-05-29': phys_metadata_utils.Session(
        phys_params=phys_params_perle.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=8.,
            from_right=1.5,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=1.000,
            end=2.700,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_posterior=4.5,
            top_from_right=1.,
            bottom_from_posterior=6.5,
            bottom_from_left=5.75,
        ),
        v_probe_1_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_posterior=2.5,
            top_from_right=0.75,
            bottom_from_posterior=4.5,
            bottom_from_left=5.75,
        ),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            through_dura=0.,
            first_spikes=1.,
            end=8.,
        ),
        notes=(
            'V-probe good neurons. Neuropixel okay neurons but not great. '
            'Raising V-probe: probe tip 9mm below guide tube tip.'
        ),
    ),
    '2022-05-30': phys_metadata_utils.Session(
        phys_params=phys_params_perle.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=4.,
            from_right=4.5,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=0.000,
            end=2.000,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_posterior=4.5,
            top_from_right=1.,
            bottom_from_posterior=6.5,
            bottom_from_left=5.75,
        ),
        v_probe_1_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_posterior=2.5,
            top_from_right=0.75,
            bottom_from_posterior=4.5,
            bottom_from_left=5.75,
        ),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            through_dura=0.,
            first_spikes=1.,
            end=10.,
        ),
        notes=(
            'Same V-probe holes as yesterday. Tons of neurons at tip of '
            'V-probe, maybe in posterior wall of genu of arcuate.'
        ),
    ),
    '2022-05-31': phys_metadata_utils.Session(
        phys_params=phys_params_perle.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=4.,
            from_right=3.5,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=0.,
            end=2.050,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_posterior=2.5,
            top_from_right=0.,
            bottom_from_posterior=9.,
            bottom_from_left=6.5,
        ),
        v_probe_1_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_posterior=0.5,
            top_from_right=0.,
            bottom_from_posterior=7.,
            bottom_from_left=6.25,
        ),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            through_dura=0.,
            first_spikes=1.,
            end=6.5,
        ),
        notes=(
            'Neuropixel great neurons. Trouble penetrating V-probe, maybe hit '
            'Kwiksil from titanium dams in chamber. Raising V-probe: probe tip '
            '9mm below guide tube tip (raised guide tubes 2mm when lowering).'
        ),
    ),
    '2022-06-01': phys_metadata_utils.Session(
        phys_params=phys_params_perle.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=3.75,
            from_right=3.5,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=0.600,
            end=2.500,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_posterior=3.,
            top_from_right=0.5,
            bottom_from_posterior=7.75,
            bottom_from_left=6.75,
        ),
        v_probe_1_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_posterior=1.,
            top_from_right=0.25,
            bottom_from_posterior=5.75,
            bottom_from_left=6.75,
        ),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            through_dura=-1.5,
            first_spikes=0.,
            end=4.5,
        ),
        notes=(
            'V-probe okay neurons. Neuropixel fantastic neurons. Raised probes '
            'before disabling recording.'
        ),
    ),
    '2022-06-03': phys_metadata_utils.Session(
        phys_params=phys_params_perle.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=3.75,
            from_right=4.5,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=0.400,
            end=2.600,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_posterior=3.25,
            top_from_right=1.5,
            bottom_from_posterior=7.5,
            bottom_from_left=6.,
        ),
        v_probe_1_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_posterior=1.25,
            top_from_right=1.5,
            bottom_from_posterior=5.5,
            bottom_from_left=6.,
        ),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            through_dura=0.,
            first_spikes=1.,
            end=8.25,
        ),
        notes=(
            'Set Eyelink to 60/155 for pupil detection threshold. V-probe good '
            'neurons, probably medial arcuate. Neuropixel great neurons.'
        ),
    ),
    '2022-06-04': phys_metadata_utils.Session(
        phys_params=phys_params_perle.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=9.,
            from_right=2.25,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=0.300,
            end=2.550,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_posterior=3.25,
            top_from_right=1.5,
            bottom_from_posterior=7.5,
            bottom_from_left=6.,
        ),
        v_probe_1_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_posterior=1.25,
            top_from_right=1.5,
            bottom_from_posterior=5.5,
            bottom_from_left=6.,
        ),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            through_dura=0.,
            first_spikes=1.,
            end=10.5,
        ),
        notes=(
            'V-probe holes same as yesterday. Neuropixel good neurons. V-probe '
            'good neurons. V-probe definitely in FEF (see saccade burst '
            'neurons). Raising V-probe: probe tip 13mm below guide tube tip '
            '(but guide tubes were not through dura).'
        ),
    ),
    '2022-06-05': phys_metadata_utils.Session(
        phys_params=phys_params_perle.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=10.75,
            from_right=4.,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=0.,
            end=1.751,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_anterior=1.,
            top_from_right=-1.5,
            bottom_from_anterior=5.,
            bottom_from_left=8.,
        ),
        v_probe_1_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_anterior=3.25,
            top_from_right=-1.75,
            bottom_from_anterior=7.,
            bottom_from_left=8.,
        ),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            through_dura=0.,
            first_spikes=1.,
            end=8.,
        ),
        notes=(
            'New V-probe guide tube that angles A->P instead of P->A. V-probe '
            'okay neurons, but probably slightly too far medial. Neuropixel '
            'good neurons initially, but Perle pushed and the neurons went '
            'away.'
        ),
    ),
    '2022-06-06': phys_metadata_utils.Session(
        phys_params=phys_params_perle.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=10.,
            from_right=3.5,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=0.,
            end=1.650,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_anterior=2.5,
            top_from_right=-0.5,
            bottom_from_anterior=5.25,
            bottom_from_left=7.,
        ),
        v_probe_1_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_anterior=4.75,
            top_from_right=-0.5,
            bottom_from_anterior=7.5,
            bottom_from_left=6.75,
        ),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            through_dura=0.,
            first_spikes=1.,
            end=7.5,
        ),
        notes=(
            '2-static. V-probe good neurons, probably lateral wall of superior '
            'arcuate. Neuropixel not great neurons.'
        ),
    ),
    '2022-06-07': phys_metadata_utils.Session(
        phys_params=phys_params_perle.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=10.,
            from_right=3.5,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=0.200,
            end=1.500,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_anterior=2.5,
            top_from_right=0.,
            bottom_from_anterior=4.,
            bottom_from_left=7.,
        ),
        v_probe_1_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_anterior=4.75,
            top_from_right=0.,
            bottom_from_anterior=6.25,
            bottom_from_left=7.,
        ),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            through_dura=0.,
            first_spikes=1.,
            end=10.75,
        ),
        notes=(
            'V-probe pretty good neurons. Neuropixel okay neurons but Perle '
            'pushed. Something went wrong with the OpenEphys sync signals. '
            'Unrecoverable V-Probe data. Neuropixel is unaffected.'
        ),
    ),
    '2022-06-08': phys_metadata_utils.Session(
        phys_params=phys_params_perle.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=10.,
            from_right=3.5,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=1.000,
            end=2.700,
        ),
        notes=(
            'V-probe guide tube hit bone, no penetration. Neuropixel same hole '
            'as yesterday. Neuropixel very few neurons.'
        ),
    ),
    '2022-06-09': phys_metadata_utils.Session(
        phys_params=phys_params_perle.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=7.,
            from_right=3.5,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=0.800,
            end=2.800,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_posterior=5.5,
            top_from_right=1.,
            bottom_from_posterior=7.5,
            bottom_from_left=7.25,
        ),
        v_probe_1_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_posterior=3.5,
            top_from_right=0.5,
            bottom_from_posterior=5.5,
            bottom_from_left=7.25,
        ),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            through_dura=0.,
            first_spikes=1.,
            end=8.,
        ),
        notes=(
            'Switched back to old V-probe guide tube holders today with P->A '
            'angle and V-probe posterior of neuropixel. V-probe good neurons. '
            'Neuropixel okay neurons. Raising V-probe: probe tip 9mm below '
            'guide tube tip. V-probe 0 less deep than V-probe 1 today (forgot '
            'to re-adjust relative V-probe depths in the holder to account for '
            'the new angle).'
        ),
    ),
    '2022-06-10': phys_metadata_utils.Session(
        phys_params=phys_params_perle.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=4.75,
            from_right=3.,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=1.400,
            end=2.900,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_posterior=5.5,
            top_from_right=0.75,
            bottom_from_posterior=7.5,
            bottom_from_left=7.5,
        ),
        v_probe_1_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_posterior=3.5,
            top_from_right=0.5,
            bottom_from_posterior=5.5,
            bottom_from_left=7.5,
        ),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            surface=0.,
            first_spikes=1.,
            end=8.,
        ),
        notes=(
            'Like yesterday, V-probe 1 is deeper than V-probe 0. V-probe good '
            'neurons. Neuropixel fantastic neurons. Raising V-probe: probe tip '
            '8mm below guide tube tip.'
        ),
    ),
    '2022-06-11': phys_metadata_utils.Session(
        phys_params=phys_params_perle.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=4.75,
            from_right=3.25,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=1.500,
            end=3.500,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_posterior=5.5,
            top_from_right=0.75,
            bottom_from_posterior=7.5,
            bottom_from_left=7.5,
        ),
        v_probe_1_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_posterior=3.5,
            top_from_right=0.5,
            bottom_from_posterior=5.5,
            bottom_from_left=7.5,
        ),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            through_dura=0.,
            first_spikes=1.,
            end=10.5,
        ),
        notes=(
            'Readjusted V-probes so that both are approximately the same '
            'depth. V-probe good neurons. Neuropixel good neurons. V-probe '
            'same grid holes as yesterady. Raising V-probe: probe tip 10mm '
            'below guide tube tip.'
        ),
    ),
    '2022-06-12': phys_metadata_utils.Session(
        phys_params=phys_params_perle.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=7.25,
            from_right=2.25,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=-1.000,
            end=1.000,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_posterior=3.25,
            top_from_right=1.5,
            bottom_from_posterior=7.5,
            bottom_from_left=6.,
        ),
        v_probe_1_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_posterior=1.25,
            top_from_right=1.5,
            bottom_from_posterior=5.5,
            bottom_from_left=6.,
        ),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            through_dura=0.,
            first_spikes=1.,
            end=11.,
        ),
        notes=(
            'Surface tissue damaged around V_probe (few spikes between first '
            'spikes and 3mm lower than that). V-probe okay neurons. Neuopixel '
            'pretty good neurons. Raising V-probe: probe tip 11mm below guide '
            'tube tip.'
        ),
    ),
    '2022-06-13': phys_metadata_utils.Session(
        phys_params=phys_params_perle.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=9.,
            from_right=1.5,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=1.200,
            end=3.000,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_posterior=3.5,
            top_from_right=1.5,
            bottom_from_posterior=6.5,
            bottom_from_left=6.,
        ),
        v_probe_1_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_posterior=1.5,
            top_from_right=1.25,
            bottom_from_posterior=4.5,
            bottom_from_left=6.,
        ),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            through_dura=0.,
            first_spikes=2.,
            end=9.75,
        ),
        notes=(
            'Neuropixel great neurons. Raising V-probe: probe tip 10mm below '
            'guide tube tip.'
        ),
    ),
    '2022-06-14': phys_metadata_utils.Session(
        phys_params=phys_params_perle.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=8.5,
            from_right=1.5,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=0.,
            end=1.850,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_posterior=3.5,
            top_from_right=1.25,
            bottom_from_posterior=6.5,
            bottom_from_left=5.75,
        ),
        v_probe_1_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_posterior=1.5,
            top_from_right=1.,
            bottom_from_posterior=4.5,
            bottom_from_left=6.,
        ),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            through_dura=0.,
            first_spikes=2.,
            end=12.,
        ),
        notes=('V-probe not great neurons. Neuropixel great neurons.'),
    ),
    '2022-06-15': phys_metadata_utils.Session(
        phys_params=phys_params_perle.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=5.5,
            from_right=2.,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(first_spikes=1.000, end=2.400),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_posterior=3.5,
            top_from_right=-0.25,
            bottom_from_posterior=9.,
            bottom_from_left=7.,
        ),
        v_probe_1_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_posterior=1.5,
            top_from_right=-0.25,
            bottom_from_posterior=7.,
            bottom_from_left=6.75,
        ),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            through_dura=0.,
            first_spikes=1.,
            end=7.75,
        ),
        notes=(
            'Stopped recording about 1 minute after raising probes. Neuropixel '
            'great neurons. Raising V-probe: probe tip 10mm below guide tube '
            'tip. Kilosort failed on V-Probe 1 for some unknown reason.'
        ),
    ),
    '2022-06-16': phys_metadata_utils.Session(
        phys_params=phys_params_perle.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=3.75,
            from_right=3.,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=1.800,
            end=3.460,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_posterior=5,
            top_from_right=1.5,
            bottom_from_posterior=8.,
            bottom_from_left=5.75,
        ),
        v_probe_1_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_posterior=3.,
            top_from_right=1.25,
            bottom_from_posterior=6.,
            bottom_from_left=5.75,
        ),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            through_dura=0.,
            first_spikes=1.5,
            end=9.75,
        ),
        notes=(
            'Still 2-static. Good saccade-bursting FEF neurons. Raising '
            'V-probe: probe tip 10mm below guide tube tip.'
        )
    ),
    '2022-06-17': phys_metadata_utils.Session(
        phys_params=phys_params_perle.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=3.75,
            from_right=4.,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=1.000,
            end=3.500,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_posterior=6,
            top_from_right=1.5,
            bottom_from_posterior=9.,
            bottom_from_left=5.75,
        ),
        v_probe_1_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_posterior=4.,
            top_from_right=1.25,
            bottom_from_posterior=7.,
            bottom_from_left=5.75,
        ),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=0.,
            end=10.5,
        ),
        notes=(
            'Generalization KPT. V-probe good bursting saccade FEF neurons. '
            'Neuropixel okay neurons.'
        ),
    ),
    '2022-06-18': phys_metadata_utils.Session(
        phys_params=phys_params_perle.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=7.25,
            from_right=1.,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=1.500,
            end=3.650,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_posterior=7,
            top_from_right=2.5,
            bottom_from_posterior=7.5,
            bottom_from_left=6.,
        ),
        v_probe_1_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_posterior=5.,
            top_from_right=2.,
            bottom_from_posterior=5.5,
            bottom_from_left=6.75,
        ),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            through_dura=0.,
            first_spikes=1.5,
            end=9.75,
        ),
        notes=(
            'Generalization KPT. V-probe good neurons. Neuropixel great '
            'neurons.'
        ),
    ),
    '2022-06-19': phys_metadata_utils.Session(
        phys_params=phys_params_perle.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=5.,
            from_right=1.75,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=1.200,
            end=3.000,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_posterior=3.5,
            top_from_right=1.,
            bottom_from_posterior=7.5,
            bottom_from_left=6.,
        ),
        v_probe_1_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_posterior=1.5,
            top_from_right=1.,
            bottom_from_posterior=5.5,
            bottom_from_left=6.,
        ),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            through_dura=0.,
            first_spikes=2.,
            end=9.25,
        ),
        notes=('2-static. V-probe great bursting FEF neurons.'),
    ),
}
