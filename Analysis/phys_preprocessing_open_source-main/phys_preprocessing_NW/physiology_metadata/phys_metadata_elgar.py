"""Physiology notes."""

import phys_metadata_utils
import phys_params_elgar


ELGAR = {
    '2022-05-01': phys_metadata_utils.Session(
        phys_params=phys_params_elgar.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=14.,
            from_left=14.5,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            surface=1.575,
            end=1.875,
        ),
        notes='Noise issues neuropixel.',
    ),    
    '2022-05-02': phys_metadata_utils.Session(
        phys_params=phys_params_elgar.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=13.75,
            from_left=15.8,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=-0.005,
            end=1.420,
        ),
        notes='No V-probe recording.'
    ),
    '2022-05-04': phys_metadata_utils.Session(
        phys_params=phys_params_elgar.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=9.5,
            from_left=15.75,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=0.0054,
            end=1.4083,
        ),
        notes='No V-probe recording.'
    ),
    '2022-05-05': phys_metadata_utils.Session(
        phys_params=phys_params_elgar.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=14.5,
            from_left=12.,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=0.0064,
            end=1.7560,
        ),
        notes='No V-probe recording.'
    ),
    '2022-05-06': phys_metadata_utils.Session(
        phys_params=phys_params_elgar.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=12.96,
            from_left=14.28,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=0.0044,
            end=1.8684,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_anterior=1.,
            bottom_from_anterior=7.,
            top_from_left=17.5,
            bottom_from_left=7.5,
        ),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            through_dura=0.,
            end=5.2,
        ),
        notes=(
            '32-channel V-probe. Rasters did not work, so un-sortable. Might '
            'be no open ephys sync signals.'
        ),
    ),
    '2022-05-07': phys_metadata_utils.Session(
        phys_params=phys_params_elgar.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=16.,
            from_left=12.5,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=0.0064,
            end=1.8714,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_anterior=1.,
            bottom_from_anterior=7.,
            top_from_left=18.5,
            bottom_from_left=8.5,
        ),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            through_dura=0.,
            end=6.,
        ),
        notes=(
            '32-channel V-probe. V-probe is bad. Tons of noise. No neurons. '
            'Un-sortable.'
        ),
    ),
    '2022-05-08': phys_metadata_utils.Session(
        phys_params=phys_params_elgar.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=9.,
            from_left=14.72,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=0.0040,
            end=1.6340,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_anterior=1.,
            bottom_from_anterior=7.,
            top_from_left=18.5,
            bottom_from_left=8.5,
        ),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            through_dura=0.,
            end=4.,
        ),
    ),
    '2022-05-09': phys_metadata_utils.Session(
        phys_params=phys_params_elgar.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=9.,
            from_left=14.72,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=0.0040,
            end=1.7370,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_anterior=1.,
            bottom_from_anterior=7.,
            top_from_left=18.5,
            bottom_from_left=8.5,
        ),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            through_dura=0.,
            end=5.5,
        ),
        notes='No mworks behavior.  Confirmed lost.',
    ),
    '2022-05-10': phys_metadata_utils.Session(
        phys_params=phys_params_elgar.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=14.12,
            from_left=15.26,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=0.0070,
            end=1.8320,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_anterior=1.,
            bottom_from_anterior=7.,
            top_from_left=18.5,
            bottom_from_left=8.5,
        ),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            surface=35.,
            through_dura=37.5,
            end=40.0,
        ),
        notes='V-probe: Superficial recording due to guide tube issue.',
    ),
    '2022-05-12': phys_metadata_utils.Session(
        phys_params=phys_params_elgar.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=16.2,
            from_left=15.24,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=0.0050,
            end=1.5500,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_anterior=1.,
            bottom_from_anterior=7.,
            top_from_left=18.5,
            bottom_from_left=8.5,
        ),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            through_dura=0.,
            first_spikes=0.,
            end=6.25,
        ),
    ),
    '2022-05-14': phys_metadata_utils.Session(
        phys_params=phys_params_elgar.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=14.7,
            from_left=13.9,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=0.0120,
            end=1.3070,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_anterior=4.,
            bottom_from_anterior=10.,
            top_from_left=17.5,
            bottom_from_left=7.5,
        ),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            through_dura=0.,
            end=7.8,
        ),
        notes='V-probe: Active. Neuropixel: Not much activity',
    ),
    '2022-05-16': phys_metadata_utils.Session(
        phys_params=phys_params_elgar.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=14.58,
            from_left=16.31,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=0.9520,
            end=2.8747,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_anterior=1.,
            bottom_from_anterior=8.,
            top_from_left=16.5,
            bottom_from_left=6.5,
        ),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            surface=0.,
            end=7.,
        ),
        notes='V-probe: Very few neurons. Neuropixel: Not great but decent',
    ),
    '2022-05-17': phys_metadata_utils.Session(
        phys_params=phys_params_elgar.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=13.5,
            from_left=15.,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=0.0147,
            end=1.6200,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_anterior=2.,
            bottom_from_anterior=9.,
            top_from_left=17.5,
            bottom_from_left=6.5,
        ),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            through_dura=0.,
            first_spikes=0.,
            end=9.03,
        ),
        notes=(
            'V-probe: Amazing, best seen. 5.1mm starts looking like FEF. '
            'Neuropixel: Pretty good.'
        ),
    ),
    '2022-05-18': phys_metadata_utils.Session(
        phys_params=phys_params_elgar.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=13.5,
            from_left=15.,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=0.0120,
            end=1.4610,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_anterior=2.,
            bottom_from_anterior=9.,
            top_from_left=17.5,
            bottom_from_left=6.5,
        ),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            surface=18.75,
            first_spikes=24.2,
            end=35.7,
        ),
        notes='V-probe: Pretty good. Neuropixel: Some neurons but not many.',
    ),
    '2022-05-19': phys_metadata_utils.Session(
        phys_params=phys_params_elgar.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=12.,
            from_left=15.6,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=0.0150,
            end=1.2332,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_anterior=2.,
            bottom_from_anterior=9.,
            top_from_left=17.5,
            bottom_from_left=6.5,
        ),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            through_dura=3.22,
            first_spikes=3.22,
            end=10.37,
        ),
    ),
    '2022-06-02': phys_metadata_utils.Session(
        phys_params=phys_params_elgar.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=12.12,
            from_left=15.35,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=0.0130,
            end=1.2620,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_anterior=3.,
            bottom_from_anterior=9.,
            top_from_left=17.5,
            bottom_from_left=7.5,
        ),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            surface=15.,
            first_spikes=21.,
            end=25.55,
        ),
        notes=(
            'V-probe: Lots of neurons. Can see at least one FEF neuron. '
            'Neuropixel: Lots of neurons.'
        ),
    ),
    '2022-06-03': phys_metadata_utils.Session(
        phys_params=phys_params_elgar.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=16.5,
            from_left=16.79,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=0.0130,
            end=1.1190,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_anterior=3.,
            bottom_from_anterior=9.,
            top_from_left=17.5,
            bottom_from_left=7.5,
        ),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            surface=22.9,
            end=30.25 + 6.85,
        ),
    ),
    '2022-06-04': phys_metadata_utils.Session(
        phys_params=phys_params_elgar.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=17.74,
            from_left=13.2,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=0.0150,
            end=1.1330,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_anterior=3.,
            bottom_from_anterior=9.,
            top_from_left=16.5,
            bottom_from_left=6.5,
        ),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            surface=35.,
            through_dura=38.,
            first_spikes=38.25,
            end=38.25 + 6.95,
        ),
        notes='V-probe: Bad. Neuropixel: Bad.',
    ),
    '2022-06-05': phys_metadata_utils.Session(
        phys_params=phys_params_elgar.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=17.1,
            from_left=16.77,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=0.0150,
            end=1.1687,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_anterior=3.,
            bottom_from_anterior=9.,
            top_from_left=17.5,
            bottom_from_left=7.5,
        ),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            surface=23.,
            first_spikes=28.2,
            end=28.2 + 7.76,
        ),
        notes='V-probe: Fantastic. Neuropixel: Fantastic, best day yet.',
    ),
    '2022-06-06': phys_metadata_utils.Session(
        phys_params=phys_params_elgar.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=15.,
            from_left=17.67,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=0.0197,
            end=2.4039,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_anterior=3.,
            bottom_from_anterior=9.,
            top_from_left=16.5,
            bottom_from_left=6.5,
        ),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            surface=11.,
            first_spikes=17.,
            end=17. + 6.1,
        ),
        notes='V-probe: Crazy good neurons. Neuropixel: Best day yet.',
    ),
    '2022-06-07': phys_metadata_utils.Session(
        phys_params=phys_params_elgar.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=15.8,
            from_left=17.5,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=0.0219,
            end=1.9120,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_anterior=3.,
            bottom_from_anterior=9.,
            top_from_left=18.5,
            bottom_from_left=8.5,
        ),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            surface=17.2,
            first_spikes=22.2,
            end=23.2 + 8.25,
        ),
        notes='V-probe: Meh. Neuropixel: Best day ever seen.',
    ),
    '2022-06-09': phys_metadata_utils.Session(
        phys_params=phys_params_elgar.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=18.,
            from_left=16.8,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=0.0229,
            end=2.2129,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_anterior=3.,
            bottom_from_anterior=9.,
            top_from_left=16.5,
            bottom_from_left=6.5,
        ),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            surface=19.1,
            first_spikes=23.1,
            end=23.1 + 5.05,
        ),
        notes='V-probe: Pretty good. Monkey did not work many trials.',
    ),
    '2022-06-13': phys_metadata_utils.Session(
        phys_params=phys_params_elgar.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=10.5,
            from_left=15.6,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=0.0370,
            end=1.0010,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_anterior=3.,
            bottom_from_anterior=9.,
            top_from_left=16.5,
            bottom_from_left=6.5,
        ),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            surface=27.,
            first_spikes=34.5,
            end=34.5 + 2.,
        ),
        notes='V-probe: Good neurons. Neuropixel: Terrible.',
    ),
    '2022-06-14': phys_metadata_utils.Session(
        phys_params=phys_params_elgar.Params,
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_anterior=4.,
            bottom_from_anterior=10.,
            top_from_left=17.5,
            bottom_from_left=7.5,
        ),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            surface=31.9,
            first_spikes=37.3,
            end=38.6 + 1.7,
        ),
        notes='No neuropixel recording. Kilosort will not run on V-probe.',
    ),
    '2022-06-15': phys_metadata_utils.Session(
        phys_params=phys_params_elgar.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=15.79,
            from_left=15.18,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=0.0210,
            end=2.0820,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_anterior=4.,
            bottom_from_anterior=10.,
            top_from_left=17.5,
            bottom_from_left=7.5,
        ),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            surface=25.9,
            end=32.2 + 4.59,
        ),
        notes='V-probe: Pretty good. Neuropixel: Amazing.',
    ),
    '2022-06-16': phys_metadata_utils.Session(
        phys_params=phys_params_elgar.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=17.95,
            from_left=14.7,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=0.0230,
            end=1.8500,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_anterior=4.,
            bottom_from_anterior=10.,
            top_from_left=17.5,
            bottom_from_left=7.5,
        ),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            surface=22.9,
            through_dura=26.,
            first_spikes=29.1,
            end=29.1 + 10.25,
        ),
        notes=(
            'V-probe: Amazing. Remeasured baseline VP guide tube. AP angle is '
            '16 degrees, and LR angle is 31 degrees. Neuropixel: Among best '
            'neurons yet.'
        ),
    ),
    '2022-06-17': phys_metadata_utils.Session(
        phys_params=phys_params_elgar.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=16.95,
            from_left=14.7,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=0.2201,
            end=2.2500,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_anterior=5.,
            bottom_from_anterior=11.,
            top_from_left=14.5,
            bottom_from_left=4.5,
        ),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            surface=32.3,
            through_dura=35.,
            first_spikes=36.5,
            end=36.5 + 6.5,
        ),
        notes='V-probe: Looks like FEF. Neuropixel: Pretty Good.',
    ),
    '2022-06-18': phys_metadata_utils.Session(
        phys_params=phys_params_elgar.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=17.5,
            from_left=16.63,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=0.0320,
            end=1.9090,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_anterior=5.,
            bottom_from_anterior=11.,
            top_from_left=14.5,
            bottom_from_left=4.5,
        ),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            surface=25.2,
            first_spikes=26.8,
            end=26.8 + 8.25,
        ),
        notes=(
            'V-probe: Good. No saccadic bursting cells observed. Neuropixel: '
            'Meh.'
        ),
    ),
    '2022-06-19': phys_metadata_utils.Session(
        phys_params=phys_params_elgar.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=11.5,
            from_left=17.9,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=0.0340,
            end=2.5770,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_anterior=5.,
            bottom_from_anterior=11.,
            top_from_left=14.5,
            bottom_from_left=4.5,
        ),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            surface=28.,
            first_spikes=31.,
            end=32.5 + 8.6,
        ),
        notes='V-probe: Solid. Neuropixel: Really good.',
    ),
    '2022-06-20': phys_metadata_utils.Session(
        phys_params=phys_params_elgar.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=13.,
            from_left=14.77,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=0.0310,
            end=2.0980,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_anterior=3.,
            bottom_from_anterior=9.,
            top_from_left=16.5,
            bottom_from_left=5.5,
        ),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            surface=36.,
            through_dura=39.,
            first_spikes=39.5,
            end=41. + 6.6,
        ),
        notes=(
            'V-probe: Solid. Probably hit FEF ~2mm from probe tip. Neuropixel: '
            'Solid.'
        ),
    ),
    '2022-06-21': phys_metadata_utils.Session(
        phys_params=phys_params_elgar.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=13.1,
            from_left=15.6,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=0.0390,
            end=2.0000,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_anterior=3.,
            bottom_from_anterior=9.,
            top_from_left=16.5,
            bottom_from_left=5.5,
        ),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            surface=29.9,
            through_dura=31.8,
            end=34. + 9.,
        ),
        notes='V-probe: Pretty bad. Neuropixel: Fantastic.',
    ),
    '2022-06-22': phys_metadata_utils.Session(
        phys_params=phys_params_elgar.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=14.56,
            from_left=17.2,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=0.0310,
            end=2.1410,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_anterior=4.,
            bottom_from_anterior=10.,
            top_from_left=16.5,
            bottom_from_left=5.5,
        ),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            surface=29.4,
            end=34.1 + 6.9,
        ),
        notes='V-probe: Pretty good. Neuropixel: Amazing.',
    ),
    '2022-06-23': phys_metadata_utils.Session(
        phys_params=phys_params_elgar.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=15.6,
            from_left=16.3,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=0.0350,
            end=2.6000,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_anterior=5.,
            bottom_from_anterior=11.,
            top_from_left=16.5,
            bottom_from_left=5.5,
        ),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            surface=35.2,
            end=40.6 + 4.55,
        ),
        notes=(
            'V-probe: Pretty good. May have hit FEF at 1.5mm from probe tip. '
            'Neuropixel: Best day yet. Spike-sorting: Kilosort will not run on '
            'V-probe .'
        ),
    ),
    '2022-06-24': phys_metadata_utils.Session(
        phys_params=phys_params_elgar.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=16.08,
            from_left=15.9,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=0.0370,
            end=2.2860,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_anterior=5.,
            bottom_from_anterior=11.,
            top_from_left=16.5,
            bottom_from_left=5.5,
        ),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            surface=32.9,
            first_spikes=37.6,
            end=37.6 + 5.62,
        ),
        notes=(
            'V-probe: Awesome. Neuropixel: Bad. Juice tube became clogged so '
            'juice rate was low until ~trial 250. At that point, made the '
            'juice amount larger to compensate.'
        ),
    ),
    '2022-06-25': phys_metadata_utils.Session(
        phys_params=phys_params_elgar.Params,
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_anterior=5.,
            bottom_from_anterior=11.,
            top_from_left=16.5,
            bottom_from_left=5.5,
        ),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            surface=34.7,
            first_spikes=40.9,
            end=40.9 + 6.,
        ),
        notes='V-probe: Pretty bad. No neuropixel data recorded.',
    ),
    '2022-06-29': phys_metadata_utils.Session(
        phys_params=phys_params_elgar.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=16.,
            from_left=17.4,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=0.0430,
            end=2.6130,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_anterior=4.,
            bottom_from_anterior=10.,
            top_from_left=16.5,
            bottom_from_left=5.5,
            height_above_grid=1.,
        ),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            surface=24.1,
            first_spikes=28.5,
            end=29. + 4.,
        ),
        notes='V-probe: Good. Neuropixel: Really good.',
    ),
    '2022-06-30': phys_metadata_utils.Session(
        phys_params=phys_params_elgar.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=15.1,
            from_left=17.6,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=0.0430,
            end=1.7850,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_anterior=4.,
            bottom_from_anterior=10.,
            top_from_left=16.5,
            bottom_from_left=5.5,
            height_above_grid=1.,
        ),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            surface=26.3,
            first_spikes=31.5,
            end=31.5 + 5.8,
        ),
        notes='V-probe: Pretty good. Neuropixel: Pretty good.',
    ),
    '2022-07-01': phys_metadata_utils.Session(
        phys_params=phys_params_elgar.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=15.,
            from_left=18.3,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=0.0470,
            end=2.8630,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_anterior=4.,
            bottom_from_anterior=10.,
            top_from_left=16.5,
            bottom_from_left=5.5,
            height_above_grid=1.,
        ),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            surface=40.,
            first_spikes=45.4,
            end=45.4 + 6.4,
        ),
        notes=(
            'V-probe: Really good. Hit white matter at the end. Neuropixel: '
            'Best day ever seen.'
        ),
    ),
    '2022-07-02': phys_metadata_utils.Session(
        phys_params=phys_params_elgar.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=14.,
            from_left=18.23,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=0.0500,
            end=2.2830,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_anterior=3.,
            bottom_from_anterior=9.,
            top_from_left=16.5,
            bottom_from_left=5.5,
            height_above_grid=2.,
        ),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            surface=33.2,
            first_spikes=38.3,
            end=38.3 + 4.55,
        ),
        notes='V-probe: Solid. Neuropixel: Really good.',
    ),
    '2022-08-19': phys_metadata_utils.Session(
        phys_params=phys_params_elgar.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=9.2,
            from_left=18.67,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=0.0595,
            end=3.3195,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_anterior=3.,
            bottom_from_anterior=9.,
            top_from_left=16.5,
            bottom_from_left=5.5,
            height_above_grid=2.,
        ),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            surface=32.,
            first_spikes=37.,
            end=38.6 + 6.,
        ),
        notes='V-probe: Decent. Neuropixel: Amazing.',
    ),
    '2022-08-20': phys_metadata_utils.Session(
        phys_params=phys_params_elgar.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=15.34,
            from_left=15.87,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=0.0525,
            end=2.7883,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_anterior=3.,
            bottom_from_anterior=9.,
            top_from_left=16.5,
            bottom_from_left=5.5,
            height_above_grid=2.,
        ),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            surface=38.3,
            through_dura=41.1,
            first_spikes=43.,
            end=45.1 + 5.,
        ),
        notes=(
            'V-probe: Amazing. Forgot to stop recording after session '
            'stoppped. Neuropixel: Alright.'
        ),
    ),
    '2022-08-21': phys_metadata_utils.Session(
        phys_params=phys_params_elgar.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=16.5,
            from_left=14.25,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=0.0543,
            end=2.6043,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_anterior=3.,
            bottom_from_anterior=9.,
            top_from_left=16.5,
            bottom_from_left=5.5,
            height_above_grid=2.,
        ),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            through_dura=33.,
            end=38. + 6.25,
        ),
        notes='V-probe: Amazing. Neuropixel: Meh.',
    ),
    '2022-08-22': phys_metadata_utils.Session(
        phys_params=phys_params_elgar.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=18.,
            from_left=16.18,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=0.1380,
            end=2.1153,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_anterior=2.,
            bottom_from_anterior=8.,
            top_from_left=16.5,
            bottom_from_left=5.5,
            height_above_grid=2.,
        ),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            surface=36.,
            first_spikes=41.,
            end=42.3 + 6.1,
        ),
        notes='V-probe: Meh. Neuropixel: good.',
    ),
    '2022-08-23': phys_metadata_utils.Session(
        phys_params=phys_params_elgar.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=11.73,
            from_left=15.1,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=0.0650,
            end=2.7056,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_anterior=2.,
            bottom_from_anterior=8.,
            top_from_left=16.5,
            bottom_from_left=5.5,
            height_above_grid=2.,
        ),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            surface=40.5,
            through_dura=43.,
            end=45.5 + 8.8,
        ),
        notes='V-probe: Not bad; really good at tip. Neuropixel: Pretty good.',
    ),
    '2022-08-24': phys_metadata_utils.Session(
        phys_params=phys_params_elgar.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=11.7,
            from_left=14.6,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=0.0646,
            end=2.4956,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_anterior=2.,
            bottom_from_anterior=8.,
            top_from_left=16.5,
            bottom_from_left=5.5,
            height_above_grid=2.,
        ),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            surface=37.5,
            end=45.3 + 8.5,
        ),
        notes=(
            'V-probe: Pretty good. Neuropixel: Meh. Spike-sorting: Double '
            'session because of SpikeGLX restart. Only sorted the first '
            'session.'
        ),
    ),
    '2022-08-25': phys_metadata_utils.Session(
        phys_params=phys_params_elgar.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=20.06,
            from_left=16.81,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=0.0651,
            end=3.1847,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_anterior=6.,
            bottom_from_anterior=12.,
            top_from_left=16.5,
            bottom_from_left=5.5,
            height_above_grid=2.,
        ),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            surface=11.,
            through_dura=15.,
            first_spikes=15.2,
            end=15.2 + 7.8,
        ),
        notes=(
            'V-probe: Decent. Tip of VP definitely in FEF. Neuropixel: Decent.'
        ),
    ),
    '2022-08-26': phys_metadata_utils.Session(
        phys_params=phys_params_elgar.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=18.72,
            from_left=15.5,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=0.0667,
            end=2.7107,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_anterior=6.,
            bottom_from_anterior=12.,
            top_from_left=16.5,
            bottom_from_left=5.5,
            height_above_grid=2.,
        ),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            surface=33.,
            first_spikes=38.3,
            end=39.2 + 7.15,
        ),
        notes='V-probe: Meh. Neuropixel: Bad.',
    ),
    '2022-08-31': phys_metadata_utils.Session(
        phys_params=phys_params_elgar.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=18.,
            from_left=13.99,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=0.0707,
            end=2.4320,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_anterior=6.,
            bottom_from_anterior=12.,
            top_from_left=16.5,
            bottom_from_left=6.5,
        ),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            surface=18.1,
            first_spikes=23.8,
            end=25. + 7.,
        ),
        notes='V-probe: Alright. Neuropixel: Bad.',
    ),
    '2022-09-01': phys_metadata_utils.Session(
        phys_params=phys_params_elgar.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=17.3,
            from_left=15.5,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=0.0687,
            end=2.2602,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_anterior=6.,
            bottom_from_anterior=12.,
            top_from_left=16.5,
            bottom_from_left=6.5,
        ),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            surface=29.,
            end=37. + 2.,
        ),
        notes=(
            'V-probe: Pretty good. Neuropixel: Pretty good. Completed trials '
            '1-88 had no juice.'
        ),
    ),
    '2022-09-02': phys_metadata_utils.Session(
        phys_params=phys_params_elgar.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=14.2,
            from_left=14.8,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=0.0754,
            end=2.7760,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_anterior=6.,
            bottom_from_anterior=12.,
            top_from_left=15.5,
            bottom_from_left=5.5,
        ),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            surface=26.2,
            end=36.2 + 0.5,
        ),
        notes=(
            'V-probe: Good. Neuropixel: Good. Spike-sorting V-probe: Phy is '
            'messed up. Timescale is messed up, data is corrupt.'
        ),
    ),
    '2022-09-03': phys_metadata_utils.Session(
        phys_params=phys_params_elgar.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=15.3,
            from_left=15.5,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=0.0770,
            end=2.6435,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_anterior=6.,
            bottom_from_anterior=12.,
            top_from_left=15.5,
            bottom_from_left=5.5,
        ),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            surface=28.,
            end=37.2 + 2.5,
        ),
        notes='V-probe: Alright. Neuropixel: Not very good.',
    ),
    '2022-09-04': phys_metadata_utils.Session(
        phys_params=phys_params_elgar.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=13.2,
            from_left=15.68,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=0.0735,
            end=2.6321,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_anterior=5.,
            bottom_from_anterior=11.,
            top_from_left=15.5,
            bottom_from_left=5.5,
        ),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            surface=15.,
            end=22.2 + 4.,
        ),
        notes='V-probe: Decent. Neuropixel: Decent.',
    ),
    '2022-09-05': phys_metadata_utils.Session(
        phys_params=phys_params_elgar.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=17.4,
            from_left=15.6,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=0.0756,
            end=2.7936,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_anterior=4.,
            bottom_from_anterior=9.,
            top_from_left=15.5,
            bottom_from_left=5.5,
        ),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            surface=30.2,
            end=35.5 + 6.9,
        ),
        notes='V-probe: Decent. Neuropixel: Not great.',
    ),
    '2022-09-06': phys_metadata_utils.Session(
        phys_params=phys_params_elgar.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=8.38,
            from_left=16.6,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=0.0819,
            end=3.0173,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_anterior=3.,
            bottom_from_anterior=8.,
            top_from_left=15.5,
            bottom_from_left=5.5,
        ),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            surface=20.6,
            first_spikes=26.5,
            end=26.5 + 6.55,
        ),
        notes=(
            'V-probe: Pretty good. Neuropixel: Alright. Juice tube was not '
            'attached until trial number 11. Also changed the juice amount at '
            'some point early on in the session.'
        ),
    ),
    '2022-09-07': phys_metadata_utils.Session(
        phys_params=phys_params_elgar.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=10.16,
            from_left=18.6,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=0.0890,
            end=3.8888,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_anterior=2.,
            bottom_from_anterior=7.,
            top_from_left=15.5,
            bottom_from_left=5.5,
        ),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            through_dura=32.8,
            end=43.8 + 2.45,
        ),
        notes=(
            'V-probe: Good. Also, ran out of space on the hard drive. At some '
            'point towards the end of the session, cleared out space and '
            'restarted the V-probe recording. Neuropixel: Good.'
        ),
    ),
    '2022-09-08': phys_metadata_utils.Session(
        phys_params=phys_params_elgar.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=15.18,
            from_left=14.65,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=0.0839,
            end=3.0510,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_anterior=2.,
            bottom_from_anterior=7.,
            top_from_left=17.5,
            bottom_from_left=7.5,
        ),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            surface=29.,
            end=33.5 + 4.5,
        ),
        notes='V-probe: Great. No moog behavior log files: Confirmed missing.',
    ),
    '2022-09-09': phys_metadata_utils.Session(
        phys_params=phys_params_elgar.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=10.,
            from_left=15.3,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=0.0856,
            end=3.0766,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_anterior=2.,
            bottom_from_anterior=7.,
            top_from_left=17.5,
            bottom_from_left=7.5,
        ),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            surface=25.4,
            first_spikes=30.5,
            end=30.5 + 5.86,
        ),
        notes=(
            'V-probe: Good. Neuropixel: Awesome. V-probe data lost for good: '
            'It was on seagate hard drive that was corrupted.'
        ),
    ),
    '2022-09-10': phys_metadata_utils.Session(
        phys_params=phys_params_elgar.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=17.34,
            from_left=15.16,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=0.0871,
            end=1.8219,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_anterior=6.,
            bottom_from_anterior=11.,
            top_from_left=16.5,
            bottom_from_left=4.5,
            height_above_grid=1.,
        ),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            surface=30.5,
            end=39.8 + 2.45,
        ),
        notes='V-probe: Good. In FEF. Neuropixel: Meh.',
    ),
    '2022-09-11': phys_metadata_utils.Session(
        phys_params=phys_params_elgar.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=17.12,
            from_left=15.43,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=0.0904,
            end=2.6354,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_anterior=7.,
            bottom_from_anterior=12.,
            top_from_left=16.5,
            bottom_from_left=4.5,
            height_above_grid=1.,
        ),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            surface=32.2,
            end=36.8 + 4.62,
        ),
        notes='V-probe: Solid. Neuropixel: Solid.',
    ),
    '2022-09-12': phys_metadata_utils.Session(
        phys_params=phys_params_elgar.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=18.12,
            from_left=14.43,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=0.0910,
            end=1.1693,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_anterior=7.,
            bottom_from_anterior=12.,
            top_from_left=16.5,
            bottom_from_left=4.5,
            height_above_grid=1.,
        ),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            surface=25.,
            first_spikes=30.,
            end=32.5 + 2.57,
        ),
        notes=(
            'V-probe: Decent. Tip was likely in FEF, not sure about the rest. '
            'Neuropixel: Bad.'
        ),
    ),
    '2022-09-13': phys_metadata_utils.Session(
        phys_params=phys_params_elgar.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=17.35,
            from_left=15,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=0.0944,
            end=2.5916,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_anterior=6.,
            bottom_from_anterior=12.,
            top_from_left=17.5,
            bottom_from_left=6.5,
        ),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            surface=32.8,
            through_dura=36.8,
            first_spikes=37.1,
            end=37.1 + 7.5,
        ),
        notes=(
            'V-probe: Fine. Neuropixel: Fine. No neuropixel task data recorded.'
        ),
    ),
    '2022-09-14': phys_metadata_utils.Session(
        phys_params=phys_params_elgar.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=13.71,
            from_left=17.5,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=0.0974,
            end=3.1459,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_anterior=6.,
            bottom_from_anterior=12.,
            top_from_left=16.5,
            bottom_from_left=6.5,
        ),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            surface=30.5,
            through_dura=34.,
            first_spikes=35.4,
            end=35.4 + 7.6,
        ),
        notes=(
            'V-probe: Okay. Neuropixel: Great. Spike-sorting: Phys computers '
            'ran out of memory halfway through recording.'
        ),
    ),
    '2022-09-15': phys_metadata_utils.Session(
        phys_params=phys_params_elgar.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=13.66,
            from_left=17.8,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=0.0959,
            end=1.8537,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_anterior=8.,
            bottom_from_anterior=12.,
            top_from_left=16.3,
            bottom_from_left=4.5,
            height_above_grid=1.,
        ),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            surface=26.8,
            first_spikes=30.8,
            end=32. + 7.54,
        ),
        notes='V-probe: Okay.',
    ),
    '2022-09-16': phys_metadata_utils.Session(
        phys_params=phys_params_elgar.Params,
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_anterior=7.,
            bottom_from_anterior=12.,
            top_from_left=16.5,
            bottom_from_left=4.5,
        ),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            surface=38.2,
            first_spikes=43.,
            end=43.8 + 4.1,
        ),
        notes=(
            'V-probe: Fantastic. No neuropixel data.  It was lost.  V-probe '
            'data was not fully copied to openmind either.  It is confirmed '
            'lost.'
        ),
    ),
    '2022-09-18': phys_metadata_utils.Session(
        phys_params=phys_params_elgar.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=16.15,
            from_left=14.3,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=0.1060,
            end=0.7752,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_anterior=7.,
            bottom_from_anterior=12.,
            top_from_left=16.5,
            bottom_from_left=5.5,
        ),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            surface=29.2,
            end=37.9 + 4.,
        ),
        notes='V-probe: Bad. Neuropixel: Bad.',
    ),
    '2022-09-19': phys_metadata_utils.Session(
        phys_params=phys_params_elgar.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=16.6,
            from_left=16.53,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=0.1150,
            end=1.9200,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_anterior=7.,
            bottom_from_anterior=12.,
            top_from_left=16.5,
            bottom_from_left=6.5,
        ),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            surface=27.2,
            first_spikes=31.9,
            end=35. + 3.51,
        ),
        notes=(
            'V-probe: Pretty good. Neuropixel: Amazing. Spike-sorting: '
            'OpenEphys drops timesteps for an interval in the session.'
        ),
    ),
    '2022-09-20': phys_metadata_utils.Session(
        phys_params=phys_params_elgar.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=17.7,
            from_left=14.7,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=0.1170,
            end=2.0664,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_anterior=7.,
            bottom_from_anterior=12.,
            top_from_left=16.5,
            bottom_from_left=6.5,
        ),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            surface=24.5,
            first_spikes=29.1,
            end=30. + 6.5,
        ),
        notes=(
            'V-probe: Decent. For sure in FEF. Neuropixel: Not good. '
            'Spike-sorting: CUDA error for neuropixel.'
        ),
    ),
    '2022-09-21': phys_metadata_utils.Session(
        phys_params=phys_params_elgar.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=17.17,
            from_left=13.85,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=0.1220,
            end=2.4089,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_anterior=4.,
            bottom_from_anterior=10.,
            top_from_left=17.5,
            bottom_from_left=6.5,
            height_above_grid=1.,
        ),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            surface=34.,
            end=40.2 + 4.02,
        ),
        notes='V-probe: Pretty good. Neuropixel: Decent.',
    ),
    '2022-09-22': phys_metadata_utils.Session(
        phys_params=phys_params_elgar.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=15.38,
            from_left=13.9,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=0.1264,
            end=2.0654,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_anterior=4.,
            bottom_from_anterior=10.,
            top_from_left=17.5,
            bottom_from_left=6.5,
            height_above_grid=1.,
        ),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            surface=30.7,
            end=40.9 + 2.61,
        ),
        notes='V-probe: Amazing. Looks like FEF. Neuropixel: Decent.',
    ),
    '2022-09-23': phys_metadata_utils.Session(
        phys_params=phys_params_elgar.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=14.1,
            from_left=18.2,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=0.1269,
            end=2.4940,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_anterior=2.,
            bottom_from_anterior=8.,
            top_from_left=17.5,
            bottom_from_left=6.5,
        ),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            surface=23.,
            end=33.75,
        ),
        notes=(
            'V-probe: Fantastic. No oil drive lowering for V-probe. '
            'Neuropixel: Fantastic.'
        ),
    ),
    '2022-10-02': phys_metadata_utils.Session(
        phys_params=phys_params_elgar.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=16.4,
            from_left=17.3,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=0.1246,
            end=2.2367,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_anterior=2.,
            bottom_from_anterior=8.,
            top_from_left=17.5,
            bottom_from_left=6.5,
        ),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            surface=39.8,
            end=45.5 + 3.75,
        ),
        notes='V-probe: Pretty good. Neuropixel: Amazing.',
    ),
    '2022-10-03': phys_metadata_utils.Session(
        phys_params=phys_params_elgar.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=16.6,
            from_left=17.05,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=0.1267,
            end=2.5331,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_anterior=2.,
            bottom_from_anterior=8.,
            top_from_left=17.5,
            bottom_from_left=6.5,
        ),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            surface=19.1,
            first_spikes=23.5,
            end=25. + 5.8,
        ),
        notes='V-probe: Meh. Neuropixel: Great.',
    ),
    '2022-10-04': phys_metadata_utils.Session(
        phys_params=phys_params_elgar.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=16.62,
            from_left=14.4,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=0.1300,
            end=2.8898,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_anterior=4.,
            bottom_from_anterior=10.,
            top_from_left=17.5,
            bottom_from_left=6.5,
        ),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            surface=34.9,
            through_dura=39.2,
            end=40.5 + 4.95,
        ),
        notes=(
            'V-probe: Decent. Neuropixel: Decent. Spike-sorting: Highly, '
            'unstable V_probe neurons.'
        ),
    ),
    '2022-10-05': phys_metadata_utils.Session(
        phys_params=phys_params_elgar.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=16.1,
            from_left=16.6,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=0.1320,
            end=2.5563,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_anterior=5.,
            bottom_from_anterior=11.,
            top_from_left=19.5,
            bottom_from_left=8.5,
        ),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            surface=18.9,
            through_dura=22.1,
            first_spikes=23.,
            end=24. + 4.75,
        ),
        notes=(
            'V-probe: Great. Neuropixel: Great. Spike-sorting: Highly, '
            'unstable V_probe neurons. Spike-sorting: imec ap.meta file is '
            'empty, so compute sample rate and start time based on lf.meta '
            'file, accounting for difference in sample rate. Assumes the same '
            'imec sample rate as 2022-10-04 and 2022-10-06.'
        ),
    ),
    '2022-10-06': phys_metadata_utils.Session(
        phys_params=phys_params_elgar.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=17.23,
            from_left=14.23,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=0.1343,
            end=1.7883,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_anterior=5.,
            bottom_from_anterior=11.,
            top_from_left=18.5,
            bottom_from_left=8.5,
        ),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            surface=21.,
            end=27.1 + 4.05,
        ),
        notes='V-probe: Meh. Neuropixel: Bad.',
    ),
    '2022-10-07': phys_metadata_utils.Session(
        phys_params=phys_params_elgar.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=15.2,
            from_left=15.2,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=0.1383,
            end=1.8353,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_anterior=6.,
            bottom_from_anterior=11.,
            top_from_left=18.5,
            bottom_from_left=6.5,
        ),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            surface=36.2,
            end=44. + 4.3,
        ),
        notes=(
            'V-probe: Bad. Neuropixel: Bad. After total_trial_num 797, doubled '
            'reward size. Stopped session early becuase monkey had water in '
            'the AM. Neuropixel broke during settling.'
        ),
    ),
    '2022-10-08': phys_metadata_utils.Session(
        phys_params=phys_params_elgar.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=16.89,
            from_left=15.,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=1.8385,
            end=3.4766,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_anterior=2.,
            bottom_from_anterior=8.,
            top_from_left=18.5,
            bottom_from_left=8.5,
        ),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            surface=24.5,
            first_spikes=29.95,
            end=31. + 4.2,
        ),
        notes=(
            'V-probe: Great. Neuropixel: Pretty good. No juice until '
            'total_trial_num 25. Spike-sorting: Open Ephys begins to drop '
            'timesteps partway through session.'
        ),
    ),
    '2022-10-11': phys_metadata_utils.Session(
        phys_params=phys_params_elgar.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=17.65,
            from_left=14.2,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=0.1366,
            end=1.9476,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_anterior=3.,
            bottom_from_anterior=9.,
            top_from_left=18.5,
            bottom_from_left=8.5,
        ),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            surface=16.,
            through_dura=18.8,
            end=23. + 1.75,
        ),
        notes=(
            'V-probe: Decent. Neuropixel: Decent. Spike-sorting: Kilosort '
            'worked with CatGT preprocessing, but cuda error without. With '
            'temporal re-alignment, could use CatGT kilosort results.'
        ),
    ),
    '2022-10-12': phys_metadata_utils.Session(
        phys_params=phys_params_elgar.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=18.82,
            from_left=14.7,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=0.1376,
            end=2.0126,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_anterior=5.,
            bottom_from_anterior=12.,
            top_from_left=16.5,
            bottom_from_left=6.5,
        ),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            surface=25.2,
            through_dura=29.8,
            first_spikes=30.6,
            end=32. + 3.65,
        ),
        notes='V-probe: Decent. Neuropixel: Decent.',
    ),
    '2022-10-13': phys_metadata_utils.Session(
        phys_params=phys_params_elgar.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=17.5,
            from_left=14.7,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=0.1395,
            end=2.2205,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_anterior=1.,
            bottom_from_anterior=7.,
            top_from_left=18.,
            bottom_from_left=7.,
        ),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            surface=31.,
            through_dura=34.,
            first_spikes=34.5,
            end=36. + 4.,
        ),
        notes='V-probe: Double check that channel map looks right.',
    ),
    '2022-10-14': phys_metadata_utils.Session(
        phys_params=phys_params_elgar.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=18.74,
            from_left=16.1,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=0.1406,
            end=1.9836,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_anterior=1.,
            bottom_from_anterior=7.,
            top_from_left=17.,
            bottom_from_left=7.,
        ),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            surface=24.,
            first_spikes=28.5,
            end=31. + 1.,
        ),
        notes='V-probe: Okay but not great. Neuropixel: Not great.',
    ),
    '2022-10-16': phys_metadata_utils.Session(
        phys_params=phys_params_elgar.Params,
        np_0_coords=phys_metadata_utils.Neuropixel_Coords(
            from_anterior=17.68,
            from_left=15.3,
        ),
        np_depth=phys_metadata_utils.Probe_Depth(
            first_spikes=0.1436,
            end=1.7670,
        ),
        v_probe_0_coords=phys_metadata_utils.V_Probe_Coords(
            top_from_anterior=1.,
            bottom_from_anterior=8.,
            top_from_left=17.,
            bottom_from_left=7.
        ),
        v_probe_depth=phys_metadata_utils.Probe_Depth(
            surface=28.8,
            end=37. + 1.8,
        ),
        notes='Used DBC probe instead of V-probe. DBC: Bad.',
    ),
}
