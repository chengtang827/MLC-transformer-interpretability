"""Runs ironclust spike sorting algorithm on preprocessed spikeglx data."""
import sys
from pathlib import Path

import spikeinterface.core as sc
import spikeinterface.sorters as ss
import spikeinterface.extractors as se
import os

_KILOSORT3_PATH = '/om2/user/c_tang/Sorting/Kilosort3'

def main(session_dir):
    sorter = 'kilosort3'
    params = ss.get_default_sorter_params(sorter)
    params_description = ss.get_sorter_params_description(sorter)


    params['n_jobs'] = 24
    params['AUCsplit'] = 0.89
    params['minfr_goodchannels'] = 0
    params['minFR'] = 0.02
    params['projection_threshold'] = [9, 4]


    print('sorter params', params)
    ss.Kilosort3Sorter.set_kilosort3_path(_KILOSORT3_PATH)

    session_dir = Path(session_dir)
    # recording = sc.load_extractor(session_dir / 'preprocess/')
    recording = se.read_spikeglx(folder_path=session_dir, stream_id='imec0.ap')

    print('running sorting algorithm')
    output_dir = session_dir / 'si_ks3_direct'
    sorting = ss.run_sorter(sorter, recording,
        output_folder=output_dir,
        docker_image=False, verbose=True, **params)

    # os.remove(output_dir/'sorter_output'/'recording.dat')



if __name__ == '__main__':
    session_dir = sys.argv[1]
    # session_dir ='/om2/user/c_tang/Sorting/Data/Faure/NP/20230508_F_g0/20230508_F_g0_imec0'
    main(session_dir)