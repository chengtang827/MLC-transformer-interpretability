"""Runs ironclust spike sorting algorithm on preprocessed spikeglx data."""
import sys
from pathlib import Path

import spikeinterface.core as sc
import spikeinterface.sorters as ss
import spikeinterface.extractors as se


_IRONCLUST_PATH = '/om2/user/c_tang/Sorting/ironclust'

def main(session_dir):
    sorter = 'ironclust'
    params = ss.get_default_sorter_params(sorter)
    params_description = ss.get_sorter_params_description(sorter)
    # params['adjacency_radius'] = -1
    # params['adjacency_radius_out'] = -1
    params['filter'] = False
    # params['batch_sec_drift'] =
    # params['step_sec_drift'] =
    params['n_jobs'] = 24
    print('sorter params', params)
    ss.IronClustSorter.set_ironclust_path(_IRONCLUST_PATH)

    session_dir = Path(session_dir)
    recording = sc.load_extractor(session_dir / 'preprocess/')
    # recording = se.read_spikeglx(folder_path=session_dir, stream_id='imec0.ap')

    print('running sorting algorithm')
    sorting = ss.run_sorter(sorter, recording,
        output_folder=session_dir / 'ironclust_output',
        docker_image=False, verbose=True, **params)


if __name__ == '__main__':
    session_dir = sys.argv[1]
    # session_dir ='/om2/user/c_tang/Sorting/Data/Faure/NP/20230508_F_g0/20230508_F_g0_imec0'
    main(session_dir)