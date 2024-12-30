"""Performs automatic curation of spike sorting output"""
import sys
import numpy as np
from pathlib import Path
import json

import spikeinterface.extractors as se
import spikeinterface.core as sc
import spikeinterface.qualitymetrics as sq
import spikeinterface.widgets as sw
from spikeinterface.postprocessing import *
from spikeinterface.exporters import *

def main(session_dir):
    """Performs postprocessing for kilosort3 output - waveform extraction and 
    automatic curation. 
    Args:
        session_dir (str): Path to om2 directory for given session.
    """

    session_dir = Path(session_dir)
    ironclust_dir = session_dir / 'ironclust_output'
    preprocess_dir = session_dir / 'preprocess'

    waveform_dir = ironclust_dir / 'waveforms'
    output_dir = ironclust_dir / 'sorter_output/tmp/firings.mda'
    phy_dir = ironclust_dir / 'convert_to_phy'

    sorting = se.read_mda_sorting(output_dir, sampling_frequency=30000.021348)
    print('finish1')
    rec = sc.load_extractor(preprocess_dir)

    # Extracts waveforms
    print('extracting waveforms')
    we = sc.extract_waveforms(rec, sorting, waveform_dir, sparse=True,
                            ms_before=1, ms_after=2., max_spikes_per_unit=500,
                            n_jobs=4, overwrite=True,  chunk_size=30000)

    # some computations are done before to control all options
    #compute_spike_amplitudes(we)
    #compute_principal_components(we, n_components=3, mode='by_channel_global')
    print('finish2')
    # the export process is fast because everything is pre-computed
    export_to_phy(we, output_folder=phy_dir)
                                                

if __name__ == "__main__":
    # session_dir = sys.argv[1]
    session_dir = '/om2/user/c_tang/Sorting/Data/Faure/NP/20230508_F_g0/20230508_F_g0_imec0'
    main(session_dir)