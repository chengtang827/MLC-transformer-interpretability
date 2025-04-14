import os
import sys
from pathlib import Path
src_dir = os.getcwd()
src_dir = src_dir + '/src'  
sys.path.append(src_dir)
print(src_dir)


import torch
import MLC_utils
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import MLC_utils as utils
from MI.hook_functions import *
import MI.model_backtrack as backtrack
import MI.model_perturbation as perturbation
from MI import hook_functions




"""

"""
if __name__ == "__main__":

        project_dir = Path(os.getcwd()) / 'src'
        model_path = project_dir / 'out_models' / 'net-HookedBIMLSmall.pt'
        if not model_path.exists():
             raise Exception('filename '+model_path+' not found')

        # # construct mean activity dataset
        dataset_path = project_dir / 'test_data_long'
        null_dataset_path = dataset_path / 'null_dataset.pt'
        # generate_null_dataset(mean_dataset_path=null_dataset_path, rewrite=0)


        plot_dir = dataset_path / 'plots'

        net, dataset = MLC_utils.load_net_and_dataset(model_path=model_path, 
                                                      dataset_path=dataset_path, 
                                                      null_dataset_path=null_dataset_path)


        backtrack_analyzer = backtrack.Analyzer(dataset=dataset, net=net, plot_dir=plot_dir)
        backtrack_analyzer.build_graph()


        backtrack_analyzer.backtrack_dec_cross_1_5_qk(mode='q', rewrite=0)

        backtrack_analyzer.backtrack_dec_cross_1_5_qk(mode='k', rewrite=0)
















                             
  
            

            




