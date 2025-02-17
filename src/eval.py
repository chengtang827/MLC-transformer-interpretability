import torch
import MLC_utils
import os

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from pathlib import Path
import MLC_utils as utils
# from MLC_utils import plot_attention_patterns, get_module_names_by_regex
from MI.hook_functions import *
import MI.model_backtrack as backtrack
import MI.model_perturbation as perturbation
from MI import hook_functions

## Evaluate a pre-trained model

if __name__ == "__main__":

        project_dir = Path(os.getcwd())
        model_path = project_dir / 'out_models' / 'net-HookedBIMLSmall.pt'
        if not model_path.exists():
             raise Exception('filename '+model_path+' not found')

        # # construct mean activity dataset
        dataset_path = project_dir / 'test_data_long' 
        null_dataset_path = dataset_path / 'null_dataset.pt'
        # generate_null_dataset(mean_dataset_path=null_dataset_path, rewrite=0)


        do_plot_attn = 1
        do_logit_attribution = 1
        plot_dir = dataset_path / 'plots'

        net, dataset = MLC_utils.load_net_and_dataset(model_path=model_path, 
                                                      dataset_path=dataset_path, 
                                                      null_dataset_path=null_dataset_path)
        if 1:
            perturb_analyzer = perturbation.Analyzer(dataset=dataset, net=net, plot_dir=plot_dir)
            
            perturb_analyzer.perturb_dec_cross_1_5_k(block = 'dec',layer = 1, type = 'cross', head = 5)

            perturb_analyzer.perturb_dec_cross_1_5_q(block = 'dec',layer = 1, type = 'cross', head = 5)


            perturb_analyzer.perturb_dec_cross_1_5_k_shortcut(block = 'dec',layer = 1, type = 'cross', head = 5)




        backtrack_analyzer = backtrack.Analyzer(dataset=dataset, net=net, plot_dir=plot_dir)
        
        # prune_names = analysis.prune_circuit(rewrite=0)


        # plot_attention_patterns(val_dataloader, net, langs, save_dir=plot_dir, rewrite=1)     

        prune_names=None
        backtrack_analyzer.build_graph(prune_names=prune_names, bias=False)

        if 1:

            utils.plot_attention_patterns(dataset=dataset, net=net, save_dir=plot_dir, rewrite=1)     

        if 0: #
            circuit = backtrack_analyzer.back_track(pred_arg=[0,1], plot_score=1, metric='VAF', sequential=1, n_track=1, threshold=0.3, rewrite=1)

        if False:
            backtrack_analyzer.analyze_enc_self_0_5(block = 'enc',layer = 0,type = 'self', head = 5)

            backtrack_analyzer.analyze_enc_self_1_1(block = 'enc',layer = 1,type = 'self', head = 1)

        backtrack_analyzer.backtrack_dec_cross_1_5(block = 'dec',layer = 1, type = 'cross', head = 5, rewrite=0)

        backtrack_analyzer.analyze_dec_cross_1_5(block = 'dec',layer = 1, type = 'cross', head = 5, rewrite=1)

        backtrack_analyzer.analyze_dec_cross_0_3(block = 'dec',layer = 0, type = 'cross', head = 3)

        backtrack_analyzer.analyze_enc_self_1_0(block = 'enc',layer = 1,type = 'self', head = 0)










        # for pred_arg in range(3):




        output = backtrack_analyzer.run_minimal_circuit(circuit, rewrite=1)






                             
  
            

            




