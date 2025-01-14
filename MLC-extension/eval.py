import torch

import os
import argparse
from copy import deepcopy
from hooked_model import HookedBIML, describe_model
import MLC_datasets as dat
from train_lib import seed_all, extract, display_input_output, assert_consist_langs
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from pathlib import Path
from torch.utils.data import DataLoader
from MI.analysis import \
    run_with_cache, plot_attention_patterns, \
    get_ablation_scores, run_with_cache_batch, plot_path_patching_scores,\
    vector_alignment_circuit, get_module_names_by_regex
from MI.hook_functions import *
import analysis.plot as plot
import MI.model2_2_analysis as model2_2_analysis
from MI import hook_functions

## Evaluate a pre-trained model

def generate_null_dataset(mean_dataset_path, rewrite=False):

    if mean_dataset_path.exists() and not rewrite:
        print('Mean dataset already exists.')
        return
    
    _, D_val = dat.get_dataset('algebraic_noise')        
    val_dataloader = DataLoader(D_val,batch_size=batch_size,collate_fn=lambda x:dat.make_biml_batch(x,D_val.langs),
                                shuffle=False)
    langs = D_val.langs
    net = HookedBIML(emb_size, input_size, output_size,
        langs['input'].PAD_idx, langs['output'].PAD_idx,
        nlayers_encoder=nlayers_encoder, nlayers_decoder=nlayers_decoder, 
        dropout_p=dropout_p, activation=myact, ff_mult=ff_mult)        
    net.load_state_dict(nets_state_dict)
    net = net.to(device=DEVICE)

    cache = run_with_cache_batch(val_dataloader, net, langs, wanted_hooks=['*hook*'])
    torch.save(cache, mean_dataset_path)
    print('Generated and saved mean dataset at', mean_dataset_path)

    return

if __name__ == "__main__":

        # Adjustable parameters
        parser = argparse.ArgumentParser()
        parser.add_argument('--fn_out_model', type=str, default='', help='*REQUIRED*. Filename for loading the model')
        parser.add_argument('--dir_model', type=str, default='out_models', help='Directory for loading the model file')
        parser.add_argument('--max_length_eval', type=int, default=10, help='Maximum generated sequence length')
        parser.add_argument('--batch_size', type=int, default=-1, help='Number of episodes in batch')                                
        parser.add_argument('--episode_type', type=str, default='', help='What type of episodes do we want? See datasets.py for options')
        parser.add_argument('--dashboard', default=False, action='store_true', help='Showing loss curves during training.')
        parser.add_argument('--ll', default=False, action='store_true', help='Evaluate log-likelihood of validation (val) set')
        parser.add_argument('--max', default=False, action='store_true', help='Find best outputs for val commands (greedy decoding)')
        parser.add_argument('--sample', default=False, action='store_true', help='Sample outputs for val commands')
        parser.add_argument('--sample_html', default=False, action='store_true', help='Sample outputs for val commands in html format (using unmap to canonical text)')
        parser.add_argument('--sample_iterative', default=False, action='store_true', help='Sample outputs for val commands iteratively. Output in html format')
        parser.add_argument('--fit_lapse', default=False, action='store_true', help='Fit the best lapse rate according to log-likelihood on validation')
        parser.add_argument('--ll_nrep', type=int, default=1, help='Evaluate each episode this many times when computing log-likelihood (needed for stochastic remappings)')
        parser.add_argument('--ll_p_lapse', type=float, default=0., help='Lapse rate when evaluating log-likelihoods')
        parser.add_argument('--verbose', default=False, action='store_true', help='Inspect outputs in more detail')
        #ADDED CODE: ARGS
        parser.add_argument('--make_plots', default=False, action='store_true',  help='To make attention plots')       
        parser.add_argument('--averaged_weights', default=False, action='store_true',  help='Whether retrieved attention scores are averaged or not')

        args = parser.parse_args()
        model_path = args.fn_out_model
        dir_model = args.dir_model
        max_length_eval = args.max_length_eval
        episode_type = args.episode_type
        do_dashboard = args.dashboard
        batch_size = args.batch_size
        do_ll = args.ll
        do_max_acc = args.max
        do_sample_acc = args.sample
        do_sample_html = args.sample_html
        do_sample_iterative = args.sample_iterative
        do_fit_lapse = args.fit_lapse
        ll_nrep = args.ll_nrep
        ll_p_lapse = args.ll_p_lapse
        verbose = args.verbose

        averaged_weights = args.averaged_weights #ADDED CODE
        make_plots = args.make_plots #ADDED CODE

        project_dir = Path(os.getcwd())
        model_path = project_dir / 'out_models' / 'net-HookedBIMLSmall.pt'
        if not model_path.exists():
             raise Exception('filename '+model_path+' not found')

        seed_all()
        print('Loading model:',model_path,'on', DEVICE)
        checkpoint = torch.load(model_path, map_location=DEVICE)

        plot_loss = 0
        if plot_loss:
            plot.plot_train_history(checkpoint['train_tracker'])


        if not episode_type: episode_type = checkpoint['episode_type']
        if batch_size<=0: batch_size = checkpoint['batch_size']
        nets_state_dict = checkpoint['nets_state_dict']
        if list(nets_state_dict.keys())==['net']: nets_state_dict = nets_state_dict['net'] # for compatibility with legacy code
        input_size = checkpoint['langs']['input'].n_symbols
        output_size = checkpoint['langs']['output'].n_symbols
        emb_size = checkpoint['emb_size']
        dropout_p = checkpoint['dropout']
        ff_mult = checkpoint['ff_mult']
        myact = checkpoint['activation']
        nlayers_encoder = checkpoint['nlayers_encoder']
        nlayers_decoder = checkpoint['nlayers_decoder']
        train_tracker = checkpoint['train_tracker']
        best_val_loss = -float('inf')
        if 'best_val_loss' in checkpoint: best_val_loss = checkpoint['best_val_loss']
            
        print('  Loading model that has completed (or started) ' + str(checkpoint['epoch']) + ' of ' + str(checkpoint['nepochs']) + ' epochs')
        print('  test episode_type:',episode_type)
        print('  batch size:',checkpoint['batch_size'])
        print('  max eval length:', max_length_eval)
        print('  number of steps:', checkpoint['step'])
        print('  best val loss achieved: {:.4f}'.format(best_val_loss))

        ### ADDED CODE
        # iterate though all test cases and create a plot folder for each case


        # construct mean activity dataset
        null_dataset_path = project_dir / 'my_data' / 'null_dataset.pt'
        generate_null_dataset(mean_dataset_path=null_dataset_path, rewrite=0)

        # Load validation dataset
        episode_type = 'my_test'

        # case_path = project_dir / 'my_data' 
        case_path = project_dir / 'my_data' / '4'
        _,val_dataset = dat.get_dataset(episode_type, **{'case': case_path})
        langs = val_dataset.langs

        # Load model parameters         
        net = HookedBIML(emb_size, input_size, output_size,
            langs['input'].PAD_idx, langs['output'].PAD_idx,
            nlayers_encoder=nlayers_encoder, nlayers_decoder=nlayers_decoder, 
            dropout_p=dropout_p, activation=myact, ff_mult=ff_mult)        
        net.load_state_dict(nets_state_dict)
        net = net.to(device=DEVICE)
        describe_model(net)


        do_plot_attn = 1
        do_logit_attribution = 1
        plot_dir = case_path / 'plots'

        # for k,v in net.named_modules():
        #     print(k)
        #...............................................
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size,
                            collate_fn=lambda x:dat.make_biml_batch(x,langs),shuffle=False)
        
        analysis = model2_2_analysis.Analysis(dataloader=val_dataloader, net=net, langs=langs, plot_dir=plot_dir,
                                              null_dataset_path=null_dataset_path)
        


        # plot_attention_patterns(val_dataloader, net, langs, save_dir=plot_dir, rewrite=1)     


        analysis.build_graph()
        # for pred_arg in range(3):
        circuit = analysis.back_track(pred_arg=0, plot_score=0, rewrite=1)

        output = analysis.run_minimal_circuit(circuit, rewrite=1)





        path_patching_scores = vector_alignment_circuit(val_dataloader, net, langs,
                                                            null_dataset_path=null_dataset_path,
                                                            circuit=
                                                            {
                                                            'sender':[
                                                                 {'module':'*decoder*z_hook*','head':'*'},
                                                                #  {'module':'*decoder*layer*1*self*z_hook*','head':'*'},
                                                                 ],
                                                            'receiver':[{'module':'*decoder*layer*1*multi*z_hook*','head':'1'}],
                                                            'freeze':[
                                                                 {'module':'*encoder*z_hook*','head':'*'},
                                                                 {'module':'*decoder*z_hook*','head':'*'},
                                                                ],
                                                            'knockout':[{'module':'*encoder*layer*1*z_hook*','head':'0'}]
                                                            },
                                                            save_path=plot_dir/'path_patching'/'VA_decoder_to_unembed.png',
                                                            rewrite=1,
                                                        )



                             

        if do_plot_attn:
            null_activations = torch.load(null_dataset_path)
            ablate_name = get_module_names_by_regex(net, [{'module':'*encoder*layer*1*z_hook*','head':'1|2|3|4|5|6|7'},
                                                        #   {'module':'*decoder*layer*0*multi*z_hook*','head':'1|5'}
                                                          ])
            hook_functions.add_hooks(net, mode='patch', hook_names=ablate_name, 
                                     patch_activation=[null_activations[str(name)] for name in ablate_name])
            plot_attention_patterns(val_dataloader, net, langs, save_dir=plot_dir, rewrite=1)     
            

            




