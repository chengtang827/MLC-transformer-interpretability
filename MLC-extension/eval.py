import torch

import os
import sys
import argparse
import numpy as np
from MI import analysis
from copy import deepcopy
from hooked_model import HookedBIML, describe_model
import MLC_datasets as dat
from train_lib import seed_all, extract, display_input_output, assert_consist_langs
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from pathlib import Path
from torch.utils.data import DataLoader
from MI.analysis import run_with_cache, plot_attention_patterns, logit_attribution
from MI.hook_functions import *
import analysis.plot as plot
import shutil
## Evaluate a pre-trained model



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
        model_path = project_dir / 'out_models' / 'net-HookedBIML.pt'
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

        case_path = project_dir / 'my_data' / '2'
        # TODO Draw model diagram
        # TODO Although models doesn't match, I can write utility functions first
        plot_dir = case_path / 'plots'



        # Load validation dataset
        _,val_datasets = dat.get_dataset(episode_type, **{'case': case_path})
        assert len(val_datasets) in [1,2]
        langs = val_datasets[0].langs
        assert_consist_langs(langs, checkpoint['langs'])

        # Load model parameters         
        net = HookedBIML(emb_size, input_size, output_size,
            langs['input'].PAD_idx, langs['output'].PAD_idx,
            nlayers_encoder=nlayers_encoder, nlayers_decoder=nlayers_decoder, 
            dropout_p=dropout_p, activation=myact, ff_mult=ff_mult)        
        net.load_state_dict(nets_state_dict)
        net = net.to(device=DEVICE)
        describe_model(net)

        activations = []
        batch_records = []

        do_plot_attn = 1
        do_logit_attribution = 1
        for val_episode in val_datasets:
            #...............................................
            val_dataloader = DataLoader(val_episode,batch_size=batch_size,
                                collate_fn=lambda x:dat.make_biml_batch(x,langs),shuffle=False)
            
            logits, tokens, cache = run_with_cache(val_dataloader, net, langs, max_length_eval, eval_type='max', hook_names=['out_hook','z_hook','attn_weight_hook'])
            
            if do_plot_attn:
                plot_attention_patterns(net, tokens, cache)
            
            if do_logit_attribution:
                logit_attribution(net, cache, langs=langs)


        plot_dir.mkdir(parents=True, exist_ok=True)
        
        if make_plots:
            kwargs = {
                'plot_diff': False,
                'plot_inv': False,
            }
            make_attn_plots(plot_dir, activations, batch_records, type_ = 'self', module_ = 'encoder', averaged = averaged_weights, **kwargs)
            make_attn_plots(plot_dir, activations, batch_records, type_ = 'multihead', module_ = 'decoder', averaged = averaged_weights, **kwargs)
