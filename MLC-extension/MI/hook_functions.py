import torch
import matplotlib.pyplot as plt
import math
import numpy as np
import itertools

def inverse_softmax(probabilities, axis=0):
    """
    Calculate the original logits from softmax probabilities.
    
    Args:
        probabilities (np.array): Array of probabilities from softmax
        
    Returns:
        np.array: Original logits (up to an additive constant)
    """
    # Take the natural log of the probabilities
    logits = np.log(probabilities)
    
    # Note: The original logits are only recoverable up to an additive constant
    # We can choose to make the mean zero as one possible solution
    return logits - np.mean(logits, axis=axis, keepdims=True)

class ActivationCache:
    def __init__(self):
        self.cache = {}

    def store(self, activation, module_name):
        self.cache[module_name] = activation


    def get(self):
        return self.cache

    def clear(self):
        self.cache = {}


                    
def build_hook_func(module_name=None, activation_cache=None, mode='cache'):
    """
    Build a hook function that saves the activation of a module to a cache.

    Args:
        module_name (str): Name of the module
        activation_cache (ActivationCache): Cache to save the activation to 

    Returns:
        function: Hook function
    """
    assert mode in ['cache', 'ablation']

    if mode == 'cache':
        def hook_func(activation, hook):
            # 0 for output embedding, 1 for attention scores
            activation_cache.store(activation, module_name)
            return activation
        
        return hook_func
    
    if mode == 'ablation':
        def hook_func(activation, hook):
            # 0 for output embedding, 1 for attention scores
            activation[:,:-1,:,:] = 0
            activation_cache.store(activation, module_name)
            return activation

        return hook_func

def make_attn_plots(folder, activations_list, batch_records_list, type_ , module_, averaged=False, model_specs = {'layers':3, 'heads':8}, **kwargs):
    #folder: name of folder where images are saved
    #activations: dictionary of model activations (generated via SaveOutput)
    #type_: "multihead" or "self"
    #module_:"encoder" or "decoder"
    #batch_record: one dictionary from evaluate_acc list, all inputs/outputs of model. Currently only assumes one batch
    #averaged: Boolean, if attn weights are averaged or not
    #model_specs: architecture of model

    assert module_ in ['encoder', 'decoder']
    plot_diff = kwargs.get('plot_diff', False)
    plot_inv = kwargs.get('plot_inv', False)

    n_layer = model_specs['layers']
    n_episodes = len(activations_list)
    query_labels_list = []
    key_labels_list = []

    for e in range(n_episodes):
        batch_record = batch_records_list[e]
        if module_ == 'encoder':
            type_ = 'self'
            query_labels_list.append(batch_record['xq_context'])
            key_labels_list.append(batch_record['xq_context'])

            n_row = math.ceil(model_specs['heads']/2)
            n_col = 2
            width = 3.2*n_layer
            height = 7.2

        elif module_ == 'decoder': 
            if type_ == 'multihead':
                #only keep until <eos> token
                key_labels = batch_record['yq_predict_cont']
                for h in range(len(key_labels)):
                    key_labels[h] = key_labels[h][:key_labels[h].index('EOS')+1]

            query_labels_list.append(batch_record['xq_context'])
            key_labels_list.append(batch_record['yq_predict_cont'])

            n_row = 2
            n_col = math.ceil(model_specs['heads']/2)
            width = 4.8*n_layer
            height = 6.4

    do_extra = True
    if do_extra:
        flag_inv = [0,1]
        flag_diff = [0,1]
        args = list(itertools.product(flag_inv, flag_diff))

        for flag_i, flag_d in args:
            suffix = ''
            data_func = lambda x: x

            if flag_i: # do inverse
                if not plot_inv:
                    continue
                suffix += 'inv'
                data_func = inverse_softmax
            
            if flag_d: # do difference
                if not plot_diff:
                    continue
                suffix += 'diff'
                # data_func = lambda x, axis: np.diff(x, axis=axis)

            q = len(query_labels_list[0])-1
            for e in range(n_episodes):
                activations = activations_list[e]
                query_labels = query_labels_list[e]
                key_labels = key_labels_list[e]

                fig, ax = plt.subplots(n_row, n_col*n_layer, constrained_layout = True)
                for l in range(model_specs['layers']):
                    key = 'transformer.'+module_+'.layers.' + str(l) + '.'+type_+'_attn.attn_weight_hook'


                    for h in range(model_specs['heads']):
                        if flag_d:
                            attn_score =  activations_list[0][key][-1]-activations_list[1][key][-1]
                        else:
                            attn_score =  activations[key]
                        # only keep the last time step
                        # [n_batch, n_head, n_query, n_key]
                        data_all = attn_score.cpu().detach().numpy()
                        ax_i = ax[h//n_col, h%n_col+n_col*l]
                        data = data_all[q,h,:,:]
                        data = data[:len(key_labels[q]), :].T

                        # data = data_func(data, axis=0)

                        ax_i.imshow(data)
                        ax_i.set_title(f"L {l} H {h}")
                        ax_i.set_yticks(list(range(len(query_labels[q]))))
                        ax_i.set_xticks(list(range(len(key_labels[q]))))
                        ax_i.set_yticklabels(query_labels[q], fontsize=5)
                        ax_i.set_xticklabels(key_labels[q], rotation =90, fontsize=5)

                fig.set_size_inches(width, height)

                save_name = folder / f"{module_} E{e} Q{q} L{l} {suffix}.png"

                fig.savefig(save_name, dpi = 400)
                print(save_name, ' saved')


def add_hooks(net, mode='cache', hooks_named=None):
    activation_cache = ActivationCache()


    for name, module in net.named_modules():
        if any(spec in name for spec in hooks_named):
            print(f'Adding hook to {name}')
            module.add_hook(hook=build_hook_func(module_name=name, activation_cache=activation_cache, mode=mode))
    return activation_cache