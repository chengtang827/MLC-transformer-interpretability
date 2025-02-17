from MI import hook_functions
import MLC_datasets as dat
import numpy as np
import torch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import einops
linear = torch._C._nn.linear
import matplotlib.pyplot as plt
from MI.hook_functions import regex_match
from tqdm import tqdm
from matplotlib.colors import TwoSlopeNorm
import multiprocessing
import concurrent
import re
from train_lib import seed_all
import torch
import MLC_datasets as dat
from hooked_model import HookedBIML, describe_model
from torch.utils.data import DataLoader

MAX_LENGTH_EVAL = 10
max_length = MAX_LENGTH_EVAL


def eval_model(val_batch, net, langs):

    val_batch = dat.set_batch_to_device(val_batch)
    out_mask_allow = dat.get_batch_output_pool(val_batch)

    net.eval()
    emission_lang = langs['output']
    use_mask = len(out_mask_allow)>0

    memory, memory_padding_mask = net.encode(val_batch) 
    n_batch = len(val_batch['yq']) # b*nq
    z_padded = torch.tensor([emission_lang.symbol2index[dat.SOS_token]]*n_batch) # b*nq length tensor
    z_padded = z_padded.unsqueeze(1) # [b*nq x 1] tensor
    z_padded = z_padded.to(device=DEVICE)
    max_length_target = val_batch['yq_padded'].shape[1]-1 # length without EOS
    target_batch = val_batch['yq_padded']
    assert max_length >= max_length_target # make sure that the net can generate targets of the proper length

    # make the output mask if certain emissions are restricted
    if use_mask:
        assert dat.EOS_token in out_mask_allow # EOS must be included as an allowed symbol
        additive_out_mask = -torch.inf * torch.ones((n_batch,net.output_size), dtype=torch.float)
        additive_out_mask = additive_out_mask.to(device=DEVICE)
        for s in out_mask_allow:
            sidx = langs['output'].symbol2index[s]
            additive_out_mask[:,sidx] = 0.

    # Run through decoder
    all_decoder_outputs = torch.zeros((n_batch, max_length), dtype=torch.long)
    all_decoder_outputs = all_decoder_outputs.to(device=DEVICE)
    logits = []
    loss = []
    loss_fn = torch.nn.CrossEntropyLoss()
    for t in range(max_length):
        decoder_output = net.decode(z_padded, memory, memory_padding_mask) # these are logits
            # decoder_output is b*nq x (t+1) x output_size
        decoder_output = decoder_output[:,-1] # get the last step's output (batch_size x output_size)
        logits.append(decoder_output)

        if use_mask: decoder_output += additive_out_mask
        # Choose the symbols at next timestep

        topi = torch.argmax(decoder_output,dim=1)
        emissions = topi.view(-1)

        all_decoder_outputs[:,t] = emissions
        z_padded = torch.cat([z_padded, emissions.unsqueeze(1)], dim=1)

    logits = torch.stack(logits, dim=1) # batch seq n_vocab
    logits = logits[:, :max_length_target+1,:] # batch seq n_vocab

    loss = np.zeros((logits.shape[0], logits.shape[1])) # batch seq
    for b in range(n_batch):
        for n_seq in range(loss.shape[1]):
            loss[b, n_seq] = loss_fn(logits[b,n_seq,:], target_batch[b,n_seq]).detach().cpu().numpy()

    # Get predictions as strings and see if they are correct
    all_decoder_outputs = all_decoder_outputs.detach()
    yq_predict = [] # list of all predicted query outputs as strings
    yq_predict_cont = [] #list of all predicted query outputs with EOS tokens -- ADDED CODE
    logits_correct = []
    v_acc = np.zeros(n_batch)
    for b in range(n_batch):
        #print(all_decoder_outputs[q,:].view(-1))
        pred_seq = emission_lang.tensor_to_symbols(all_decoder_outputs[b,:].view(-1))
        pred_seq_cont = emission_lang.tensor_to_symbols(all_decoder_outputs[b,:].view(-1), break_=False) #ADDED CODE

        pred_seq.append('EOS')
        yq_predict.append(pred_seq)
        yq_predict_cont.append(pred_seq_cont) #ADDED CODE
        
        # only keep the logit for correct symbol
        correct_symbol_id = [langs['output'].symbol2index[i] for i in val_batch['yq'][b]+['EOS']]
        logits_correct.append(logits[b, np.arange(len(correct_symbol_id)), correct_symbol_id].detach().cpu().numpy())

        v_acc[b] = yq_predict[b] == (val_batch['yq'][b]+['EOS']) # for each query, did model get it right?
        


    in_support = np.array(val_batch['in_support']) # which queries are also support items
    for yq in yq_predict:
        yq.insert(0, 'SOS')
    
    out = {'yq':val_batch['yq'], 'yq_predict':yq_predict, 'v_acc':v_acc, 'in_support':in_support, 
            'yq_predict_cont': yq_predict_cont, 'xq_context': val_batch['xq_context'],
            'grammar': val_batch['list_samples'],
            'logits_nvocab': logits.detach().cpu().numpy(), 
            'logits_correct': logits_correct,
            'loss': loss}

    return out


def simplify_tokens(tokens):
    """
    Simplify tokens by removing the EOS token and any padding
    """
    simple_dict = {'RED': 'red',
                   'BLUE': 'blue',
                   'GREEN': 'green',
                   'YELLOW': 'yellow',
                   'PINK': 'pink', 
                   'PURPLE': 'purple',
                   'IO': '=',
                   'blicket': 'A',
                   'dax': 'B',
                   'fep': 'C',
                   'gazzer': 'D',
                   'kiki': 'E',
                   'tufa': 'F',
                   'wif': 'G',
                   'zup': 'H',
                    }
    for i, token in enumerate(tokens):
        if token in simple_dict:
            tokens[i] = simple_dict[token]
    return tokens


def plot_attention_patterns_by_head(dataset, net, head_kwargs, save_dir=None, rewrite=False, display=False):

    dataloader = dataset['dataloader']
    langs = dataset['langs']

    attention_dir = save_dir / 'attention'
    attention_dir.mkdir(parents=True, exist_ok=True)

    block = head_kwargs['block']
    layer = head_kwargs['layer']
    head = head_kwargs['head']
    type = head_kwargs['type']
    if type=='cross':
        type ='multi'


    attention_path = attention_dir / f'{block}_{type}_{layer}_{head}.png'
    if attention_path.exists() and not rewrite:
        print(f'{attention_path} exists')
        return


    val_batch = next(iter(dataloader))
    attention_name = get_module_names_by_regex(net,[{'module':f'*{block}*{layer}*{type}*attn_weight*', 'head':f'{head}'}])
    cache, attn_modules = hook_functions.add_hooks(net, mode='cache', hook_names=attention_name)
    out = eval_model(val_batch, net, langs)

    support_tokens = val_batch['xq_context'][-1]
    support_tokens.append('EOS')
    pred_tokens = out['yq_predict'][-1]

    support_tokens = simplify_tokens(support_tokens)
    pred_tokens = simplify_tokens(pred_tokens)

    [attn_module.remove_hooks() for attn_module in attn_modules]

    fontsize = 400/len(support_tokens)
    cmap='hot'
    fig, ax= plt.subplots(1, 1, constrained_layout = True)

    key_string = str(attention_name[0])
    if type == 'multi':
        im = ax.imshow(cache.cache[key_string][-1,0,:len(pred_tokens), :len(support_tokens)].squeeze(), cmap=cmap)
        fig.colorbar(im, ax=ax)
        ax.set_title(f'Decoder L {layer} H {head}')
        ax.set_yticks(list(range(len(pred_tokens))))
        ax.set_xticks(list(range(len(support_tokens))))
        ax.set_yticklabels(pred_tokens, fontsize=fontsize)
        ax.set_xticklabels(support_tokens, rotation =90, fontsize=fontsize)
    elif type == 'self':
        ax.imshow(cache.cache[key_string][-1,0,:len(support_tokens), :len(support_tokens)].squeeze(), cmap=cmap)
        ax.set_title(f'Decoder L {layer} H {head}')
        ax.set_yticks(list(range(len(support_tokens))))
        ax.set_xticks(list(range(len(support_tokens))))
        ax.set_yticklabels(support_tokens, fontsize=fontsize)
        ax.set_xticklabels(support_tokens, rotation =90, fontsize=fontsize)
    if display:
        plt.show()
    else:
        fig.savefig(attention_path, dpi = 400)
        plt.close()
    print(f'{attention_path} saved')



def plot_attention_patterns(dataset, net, save_dir=None, rewrite=False):

    dataloader = dataset['dataloader']
    langs = dataset['langs']

    attention_dir = save_dir / 'attention'
    attention_dir.mkdir(parents=True, exist_ok=True)

    attention_path = attention_dir / 'attention.png'
    if attention_path.exists() and not rewrite:
        print(f'{attention_path} exists')
        return

    n_head = net.nhead
    nlayers_encoder = net.nlayers_encoder
    nlayers_decoder = net.nlayers_decoder

    # support_tokens = cache['xq_context']
    # pred_tokens = cache['yq_predict']

    val_batch = next(iter(dataloader))
    attention_names = get_module_names_by_regex(net,[{'module':'*attn_weight*', 'head':'*'}])
    cache, attn_modules = hook_functions.add_hooks(net, mode='cache', hook_names=attention_names)
    out = eval_model(val_batch, net, langs)

    support_tokens = val_batch['xq_context'][-1]
    support_tokens.append('EOS')
    pred_tokens = out['yq_predict'][-1]

    [attn_module.remove_hooks() for attn_module in attn_modules]

    fontsize = 200/len(support_tokens)
    cmap='hot'
    nlayers_total = nlayers_encoder+nlayers_decoder*2
    fig, ax= plt.subplots(nlayers_total, n_head, 
                          figsize=(3.2*n_head, 3.2*nlayers_total),
                          constrained_layout = True)
    for l in range(nlayers_encoder):
        for h in range(n_head):
            key_string = f"{{'module': 'transformer.encoder.layers.{l}.self_attn.attn_weight_hook', 'head': {h}}}"
            ax_i = ax[nlayers_total-l-1,h] # reverse the order
            ax_i.imshow(cache.cache[key_string][-1,0,:len(support_tokens), :len(support_tokens)].squeeze(), cmap=cmap)
            ax_i.set_title(f'Encoder L {l} H {h}')
            ax_i.set_yticks(list(range(len(support_tokens))))
            ax_i.set_xticks(list(range(len(support_tokens))))
            ax_i.set_yticklabels(support_tokens, fontsize=fontsize)
            ax_i.set_xticklabels(support_tokens, rotation =90, fontsize=fontsize)
    
    for l in range(nlayers_decoder):
        for h in range(n_head):


            key_string = f"{{'module': 'transformer.decoder.layers.{l}.self_attn.attn_weight_hook', 'head': {h}}}"
            ax_i = ax[nlayers_total-nlayers_encoder-l*2-1,h]
            ax_i.imshow(cache.cache[key_string][-1,0,:len(pred_tokens), :len(pred_tokens)].squeeze(), cmap=cmap)
            ax_i.set_title(f'Decoder L {l} H {h}')
            ax_i.set_yticks(list(range(len(pred_tokens))))
            ax_i.set_xticks(list(range(len(pred_tokens))))
            ax_i.set_yticklabels(pred_tokens, fontsize=fontsize)
            ax_i.set_xticklabels(pred_tokens, rotation =90, fontsize=fontsize)

            key_string = f"{{'module': 'transformer.decoder.layers.{l}.multihead_attn.attn_weight_hook', 'head': {h}}}"
            ax_i = ax[nlayers_total-nlayers_encoder-l*2-2,h]
            ax_i.imshow(cache.cache[key_string][-1,0,:len(pred_tokens), :len(support_tokens)].squeeze(), cmap=cmap)
            ax_i.set_title(f'Decoder L {l} H {h}')
            ax_i.set_yticks(list(range(len(pred_tokens))))
            ax_i.set_xticks(list(range(len(support_tokens))))
            ax_i.set_yticklabels(pred_tokens, fontsize=fontsize)
            ax_i.set_xticklabels(support_tokens, rotation =90, fontsize=fontsize)
        
    fig.savefig(attention_path, dpi = 400)
    print(f'{attention_path} saved')




def get_module_names_by_regex(net:torch.nn.Module, wanted_hooks:list[dict]):
    wanted_names = []
    wanted_heads = []
    # for each dictionary of wanted hooks
    for module_dict in wanted_hooks: 
        wanted_name = module_dict.get('module', None)
        wanted_head = module_dict.get('head', None)
        for net_name, net_module in net.named_modules():
            if regex_match(net_name, wanted_name):
                wanted_names.append(net_name)
                wanted_heads.append(wanted_head)
                continue


    expanded_names = []
    modules_to_expand = ['q_hook','k_hook','v_hook','z_hook', 'attn_weight']
    for wanted_name, wanted_head in zip(wanted_names, wanted_heads):
        if any([module in wanted_name for module in modules_to_expand]):
            for h in range(net.nhead):
                if regex_match(str(h), wanted_head):
                    expanded_names.append({'module':wanted_name, 'head':h})
        else:
            expanded_names.append({'module':wanted_name})
        
    return expanded_names

def get_activations_by_regex(net, cache: dict, hook_regex:list[dict]):
    hook_names = get_module_names_by_regex(net, hook_regex)
    activations = []
    for hook_name in hook_names:
        activations.append(cache.cache[str(hook_name)])
    return activations


def get_activations_by_name(cache: dict, hook_names:tuple):


    names = []
    activations = []
    for k,v in cache.items():
        for hook_name in hook_names:

            if len(k)>=len(hook_name):
                if all([hook_functions.regex_match(str(a),b) for a,b in zip(k[:len(hook_name)],hook_name)]):
                    names.append(k)
                    activations.append(v)
    
    return names, activations


def apply_ln_to_stack(act_before_ln:np.ndarray, ln:torch.nn.LayerNorm, stack_activations:list[np.ndarray],axis=None):

    mu = np.mean(act_before_ln, axis=axis, keepdims=True)
    sigma = np.std(act_before_ln, axis=axis, keepdims=True)
    gamma = ln.weight.detach().cpu().numpy()
    beta = ln.bias.detach().cpu().numpy()
    stack_activations_ln = [(act-mu)/sigma*gamma+beta for act in stack_activations]
    # stack_activations_ln = [(act)/sigma*gamma for act in stack_activations]

    return stack_activations_ln


def logit_attribution(val_dataloader, net, langs=None, save_dir=None, rewrite=False):

    logit_attribution_dir = save_dir / 'logit_attribution'
    logit_attribution_dir.mkdir(parents=True, exist_ok=True)

    logit_attribution_path = logit_attribution_dir / 'logit_attribution.png'
    if logit_attribution_path.exists() and not rewrite:
        return


    # define the correct token and the position (say the first token)
    pred_tokens = cache['yq_predict']
    symbol_id = [langs['output'].symbol2index[i] for i in pred_tokens] 

    # center the W_u matrix
    unembed_mat = net.out.weight.detach().cpu().numpy() # n_vocab x hidden_size
    unembed_mat = unembed_mat - np.mean(unembed_mat, axis=0)

    # find out the direction for predicted token (correct)
    # skip the first SOS token
    token_directions = unembed_mat[symbol_id[1:], :]    



    # get the stack of out activations to residual stream
    stack_names, stack_activations = get_activations_by_name(cache, hook_names=(('decoder','*','*out'),
                                                                                ('decoder_embedding'),
                                                                                ))

    # apply the final layer norm to the stacked activations
    ln = net.transformer.decoder.norm
    act_before_ln = cache['decoder', net.nlayers_decoder-1, 'resid_post']

    stack_activations = apply_ln_to_stack(act_before_ln, ln, stack_activations)

    """    
    # sanity check
    act_after_pred = apply_ln_to_stack(act_before_ln, ln, [act_before_ln])[0]
    act_after_ln = cache['decoder_hook']
    they should be equal
    """

    # do the dot product
    stack_activations = np.stack(stack_activations, axis=0) # n_layer x seq x hidden_size

    # append the final stream to the stack (without layer norm)
    stack_activations = np.concatenate((stack_activations, cache['decoder_hook'][None, :]), axis=0) # n_layer+1 x seq x hidden_size
    stack_names.append('decoder_hook')


    logit_attribution = []
    for n_seq in range(token_directions.shape[0]):
        logit_attribution.append(np.einsum('l h, h->l', stack_activations[:, n_seq, :], token_directions[n_seq, :]))

    logit_attribution = np.stack(logit_attribution)


    # plot heatmap of logit_attribution
    fig, ax = plt.subplots(1,1)
    handle=ax.imshow(logit_attribution.T, cmap='hot')
    fig.colorbar(handle, ax=ax)  # Associate the colorbar with the Axes

    ax.set_title('Logit Attribution')
    ax.set_xticks(np.arange((len(pred_tokens))))
    ax.set_xticklabels(pred_tokens, fontsize=5, rotation=90)
    ax.set_yticks(list(range(stack_activations.shape[0])))
    ax.set_yticklabels(stack_names, fontsize=5, rotation=0)
    fig.savefig(logit_attribution_path, dpi = 400)
    print(f'{logit_attribution_path} saved')


    return 


def run_with_cache(val_dataloader, net, langs, wanted_hooks=None):
    # Evaluate accuracy (exact match) across entire validation set


    # Add retreival hooks
    hook_names = get_module_names_by_regex(net, wanted_hooks)
    cache, _ = hook_functions.add_hooks(net, mode='cache', hook_names=hook_names)
    logit = None
    for _, val_batch in enumerate(val_dataloader): # each batch
        
        out = eval_model(val_batch, net, langs)
        in_support = out['in_support']
        yq_predict = out['yq_predict']

        cache_organized = {}

        # only keep the last batch for query, not support
        ind_query = np.where(in_support==0)[0][0]
        for k,v in cache.cache.items():
            cache.cache[k] = v[ind_query, :, :].squeeze(0)

        """
        The first token is actually SOS, the last token is EOS
        """
        cache_organized['yq_predict'] = yq_predict[ind_query]


        """
        Why is support_tokens 1 longer than the input?
        Appended EOS token to the end; additionally, Padding added to shorter eposides
        """
        cache_organized['xq_context'] = val_batch['xq_context'][ind_query]
        cache_organized['xq_context'].append('EOS')


        # encoder modules
        n_head = net.nhead
        d_head = net.hidden_size // n_head
        nlayer_encoder = net.nlayers_encoder

        
        """
        # # How to split the contribution of each head
        # attn_out_cache = cache.cache['transformer.encoder.layers.0.attn_out_hook'] # seq, d_model
        # z = cache.cache['transformer.encoder.layers.0.self_attn.z_hook'] # n_head, seq, d_head
        # W_out_weight = net.transformer.encoder.layers[0].self_attn.out_proj.weight # d_model x d_model
        # W_out_bias = net.transformer.encoder.layers[0].self_attn.out_proj.bias # d_model

        # W_o_weight_split = einops.rearrange(W_out_weight, 'd_model (n_head d_head) -> d_model n_head d_head', n_head=8, d_head=16) # n_head d_head d_model
        # attn_out_split = einops.einsum(z, W_o_weight_split, 'n_head seq d_head, d_model n_head d_head -> seq n_head d_model') # seq, n_head, d_model
        # attn_out_split.sum(1)-attn_out_cache+W_out_bias == 0 # should be zero
        """
        for l in range(nlayer_encoder):
            W_out_self = net.transformer.encoder.layers[l].self_attn.out_proj.weight # d_model x d_model
            z_self = cache.cache[f'transformer.encoder.layers.{l}.self_attn.z_hook'] # n_head, seq, d_head
            W_o_self_split = einops.rearrange(W_out_self, 'd_model (n_head d_head) -> d_model n_head d_head', n_head=n_head, d_head=d_head) # n_head d_head d_model
            self_attn_out_split = einops.einsum(z_self, W_o_self_split, 'n_head seq d_head, d_model n_head d_head -> seq n_head d_model')  # seq, n_head, d_model

            cache_organized['encoder',l,'self_attn_out'] = cache.cache[f'transformer.encoder.layers.{l}.attn_out_hook'].detach().cpu().numpy()
            mlp_out = cache.cache[f'transformer.encoder.layers.{l}.mlp_out_hook']

            cache_organized['encoder', l, 'mlp_out'] = mlp_out.detach().cpu().numpy()

            for h in range(n_head):
                cache_organized['encoder', l, 'self_attn_out', h] = self_attn_out_split[:, h, :].detach().cpu().numpy()
                cache_organized['encoder', l, 'self_attn_patten', h] = cache.cache[f'transformer.encoder.layers.{l}.self_attn.attn_weight_hook'][h].detach().cpu().numpy()
        
        nlayer_decoder = net.nlayers_decoder

        for l in range(nlayer_decoder):
            W_out_self = net.transformer.decoder.layers[l].self_attn.out_proj.weight # d_model x d_model
            z_self = cache.cache[f'transformer.decoder.layers.{l}.self_attn.z_hook']
            W_o_self_split = einops.rearrange(W_out_self, 'd_model (n_head d_head) -> d_model n_head d_head', n_head=n_head, d_head=d_head)
            self_attn_out_split = einops.einsum(z_self, W_o_self_split, 'n_head seq d_head, d_model n_head d_head -> seq n_head d_model')

            W_out_cross = net.transformer.decoder.layers[l].multihead_attn.out_proj.weight # d_model x d_model
            z_cross = cache.cache[f'transformer.decoder.layers.{l}.multihead_attn.z_hook']
            W_o_cross_split = einops.rearrange(W_out_cross, 'd_model (n_head d_head) -> d_model n_head d_head', n_head=n_head, d_head=d_head)
            cross_attn_out_split = einops.einsum(z_cross, W_o_cross_split, 'n_head seq d_head, d_model n_head d_head -> seq n_head d_model')

            mlp_out = cache.cache[f'transformer.decoder.layers.{l}.mlp_out_hook']
            resid_post = cache.cache[f'transformer.decoder.layers.{l}.resid_post_hook']

            cache_organized['decoder', l, 'mlp_out'] = mlp_out.detach().cpu().numpy()
            cache_organized['decoder', l, 'resid_post'] = resid_post.detach().cpu().numpy()

            cache_organized['decoder', l, 'self_attn_out'] = cache.cache[f'transformer.decoder.layers.{l}.self_attn_out_hook'].detach().cpu().numpy()
            cache_organized['decoder', l, 'cross_attn_out'] = cache.cache[f'transformer.decoder.layers.{l}.cross_attn_out_hook'].detach().cpu().numpy()

            for h in range(n_head):
                cache_organized['decoder', l, 'self_attn_out', h] = self_attn_out_split[:, h, :].detach().cpu().numpy()
                cache_organized['decoder', l, 'cross_attn_out', h] = cross_attn_out_split[:, h, :].detach().cpu().numpy()
                cache_organized['decoder', l, 'self_attn_patten', h] = cache.cache[f'transformer.decoder.layers.{l}.self_attn.attn_weight_hook'][h].detach().cpu().numpy()
                cache_organized['decoder', l, 'cross_attn_patten', h] = cache.cache[f'transformer.decoder.layers.{l}.multihead_attn.attn_weight_hook'][h].detach().cpu().numpy()

            cache_organized['decoder_hook'] = cache.cache[f'transformer.decoder_hook'].detach().cpu().numpy()
            cache_organized['decoder_embedding'] = cache.cache[f'transformer.decoder.layers.0.resid_pre_hook'].detach().cpu().numpy()

        return logit, cache_organized


def run_with_cache_batch(val_dataloader, net, langs, wanted_hooks=None):
    # Evaluate accuracy (exact match) across entire validation set
    #
    # Input
    #   val_dataloader : 
    #   net : BIML model
    #   langs : dict of dat.Lang classes
    #   max_length : maximum length of output sequences
    #   langs : dict of dat.Lang classes
    #   eval_type : 'max' for greedy decoding, 'sample' for sample from distribution
    #   out_mask_allow : default=[]; set of emission symbols we want to allow. Default of [] allows all output emissions
    # Evaluate exact match accuracy for a given batch


    # Add retreival hooks
    wanted_hooks_list = get_module_names_by_regex(net, wanted_hooks)
    cache, _ = hook_functions.add_hooks(net, mode='cache', hook_names=wanted_hooks_list)
    cache_mean = {}
    n_batch = 0
    for _, val_batch in enumerate(val_dataloader): # each batch
        
        out = eval_model(val_batch, net, langs)
        
        # for each hook, take the mean over the batch and seq dimension
        for k,v in cache.cache.items():

            if 'attn_weight_hook' in k:
                # can't average over attention maps with different sizes
                continue

            if k not in cache_mean:
                cache_mean[k] = [v.mean(axis=(0,-2))]
            else:
                cache_mean[k].append(v.mean(axis=(0,-2)))
            print(k, v.shape, cache_mean[k][-1].shape)
        
        n_batch += 1
        print(n_batch)

    # do compressing
    for k,v in cache_mean.items():
        cache_mean[k] = np.stack(v, axis=0).mean(axis=0)
    return cache_mean


def plot_heatmap(data, xticklabels, yticklabels, cmap='coolwarm', title=None, balance=True):
    fig, ax = plt.subplots(1,1)

    data = np.flipud(data)
    yticklabels = yticklabels[::-1]
    extreme = np.max(np.abs(data))
    norm = TwoSlopeNorm(vmin=-extreme, vcenter=0, vmax=extreme)
    if balance:
        handle=ax.imshow(data, cmap=cmap, norm=norm)
    else:
        handle=ax.imshow(data, cmap=cmap)
    fig.colorbar(handle, ax=ax)  # Associate the colorbar with the Axes
    ax.set_title(title)
    ax.set_xticks(np.arange(len(xticklabels)))
    ax.set_xticklabels(xticklabels, fontsize=5, rotation=90)
    ax.set_yticks(np.arange(len(yticklabels)))
    ax.set_yticklabels(yticklabels, fontsize=5, rotation=0)
    return fig


def get_ablation_scores(val_dataloader, net, langs, save_dir,
                        null_dataset_path, rewrite, ablation_hooks=None):

    ablation_dir = save_dir / 'ablation'
    ablation_dir.mkdir(parents=True, exist_ok=True)
    ablation_path = ablation_dir / 'ablation.png'
    if ablation_path.exists() and not rewrite:
        return
    
    null_activations = torch.load(null_dataset_path)

 


    val_batch = next(iter(val_dataloader))

    # clean run
    out_clean = eval_model(val_batch, net, langs)
    # is_query = ~out_clean['in_support']
    correct_tokens = out_clean['yq_predict'][-1]


    loss_clean = out_clean['loss'][-1]

    logits_clean = out_clean['logits_correct']
    loss_diff = []
    logits_diff = []
    pred_tokens = []

    expanded_names = get_module_names_by_regex(net, ablation_hooks)

    # for each of the modules, add an ablation hook, run the model, get result, reset hook
    for name in tqdm(expanded_names, total=len(expanded_names)):
        ablate_mean = [null_activations[str(name)]]
        _, hooked_modules = hook_functions.add_hooks(net, mode='patch', hook_names=[name],
                                                        patch_activation=ablate_mean)

        out_ablate = eval_model(val_batch, net, langs)
        loss_ablate = out_ablate['loss'][-1]
        loss_diff.append((loss_ablate-loss_clean)/loss_clean)

        logits_ablate = out_ablate['logits_correct']
        logits_diff.append(logits_ablate-logits_clean)

        pred_tokens.append(out_ablate['yq_predict'])
        [hooked_module.remove_hooks() for hooked_module in hooked_modules]


    loss_diff = np.stack(loss_diff, axis=0).squeeze()
    logits_diff = np.stack(logits_diff, axis=0).squeeze()
    fig = plot_heatmap(data=loss_diff,
                       xticklabels=correct_tokens,
                       yticklabels=expanded_names,
                       cmap='coolwarm',
                       title='Ablation Scores (loss diff ratio)')

    fig.savefig(ablation_path, dpi = 400)
    print(f'{ablation_path} saved')

    return


def run_path_patching(val_dataloader, net, langs, rewrite=1,
                             null_dataset_path=None,
                             circuit: dict={},
                             save_path=None):
    
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if save_path.exists() and not rewrite:
        return

    null_activations = torch.load(null_dataset_path)
    # perma_ablate_names = get_module_names_by_regex(net, 
    #                                                [{'module':'*encoder*layer*0*z_hook*','head':'3'},
    #                                                 {'module':'*encoder*layer*0*z_hook*','head':'4'}
    #                                                 ])

    # metric = []
    # ylabels = []

    sender_hooks = circuit['sender']
    receiver_hooks = circuit['receiver']
    freeze_hooks = circuit['freeze']
    knockout_hooks = circuit.get('knockout', [])

    # list senders
    sender_names = get_module_names_by_regex(net, sender_hooks)
    receiver_names = get_module_names_by_regex(net, receiver_hooks)
    freeze_names = get_module_names_by_regex(net, freeze_hooks)
    knockout_names = get_module_names_by_regex(net, knockout_hooks)

    # clean input
    val_batch = next(iter(val_dataloader))

    #------first run------
    # forward clean run
    all_hook_names = get_module_names_by_regex(net, [{'module':'*hook*', 'head':'*'}])
    cache_clean, hooked_cache_modules = hook_functions.add_hooks(net, mode='cache', hook_names=all_hook_names)
    output_clean = eval_model(val_batch, net, langs)
    # logits_clean = output_clean['logits_correct']  
    loss_clean = output_clean['loss'] 
    [hooked_module.remove_hooks() for hooked_module in hooked_cache_modules]

    # get corrupted activation at the sender    

    #------second run------
    
    loss_diff = []
    pred_tokens_patch = []



    for i in tqdm(range(len(sender_names))):
        # patch corrupted activation at the sender
        sender_name = [sender_names[i]]
        corrupt_sender_activation = [null_activations[str(name)] for name in sender_name]
        _, sender_modules = hook_functions.add_hooks(net, mode='patch', hook_names=sender_name, 
                                                    patch_activation=corrupt_sender_activation)

        # patch clean activations to the freeze hooks
        # need to remove senders and receiver from frozen hooks
        freeze_names_exclusive = [name for name in freeze_names if name not in sender_name+receiver_names+knockout_names]
        clean_freeze_activation = [cache_clean.cache[str(name)] for name in freeze_names_exclusive]
        _, freeze_modules = hook_functions.add_hooks(net, mode='patch', hook_names=freeze_names_exclusive,
                                                    patch_activation=clean_freeze_activation)
        
        # knockout patch
        _, knockout_modules = hook_functions.add_hooks(net, mode='patch', hook_names=knockout_names, 
                                       patch_activation=[null_activations[str(name)] for name in knockout_names])
        
        # cache the receiver activations
        cache_receiver, receiver_modules = hook_functions.add_hooks(net, mode='cache', hook_names=receiver_names)

        output_patch = eval_model(val_batch, net, langs)
        loss_patch = output_patch['loss'][-1]
        [hooked_module.remove_hooks() for hooked_module in sender_modules+freeze_modules+knockout_modules+receiver_modules]
        


        #------third run------
        patched_receiver_activation = [cache_receiver.cache[str(name)] for name in receiver_names]
        # patch the receiver activations
        _, receiver_modules = hook_functions.add_hooks(net, mode='patch', hook_names=receiver_names,
                                                    patch_activation=patched_receiver_activation)
        
        # patch clean activations to the freeze hooks
        freeze_names_exclusive = [name for name in freeze_names if name not in receiver_names+knockout_names]
        clean_freeze_activation = [cache_clean.cache[str(name)] for name in freeze_names_exclusive]
        _, freeze_modules = hook_functions.add_hooks(net, mode='patch', hook_names=freeze_names_exclusive,
                                            patch_activation=clean_freeze_activation)
        
        # knockout patch
        _, knockout_modules = hook_functions.add_hooks(net, mode='patch', hook_names=knockout_names,
                                        patch_activation=[null_activations[str(name)] for name in knockout_names])
                                                       

        output_patch = eval_model(val_batch, net, langs)
        loss_patch = output_patch['loss'][-1]

        [hooked_module.remove_hooks() for hooked_module in receiver_modules+freeze_modules+knockout_modules]

        pred_tokens_patch.append(output_patch['yq_predict'])
        loss_diff.append(loss_patch-loss_clean)
        

    # logits_diff = np.stack(logits_diff, axis=0)
    loss_diff = np.stack(loss_diff, axis=0) # n_hooks n_batch n_seq

    nhead = net.nhead


    # only the first token
    loss_diff = loss_diff[:,:,0].mean(axis=1)
    print(loss_diff)
    # logits_diff = logits_diff[:,0]
    loss_diff = einops.rearrange(loss_diff, ' (n_layer n_head) -> n_layer n_head', n_head=nhead)
    # logits_diff = einops.rearrange(logits_diff, ' (n_layer n_head) -> n_layer n_head', n_head=nhead)



    ylabels = []
    for sender_name in sender_names:
        if sender_name['module'] not in ylabels:

            ylabels.append(sender_name['module'])
    
    metric=loss_diff


    xlabels = [f'Head {i}' for i in range(loss_diff.shape[1])]

    fig = plot_heatmap(data=metric,
                        xticklabels=xlabels,
                        yticklabels=ylabels,
                        cmap='coolwarm',
                        title='Path Patching Scores (loss diff)')

    fig.savefig(save_path, dpi = 400)
    print(f'{save_path} saved')






def grammar_to_dict(input_string):

    """
    Parses a string to extract mappings of words to colors and functions to outputs.
    Produces both a forward dictionary and a reverse dictionary (color to word).
    The function names will be extracted as words (e.g., 'dax', 'lug').

    Args:
        input_string (str): The input string to parse.

    Returns:
        tuple: A tuple containing:
            - forward_dict (dict): Maps words/functions to their outputs.
            - reverse_dict (dict): Maps colors to words.
    """
    # Initialize the dictionaries
    forward_dict = {}
    reverse_dict = {}
    function_dict = {}
    # Split the string by lines
    lines = input_string.strip().split('\n')

    for line in lines:
        # Match patterns for direct word-to-color mappings (e.g., "blicket -> GREEN")
        word_color_match = re.match(r"^([\w]+) -> (\w+)$", line.strip())
        if word_color_match:
            word, color = word_color_match.groups()
            forward_dict[word] = color

            # Populate the reverse dictionary
            if color not in reverse_dict:
                reverse_dict[color] = []
            reverse_dict[color]= word
            continue

        # Match patterns for function-to-output mappings (e.g., "x1 dax -> [x1] [x1] [x1]")
        function_match = re.match(r"^[\w]+ ([\w]+)(?: [\w]+)? -> (.+)$", line.strip())
        if function_match:
            function, output = function_match.groups()
            function_dict[function] = output
            continue

    return forward_dict, reverse_dict, function_dict


def load_net_and_dataset(model_path, dataset_path, null_dataset_path):
        episode_type = 'my_test'
        batch_size = -1

        seed_all()
        print('Loading model:',model_path,'on', DEVICE)
        checkpoint = torch.load(model_path, map_location=DEVICE)


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

        if 'best_val_loss' in checkpoint: best_val_loss = checkpoint['best_val_loss']

        _,val_dataset = dat.get_dataset(episode_type, **{'case': dataset_path})
        langs = val_dataset.langs

        # Load model parameters         
        net = HookedBIML(emb_size, input_size, output_size,
                langs['input'].PAD_idx, langs['output'].PAD_idx,
                nlayers_encoder=nlayers_encoder, nlayers_decoder=nlayers_decoder, 
                dropout_p=dropout_p, activation=myact, ff_mult=ff_mult)        
        net.load_state_dict(nets_state_dict)
        net = net.to(device=DEVICE)
        describe_model(net)

        val_dataloader = DataLoader(val_dataset, batch_size=batch_size,
                        collate_fn=lambda x:dat.make_biml_batch(x,langs),shuffle=False)
        dataset = {
                'dataloader': val_dataloader,
                'langs': langs,
                'null_dataset_path': null_dataset_path
        }

        return net, dataset
                
