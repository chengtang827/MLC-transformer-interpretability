from MI import hook_functions
import MLC_datasets as dat
import numpy as np
from train_lib import seed_all, extract, display_input_output, assert_consist_langs
import torch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import einops
linear = torch._C._nn.linear
from eval_org import evaluate_acc
import matplotlib.pyplot as plt


def plot_attention_patterns(net, tokens, cache):
    n_head = net.nhead
    nlayers_encoder = net.nlayers_encoder
    nlayers_decoder = net.nlayers_decoder

    # do encoder self attention
    """
    Why is support_tokens 1 longer than the input?
    Appended EOS token to the end; additionally, Padding added to shorter eposides
    """
    support_tokens = tokens['xq_context']
    support_tokens.append('EOS')
    """
    The first token is actually SOS, the last token is EOS
    """
    pred_tokens = tokens['yq_predict']
    pred_tokens.insert(0, 'SOS')
    pred_tokens.append('EOS')

    do_encoder_self_attn=1
    if do_encoder_self_attn:
        fig, ax= plt.subplots(nlayers_encoder, n_head, figsize=(3.2*n_head, 3.2*nlayers_encoder),constrained_layout = True)
        for l in range(nlayers_encoder):
            for h in range(n_head):
                ax_i = ax[nlayers_encoder-l-1,h] # reverse the order
                ax_i.imshow(cache['encoder', l, 'self_attn_patten', h][:len(support_tokens), :len(support_tokens)])
                ax_i.set_title(f'Encoder L {l} H {h}')
                ax_i.set_yticks(list(range(len(support_tokens))))
                ax_i.set_xticks(list(range(len(support_tokens))))
                ax_i.set_yticklabels(support_tokens, fontsize=5)
                ax_i.set_xticklabels(support_tokens, rotation =90, fontsize=5)
        plt.show()
    
    do_decoder_cross_attn=1
    if do_decoder_cross_attn:
        fig, ax= plt.subplots(nlayers_decoder, n_head, figsize=(3.2*n_head, 3.2*nlayers_decoder),constrained_layout = True)
        for l in range(nlayers_decoder):
            for h in range(n_head):
                ax_i = ax[nlayers_decoder-l-1,h]
                ax_i.imshow(cache['decoder', l, 'cross_attn_patten', h][:len(pred_tokens), :len(support_tokens)])
                ax_i.set_title(f'Decoder L {l} H {h}')
                ax_i.set_yticks(list(range(len(pred_tokens))))
                ax_i.set_xticks(list(range(len(support_tokens))))
                ax_i.set_yticklabels(pred_tokens, fontsize=5)
                ax_i.set_xticklabels(support_tokens, rotation =90, fontsize=5)
        plt.show()

    do_decoder_self_attn=1
    if do_decoder_self_attn:
        fig, ax= plt.subplots(nlayers_decoder, n_head, figsize=(3.2*n_head, 3.2*nlayers_decoder),constrained_layout = True)
        for l in range(nlayers_decoder):
            for h in range(n_head):
                ax_i = ax[nlayers_decoder-l-1,h]
                ax_i.imshow(cache['decoder', l, 'self_attn_patten', h][:len(pred_tokens), :len(pred_tokens)])
                ax_i.set_title(f'Decoder L {l} H {h}')
                ax_i.set_yticks(list(range(len(pred_tokens))))
                ax_i.set_xticks(list(range(len(pred_tokens))))
                ax_i.set_yticklabels(pred_tokens, fontsize=5)
                ax_i.set_xticklabels(pred_tokens, rotation =90, fontsize=5)
        plt.show()



def logit_attribution(net, cache, langs=None):
    # val_dataloader = DataLoader(val_episode,batch_size=batch_size,
    #                             collate_fn=lambda x:dat.make_biml_batch(x,langs),shuffle=False)


    # define the correct token and the position (say the first token)
    pos = 0
    symbol = 'PURPLE'
    symbol_id = [i for i,v in langs['output'].index2symbol.items() if v == symbol][0]

    # center the W_u matrix
    unembed_mat = net.out.weight.detach().cpu().numpy() # n_vocab x hidden_size
    unembed_mat = unembed_mat - np.mean(unembed_mat, axis=0)

    # find out the direction for correct token
    token_direction = unembed_mat[symbol_id, :]    


    # find and apply the final layer norm to the stacked activations
    # TODO the ln is applied separately 
    ln = net.norm
    for k,v in cache.items():
        if 'out' in k:
            cache[k] = ln(v)

    # do the dot product
    logit_attribution = {}
    for k,v in cache.items():
        logit_attribution[k]=np.dot(v[pos,:], token_direction)

    # TODO visualize attn patterns
    # TODO visualize the logit attribution
    # TODO zeros ablation
    # need a working version of model
    return 


def run_with_cache(val_dataloader, net, langs, max_length, eval_type='max', hook_names=None):
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

    assert eval_type in ['max','sample']

    # Add retreival hooks
    cache = hook_functions.add_hooks(net, mode='cache', hooks_named=hook_names)
    logit = None
    for _, val_batch in enumerate(val_dataloader): # each batch
        val_batch = dat.set_batch_to_device(val_batch)

        out_mask_allow = dat.get_batch_output_pool(val_batch)

        net.eval()
        emission_lang = langs['output']
        use_mask = len(out_mask_allow)>0



        memory, memory_padding_mask = net.encode(val_batch) 
            # memory : b*nq x maxlength_src x hidden_size
            # memory_padding_mask : b*nq x maxlength_src (False means leave alone)
        m = len(val_batch['yq']) # b*nq
        z_padded = torch.tensor([emission_lang.symbol2index[dat.SOS_token]]*m) # b*nq length tensor
        z_padded = z_padded.unsqueeze(1) # [b*nq x 1] tensor
        z_padded = z_padded.to(device=DEVICE)
        max_length_target = val_batch['yq_padded'].shape[1]-1 # length without EOS
        assert max_length >= max_length_target # make sure that the net can generate targets of the proper length

        # make the output mask if certain emissions are restricted
        if use_mask:
            assert dat.EOS_token in out_mask_allow # EOS must be included as an allowed symbol
            additive_out_mask = -torch.inf * torch.ones((m,net.output_size), dtype=torch.float)
            additive_out_mask = additive_out_mask.to(device=DEVICE)
            for s in out_mask_allow:
                sidx = langs['output'].symbol2index[s]
                additive_out_mask[:,sidx] = 0.

        # Run through decoder
        all_decoder_outputs = torch.zeros((m, max_length), dtype=torch.long)
        all_decoder_outputs = all_decoder_outputs.to(device=DEVICE)
        for t in range(max_length):
            decoder_output = net.decode(z_padded, memory, memory_padding_mask)
                # decoder_output is b*nq x (t+1) x output_size
            decoder_output = decoder_output[:,-1] # get the last step's output (batch_size x output_size)
            if use_mask: decoder_output += additive_out_mask
            # Choose the symbols at next timestep
            if eval_type == 'max': # pick the most likely
                topi = torch.argmax(decoder_output,dim=1)
                emissions = topi.view(-1)

            all_decoder_outputs[:,t] = emissions
            z_padded = torch.cat([z_padded, emissions.unsqueeze(1)], dim=1)

        # Get predictions as strings and see if they are correct
        all_decoder_outputs = all_decoder_outputs.detach()
        yq_predict = [] # list of all predicted query outputs as strings
        yq_predict_cont = [] #list of all predicted query outputs with EOS tokens -- ADDED CODE
        v_acc = np.zeros(m)
        for q in range(m):
            #print(all_decoder_outputs[q,:].view(-1))
            myseq = emission_lang.tensor_to_symbols(all_decoder_outputs[q,:].view(-1))
            cont = emission_lang.tensor_to_symbols(all_decoder_outputs[q,:].view(-1), break_=False) #ADDED CODE
            yq_predict.append(myseq)
            yq_predict_cont.append(cont) #ADDED CODE
            v_acc[q] = yq_predict[q] == val_batch['yq'][q] # for each query, did model get it right?
        in_support = np.array(val_batch['in_support']) # which queries are also support items
        out = {'yq_predict':yq_predict, 'v_acc':v_acc, 'in_support':in_support, 
                'yq_predict_cont': yq_predict_cont, 'xq_context': val_batch['xq_context']} #ADDED CODE: last two keys
        

        # only keep the last batch for query, not support
        ind_query = np.where(in_support==0)[0][0]
        for k,v in cache.cache.items():
            cache.cache[k] = v[ind_query, :, :].squeeze(0)

        tokens = {}
        tokens['yq_predict'] = yq_predict[ind_query]
        tokens['xq_context'] = val_batch['xq_context'][ind_query]


        # organize the cache into 
        # TODO: L0 self H0, L0 cross H0, L0 MLP... 

        cache_organized = {}
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

            cache_organized['decoder', l, 'mlp_out'] = mlp_out.detach().cpu().numpy()

            for h in range(n_head):
                cache_organized['decoder', l, 'self_attn_out', h] = self_attn_out_split[:, h, :].detach().cpu().numpy()
                cache_organized['decoder', l, 'cross_attn_out', h] = cross_attn_out_split[:, h, :].detach().cpu().numpy()
                cache_organized['decoder', l, 'self_attn_patten', h] = cache.cache[f'transformer.decoder.layers.{l}.self_attn.attn_weight_hook'][h].detach().cpu().numpy()
                cache_organized['decoder', l, 'cross_attn_patten', h] = cache.cache[f'transformer.decoder.layers.{l}.multihead_attn.attn_weight_hook'][h].detach().cpu().numpy()


        return logit, tokens, cache_organized

